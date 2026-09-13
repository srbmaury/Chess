# Player Profiles and Community Username Isolation

## Goal

Turn the current single-user local Chess ML Coach into a multi-profile local application suitable for sharing with the chess community while preserving the existing `srbmaury` analysis, model artifacts, puzzle bank, explanation cache, review history, and progress.

The core requirement is strict local isolation: selecting or adding another Chess.com username must never read, overwrite, reuse, retire, retrain, or mutate another player's artifacts.

## Current problem

Today `Settings` points every command and the web app at global roots such as `data/raw`, `data/processed`, `data/engine`, `data/training`, and `models/`. The web app captures one `Settings` instance when it starts, and the pipeline manager also owns one fixed settings object.

That makes `--username` change the Chess.com identity while still sharing artifact paths. A second username can therefore collide with the first user's normalized games, Stockfish cache, model, puzzles, explanation cache, and review history.

## Chosen architecture

Introduce a profile layer between the root configuration and all user-specific artifacts.

Root directories remain configurable:

```text
data/
models/
```

Each Chess.com username resolves to its own safe namespace:

```text
data/
  users/
    srbmaury/
      raw/
      processed/
      engine/
      training/training.db
    another_user/
      raw/
      processed/
      engine/
      training/training.db

models/
  users/
    srbmaury/
      mistake_model.joblib
      mistake_model.metadata.json
    another_user/
      mistake_model.joblib
      mistake_model.metadata.json
```

The application continues to pass an ordinary `Settings` object to existing pipeline functions. A new profile resolver constructs that settings object with profile-scoped `data_dir` and `model_dir`, so sync/analyze/features/puzzles/train/report/practice/explanations do not need duplicate implementations.

## Username identity and path safety

The Chess.com username remains the logical identity and is preserved for API calls and report labels.

Filesystem paths use a canonical profile key derived from the username:

- trim surrounding whitespace;
- lowercase for profile identity because Chess.com usernames are treated case-insensitively by this application;
- require at least three characters;
- permit only letters, numbers, underscores (`_`), and dashes (`-`);
- require the first and last character to be a letter or number;
- reject usernames consisting only of numbers;
- reject spaces, slashes, backslashes, control characters, and path-traversal values;
- never concatenate an unchecked username directly into a filesystem path.

These validation rules mirror the current published Chess.com username syntax. Two inputs differing only by case map to the same local profile.

## Preserving the existing `srbmaury` workspace

Migration must preserve current progress before any new username can become active.

Legacy artifacts currently live directly under:

```text
data/raw
data/processed
data/engine
data/training
models/mistake_model.*
```

On first profile-aware startup/command, the migration layer detects legacy artifacts. If the destination profile does not already contain conflicting artifacts, it migrates the legacy workspace into the `srbmaury` profile namespace.

Migration rules:

1. Default legacy owner is the current configured/default username (`srbmaury` unless explicitly overridden for migration tests).
2. Migration is idempotent and resumable. Re-running after a partial move must continue safely.
3. Existing destination files are never silently overwritten.
4. `training.db` moves with its puzzle reviews and `puzzle_explanations` table intact.
5. Existing Stockfish `analysis.parquet`, normalized parquet files, report, raw PGNs, manifest, and trained model files are reused; migration does not trigger recomputation.
6. A small migration marker records successful completion.
7. If both legacy and destination copies contain conflicting data, stop with a clear error rather than guessing which is authoritative.

This means pulling the profile-aware release and launching the app should preserve the user's current progress automatically.

## Profile registry

Add a lightweight local registry under the root data directory, for example:

```text
data/profiles.json
```

It stores only non-sensitive local profile metadata:

- canonical username;
- display username;
- created timestamp;
- last-used timestamp;
- active username.

Actual games, analysis, models, puzzles, and reviews remain inside profile directories. The registry is local and gitignored with `data/`.

Profiles can also be discovered from `data/users/*` to recover gracefully if the registry is deleted while profile directories remain.

## Backend profile manager

Add a `ProfileManager` responsible for:

- validation/canonicalization;
- legacy migration;
- listing known profiles;
- creating a profile namespace;
- reading and updating the active profile;
- producing profile-scoped `Settings` from root settings;
- preventing a profile switch while a pipeline operation is running.

The manager is the only code allowed to map usernames to storage paths.

`Settings` itself remains the execution configuration used by the existing services. Root UI settings are not directly passed into player-specific services.

## CLI behavior

All existing commands keep supporting `--username`.

With profile support:

```bash
chess-coach sync --username hikaru
```

resolves to the `hikaru` profile automatically, so all later commands using the same username operate in that user's namespace.

For backward compatibility, omitting `--username` continues to use the configured/default username.

Explicit `--data-dir` and `--model-dir` continue to mean root directories; profile scoping is applied beneath them. This prevents two usernames passed with the same custom root from colliding.

No command should require users to manually build `data/users/<username>` paths.

## Web API

Add profile endpoints:

```text
GET  /api/profiles
POST /api/profiles
POST /api/profiles/{username}/activate
```

`GET /api/profiles` returns the active username and known local profiles with lightweight readiness/progress summaries.

`POST /api/profiles` accepts a Chess.com username, validates it, creates the isolated namespace/registry entry, and optionally makes it active. It does not need to download games immediately; the Pipeline screen remains responsible for Sync and later stages.

`POST /api/profiles/{username}/activate` switches the local UI context only when no pipeline job is running.

After activation, all existing endpoints (`dashboard`, `practice`, `puzzles`, `progress`, `explanation`, pipeline actions) resolve settings from the active profile at request time rather than from a fixed startup `Settings` closure.

The health endpoint reports the active username and scoped directories.

## Pipeline manager

There remains one pipeline worker per local application process.

The pipeline manager gains a safe rebind operation or equivalent context lookup so:

- each job snapshots the active profile settings at start;
- switching profiles while a job is running returns HTTP 409;
- a job continues against the profile it started with even if the UI later changes after completion;
- progress events/status identify the username/profile they belong to;
- a job for one player can never write into another player's directories.

## Web UI

Add a profile selector to the application shell, visible on every screen.

Primary flows:

### First/community user

```text
Chess.com username [____________]
Continue
```

Creating/activating the username opens that player's dashboard. If they have no artifacts yet, the UI clearly guides them to Pipeline → Sync → Analyze → Features → Puzzles.

### Existing user

The shell displays the active player prominently, for example:

```text
Player: srbmaury ▾
```

The menu supports:

- switching among local profiles;
- adding another Chess.com username.

Switching profile refreshes Dashboard/Practice/Mistakes/Progress/Pipeline data and clears stale page state so another player's puzzle or feedback cannot remain visible.

The UI must never expose best moves, puzzle records, progress, or pipeline status from the previously active player after a switch.

## Community/local privacy model

This remains a local-first app. Multiple profiles on one machine are local namespaces, not authenticated multi-user accounts.

There is still no public-server authentication model. The app remains bound to localhost by default. Sharing the repository with the community means each person runs their own local instance and enters their own Chess.com username.

A future hosted SaaS version would require real accounts, server-side authorization, remote storage isolation, quotas, job scheduling, and secrets management; that is explicitly out of scope.

## Isolation invariants

Tests must enforce these invariants:

1. Different usernames resolve to different data/model roots.
2. Case variants of one username resolve to the same profile.
3. Unsafe/path-traversal usernames are rejected.
4. Legacy `srbmaury` artifacts migrate without losing SQLite review/explanation history.
5. Migration is idempotent.
6. Conflicting legacy/destination artifacts fail safely.
7. Sync/analyze/features/puzzles/train/report for profile B cannot modify profile A files.
8. Web profile activation changes all subsequent API reads to the selected profile.
9. Profile switch is rejected while a pipeline job is running.
10. Pipeline events/results are tagged with the profile that started the job.
11. Practice/explanation state never leaks across profile switches.
12. Existing single-user CLI behavior remains compatible when no username is supplied.

## Testing strategy

Implementation is test-first.

Python tests cover profile paths, validation, migration, service/CLI scoping, API activation, pipeline isolation, and preservation of `training.db` history.

Frontend tests cover first-run username entry, add/switch player, active-player labeling, profile refresh, and clearing stale Practice feedback/explanations when switching.

CI continues to require:

```bash
ruff check src tests
pytest --cov=chess_ml_coach --cov-report=term-missing
cd web
npm install
npm run test:run
npm run build
```

## Rollout

After merge, the existing user should:

```bash
git pull
pip install -e '.[dev]'
cd web && npm install && npm run build && cd ..
chess-coach ui
```

On first launch the app performs the safe legacy migration for `srbmaury`. No full Stockfish reanalysis, feature rebuild, model retraining, or puzzle rebuild is required solely because of the storage migration.

After confirming the migrated profile works, a second username can be created from the same UI and used as the isolation test. A second repository clone should no longer be necessary.
