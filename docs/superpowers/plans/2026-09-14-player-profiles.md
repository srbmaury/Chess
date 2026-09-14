# Player Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Isolate every user's games, analysis, models, puzzles, explanation cache, and review history by Chess.com username while preserving the existing `srbmaury` workspace.

**Architecture:** Add a `ProfileManager` as the only username-to-path mapper. Existing pipeline functions keep consuming ordinary `Settings`; profile-scoped settings point at `data/users/<key>` and `models/users/<key>`. The web app resolves the active profile at request time, and every pipeline job snapshots one profile before execution.

**Tech Stack:** Python 3.11, pathlib/json/shutil, FastAPI, Typer, SQLite, React/TypeScript/Vite, pytest, Vitest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-14-player-profiles-design.md`

## Global Constraints

- Keep `data/` and `models/` as configurable roots.
- Canonicalize profile identity case-insensitively and validate usernames before path construction.
- Existing legacy artifacts migrate to `srbmaury` without recomputation.
- Migration is idempotent and never overwrites conflicting destination content.
- Profile switching is rejected while a pipeline job is running.
- Existing CLI `--username` remains supported and becomes storage-isolated.
- Local-only security model remains unchanged.

---

## Task 1: Profile identity, registry, and migration

**Files**
- Create `src/chess_ml_coach/profiles.py`
- Create `tests/test_profiles.py`
- Modify `src/chess_ml_coach/config.py` only if a root/scoped distinction is needed

**Produces**
- `canonicalize_username(username: str) -> str`
- `ProfileRecord`
- `ProfileManager.settings_for(username: str) -> Settings`
- `ProfileManager.list_profiles()`
- `ProfileManager.active_username()`
- `ProfileManager.create_or_activate(username, activate=True)`
- `ProfileManager.activate(username)`
- `ProfileManager.migrate_legacy(owner_username)`

- [ ] Write RED tests proving two usernames resolve to different data/model roots, case variants resolve to one profile, invalid/path-like usernames are rejected, and profile paths are always beneath the configured roots.
- [ ] Implement canonicalization and scoped settings.
- [ ] Write RED migration tests that create legacy raw/processed/engine/training/model artifacts, including a byte-for-byte `training.db`, then assert migration preserves all bytes and is idempotent.
- [ ] Implement `data/profiles.json` registry and safe migration. Existing destination files with identical bytes are accepted; differing files raise `ProfileMigrationConflictError`; a migration marker is written only after successful reconciliation.
- [ ] Run `pytest tests/test_profiles.py -q` and make it green.
- [ ] Commit `feat: add isolated player profiles`.

## Task 2: Scope CLI and services by profile

**Files**
- Modify `src/chess_ml_coach/cli.py`
- Modify `src/chess_ml_coach/services.py` only where path assumptions need clarification
- Modify `tests/test_cli.py`
- Create `tests/test_profile_services.py`

**Produces**
- One central CLI helper that starts from root configuration, performs legacy migration once, selects username/default profile, and returns profile-scoped `Settings`.

- [ ] Write RED tests proving `--username beta_user --data-dir ROOT --model-dir MODELS` writes below `ROOT/users/beta_user` and `MODELS/users/beta_user`, while the default username resolves to its own profile.
- [ ] Implement central profile resolution; apply engine depth/threshold overrides with `dataclasses.replace` after scoping so path isolation cannot be bypassed.
- [ ] Run `pytest tests/test_profile_services.py tests/test_cli.py -q` and make it green.
- [ ] Commit `feat: scope CLI pipelines by player`.

## Task 3: Profile-aware FastAPI and pipeline isolation

**Files**
- Create `src/chess_ml_coach/web/profile_routes.py`
- Modify `src/chess_ml_coach/web/app.py`
- Modify `src/chess_ml_coach/web/pipeline.py`
- Modify `src/chess_ml_coach/web/explanation_routes.py`
- Modify `src/chess_ml_coach/web/schemas.py`
- Create `tests/test_web_profiles.py`
- Modify `tests/test_web_pipeline.py`, `tests/test_web_api.py`, and explanation API tests

**API**
- `GET /api/profiles`
- `POST /api/profiles` with username and optional activate flag
- `POST /api/profiles/{username}/activate`

- [ ] Write RED API tests proving activation changes health/dashboard/practice storage, profile A puzzles cannot be read after switching to profile B, and activation returns 409 during a running pipeline job.
- [ ] Store root settings and `ProfileManager` on app state; resolve active profile settings inside each request rather than closing over startup settings.
- [ ] Change pipeline start so each job receives/snapshots explicit profile settings and emits username in status/progress/terminal events.
- [ ] Update explanation routing to use active profile settings at request time.
- [ ] Run the targeted web tests and make them green.
- [ ] Commit `feat: add profile-aware web API`.

## Task 4: React add/switch-player UI

**Files**
- Modify `web/src/App.tsx`
- Modify `web/src/App.test.tsx`
- Modify the existing web stylesheet as needed

- [ ] Write RED tests for first-run username entry, active-player label, Add player, switching between existing players, and clearing old puzzle feedback/explanation after a switch.
- [ ] Add typed API calls for list/create/activate profile.
- [ ] Add an always-visible player selector in the shell and an Add player flow.
- [ ] On activation, increment a profile revision and remount the routed page subtree so no stale puzzle/best-move/explanation state survives.
- [ ] Run `npm run test:run` and `npm run build` and make both green.
- [ ] Commit `feat: add player switcher to web UI`.

## Task 5: Docs, full isolation review, and merge

**Files**
- Modify `README.md`
- Update tests only for issues discovered during final review

- [ ] Document automatic legacy migration, per-player storage paths, and the community workflow.
- [ ] State explicitly that a second clone is no longer necessary after migration is verified.
- [ ] Run targeted migration/isolation tests.
- [ ] Run full Python gate: `ruff check src tests` then `pytest --cov=chess_ml_coach --cov-report=term-missing`.
- [ ] Run full frontend gate: `npm install`, `npm run test:run`, `npm run build`.
- [ ] Review the final diff against all isolation invariants: no unchecked username path construction, no stale startup profile settings, no cross-profile pipeline writes, no conflicting migration overwrite.
- [ ] Mark the PR ready and squash-merge only the verified head SHA.
