# Chess ML Coach

Turn Chess.com game history into a personalized training plan.

Chess ML Coach is a local-first chess improvement tool that combines:

- **Stockfish** for objective position evaluation, best moves, principal variations, and centipawn loss;
- **LightGBM** for learning the kinds of positions in which a specific player is historically most likely to make a significant mistake;
- a **personal puzzle trainer** generated from that player's own mistakes;
- **spaced repetition** and persistent mastery/progress tracking;
- an interactive **FastAPI + React** web UI;
- engine-grounded **"Why is this best?"** explanations after a puzzle attempt.

It does not train a chess engine from a player's games and does not replace Stockfish. The personalization layer learns *where a player struggles*; Stockfish remains the chess authority.

## How it works

```text
Chess.com PubAPI
      |
      v
sync -> PGN corpus -> Stockfish analysis -> features -> LightGBM -> coaching report
                                          |
                                          v
                                  personal puzzles
                                          |
                                          v
                              spaced-repetition practice
                                          |
                                          v
                                   progress tracking
```

Every Chess.com username has an isolated local profile, so multiple players can use the same installation without sharing games, Stockfish caches, models, puzzles, explanations, or review history.

## Requirements

- Python 3.11+
- Stockfish installed locally for analysis/explanations
- Internet access for Chess.com sync
- Node.js 22+ to build the web UI

Chess.com ingestion uses the public, read-only PubAPI.

## Setup

```bash
git clone https://github.com/srbmaury/Chess.git
cd Chess
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Configure Stockfish:

```bash
export STOCKFISH_PATH=/absolute/path/to/stockfish
```

On Windows PowerShell:

```powershell
$env:STOCKFISH_PATH = "C:\path\to\stockfish.exe"
```

You can also pass `--stockfish-path` to `chess-coach analyze`.

## Local web UI

Build the frontend once:

```bash
cd web
npm install
npm run build
cd ..
```

Start the application:

```bash
chess-coach ui
```

It binds to `127.0.0.1:8000` by default and opens the browser. To avoid opening a browser automatically:

```bash
chess-coach ui --no-open
```

The web UI contains:

- **Player controls** — enter a Chess.com username, add another player, or switch between local profiles;
- **Dashboard** — analyzed moves, due/mastered puzzles, review accuracy, and artifact readiness;
- **Practice** — interactive chessboard puzzles generated from the active player's own mistakes;
- **Why is this best?** — after an attempt, lazily analyze one position and explain the best move using Stockfish's line plus deterministic board facts;
- **Mistakes** — browse active mistake/blunder positions and evaluation losses;
- **Progress** — review activity, mastery, motif accuracy, and opening accuracy;
- **Pipeline** — Sync, Analyze, Features, Puzzles, Train, and Report with live progress.

### Multiple players

Each username receives independent storage. Adding or switching a player never reuses another player's analysis or training history.

Example:

```text
data/
  users/
    alice/
      raw/
      processed/
      engine/
      training/training.db
    bob/
      raw/
      processed/
      engine/
      training/training.db

models/
  users/
    alice/
      mistake_model.joblib
      mistake_model.metadata.json
    bob/
      mistake_model.joblib
      mistake_model.metadata.json
```

The active profile is recorded locally in `data/profiles.json`. Usernames are validated and canonicalized before being used for filesystem paths.

A second repository clone is **not required** to test another Chess.com account. Use **Add player** / **Switch player** in the UI, or pass `--username` to CLI commands.

### Permanently delete a player's local data

Stop the web UI and any pipeline work first. Then delete one player's raw games,
processed datasets, Stockfish cache, puzzle/review/explanation history, trained
model, and local profile metadata:

```bash
chess-coach delete-user-data --username alice
```

The interactive command requires typing the canonical username exactly. For
non-interactive local scripts, explicitly bypass the prompt:

```bash
chess-coach delete-user-data --username alice --yes
```

Deletion is limited to that username's isolated directories. It refuses unknown
profiles and profiles with an active Stockfish analysis lock. This removes only
Chess ML Coach's local copies; it does not delete public games from Chess.com.

### Existing single-user installations

Older versions stored artifacts directly under:

```text
data/raw
data/processed
data/engine
data/training
models/mistake_model.*
```

On the first profile-aware command/startup, existing legacy artifacts are migrated to the configured/default player's profile (normally `srbmaury`) without rerunning Stockfish or retraining the model.

Migration is conservative:

- files are moved byte-for-byte, including `training.db` and its review/explanation history;
- already-migrated identical files are accepted;
- conflicting legacy/destination files stop migration with an error instead of being overwritten;
- the operation is safe to rerun after a partial migration.

### Stopping a long Stockfish analysis

While **Analyze** is running, the UI exposes **Stop analysis**. Cancellation is cooperative: the worker stops at a safe progress boundary rather than killing the process during a write.

Already completed analysis rows remain cached. Starting Analyze again resumes from the persisted cache instead of throwing away completed work.

Player switching is disabled while a pipeline job is running or stopping so a job cannot cross profile boundaries.

## CLI workflow

All CLI pipeline commands are profile-aware. `--data-dir` and `--model-dir` remain *root* directories; username scoping is applied beneath them automatically.

For a particular Chess.com account:

```bash
chess-coach sync --username alice
chess-coach analyze --username alice
chess-coach features --username alice
chess-coach puzzles --username alice
chess-coach train --username alice
chess-coach report --username alice
```

Omitting `--username` uses the configured/default player.

### 1. Sync games

```bash
chess-coach sync
```

Sync is idempotent: known games are deduplicated, new games are added, and unavailable monthly Chess.com archives are skipped safely.

### 2. Analyze moves

```bash
chess-coach analyze
```

Optional controls:

```bash
chess-coach analyze \
  --stockfish-path /absolute/path/to/stockfish \
  --depth 14
```

Depth 14 is the default. Analysis is resumable, reports reused vs newly analyzed moves, and uses a single-writer guard for each profile's analysis cache.

### 3. Build features

```bash
chess-coach features
```

Features include position complexity, material, pawn structure, king-safety proxies, opening metadata, phase, rating difference, time-control category, and the pre-move engine evaluation.

Post-move engine evaluation, centipawn loss, and final game result are excluded from predictive features to avoid target leakage.

### 4. Train the personalized model

```bash
chess-coach train
```

The model estimates:

```text
P(significant mistake | position + this player's historical patterns)
```

The default significant-mistake target is CPL >= 100. Games are split chronologically at whole-game level to prevent train/test leakage between moves from one game.

### 5. Generate the coaching report

```bash
chess-coach report
```

The report highlights recurring weak contexts by color, opening, phase, time control, and model-derived risk signals, with sample-size guards for grouped statistics.

## Personal puzzle training

Build or refresh the player's puzzle bank:

```bash
chess-coach puzzles
```

Puzzle generation reuses existing engine analysis and does not launch Stockfish. Rebuilding is idempotent: attempts, streaks, due dates, mastery, reviews, and cached explanations are preserved. Positions that are no longer eligible are retired from active practice without deleting their history.

Practice due puzzles:

```bash
chess-coach practice --limit 10
```

The spaced-repetition schedule is intentionally simple:

| Result / streak | Next review |
| --- | ---: |
| Incorrect | 1 day |
| 1 correct in a row | 3 days |
| 2 correct in a row | 7 days |
| 3 correct in a row | 14 days |
| 4+ correct in a row | 30 days |

A puzzle is considered mastered after four consecutive correct reviews. A later failure resets the streak.

Track progress:

```bash
chess-coach progress
```

## Why is this best?

The practice UI keeps the answer hidden until the player attempts the puzzle. After the attempt, **Why is this best?** lazily requests an explanation.

The first request for a position/depth may run Stockfish on that single position to obtain a short principal variation. Python then derives concrete board facts and produces a deterministic explanation. The engine-grounded result is cached in the player's own `training.db` for later requests.

If Stockfish is unavailable, practice still works and the explanation falls back to clearly labeled board-only facts. If a fresh engine run disagrees with the puzzle's stored best move, the app does not attach an unrelated principal variation; it asks the user to refresh the analysis pipeline instead.

## Suggested routine

```bash
# After playing new games
chess-coach sync
chess-coach analyze
chess-coach features
chess-coach puzzles

# Most days
chess-coach practice --limit 10
chess-coach progress

# Periodically
chess-coach train
chess-coach report
```

The expensive step is Stockfish analysis. At the same engine configuration, later runs mainly analyze newly synced moves.

## Configuration

Precedence is:

1. CLI option
2. environment variable
3. project default

Useful environment variables:

```text
CHESS_COACH_USERNAME
CHESS_COACH_DATA_DIR
CHESS_COACH_MODEL_DIR
STOCKFISH_PATH
CHESS_COACH_STOCKFISH_DEPTH
CHESS_COACH_MIN_GROUP_SIZE
CHESS_COACH_INACCURACY_CPL
CHESS_COACH_MISTAKE_CPL
CHESS_COACH_BLUNDER_CPL
```

Default CPL boundaries:

| Label | CPL |
| --- | ---: |
| Good | `< 50` |
| Inaccuracy | `50–99` |
| Mistake | `100–199` |
| Blunder | `>= 200` |

## Local-only privacy model

Personal data, PGNs, engine analysis, puzzle/review history, explanation cache, trained models, frontend dependencies/build output, and `.env` files are excluded from Git.

The application is intentionally local-first and binds to localhost by default. Local player profiles are filesystem namespaces, **not authenticated server accounts**. Do not expose the current server publicly or bind it to an external interface without adding authentication, authorization, remote-storage isolation, quotas, job scheduling, and a deployment security design.

## Frontend development

Build once, then run:

```bash
# terminal 1
chess-coach ui --no-open

# terminal 2
cd web
npm run dev
```

Vite proxies `/api` to the local FastAPI process. Development CORS accepts only local Vite origins.

## Development checks

Python:

```bash
ruff check src tests
pytest --cov=chess_ml_coach --cov-report=term-missing
```

Frontend:

```bash
cd web
npm install
npm run test:run
npm run build
```

Tests mock network and engine boundaries, so CI does not need a Chess.com account, live PubAPI calls, or a Stockfish binary.

## Next improvements

High-value follow-ups include:

- multi-ply puzzle continuations instead of validating only the first best move;
- stronger tactical-motif classification;
- rolling 7/30/90-day improvement tracking;
- SHAP explanations for the personalized LightGBM risk model;
- richer board arrows, hints, and variation exploration;
- a separate authenticated architecture if this becomes a hosted service rather than a local community tool.
