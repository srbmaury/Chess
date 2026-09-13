# Chess ML Coach

A private, personalized chess-improvement pipeline for Chess.com user `srbmaury`.

The project deliberately separates **chess strength** from **personalization**:

- **Stockfish** evaluates positions, finds the best move, and measures centipawn loss (CPL).
- **LightGBM** learns which kinds of positions are risky *for you* based on your historical decisions.
- The coaching report turns those signals into recurring weakness contexts.
- The **personal puzzle trainer** turns genuine mistakes from your own games into spaced-repetition practice.

It does **not** try to train a chess engine from your games or replace Stockfish.

## How it works

```text
Chess.com PubAPI
      |
      v
sync -> PGN corpus -> Stockfish analysis -> features -> LightGBM -> report
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

Core analysis pipeline:

```bash
chess-coach sync
chess-coach analyze
chess-coach features
chess-coach train
chess-coach report
```

Training loop:

```bash
chess-coach puzzles
chess-coach practice --limit 10
chess-coach progress
```

Each stage writes local artifacts. Once Stockfish has analyzed a move at the current configuration, downstream puzzle generation and practice reuse that analysis and do **not** run Stockfish again.

## Requirements

- Python 3.11+
- Stockfish installed locally for the `analyze` stage
- Internet access for `sync`

Chess.com game ingestion uses the public, read-only PubAPI and performs monthly archive requests sequentially.

## Setup

```bash
git clone https://github.com/srbmaury/Chess.git
cd Chess
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install Stockfish using your OS package manager or from the official Stockfish distribution, then configure its executable path:

```bash
export STOCKFISH_PATH=/absolute/path/to/stockfish
```

On Windows PowerShell:

```powershell
$env:STOCKFISH_PATH = "C:\path\to\stockfish.exe"
```

You can also pass `--stockfish-path` directly to `chess-coach analyze`.

## Run the analysis pipeline

### 1. Download Chess.com games

```bash
chess-coach sync
```

The default username is `srbmaury`. To use another account:

```bash
chess-coach sync --username another_user
```

Sync is idempotent: already-known games are deduplicated, newly published games are added, and unavailable monthly archives are skipped safely.

### 2. Analyze your moves with Stockfish

```bash
chess-coach analyze
```

Optional engine controls:

```bash
chess-coach analyze \
  --stockfish-path /absolute/path/to/stockfish \
  --depth 14
```

Depth 14 is the default. Analysis is resumable and prints live progress such as reused vs newly analyzed moves. Completed `(game, move, engine configuration)` rows are reused. A single-writer lock prevents two concurrent analyzers from corrupting the same artifact.

### 3. Build the feature dataset

```bash
chess-coach features
```

Features are intentionally interpretable and include position complexity, material, pawn structure, king-safety proxies, opening metadata, game phase, rating difference, time-control category, and the pre-move engine evaluation.

### 4. Train the personalized model

```bash
chess-coach train
```

The model predicts:

```text
P(significant mistake | this position and your historical patterns)
```

The default target is:

```text
significant_mistake = 1 when CPL >= 100
```

Games are split chronologically at **game level**, so moves from one game cannot leak between training and test partitions. Training prints its major stages while it runs.

### 5. Generate the coaching report

```bash
chess-coach report
```

The report includes:

- actionable training priorities;
- overall mistake and blunder rates;
- White vs Black breakdown;
- opening, game-phase, and time-control breakdowns with minimum-sample guards;
- recurring high-risk contexts;
- human-readable evaluation losses;
- selected training positions from your own games;
- human-readable model feature importance with a correlation-not-causation warning.

## Train on your own mistakes

### 6. Build or refresh your puzzle bank

```bash
chess-coach puzzles
```

This reads `data/processed/features.parquet` and extracts genuine mistakes/blunders into:

```text
data/training/training.db
```

Puzzle generation:

- reuses existing Stockfish analysis;
- does **not** launch Stockfish;
- excludes positions where your move already equals Stockfish's best move;
- excludes moves where you delivered checkmate;
- is idempotent, so rebuilding the bank preserves attempts, streaks, due dates, mastery, and review history;
- retires positions that are no longer mistakes from active practice without deleting their history, and restores that history if they later become eligible again;
- shows scanning progress for large datasets.

Each puzzle stores the position before your move, your move, Stockfish's best move, game/opening/phase metadata, evaluation loss, source-game link, difficulty, and a heuristic tactical theme.

Tactical-theme labels such as `fork`, `pin`, `missed mate`, `forcing check`, and `winning capture` are **heuristics for training organization**, not exhaustive tactical proofs.

### 7. Practice due puzzles

```bash
chess-coach practice --limit 10
```

For each due position the CLI shows an ASCII board and asks for your move. You can enter either SAN or UCI:

```text
Your move (SAN/UCI, q to quit): d4
```

The trainer records the result and schedules the position again using a simple spaced-repetition policy:

| Result / streak | Next review |
| --- | ---: |
| Incorrect | 1 day |
| 1 correct in a row | 3 days |
| 2 correct in a row | 7 days |
| 3 correct in a row | 14 days |
| 4+ correct in a row | 30 days |

A puzzle is marked mastered after four consecutive correct reviews. A later failure resets the streak and brings it back the next day.

Use `q` to stop a practice session without recording an attempt for the current puzzle.

### 8. Track training progress

```bash
chess-coach progress
```

This reports:

- total puzzles;
- puzzles currently due;
- puzzles reviewed at least once;
- mastered puzzles;
- total review attempts and accuracy;
- accuracy grouped by heuristic motif;
- accuracy grouped by opening.

A useful regular workflow is:

```bash
# After playing new games
chess-coach sync
chess-coach analyze
chess-coach features
chess-coach puzzles

# Most days
chess-coach practice --limit 10
chess-coach progress

# Periodically, when enough new games exist
chess-coach train
chess-coach report
```

The expensive step is Stockfish analysis. Once your historical corpus is analyzed at the same depth/configuration, later runs mainly analyze newly synced moves; puzzle practice itself is lightweight.

## Local artifact layout

All personal data and trained models are intentionally ignored by Git.

```text
data/
├── raw/
│   ├── games.json
│   ├── srbmaury_all_games.pgn
│   └── sync_manifest.json
├── processed/
│   ├── games.parquet
│   ├── moves.parquet
│   ├── features.parquet
│   └── coaching_report.md
├── engine/
│   └── analysis.parquet
└── training/
    └── training.db

models/
├── mistake_model.joblib
└── mistake_model.metadata.json
```

The repository contains code, tests, synthetic fixtures, documentation, and CI only. Raw PGNs, processed personal game data, puzzle/review history, trained models, caches, and `.env` files are excluded by `.gitignore`.

## Move-quality labels

The default CPL boundaries are:

| Label | CPL |
| --- | ---: |
| Good | `< 50` |
| Inaccuracy | `50–99` |
| Mistake | `100–199` |
| Blunder | `>= 200` |

They can be overridden during analysis/feature generation:

```bash
chess-coach analyze --inaccuracy-cpl 40 --mistake-cpl 90 --blunder-cpl 180
chess-coach features --inaccuracy-cpl 40 --mistake-cpl 90 --blunder-cpl 180
```

Keep the thresholds consistent between analysis and feature generation.

## Configuration

Configuration precedence is:

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

## Development

Run the complete test/lint suite:

```bash
ruff check src tests
pytest --cov=chess_ml_coach --cov-report=term-missing
```

The tests mock network and engine boundaries, so CI does not require a Chess.com account, live PubAPI calls, or a Stockfish binary.

## Next improvements

The local training loop is intentionally simple and explainable. High-value follow-ups are:

- multi-ply puzzle continuations rather than validating only the first best move;
- stronger tactical-motif classification for overloaded defenders, skewers, discovered attacks, back-rank themes, and mating nets;
- rolling 7/30/90-day improvement tracking that compares recent games against older baselines;
- SHAP-based per-position explanations for the LightGBM risk model;
- a web chessboard/dashboard after the training loop proves useful.
