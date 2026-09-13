# Chess ML Coach

A private, personalized chess-improvement pipeline for Chess.com user `srbmaury`.

The project deliberately separates **chess strength** from **personalization**:

- **Stockfish** evaluates positions, finds the best move, and measures centipawn loss (CPL).
- **LightGBM** learns which kinds of positions are risky *for you* based on your historical decisions.
- The report turns those signals into recurring weakness contexts and positions worth revisiting.

It does **not** try to train a chess engine from your games or replace Stockfish.

## What V1 does

```text
Chess.com PubAPI
      |
      v
sync -> PGN corpus -> parse -> Stockfish analysis -> features -> LightGBM -> report
```

The CLI has five stages:

```bash
chess-coach sync
chess-coach analyze
chess-coach features
chess-coach train
chess-coach report
```

Each stage writes a local artifact and can be run independently once its prerequisites exist.

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

## Run the pipeline

### 1. Download all Chess.com games

```bash
chess-coach sync
```

The default username is `srbmaury`. To use another account:

```bash
chess-coach sync --username another_user
```

The sync is idempotent: already-known games are deduplicated, while newly published monthly games are added to the local corpus.

### 2. Parse games and analyze your moves with Stockfish

```bash
chess-coach analyze
```

Optional engine controls:

```bash
chess-coach analyze \
  --stockfish-path /absolute/path/to/stockfish \
  --depth 14
```

Analysis is resumable. Completed `(game, move, engine configuration)` rows are reused. If the engine configuration changes, stale analysis for the old configuration is replaced rather than mixed into the new dataset.

### 3. Build the ML dataset

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

Games are split chronologically at **game level**, so moves from a single game can never leak between training and test partitions.

### 5. Generate your coaching report

```bash
chess-coach report
```

The report includes:

- overall mistake and blunder rates;
- White vs Black breakdown;
- opening, game-phase, and time-control breakdowns with minimum-sample guards;
- recurring high-risk contexts;
- model feature importance with a correlation-not-causation warning;
- high-CPL positions from your own games that are good candidates for future puzzle generation.

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
└── engine/
    └── analysis.parquet

models/
├── mistake_model.joblib
└── mistake_model.metadata.json
```

The repository contains code, tests, synthetic fixtures, documentation, and CI only. Raw PGNs, processed personal game data, trained models, caches, and `.env` files are excluded by `.gitignore`.

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

## V1 limitations and likely next steps

V1 focuses on a reliable, explainable local pipeline. Useful follow-ups are:

- turn your worst historical positions into a spaced-repetition puzzle queue;
- richer tactical-pattern classification (forks, pins, overloaded defenders, mating nets);
- SHAP-based per-position explanations;
- progress tracking by rolling time window;
- a FastAPI/React dashboard after the analytics prove useful.
