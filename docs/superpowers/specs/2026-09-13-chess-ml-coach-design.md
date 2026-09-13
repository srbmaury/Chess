# Chess ML Coach — V1 Design

## Goal
Build a private, CLI-first Python project for Chess.com user `srbmaury` that downloads completed games, parses them into game/move/position records, evaluates the user's moves with Stockfish, derives ML-ready features, trains a personalized mistake-risk model, and generates actionable coaching reports.

## V1 Scope

### In scope
- Download all available Chess.com game archives for `srbmaury` through the public Chess.com PubAPI.
- Preserve raw PGNs locally as the source of truth.
- Parse PGNs with `python-chess` into normalized game and move records.
- Run Stockfish on positions immediately before and after moves played by `srbmaury`.
- Compute engine-derived labels including best move, evaluation swing, centipawn loss, and coarse move-quality class.
- Extract position/game/context features suitable for classical ML.
- Train a LightGBM model that estimates the probability that `srbmaury` makes a significant mistake in a position.
- Produce a local coaching report ranking recurring weaknesses by opening, phase, time-control/context when available, and position characteristics.
- Provide one CLI with staged commands: `sync`, `analyze`, `features`, `train`, and `report`.
- Add deterministic unit tests and GitHub Actions CI that do not depend on live Chess.com or Stockfish network access.

### Out of scope for V1
- React/Next.js dashboard.
- FastAPI service.
- Cloud deployment or scheduled jobs.
- Training a chess engine or deep neural network from scratch.
- Natural-language LLM coaching.
- Automated spaced-repetition UI.

## Architecture

```text
Chess.com PubAPI
      |
      v
  sync command
      |
      v
raw PGN archive  --->  PGN parser  --->  normalized games + moves
                                       |
                                       v
                                 Stockfish analyzer
                                       |
                                       v
                              engine analysis dataset
                                       |
                                       v
                                feature extractor
                                       |
                                       v
                                ML-ready dataset
                                       |
                                       v
                              LightGBM trainer
                                       |
                                       v
                            personalized report
```

## Repository Layout

```text
chess-ml-coach/
├── src/chess_ml_coach/
│   ├── __init__.py
│   ├── chesscom.py
│   ├── pgn.py
│   ├── engine.py
│   ├── features.py
│   ├── train.py
│   ├── report.py
│   └── cli.py
├── tests/
├── data/
│   ├── raw/
│   ├── processed/
│   └── engine/
├── models/
├── docs/superpowers/specs/
├── .github/workflows/test.yml
├── .gitignore
├── pyproject.toml
└── README.md
```

`data/` and `models/` are local artifacts and must be ignored by Git except for optional `.gitkeep` files if needed.

## Data Flow

### 1. Sync
`chess-coach sync --username srbmaury`

- Fetch `/pub/player/{username}/games/archives`.
- Fetch each monthly archive.
- Keep completed games that contain PGN data.
- Deduplicate by Chess.com game URL when present, otherwise by a stable hash of canonical game metadata + movetext.
- Write a single canonical PGN file plus a compact sync manifest.
- A rerun must be idempotent and only add newly observed games.

### 2. Parse
Parsing occurs as part of downstream commands when normalized files are absent or stale.

Game-level fields include:
- stable `game_id`
- date / UTC date if available
- white / black usernames
- white / black ratings
- result
- time control
- rated flag when present
- ECO/opening metadata when present
- source URL

Move-level fields include:
- `game_id`
- ply and fullmove number
- side to move
- FEN before move
- SAN and UCI move
- whether the move was played by `srbmaury`
- clock information when present in PGN comments/tags
- resulting FEN

### 3. Stockfish analysis
`chess-coach analyze`

For each move played by `srbmaury`:
- evaluate the position before the move;
- determine Stockfish's best move;
- evaluate the position after the played move;
- normalize scores from the user's perspective;
- compute centipawn loss with mate-score handling;
- assign a coarse label using configurable thresholds.

Default move-quality labels:
- `good`: CPL < 50
- `inaccuracy`: 50 <= CPL < 100
- `mistake`: 100 <= CPL < 200
- `blunder`: CPL >= 200

Thresholds are configuration, not hard-coded model assumptions.

Engine analysis must be resumable. Existing analyzed positions are skipped unless explicitly forced or the engine-analysis configuration changes.

### 4. Feature extraction
`chess-coach features`

Initial features should be intentionally interpretable:
- move number and game phase;
- color;
- rating and rating difference;
- time control category;
- material balance;
- total material remaining;
- legal move count as a rough complexity measure;
- check state;
- castling rights;
- queen presence;
- pawn-island / doubled-pawn / isolated-pawn counts where practical;
- king-safety proxies;
- opening/ECO categorical values;
- engine evaluation before the move;
- optional clock/time-left features when reliable data exists.

The primary supervised target for V1 is binary:
`significant_mistake = 1` when CPL >= 100, else `0`.

The raw continuous CPL and multi-class move-quality label are retained for analysis and future experiments.

### 5. Model training
`chess-coach train`

- Use LightGBM as the primary model.
- Split chronologically by game date to reduce future-to-past leakage.
- Never split rows from the same game across train/test boundaries.
- Report class balance, ROC-AUC, PR-AUC, log loss, and calibration/Brier score where meaningful.
- Save the trained model and a metadata JSON file with feature list, thresholds, dataset version, and evaluation metrics.
- If the dataset is too small or contains only one target class, fail with a clear actionable error instead of producing a misleading model.

## Coaching Report

`chess-coach report`

The report should answer practical improvement questions rather than merely restating model metrics. V1 output is Markdown/terminal text and includes:
- overall mistake/blunder rates;
- performance and mistake rate by color;
- game phase breakdown;
- opening/ECO breakdown when sample size is sufficient;
- time-control breakdown;
- strongest recurring error contexts;
- top model features using LightGBM feature importance/SHAP-compatible design, with warnings that correlation is not causation;
- a shortlist of candidate positions from the user's own games for later puzzle generation.

Small-sample groups must be suppressed or clearly marked to avoid false conclusions.

## Error Handling

- Chess.com HTTP errors: exponential backoff for transient failures; explicit error for permanent 4xx responses.
- Partial sync: preserve already downloaded data and manifest progress.
- Invalid/unsupported PGN: skip the game with a logged reason; do not abort the entire corpus.
- Missing Stockfish binary: fail before analysis with installation/configuration guidance.
- Engine crash: close/restart safely and preserve completed analysis rows.
- Missing clock data: leave clock-derived features null; never infer fabricated values.
- Corrupt local artifacts: validate schemas and fail with a message identifying the file that should be rebuilt.

## Configuration

Configuration priority:
1. CLI flag
2. environment variable
3. project default

Initial configurable values:
- Chess.com username (default `srbmaury`)
- Stockfish binary path
- Stockfish depth or time-per-position mode
- worker count if safe
- CPL thresholds
- data/model directories

No credentials are required for Chess.com public archive ingestion.

## Testing Strategy

Unit tests use fixed PGN and JSON fixtures and mock network/engine boundaries.

Required test areas:
- archive URL discovery and monthly download parsing;
- idempotent deduplication;
- PGN parsing and correct identification of `srbmaury`'s moves as White and Black;
- FEN before/after move correctness;
- score normalization from the user's perspective;
- mate-score/CPL edge cases;
- move-quality threshold boundaries;
- feature extraction from known positions;
- chronological game-level split without leakage;
- training failure for insufficient/single-class data;
- report aggregation with minimum-sample guards;
- CLI smoke tests.

CI runs formatting/linting only if configured in the initial implementation and always runs the complete unit-test suite. Stockfish-dependent behavior is tested through a fake engine adapter so CI remains portable.

## Dependencies

Runtime:
- Python 3.11+
- `httpx`
- `python-chess`
- `pandas`
- `pyarrow`
- `numpy`
- `lightgbm`
- `scikit-learn`
- `typer`
- `rich`

Development:
- `pytest`
- `pytest-cov`
- `ruff`

Stockfish is an external executable and is not committed to the repository.

## Privacy and Repository Rules

- Repository visibility must be private.
- Raw PGNs, processed personal game data, engine datasets, trained models, caches, and local environment files must be gitignored.
- The repository contains code, fixtures using synthetic/public-safe games, documentation, and CI only.
- No GitHub, Chess.com, or other credentials are stored in source files.

## Success Criteria

V1 is complete when:
1. `sync` can build/update the local corpus for `srbmaury` idempotently.
2. `analyze` can resume Stockfish evaluation and produce per-user-move CPL/labels.
3. `features` creates a deterministic ML dataset.
4. `train` produces a validated LightGBM model without game leakage.
5. `report` identifies statistically guarded recurring weakness contexts and example positions.
6. Unit tests pass locally and in GitHub Actions.
7. A new machine can reproduce the workflow from the README without access to committed personal game data.
