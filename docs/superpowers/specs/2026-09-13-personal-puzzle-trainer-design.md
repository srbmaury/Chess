# Personal Puzzle Trainer Design

## Goal

Turn the existing Chess ML Coach from an analytics-only pipeline into a training loop that converts the user's own genuine mistakes into reusable chess puzzles, schedules them with spaced repetition, records attempts, and shows whether recurring weaknesses improve over time.

## User flow

1. `chess-coach sync` and `chess-coach analyze` continue to collect and evaluate games.
2. `chess-coach features` produces the joined per-move dataset.
3. `chess-coach puzzles` extracts genuine mistakes/blunders into a persistent local puzzle bank.
4. `chess-coach practice --limit 10` serves due puzzles interactively from the user's own games.
5. The user enters a SAN or UCI move. The trainer accepts the stored Stockfish best move, records correctness, reveals the best move when needed, and schedules the next review.
6. `chess-coach progress` summarizes puzzle-bank size, due count, review accuracy, mastery, and recurring motifs/openings.

## Scope

### Included in this increment

- Persistent puzzle bank backed by local SQLite at `data/training/training.db`.
- Idempotent puzzle generation from `data/processed/features.parquet`.
- Only rows classified as `mistake` or `blunder` are eligible.
- Rows where the player's move equals Stockfish's best move are excluded.
- Rows where the player's move delivers checkmate are excluded.
- Stable puzzle IDs derived from `(game_id, ply, best_move_uci)` so rebuilds preserve review history.
- Active/inactive puzzle lifecycle: corrected or obsolete mistakes leave the active practice queue without losing historical attempts and can later be reactivated with their history intact.
- Human-readable puzzle metadata: game, move number, opening, phase, source link, actual move, best move, evaluation loss, motif, difficulty.
- Heuristic tactical motif classification using the board and Stockfish best move.
- Interactive CLI practice using an ASCII board and SAN/UCI input.
- Spaced repetition schedule based on consecutive correct attempts.
- Review history and progress summaries.
- Existing pipeline commands remain backward-compatible.

### Deliberately deferred

- Browser/React chessboard UI.
- Multi-ply Stockfish continuation storage.
- Engine-generated natural-language explanations.
- SHAP-based puzzle selection.
- Full tactical motif engine with exhaustive combinational proof.

The first trainer should be immediately useful without forcing another expensive Stockfish pass. It therefore reuses the already stored `best_move_uci` and position metadata.

## Architecture

### `training.py`

Owns persistent training state and scheduling.

Responsibilities:
- initialize and migrate the SQLite schema;
- upsert puzzles without destroying review state;
- retire puzzles that are no longer present in the latest eligible set without deleting them;
- reactivate returning puzzles with their review state intact;
- query due active puzzles;
- record attempts;
- compute next-review timestamps;
- aggregate progress metrics for active puzzles.

Tables:

`puzzles`
- `puzzle_id TEXT PRIMARY KEY`
- `game_id TEXT NOT NULL`
- `ply INTEGER NOT NULL`
- `fen_before TEXT NOT NULL`
- `color TEXT NOT NULL`
- `game_label TEXT NOT NULL`
- `move_label TEXT NOT NULL`
- `your_move_san TEXT`
- `your_move_uci TEXT`
- `best_move_san TEXT`
- `best_move_uci TEXT NOT NULL`
- `cpl INTEGER NOT NULL`
- `quality TEXT NOT NULL`
- `opening TEXT`
- `eco TEXT`
- `game_phase TEXT`
- `source_url TEXT`
- `motif TEXT NOT NULL`
- `difficulty INTEGER NOT NULL`
- `created_at TEXT NOT NULL`
- `updated_at TEXT NOT NULL`
- `attempts INTEGER NOT NULL DEFAULT 0`
- `correct_attempts INTEGER NOT NULL DEFAULT 0`
- `consecutive_correct INTEGER NOT NULL DEFAULT 0`
- `last_reviewed_at TEXT`
- `next_review_at TEXT NOT NULL`
- `mastered INTEGER NOT NULL DEFAULT 0`
- `active INTEGER NOT NULL DEFAULT 1`

`reviews`
- `id INTEGER PRIMARY KEY AUTOINCREMENT`
- `puzzle_id TEXT NOT NULL`
- `reviewed_at TEXT NOT NULL`
- `answer TEXT NOT NULL`
- `correct INTEGER NOT NULL`
- `previous_interval_days INTEGER NOT NULL`
- `next_interval_days INTEGER NOT NULL`

Puzzle refresh semantics:
- every refresh marks the existing bank inactive inside the same transaction;
- every currently eligible puzzle is inserted or reactivated;
- metadata can be refreshed, but attempts, streak, due date, mastery, and review rows are never reset;
- inactive puzzles are excluded from due queues and current progress metrics;
- inactive rows remain available internally so their history survives future reactivation.

### `puzzles.py`

Owns transformation from analyzed feature rows into training puzzles and motif/difficulty classification.

Eligibility:
- `quality` in `{mistake, blunder}`;
- `best_move_uci` present;
- actual UCI differs from `best_move_uci`;
- `fen_before` and move are valid;
- the actual move does not produce checkmate.

Motif classification is intentionally heuristic and ordered from specific to general:
1. `missed mate` if the best move itself checkmates;
2. `forcing check` if the best move gives check;
3. `fork` if after the best move the moved piece attacks at least two higher-value enemy pieces;
4. `pin` if the best move creates a newly pinned enemy piece to its king;
5. `winning capture` if the best move captures material;
6. `king safety` when the position has heavy king-ring pressure;
7. `positional / calculation` fallback.

The report must call these heuristic labels, not proven tactical truth.

Difficulty is a small 1-5 score derived from evaluation loss, move forcingness, and legal-move count. It is used only for ordering/display, not as an objective chess rating.

### CLI

New commands:

`chess-coach puzzles`
- Requires `features.parquet`.
- Builds/refreshes the puzzle bank.
- Prints extraction progress and summary: eligible, inserted, updated, skipped.

`chess-coach practice --limit 10`
- Opens the local training DB.
- Selects due active puzzles ordered by overdue status, previous failures, and difficulty.
- Shows metadata and `chess.Board(fen)`.
- Accepts SAN or UCI; `q` exits cleanly.
- Correct answer records a success and schedules a later review.
- Wrong answer records a failure, reveals the best move, and schedules it for the next day.

`chess-coach progress`
- Shows active puzzle count, due now, reviewed, mastered, overall review accuracy.
- Shows compact top motif/opening rows with attempts and accuracy.

## Spaced repetition

The schedule is deterministic and intentionally simple:

- incorrect: 1 day; consecutive-correct resets to 0;
- first consecutive correct: 3 days;
- second: 7 days;
- third: 14 days;
- fourth and later: 30 days;
- mastered after 4 consecutive correct attempts.

A correct answer after a previous failure starts again at the 3-day interval.

## Data integrity and privacy

- Training state lives under `data/`, already gitignored.
- Rebuilding puzzles is idempotent and never deletes review history.
- Existing puzzles that still exist in source data are metadata-refreshed but review counters remain intact.
- Puzzles absent from the newest eligible set are marked inactive rather than deleted.
- Invalid FENs/moves are skipped rather than breaking the whole build.
- Internal `game_id` remains in SQLite for joins but is not shown as the primary human identifier.

## Tests

Add focused tests for:
- stable puzzle IDs and idempotent upserts;
- exclusion of false/same-move/checkmating-player positions;
- SAN conversion;
- motif classification for mate/check/capture/fork cases;
- spaced repetition intervals and mastery;
- due ordering;
- persistent review history across puzzle rebuilds;
- obsolete-puzzle retirement and history-preserving reactivation;
- CLI command exposure and practice answer handling;
- progress aggregation.

Full gate remains `ruff check src tests` plus `pytest --cov=chess_ml_coach --cov-report=term-missing`.