# Personal Puzzle Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a persistent personal puzzle trainer from the user's own analyzed mistakes, with spaced repetition, interactive CLI practice, and progress tracking.

**Architecture:** Extract valid mistakes from `features.parquet` into stable puzzle records, store mutable review state in a local SQLite database, serve due puzzles interactively, and aggregate review progress. Puzzle generation reuses existing Stockfish best-move analysis and does not require another engine pass.

**Tech Stack:** Python 3.11+, python-chess, pandas, sqlite3 standard library, Typer, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-13-personal-puzzle-trainer-design.md`

## Global Constraints

- Keep all personal game/training data under `data/`; it remains gitignored.
- Existing `sync`, `analyze`, `features`, `train`, and `report` behavior must remain backward-compatible.
- Puzzle rebuilds must be idempotent and preserve review history.
- Never surface rows where the player's move equals Stockfish's best move as training mistakes.
- Never surface a move that delivers checkmate as a training mistake.
- Tactical motif labels are heuristic and must be described as such.
- No additional Stockfish pass is required for puzzle generation.

---

### Task 1: Puzzle extraction and tactical labels

**Files:**
- Create: `src/chess_ml_coach/puzzles.py`
- Create: `tests/test_puzzles.py`

**Interfaces:**
- Consumes: feature rows produced by `build_feature_dataset()`.
- Produces: `extract_puzzles(frame: pd.DataFrame) -> list[PuzzleSeed]`, `classify_motif(fen: str, best_move_uci: str, king_ring_attacks: int | None = None) -> str`, and stable puzzle IDs.

- [ ] **Step 1: Write failing tests** for same-move exclusion, delivered-mate exclusion, stable IDs, SAN best move conversion, and motif classification (`missed mate`, `forcing check`, `fork`, `winning capture`).
- [ ] **Step 2: Run** `pytest tests/test_puzzles.py -q` and verify RED.
- [ ] **Step 3: Implement** `PuzzleSeed` plus validation/extraction helpers. Parse `fen_before` with `chess.Board`; parse `best_move_uci`; derive SAN before pushing the move; use `(game_id, ply, best_move_uci)` SHA-256 prefix for identity.
- [ ] **Step 4: Implement heuristic motif classification** in the precedence defined by the spec.
- [ ] **Step 5: Run** `pytest tests/test_puzzles.py -q` and verify GREEN.
- [ ] **Step 6: Commit** `feat: extract personal training puzzles`.

### Task 2: SQLite puzzle bank and spaced repetition

**Files:**
- Create: `src/chess_ml_coach/training.py`
- Create: `tests/test_training.py`

**Interfaces:**
- Consumes: `list[PuzzleSeed]` from Task 1.
- Produces: `TrainingStore`, `ReviewResult`, `ProgressSummary`, and deterministic review scheduling.

- [ ] **Step 1: Write failing tests** for schema creation, idempotent upsert, review-history preservation, due ordering, schedule intervals (1/3/7/14/30 days), mastery at four consecutive correct reviews, and progress aggregation.
- [ ] **Step 2: Run** `pytest tests/test_training.py -q` and verify RED.
- [ ] **Step 3: Implement schema initialization** for `puzzles` and `reviews` using `sqlite3`, `PRAGMA foreign_keys=ON`, and context-managed transactions.
- [ ] **Step 4: Implement idempotent upsert** using `ON CONFLICT(puzzle_id) DO UPDATE` for puzzle metadata only; never overwrite attempts, streak, due date, mastery, or review history.
- [ ] **Step 5: Implement `record_review()`** with deterministic UTC timestamps and the schedule from the spec.
- [ ] **Step 6: Implement due query and progress aggregation** including motif/opening attempt accuracy.
- [ ] **Step 7: Run** `pytest tests/test_training.py -q` and verify GREEN.
- [ ] **Step 8: Commit** `feat: add spaced repetition training store`.

### Task 3: CLI puzzle build and interactive practice

**Files:**
- Modify: `src/chess_ml_coach/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `extract_puzzles()`, `TrainingStore`.
- Produces new CLI commands `puzzles`, `practice`, `progress`.

- [ ] **Step 1: Add failing CLI tests** asserting the three commands appear in `--help`, puzzle build requires `features.parquet`, practice can accept a correct UCI/SAN answer, `q` exits without recording a review, and progress prints human summary fields.
- [ ] **Step 2: Run** `pytest tests/test_cli.py -q` and verify RED.
- [ ] **Step 3: Implement `_run_puzzles(settings)`**: load `features.parquet`, extract seeds, initialize `data/training/training.db`, upsert, and return inserted/updated/skipped/total counts.
- [ ] **Step 4: Implement `practice --limit N`**: show game/opening/phase/motif/difficulty, render `chess.Board(fen_before)`, prompt for SAN/UCI, normalize input against legal moves, reveal best move and source link, record review.
- [ ] **Step 5: Implement `progress`**: print total/due/reviewed/mastered/accuracy plus compact motif/opening summaries.
- [ ] **Step 6: Run** `pytest tests/test_cli.py -q` and verify GREEN.
- [ ] **Step 7: Commit** `feat: add puzzle practice CLI`.

### Task 4: Documentation and end-to-end verification

**Files:**
- Modify: `README.md`
- Test: entire `tests/` suite.

**Interfaces:**
- Documents the new daily/weekly training loop.

- [ ] **Step 1: Update README** with:
  - `chess-coach puzzles`
  - `chess-coach practice --limit 10`
  - `chess-coach progress`
  - explanation that motif labels are heuristic;
  - explanation that puzzle generation reuses existing analysis and does not rerun Stockfish.
- [ ] **Step 2: Run** `ruff check src tests` and require zero errors.
- [ ] **Step 3: Run** `pytest --cov=chess_ml_coach --cov-report=term-missing` and require zero failures.
- [ ] **Step 4: Review branch diff** for accidental internal IDs in human output, unsafe training-state resets, or regressions to existing commands.
- [ ] **Step 5: Open PR**, verify GitHub Actions on the PR head, and merge only when the complete gate is green.