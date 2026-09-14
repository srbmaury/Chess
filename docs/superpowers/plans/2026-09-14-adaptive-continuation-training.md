# Adaptive Continuation Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add default adaptive multi-move puzzle training with Stockfish-guided continuation, strict one-review-per-sequence spaced repetition, resumable per-player sessions, and continuation progress metrics while preserving Quick mode.

**Architecture:** Add a focused adaptive persistence layer and `AdaptivePracticeService` beside the existing `TrainingStore` and explanation service. FastAPI routes expose start/move/abandon APIs without leaking future best moves; the React Practice page defaults to Adaptive and keeps Quick as the existing one-move flow. Adaptive Stockfish use is independent of the full-game analysis lock and reuses a per-session engine/PV in memory when possible.

**Tech Stack:** Python 3.11, python-chess, SQLite, FastAPI/Pydantic, React 19, TypeScript, react-chessboard, Vitest.

**Spec:** `docs/superpowers/specs/2026-09-14-adaptive-continuation-training-design.md`

## Global Constraints

- Adaptive is the default web Practice mode; Quick preserves the current one-move behavior.
- Ordinary alternate moves are accepted when evaluation loss is at most **30 cp**.
- A forced winning mate must remain a winning mate for the user's side.
- At most **4 accepted user decisions** and **8 plies** per adaptive session.
- Quiet/stable completion requires at least **2 accepted user decisions**, a non-forcing next PV move, no forced mate, and evaluation movement of at most **30 cp**.
- Exactly one spaced-repetition review is recorded when a session succeeds or fails; abandoned sessions record no review.
- Existing `analysis.parquet`, the analyzer single-writer lock, profile isolation, explanation cache, and Quick mode semantics must remain unchanged.
- Old per-player `training.db` files upgrade in place with `CREATE TABLE IF NOT EXISTS`/safe schema additions.

---

### Task 1: Adaptive session persistence and atomic review finalization

**Files:**
- Create: `src/chess_ml_coach/adaptive_store.py`
- Modify: `src/chess_ml_coach/training.py`
- Create: `tests/test_adaptive_store.py`
- Modify: `tests/test_training.py`

**Interfaces:**
- Produces `AdaptiveSession`, `AdaptiveStep`, `AdaptiveMetrics`, `AdaptiveSessionStore`.
- Produces `TrainingStore.record_review_on_connection(connection, puzzle_id, *, answer, correct, now) -> tuple[int, ReviewResult]` so Quick and adaptive finalization share one review algorithm.
- `AdaptiveSessionStore.finalize(session_id, *, succeeded, answer, now=None) -> tuple[AdaptiveSession, ReviewResult]` must be transactional and idempotent.

- [ ] **Step 1: Write failing persistence tests**

Add tests that construct a normal `TrainingStore`, seed one `PuzzleSeed`, then instantiate `AdaptiveSessionStore` on the same DB. Cover start/resume, step ordering, abandon-without-review, and finalization retry:

```python
session = sessions.start_or_resume(puzzle, depth=14, now=now)
assert sessions.start_or_resume(puzzle, depth=14, now=now).session_id == session.session_id

finished, review = sessions.finalize(
    session.session_id,
    succeeded=True,
    answer="d2d4",
    now=now,
)
again, again_review = sessions.finalize(
    session.session_id,
    succeeded=True,
    answer="d2d4",
    now=now,
)
assert finished.review_id == again.review_id
assert review == again_review
assert training.review_count(puzzle.puzzle_id) == 1
```

Also assert `practice_session_steps` has unique `(session_id, step_index)`, only one active session exists for one puzzle, and schema initialization does not change existing puzzle/review rows.

- [ ] **Step 2: Run RED tests**

Run:

```bash
pytest tests/test_adaptive_store.py tests/test_training.py -q
```

Expected: collection/import failure because `adaptive_store.py` and the shared review transaction helper do not exist.

- [ ] **Step 3: Extract the existing review mutation without changing semantics**

In `training.py`, move the SQL currently inside `record_review()` into a connection-aware method:

```python
@staticmethod
def record_review_on_connection(
    connection: sqlite3.Connection,
    puzzle_id: str,
    *,
    answer: str,
    correct: bool,
    now: datetime | None = None,
) -> tuple[int, ReviewResult]:
    ...
```

It returns the inserted review row id plus the existing `ReviewResult`. Keep `record_review()` as a thin wrapper that opens the connection and delegates. Existing schedule/mastery tests must remain byte-for-byte equivalent in behavior.

- [ ] **Step 4: Implement `AdaptiveSessionStore`**

Create `practice_sessions` and `practice_session_steps` exactly from the approved spec. `start_or_resume()` uses one active session per puzzle, `append_steps()` validates the expected active status/current FEN under `BEGIN IMMEDIATE`, and `finalize()` performs all of these in the same SQLite transaction:

```text
BEGIN IMMEDIATE
read session
if review_recorded: return stored terminal result
record review through TrainingStore.record_review_on_connection(...)
update status/review_recorded/review_id/finished_at
COMMIT
```

Persist the terminal review fields needed to reconstruct an idempotent API response (`review_id`, status, plus review schedule retrievable by the inserted review and puzzle state).

- [ ] **Step 5: Add adaptive aggregate query**

`AdaptiveSessionStore.metrics()` returns completed sessions, success rate, continuation accuracy excluding each session's first user move, average accepted decisions, and average terminal `current_ply`.

- [ ] **Step 6: Run GREEN tests and commit**

```bash
pytest tests/test_adaptive_store.py tests/test_training.py -q
git add src/chess_ml_coach/training.py src/chess_ml_coach/adaptive_store.py tests/test_adaptive_store.py tests/test_training.py
git commit -m "feat: persist adaptive practice sessions"
```

---

### Task 2: Hybrid Stockfish adaptive practice service

**Files:**
- Create: `src/chess_ml_coach/adaptive_practice.py`
- Create: `tests/test_adaptive_practice.py`
- Do not modify: full-game analyzer locking code in `src/chess_ml_coach/engine.py`

**Interfaces:**
- Produces `AdaptivePracticeService.start(puzzle)`, `.submit_move(session_id, move_uci)`, `.abandon(session_id)`, `.close()`.
- Produces immutable result dataclasses `AdaptiveState` and `AdaptiveMoveResult` safe for API serialization.
- Constructor accepts an injected engine factory for tests; production uses `_resolve_stockfish` + `StockfishAdapter`.

- [ ] **Step 1: Write RED service tests with a scripted fake engine**

Use `chess.engine.PovScore`/`Cp`/`Mate` values in fake `analyse()` responses. Cover:

```text
stored first best move -> accepted without candidate-comparison analysis
alternate move with 20 cp loss -> accepted
alternate move with 31 cp loss -> failed + one incorrect review
winning mate preserved -> accepted
winning mate lost -> failed
illegal move -> 422-style domain error, zero DB mutation
engine reply advances persisted FEN
quiet/stable position after 2 accepted decisions -> succeeded
4 accepted decisions / 8 plies -> succeeded
engine exception -> active session unchanged, no review
resume after constructing a new service -> persisted FEN is authoritative
```

- [ ] **Step 2: Run RED**

```bash
pytest tests/test_adaptive_practice.py -q
```

Expected: import failure for `AdaptivePracticeService`.

- [ ] **Step 3: Implement engine lifecycle and PV cache**

Maintain an in-process mapping keyed by `session_id` containing one adapter and the next expected user PV move. The first puzzle move may use `puzzle.best_move_uci` as the expected move. After an accepted user move, analyze the reply position once, play `pv[0]`, and retain `pv[1]` as the next expected user move when legal.

A cache hit accepts the exact PV move without a separate candidate comparison, but the service still performs the analysis required to choose the opponent reply/stop condition.

- [ ] **Step 4: Implement candidate scoring**

For deviations, analyze the current board for `best_info` and the board after the candidate for `candidate_info`, always converting score from the puzzle user's color. Ordinary acceptance is:

```python
loss_cp = max(0, best_eval_cp - candidate_eval_cp)
accepted = loss_cp <= 30
```

If the best current POV score is a positive mate, require the post-candidate POV score to also be a positive mate.

- [ ] **Step 5: Implement deterministic stop logic**

After an accepted user move: stop immediately for terminal board, 4 accepted decisions, or 8 plies. Otherwise play the strongest engine reply. If at least 2 user decisions are accepted, analyze the resulting user-to-move board; continue on forced mate or if the first PV move is a check/capture/promotion. Stop successfully only when that move is non-forcing and `abs(current_best_eval_cp - previous_best_eval_cp) <= 30`.

- [ ] **Step 6: Serialize session mutation**

Use a per-session `threading.Lock` around submit/abandon so concurrent requests cannot both advance one session. Database `BEGIN IMMEDIATE` remains the final mutation guard.

- [ ] **Step 7: Run GREEN and commit**

```bash
pytest tests/test_adaptive_practice.py tests/test_adaptive_store.py -q
git add src/chess_ml_coach/adaptive_practice.py tests/test_adaptive_practice.py
git commit -m "feat: add adaptive continuation engine"
```

---

### Task 3: Adaptive FastAPI routes and schemas

**Files:**
- Create: `src/chess_ml_coach/web/adaptive_routes.py`
- Modify: `src/chess_ml_coach/web/schemas.py`
- Modify: `src/chess_ml_coach/web/app.py`
- Modify: `src/chess_ml_coach/web/profile_routes.py`
- Create: `tests/test_web_adaptive_practice.py`

**Interfaces:**
- `POST /api/practice/{puzzle_id}/adaptive/start`
- `POST /api/practice/adaptive/{session_id}/move`
- `POST /api/practice/adaptive/{session_id}/abandon`

- [ ] **Step 1: Write API RED tests**

Assert start/resume returns only safe state fields and never `best_move_uci`, expected next move, or engine PV. Assert an accepted move may return the engine reply and new FEN, terminal retries do not add reviews, inactive puzzles return 409, illegal moves return 422, and two profile-scoped apps/active profiles cannot access each other's session IDs.

- [ ] **Step 2: Run RED**

```bash
pytest tests/test_web_adaptive_practice.py -q
```

- [ ] **Step 3: Add Pydantic schemas**

Add `AdaptiveStartResponse`, `AdaptiveMoveRequest`, `AdaptiveMoveResponse`, safe step history, and terminal review fields. Keep `next_expected_move` out of every response model.

- [ ] **Step 4: Add route module and app wiring**

Routes obtain the current `Settings` at request time, build/use the adaptive service for that player's `training.db`, map domain not-found/inactive/illegal/engine errors to 404/409/422/503, and return terminal state idempotently.

When a profile is activated, close/discard any in-memory adaptive engines owned by the previous profile before switching `app.state.settings`; persisted sessions remain resumable.

- [ ] **Step 5: Run GREEN and commit**

```bash
pytest tests/test_web_adaptive_practice.py tests/test_web_api.py tests/test_web_profiles.py -q
git add src/chess_ml_coach/web/adaptive_routes.py src/chess_ml_coach/web/schemas.py src/chess_ml_coach/web/app.py src/chess_ml_coach/web/profile_routes.py tests/test_web_adaptive_practice.py
git commit -m "feat: expose adaptive practice API"
```

---

### Task 4: Adaptive-by-default Practice UI with Quick fallback

**Files:**
- Modify: `web/src/LegacyApp.tsx`
- Modify: `web/src/styles.css`
- Create: `web/src/AdaptivePractice.test.tsx`
- Preserve: existing `Why is this best?` post-completion flow

**Interfaces:**
- Frontend consumes the Task 3 adaptive endpoints.
- Quick mode continues calling `/api/practice/{id}/attempt` exactly as today.

- [ ] **Step 1: Write RED frontend tests**

Mock the API and assert:

```text
Adaptive is selected on initial Practice render
Quick can be selected and uses the old one-move endpoint
Adaptive start/resume renders returned current_fen
accepted user move + engine reply updates board and leaves drill active
failed continuation displays partial result and next-review interval
successful terminal result displays Converted and Why is this best?
no response renders an expected future move
```

- [ ] **Step 2: Run RED**

```bash
cd web && npm run test:run -- AdaptivePractice.test.tsx
```

- [ ] **Step 3: Implement API types and mode state**

Add typed adaptive methods beside the existing `api` methods. Default `mode` to `'adaptive'`. Reset/abandon active adaptive state when intentionally switching puzzles/modes; browser refresh resumes by calling start for the same due puzzle.

- [ ] **Step 4: Implement incremental board flow**

The board's displayed FEN is `adaptive.current_fen` while adaptive is active. On a user drop, submit the move, show `Strong. Continuing the line…` when accepted, apply the returned engine reply/current FEN, and re-enable user input only if status remains `active`.

Terminal panel displays accepted/attempted user decisions, current ply calculation depth, maximum loss, next review, source-game link, and existing explanation button.

- [ ] **Step 5: Style the mode switch/progress without introducing theme regressions**

Use existing CSS variables and dark form-control rules. Add a compact segmented Adaptive/Quick control and continuation progress text; do not hard-code a light background.

- [ ] **Step 6: Run GREEN and commit**

```bash
cd web
npm run test:run
npm run build
cd ..
git add web/src/LegacyApp.tsx web/src/styles.css web/src/AdaptivePractice.test.tsx
git commit -m "feat: add adaptive continuation practice UI"
```

---

### Task 5: Adaptive progress metrics and documentation

**Files:**
- Modify: `src/chess_ml_coach/web/schemas.py`
- Modify: `src/chess_ml_coach/web/app.py`
- Modify: `web/src/LegacyApp.tsx`
- Modify: `web/src/App.test.tsx` or create `web/src/AdaptiveProgress.test.tsx`
- Modify: `README.md`

- [ ] **Step 1: Write RED progress tests**

Seed one succeeded and one failed adaptive session with multiple user steps. Assert `/api/progress` reports:

```text
adaptive_sessions_completed = 2
adaptive_success_rate = 0.5
continuation_accuracy = accepted continuation attempts / continuation attempts
average_accepted_decisions
average_calculation_depth_plies
```

The first user move of each session must be excluded from continuation accuracy.

- [ ] **Step 2: Expose metrics through `ProgressResponse`**

Add a nested `adaptive` object so existing top-level review/mastery fields retain their meaning.

- [ ] **Step 3: Add Progress UI panel**

Show completed adaptive drills, conversion rate, continuation accuracy, average decisions, and average depth. Keep the existing review activity chart and Mastered definition unchanged.

- [ ] **Step 4: Update README**

Replace the old “multi-ply puzzle continuations” next-improvement bullet with documentation of Adaptive vs Quick, 30-cp alternative acceptance, resumability, and the fact that adaptive practice may invoke Stockfish locally during a drill.

- [ ] **Step 5: Run targeted checks and commit**

```bash
pytest tests/test_web_adaptive_practice.py tests/test_training.py -q
cd web && npm run test:run && npm run build && cd ..
git add src/chess_ml_coach/web/schemas.py src/chess_ml_coach/web/app.py web/src/LegacyApp.tsx web/src/AdaptiveProgress.test.tsx README.md
git commit -m "feat: report adaptive training progress"
```

---

### Task 6: Full regression verification, PR, and merge gate

**Files:**
- Review all changed files; no intended production edits unless a verified failure is found.

- [ ] **Step 1: Run complete Python verification**

```bash
ruff check src tests
pytest --cov=chess_ml_coach --cov-report=term-missing
```

Expected: all tests pass; existing Quick, profiles, explanation, pipeline cancellation/SSE, and analyzer tests remain green.

- [ ] **Step 2: Run complete frontend verification**

```bash
cd web
npm install
npm run test:run
npm run build
```

Expected: all tests and TypeScript/Vite production build pass, including stale-build fingerprint generation.

- [ ] **Step 3: Review diff for invariants**

Confirm `src/chess_ml_coach/engine.py` full-game analysis lock behavior and `ANALYSIS_COLUMNS` are unchanged, profile paths remain per-player, no API response leaks the next expected user move, and Quick mode still records one review immediately after its first move.

- [ ] **Step 4: Open a draft PR and verify exact-head CI**

The existing workflow should produce one PR workflow with the `python` and `web` jobs. Wait for both jobs on the exact final head SHA; do not merge based on an older green run.

- [ ] **Step 5: Mark ready and merge only when exact-head CI is green**

Use squash merge unless repository state requires otherwise. Record the final merge SHA and the final Python/frontend counts in the completion message.
