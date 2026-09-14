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
- Old per-player `training.db` files upgrade in place with safe additive schema creation.

---

### Task 1: Adaptive persistence and atomic review finalization

**Files:**
- Create: `src/chess_ml_coach/adaptive_store.py`
- Modify: `src/chess_ml_coach/training.py`
- Create: `tests/test_adaptive_store.py`
- Modify: `tests/test_training.py`

**Interfaces:**
- Produces `AdaptiveSession`, `AdaptiveStep`, `AdaptiveMetrics`, `AdaptiveSessionStore`.
- Produces `TrainingStore.record_review_on_connection(connection, puzzle_id, *, answer, correct, now) -> tuple[int, ReviewResult]`.
- `AdaptiveSessionStore.finalize(session_id, *, succeeded, answer, now=None) -> tuple[AdaptiveSession, ReviewResult]` is transactional and idempotent.

- [ ] **Step 1: Write failing persistence tests**

Add tests for start/resume, ordered steps, abandon-without-review, schema upgrade preserving old reviews, and idempotent finalization:

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

- [ ] **Step 2: Run RED**

```bash
pytest tests/test_adaptive_store.py tests/test_training.py -q
```

Expected: import failure because the adaptive store and shared transaction helper do not exist.

- [ ] **Step 3: Extract the existing review mutation without changing behavior**

Move the current `record_review()` SQL into this concrete connection-aware interface:

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
    resolved_now = _ensure_utc(now)
    # Run the existing puzzle lookup, streak/interval calculation,
    # puzzle UPDATE, and review INSERT on this supplied connection.
    # Return (inserted_review_id, ReviewResult).
```

`record_review()` becomes a thin wrapper that opens a connection and delegates. The existing 1/3/7/14/30-day schedule and 4-correct mastery threshold must not change.

- [ ] **Step 4: Implement adaptive tables and immutable models**

`AdaptiveSessionStore` creates the approved `practice_sessions` and `practice_session_steps` tables. Add terminal review snapshot columns to `practice_sessions` so retries return the original scheduling result even after future reviews: `review_next_interval_days`, `review_next_review_at`, `review_consecutive_correct`, and `review_mastered`.

`start_or_resume()` returns the existing active row for a puzzle or creates one. `append_steps()` runs under `BEGIN IMMEDIATE`, requires `status='active'`, verifies the expected `current_fen`, assigns monotonically increasing `step_index`, and updates counters/FEN/previous evaluation atomically.

- [ ] **Step 5: Implement transactional finalization**

Use this sequence on one SQLite connection:

```text
BEGIN IMMEDIATE
read session
if review_recorded = 1: reconstruct original ReviewResult from session snapshot
else:
  call TrainingStore.record_review_on_connection(...)
  update status, review_recorded, review_id and review snapshot columns
COMMIT
```

`abandon()` marks only active sessions abandoned and never writes a review.

- [ ] **Step 6: Add aggregate metrics**

`AdaptiveSessionStore.metrics()` returns completed sessions, success rate, continuation accuracy excluding the first user step per session, average accepted decisions, and average terminal `current_ply`.

- [ ] **Step 7: Run GREEN and commit**

```bash
pytest tests/test_adaptive_store.py tests/test_training.py -q
git add src/chess_ml_coach/training.py src/chess_ml_coach/adaptive_store.py tests/test_adaptive_store.py tests/test_training.py
git commit -m "feat: persist adaptive practice sessions"
```

---

### Task 2: Hybrid Stockfish continuation service

**Files:**
- Create: `src/chess_ml_coach/adaptive_practice.py`
- Create: `tests/test_adaptive_practice.py`
- Do not change: full-game analysis locking or `ANALYSIS_COLUMNS` in `src/chess_ml_coach/engine.py`

**Interfaces:**
- `AdaptivePracticeService.start(puzzle) -> AdaptiveState`
- `AdaptivePracticeService.submit_move(session_id, move_uci) -> AdaptiveMoveResult`
- `AdaptivePracticeService.abandon(session_id) -> AdaptiveState`
- `AdaptivePracticeService.close_session(session_id) -> None`
- `AdaptivePracticeService.close() -> None`

- [ ] **Step 1: Write RED engine/service tests**

Use a scripted fake adapter returning real `chess.engine.PovScore` values. Cover exact PV acceptance, a 20-cp alternative accepted, a 31-cp alternative failed, winning mate preserved/lost, illegal move with zero mutation, engine reply FEN advancement, quiet/stable completion, terminal/hard-cap completion, persisted resume, engine failure without user penalty, and double-submit serialization.

- [ ] **Step 2: Run RED**

```bash
pytest tests/test_adaptive_practice.py -q
```

- [ ] **Step 3: Implement per-session engine/PV cache**

Keep an in-memory entry per `session_id` containing one `StockfishAdapter`, an optional expected next user UCI move, and last access time. The first expected move is the stored `puzzle.best_move_uci`. After an accepted user move, analyze the opponent-to-move board, play `pv[0]`, and retain `pv[1]` as the next expected user move only when it is legal after the reply.

An exact expected move skips the separate candidate-comparison analysis, but reply/stop analysis still runs. A resumed session after process restart simply has no PV cache and therefore re-analyzes safely from persisted FEN.

- [ ] **Step 4: Implement move scoring**

For a deviation, analyze current and post-candidate boards from the puzzle user's color:

```python
best_eval_cp = normalize_score(best_info["score"], user_color)
candidate_eval_cp = normalize_score(candidate_info["score"], user_color)
loss_cp = max(0, best_eval_cp - candidate_eval_cp)
accepted = loss_cp <= 30
```

When the current best POV score is a positive forced mate, accept only when the candidate POV score is also a positive forced mate.

- [ ] **Step 5: Implement deterministic stopping**

Stop successfully immediately for terminal board, 4 accepted user decisions, or 8 plies. Otherwise play the strongest reply. Once 2 user decisions are accepted, analyze the new user-to-move position: continue on forced mate or when the first PV move gives check, is a capture, or is a promotion; otherwise succeed when `abs(current_best_eval_cp - previous_best_eval_cp) <= 30`.

- [ ] **Step 6: Bound engine lifetime and serialize mutation**

Use a per-session `threading.Lock` around submit/abandon. Close the adapter when a session succeeds, fails, or is abandoned; `close()` shuts down all remaining adapters during app cleanup/profile switch. Persisted sessions remain active even if their in-memory adapter is closed.

- [ ] **Step 7: Run GREEN and commit**

```bash
pytest tests/test_adaptive_practice.py tests/test_adaptive_store.py -q
git add src/chess_ml_coach/adaptive_practice.py tests/test_adaptive_practice.py
git commit -m "feat: add adaptive continuation engine"
```

---

### Task 3: Adaptive FastAPI API and profile-safe lifecycle

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

Assert start/resume never exposes `best_move_uci`, expected next move, or engine PV; accepted moves may expose only the move just played and engine reply; terminal retry does not add reviews; inactive puzzle is 409; illegal move is 422; engine unavailable is 503; and a session from one player profile cannot be accessed after switching to another profile.

- [ ] **Step 2: Run RED**

```bash
pytest tests/test_web_adaptive_practice.py -q
```

- [ ] **Step 3: Add explicit schemas**

Create `AdaptiveSafeStep`, `AdaptiveStartResponse`, `AdaptiveMoveRequest`, `AdaptiveMoveResponse`, and `AdaptiveReviewResult`. Do not define any `next_expected_move`, best-PV, or hidden-answer field in response schemas.

- [ ] **Step 4: Wire one adaptive service to the app**

`create_app()` initializes an adaptive service factory/cache keyed by the active player's training DB path. Routes resolve `request.app.state.settings` at request time, use only that player's DB, and map domain errors to HTTP statuses.

On profile activation, call `app.state.adaptive_services.close_all()` before changing `app.state.settings`. Add a FastAPI lifespan/shutdown cleanup that also closes all adapters. Persisted session rows are never deleted by this cleanup.

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

**Interfaces:**
- Adaptive UI consumes Task 3 endpoints.
- Quick continues using `/api/practice/{id}/attempt` exactly as today.

- [ ] **Step 1: Write RED frontend tests**

Mock API calls and assert Adaptive is selected initially, Quick still uses the old endpoint, start/resume renders returned `current_fen`, accepted move + engine reply updates the board and continues, failed continuation shows partial result, successful completion shows review scheduling plus `Why is this best?`, and no UI text/DOM data reveals a future expected move.

- [ ] **Step 2: Run RED**

```bash
cd web && npm run test:run -- AdaptivePractice.test.tsx
```

- [ ] **Step 3: Add typed adaptive client state**

Add adaptive request/response types and API methods beside the existing client. Initialize `mode` to `'adaptive'`. When a due puzzle loads in Adaptive mode, call start; if the server returns an active persisted session, display its `current_fen` and safe step history.

- [ ] **Step 4: Implement incremental continuation UI**

Use `adaptive.current_fen` as board position. On user drop, submit once, disable the board while waiting, apply returned `current_fen`, show the engine reply already played, and re-enable only when status remains active. Terminal panel shows Converted/Continuation missed, accepted vs attempted user decisions, calculation depth, maximum evaluation loss, next review, source-game link, and the existing explanation button.

- [ ] **Step 5: Keep mode switching safe**

Switching from an active Adaptive drill to Quick explicitly calls abandon before changing mode, so it does not create a spaced-repetition failure. Loading the next puzzle abandons any still-active session first. Browser refresh does not abandon; start resumes it.

- [ ] **Step 6: Style with existing dark-theme variables**

Add a compact Adaptive/Quick segmented control and sequence progress text using `--panel`, `--line`, `--text`, `--muted`, and `--accent`. Do not introduce light native-control defaults.

- [ ] **Step 7: Run GREEN and commit**

```bash
cd web
npm run test:run
npm run build
cd ..
git add web/src/LegacyApp.tsx web/src/styles.css web/src/AdaptivePractice.test.tsx
git commit -m "feat: add adaptive continuation practice UI"
```

---

### Task 5: Adaptive Progress metrics and README

**Files:**
- Modify: `src/chess_ml_coach/web/schemas.py`
- Modify: `src/chess_ml_coach/web/app.py`
- Modify: `web/src/LegacyApp.tsx`
- Create: `web/src/AdaptiveProgress.test.tsx`
- Modify: `README.md`

- [ ] **Step 1: Write RED progress tests**

Seed one succeeded and one failed adaptive session with multiple user steps. Assert `/api/progress` returns a nested `adaptive` object with `sessions_completed=2`, `success_rate=0.5`, continuation accuracy excluding the first user move, average accepted decisions, and average calculation depth.

- [ ] **Step 2: Expose metrics without redefining existing review accuracy**

Add `AdaptiveProgressSummary` to Pydantic schemas and populate it from `AdaptiveSessionStore.metrics()`. Existing `accuracy`, `Mastered`, motif, opening, and daily-review calculations remain unchanged.

- [ ] **Step 3: Add Progress UI panel**

Show adaptive drills completed, conversion rate, continuation accuracy, average decisions, and average depth beneath the existing review activity panel. Keep chart theming untouched.

- [ ] **Step 4: Update README**

Document Adaptive default vs Quick, 30-cp alternative acceptance, local Stockfish use during an adaptive drill, resumable sessions, stricter mastery semantics, and new continuation metrics. Remove the old “multi-ply puzzle continuations” item from Next improvements.

- [ ] **Step 5: Run targeted checks and commit**

```bash
pytest tests/test_web_adaptive_practice.py tests/test_adaptive_store.py tests/test_training.py -q
cd web && npm run test:run && npm run build && cd ..
git add src/chess_ml_coach/web/schemas.py src/chess_ml_coach/web/app.py web/src/LegacyApp.tsx web/src/AdaptiveProgress.test.tsx README.md
git commit -m "feat: report adaptive training progress"
```

---

### Task 6: Full verification, PR, and merge gate

**Files:**
- Review all changed files; no intended production edits unless a verified failure is found.

- [ ] **Step 1: Run complete Python verification**

```bash
ruff check src tests
pytest --cov=chess_ml_coach --cov-report=term-missing
```

- [ ] **Step 2: Run complete frontend verification**

```bash
cd web
npm install
npm run test:run
npm run build
```

- [ ] **Step 3: Review invariants**

Confirm full-game analysis lock behavior and `ANALYSIS_COLUMNS` are unchanged, profile paths remain per-player, no adaptive API response leaks the future expected user move, terminal retries create exactly one review, explanation remains post-attempt/post-sequence, and Quick still records one review after its first move.

- [ ] **Step 4: Open draft PR and verify exact-head CI**

The deduplicated workflow must create one PR run containing the `python` and `web` jobs. Wait for both jobs on the exact final head SHA; do not use an older green run.

- [ ] **Step 5: Mark ready and merge only when exact-head CI is green**

Use squash merge unless repository state requires otherwise. Report the merge SHA and final Python/frontend test counts.

