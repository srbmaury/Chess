# Adaptive Continuation Training Design

Date: 2026-09-14

## Goal

Extend the current one-move puzzle trainer into an adaptive continuation trainer that practices calculation and conversion, not just first-move recognition.

Adaptive mode becomes the default practice mode. Quick mode preserves the existing one-move behavior for fast drills.

The design must preserve the existing spaced-repetition model, per-player isolation, explanation flow, and local-first privacy model.

## Product behavior

### Practice modes

- **Adaptive**: default. The user must find the first strong move and then continue through a short engine-guided sequence.
- **Quick**: existing one-move behavior. A review is recorded immediately after the first move.

The selected mode is a UI preference only. It does not change the puzzle bank.

### Adaptive sequence

For an adaptive puzzle:

1. Start from the puzzle's stored `fen_before`.
2. Ask the user for a move.
3. Evaluate whether the move is acceptable.
4. If acceptable and the drill should continue, Stockfish plays the opponent's strongest reply.
5. Ask the user for the next move from the resulting position.
6. Repeat until the drill succeeds, fails, or reaches an adaptive stop condition.
7. Record exactly one spaced-repetition review for the puzzle when the session ends.
8. Show sequence feedback and retain the existing explanation option for the original puzzle idea.

### Move acceptance

The service uses a hybrid strategy.

- If the user's move matches the current cached/principal-variation move, accept it without a comparison search.
- If the user deviates from the expected line, evaluate the candidate and compare it with the engine's best continuation.
- Accept alternatives whose evaluation is within **30 centipawns** of the best move in ordinary positions.
- In forced-mate positions, a move is accepted only if it preserves a winning mate for the training side. A materially shorter mate may be accepted; a move that loses the forced mate is not.
- Illegal moves are rejected without advancing or recording the review.

The evaluation is always from the puzzle user's point of view.

### Failure and success

A sequence succeeds only if:

- the initial puzzle move is acceptable; and
- every required continuation move is acceptable until the adaptive stop condition is reached.

If the first move is good but a later continuation is clearly inferior, the spaced-repetition review is recorded as incorrect. The session still records partial success such as accepted moves and maximum calculation depth.

This keeps puzzle mastery strict: "recognized the idea but failed to convert" does not count as a mastered review.

### Adaptive stop conditions

The sequence ends successfully when any of the following becomes true:

- checkmate, stalemate, draw, or another terminal game state;
- the original tactic/material gain has resolved and the position is no longer forcing;
- the evaluation advantage has stabilized after the tactical sequence;
- the engine line becomes quiet enough that further play would be general game play rather than training the original mistake;
- the user has made **4 accepted decisions**;
- the sequence reaches a hard cap of **8 plies** from the starting position.

The first implementation should use deterministic, conservative stop rules rather than a machine-learned stopping model.

A practical V1 heuristic is:

- always continue after an accepted user move if the position is forcing (check, capture sequence, promotion threat, forced mate) and the cap has not been reached;
- otherwise continue while the best line still shows a significant tactical swing/material conversion;
- stop once the position is quiet after at least two user decisions, or when the hard cap is reached.

## Architecture

Introduce a dedicated `AdaptivePracticeService` rather than extending `PuzzleExplanationService` or embedding engine logic in FastAPI routes.

Responsibilities:

- create/resume adaptive practice sessions;
- validate current-session ownership and puzzle state;
- validate legal user moves;
- reuse a cached current principal variation when possible;
- invoke Stockfish when a candidate deviates or when a fresh reply/stop decision is required;
- compare user move quality against the best move;
- select and apply the opponent's strongest reply;
- decide whether the sequence should continue;
- persist every step;
- finalize exactly one spaced-repetition review when the sequence ends;
- return UI-ready session state.

The service depends on:

- `TrainingStore` for puzzle/review persistence;
- `python-chess` for legality, FEN/SAN conversion, terminal-state detection, checks/captures and board facts;
- the existing Stockfish resolution/adapter infrastructure for engine analysis.

The full-game analysis pipeline and `analysis.parquet` remain unchanged. Adaptive practice must not acquire or mutate the full-game analysis lock.

## Engine lifecycle

Use one Stockfish process per active adaptive session when the session is being actively handled by the application process.

- The process is opened lazily when engine analysis is first needed.
- It is reused for the session's continuation decisions.
- It is closed when the session reaches a terminal state, is abandoned, expires, or application cleanup occurs.
- Persisted session state is authoritative. If the browser refreshes or the server restarts, the session can resume by starting a new Stockfish process from the persisted `current_fen`.

No engine process identity is persisted.

## Persistence

Use the existing per-player `training.db`.

### `practice_sessions`

Fields:

- `session_id TEXT PRIMARY KEY`
- `puzzle_id TEXT NOT NULL`
- `mode TEXT NOT NULL` (`adaptive` for these sessions)
- `starting_fen TEXT NOT NULL`
- `current_fen TEXT NOT NULL`
- `user_color TEXT NOT NULL`
- `status TEXT NOT NULL` (`active`, `succeeded`, `failed`, `abandoned`)
- `user_moves_attempted INTEGER NOT NULL DEFAULT 0`
- `user_moves_accepted INTEGER NOT NULL DEFAULT 0`
- `current_ply INTEGER NOT NULL DEFAULT 0`
- `engine_depth INTEGER NOT NULL`
- `max_eval_loss_cp INTEGER NOT NULL DEFAULT 0`
- `started_at TEXT NOT NULL`
- `updated_at TEXT NOT NULL`
- `finished_at TEXT`
- foreign key to `puzzles(puzzle_id)`

Only one active adaptive session per puzzle/player is needed in V1. Starting the same due puzzle should resume its active session rather than create duplicates.

### `practice_session_steps`

Fields:

- `id INTEGER PRIMARY KEY AUTOINCREMENT`
- `session_id TEXT NOT NULL`
- `step_index INTEGER NOT NULL`
- `side TEXT NOT NULL` (`user` or `engine`)
- `fen_before TEXT NOT NULL`
- `move_uci TEXT NOT NULL`
- `move_san TEXT NOT NULL`
- `accepted INTEGER NOT NULL`
- `source TEXT NOT NULL` (`pv`, `engine`, `user`)
- `eval_cp INTEGER`
- `best_eval_cp INTEGER`
- `eval_loss_cp INTEGER`
- `created_at TEXT NOT NULL`
- foreign key to `practice_sessions(session_id)`
- unique `(session_id, step_index)`

Existing `reviews` remains the spaced-repetition source of truth. Adaptive session tables supplement reviews; they do not replace them.

## Review finalization

`TrainingStore.record_review(...)` is called once when the adaptive session reaches `succeeded` or `failed`.

- `succeeded` -> `correct=True`
- `failed` -> `correct=False`
- `abandoned` -> no review in V1

The `answer` field in the review stores a compact sequence marker such as the first user move UCI or a serialized adaptive result identifier; detailed move history remains in `practice_session_steps`.

Finalization must be idempotent. Repeating the final API request must not create a second review.

## API

Keep Quick mode's existing endpoint behavior for compatibility.

Add adaptive endpoints:

### `POST /api/practice/{puzzle_id}/adaptive/start`

Creates or resumes an active adaptive session.

Response includes:

- `session_id`
- `puzzle_id`
- `status`
- `current_fen`
- `orientation`
- `user_moves_attempted`
- `user_moves_accepted`
- `current_ply`
- `max_user_decisions`
- safe prior step history without unrevealed best moves

### `POST /api/practice/adaptive/{session_id}/move`

Request:

- `move_uci`

Response includes:

- whether the move was accepted;
- candidate move SAN/UCI;
- evaluation loss where available;
- engine reply SAN/UCI if the drill continues;
- updated `current_fen`;
- session status;
- sequence progress;
- final review scheduling information only when the session finishes.

The response must never reveal the next expected user move while the session is active.

### `POST /api/practice/adaptive/{session_id}/abandon`

Marks the active session abandoned without recording a spaced-repetition review.

### Resume behavior

The Practice page can call `start` again for the current puzzle after refresh; the backend returns the active session and current board.

## UI

Practice page gets a compact **Adaptive / Quick** mode control, defaulting to Adaptive.

Adaptive feedback should be incremental rather than replacing the whole puzzle after the first move.

Example:

- `Bxh7+` -> "Strong. Continuing the line…"
- board animates engine reply `Kxh7`
- user is prompted to move again
- progress indicator: `2 / up to 4 decisions`

At completion show:

- sequence result (`Converted`, `Continuation missed`, etc.);
- accepted user moves / attempted user moves;
- calculation depth in plies;
- maximum evaluation loss in the sequence;
- next spaced-repetition interval;
- existing **Why is this best?** explanation control.

Quick mode keeps the current one-move feedback unchanged.

## Progress metrics

Add aggregate adaptive metrics without changing the definition of existing puzzle review accuracy:

- adaptive sessions completed;
- adaptive success rate;
- continuation move accuracy (`accepted user continuation moves / attempted continuation moves`);
- average accepted user decisions per completed adaptive session;
- average calculation depth in plies.

Existing `Mastered` remains based on consecutive correct spaced-repetition reviews, so adaptive practice makes mastery stricter but does not redefine it.

## Error handling

- Missing/invalid session -> 404.
- Session belongs to an inactive/missing puzzle -> 409 and session cannot continue.
- Move illegal in `current_fen` -> 422, no session mutation.
- Move submitted after terminal session -> return terminal session state idempotently; do not create another review.
- Stockfish unavailable before any adaptive evaluation -> actionable 409/503-style UI error and leave session active/unmodified.
- Stockfish crashes mid-session -> close/recreate once using the existing retry philosophy; if still unavailable, return an error without recording failure against the user.
- Browser refresh/network retry -> persisted session + idempotent finalization prevent duplicate reviews.

## Concurrency

Per-player practice sessions and pipeline jobs are independent, but Stockfish resource use should remain bounded.

V1 supports one active adaptive engine operation per session request. The service serializes mutation of a given session so two simultaneous move submissions cannot both advance it.

Do not weaken the full-game analyzer's existing single-writer lock.

## Backward compatibility

- Existing puzzles and reviews need no migration/rebuild.
- Existing one-move Practice remains available as Quick mode.
- Existing explanation cache remains valid.
- Existing per-player profile isolation remains unchanged.
- Schema creation uses `CREATE TABLE IF NOT EXISTS` so old `training.db` files upgrade in place.
- No Stockfish full-game reanalysis is required to enable adaptive training.

## Testing

### Unit/service tests

- exact PV move accepted without unnecessary candidate comparison;
- alternate move within 30 cp accepted;
- worse move beyond tolerance fails the session;
- forced mate must be preserved;
- illegal move does not mutate session;
- engine reply advances `current_fen` correctly;
- quiet/terminal/cap stop rules;
- session resume from persisted FEN;
- final review written exactly once;
- abandoned session writes no review;
- engine failure does not penalize the user;
- Quick-mode review behavior remains unchanged.

### API tests

- start/resume;
- no solution leakage;
- move progression;
- terminal idempotency;
- player/profile isolation;
- inactive puzzle handling.

### Frontend tests

- Adaptive is default;
- Quick mode still works;
- accepted move + engine reply updates board and continues;
- failed continuation shows partial sequence result;
- finished sequence shows review scheduling and explanation button;
- refresh/resume renders persisted current position;
- loading/error states do not expose an expected move.

### Verification gate

Before merge:

- Ruff clean;
- full Python test suite green;
- frontend Vitest suite green;
- production Vite build green;
- exact PR-head CI green;
- final diff review confirms no changes to full-game analysis locking or existing profile isolation.

## Non-goals for V1

- full game sparring after a puzzle;
- cloud/shared engine service;
- multiplayer/session sharing;
- LLM deciding move correctness;
- replacing the existing spaced-repetition algorithm;
- changing the full-game analysis schema to store long PVs globally.
