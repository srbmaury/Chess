# Chess ML Coach Web UI Design

## Goal

Add a local-first browser UI on top of the existing Chess ML Coach without duplicating chess logic, model logic, puzzle scheduling, or pipeline orchestration. The UI should make the trainer usable like a lightweight personal Chess.com/Lichess study tool while preserving the current CLI as a first-class interface.

## Product principles

- **One source of truth:** Python remains authoritative for chess rules, puzzle correctness, SQLite training state, Stockfish analysis, ML training, and report generation.
- **Local-first and private:** the app binds to localhost by default and reads/writes the same gitignored `data/` and `models/` artifacts as the CLI.
- **No unnecessary engine work:** puzzle practice and dashboard views reuse stored artifacts; they never invoke Stockfish unless the user explicitly starts an Analyze pipeline job.
- **Progress is visible:** long operations stream progress/events into the UI.
- **CLI compatibility:** existing commands and artifact formats continue to work unchanged.

## Recommended architecture

### Backend: FastAPI

Add a small API layer under `src/chess_ml_coach/web/`.

Responsibilities:
- expose dashboard/progress/puzzle/report data;
- validate practice moves with `python-chess`;
- record attempts through `TrainingStore`;
- invoke existing pipeline functions instead of duplicating them;
- manage one local pipeline job at a time;
- stream pipeline progress to the browser;
- serve the production React bundle when available.

New Python dependencies:
- `fastapi`
- `uvicorn`

### Frontend: React + TypeScript + Vite

Add a standalone frontend under `web/`.

Recommended libraries:
- React + TypeScript
- Vite
- `react-chessboard` for interactive board rendering/drag-drop
- `chess.js` only for lightweight client UX such as highlighting legal-looking interactions; backend remains authoritative
- `recharts` for progress visualizations

Avoid a large UI framework initially. Use a small design system built with CSS variables/components so the app stays easy to maintain.

### Runtime model

Development:

```text
Vite dev server :5173
       |
       | /api proxy
       v
FastAPI :8000
       |
       +--> existing Chess ML Coach modules
       +--> data/training/training.db
       +--> parquet/model/report artifacts
```

Production/local usage:

```text
chess-coach ui
      |
      v
FastAPI/uvicorn :8000
      |
      +--> /api/*
      +--> serves built React assets from web/dist
```

`chess-coach ui` should print the local URL and optionally open the default browser unless `--no-open` is passed.

## Screens

### 1. Dashboard

Primary purpose: tell the user what to train today.

Cards:
- analyzed moves
- current significant mistake rate
- due puzzles
- mastered puzzles
- review accuracy
- latest pipeline state

Sections:
- **Today's training:** due puzzle count and Start Practice button
- **Top current weaknesses:** top opening/phase/color contexts from the coaching report or feature dataset
- **Training progress:** reviewed/mastered totals and motif/opening accuracy
- **Pipeline status:** last artifact timestamps and actions to Sync / Analyze / Features / Puzzles / Train / Report

The dashboard should prioritize actionable information over raw metrics.

### 2. Practice

Primary purpose: replace the ASCII/SAN-only trainer with an interactive chessboard.

Layout:
- interactive board, oriented to the user's side for the puzzle;
- game/opponent and move metadata;
- opening, phase, heuristic motif, difficulty;
- progress within the current session;
- optional Hint and Skip controls;
- feedback panel after an answer.

Flow:
1. load next due puzzle;
2. user drags/clicks a move;
3. frontend submits UCI to backend;
4. backend validates legality and correctness against stored best move;
5. backend records the spaced-repetition review;
6. UI shows Correct/Incorrect, best move, original played move, evaluation loss, source-game link, and next review interval;
7. continue to next due puzzle.

Rules:
- no review is recorded until the user submits a move;
- Skip does not record a review;
- leaving the page does not record a review;
- puzzle eligibility safeguards from the CLI remain unchanged;
- source links open Chess.com in a new tab.

### 3. Mistakes Explorer

Primary purpose: browse analyzed mistakes beyond the due spaced-repetition queue.

Filters:
- color
- opening/ECO
- phase
- time-control category
- quality: mistake/blunder
- motif
- difficulty
- reviewed/unreviewed/mastered

Rows/cards show:
- game label + move
- your move
- best move
- evaluation loss in pawns
- motif
- opening/phase
- source link

Selecting a row opens a board detail drawer/modal.

Initial implementation can use SQLite active puzzles as the source. It does not need to scan the entire feature parquet on every request.

### 4. Progress

Primary purpose: show whether training is helping.

V1 charts/tables:
- total review accuracy
- mastered vs active puzzles
- accuracy by motif
- accuracy by opening
- review volume by day, derived from `reviews.reviewed_at`

The first UI release should avoid claiming causal chess improvement from review accuracy alone. A future version can add rolling 7/30/90-day in-game mistake-rate comparisons.

### 5. Pipeline

Primary purpose: run and understand the local data pipeline without a terminal.

Stages:
- Sync
- Analyze
- Features
- Puzzles
- Train
- Report

Each stage shows:
- prerequisite state
- artifact existence/last-modified time
- status: idle/running/succeeded/failed
- latest progress message
- action button

Analyze additionally shows:
- Stockfish depth
- reused vs newly analyzed moves
- current/total move count

The backend allows **one pipeline job at a time**. Starting another while one is running returns a clear conflict instead of spawning concurrent analyzers.

## API design

All routes are under `/api`.

### Health / app state

`GET /api/health`

Returns app/version/data-path summary.

`GET /api/dashboard`

Returns the aggregated dashboard payload in one request.

### Practice

`GET /api/practice/next`

Query:
- optional `limit`/session context later

Returns next due puzzle with:
- public puzzle ID
- FEN
- orientation/color
- game label
- move label
- opening/ECO
- phase
- motif
- difficulty
- source URL

Does **not** return the best move before submission.

`POST /api/practice/{puzzle_id}/attempt`

Body:

```json
{"move_uci": "e2e4"}
```

Backend:
- validates the move is legal from the stored FEN;
- compares it with `best_move_uci`;
- records review through `TrainingStore.record_review()`;
- returns correctness, best move SAN/UCI, original game move, evaluation loss, next review date/interval, and source URL.

`POST /api/practice/{puzzle_id}/skip`

Optional convenience endpoint; no review state changes.

### Puzzle explorer

`GET /api/puzzles`

Supports pagination + filters.

`GET /api/puzzles/{puzzle_id}`

Returns puzzle metadata and review summary. Best move may be returned on this non-practice detail route because the user is intentionally exploring rather than solving blind.

### Progress

`GET /api/progress`

Returns:
- current `TrainingStore.progress()` metrics;
- daily review counts/accuracy from the review table;
- motif/opening rows.

### Report

`GET /api/report`

Returns parsed/coaching-report summary data for dashboard usage. Avoid sending raw Markdown as the primary API contract when structured data is available.

### Pipeline

`GET /api/pipeline/status`

Returns stage/artifact status and current job.

`POST /api/pipeline/{stage}`

Allowed stage values:
- `sync`
- `analyze`
- `features`
- `puzzles`
- `train`
- `report`

Analyze body may contain:

```json
{"depth": 14}
```

Other commands use existing settings/defaults initially.

`GET /api/pipeline/events`

Server-Sent Events (SSE) stream with progress payloads such as:

```json
{"stage":"analyze","completed":5679,"total":211639,"reused":3333,"analyzed":2346,"ply":477}
```

SSE is preferred over WebSockets for V1 because progress is server-to-client only.

## Backend service boundaries

### `web/app.py`

Creates FastAPI app, mounts routers/static frontend, installs local-only CORS for Vite development.

### `web/schemas.py`

Pydantic request/response models. API models should hide internal database details such as raw `game_id` unless genuinely useful.

### `web/practice.py`

Small service/router around `TrainingStore`. It does not implement scheduling itself.

### `web/pipeline.py`

Wraps the existing CLI/core pipeline functions with a single-process job manager.

Important constraint: pipeline orchestration must call reusable core functions. If current CLI helpers are private to `cli.py`, extract them into a neutral service module rather than importing Typer/UI concerns into FastAPI.

### `web/dashboard.py`

Aggregates lightweight summary data from SQLite/artifacts/report metadata.

## Pipeline job manager

The local web process needs long operations to continue while HTTP/SSE requests stay responsive.

V1 design:
- one in-process worker thread via `ThreadPoolExecutor(max_workers=1)`;
- one mutable `PipelineJob` state protected by a lock;
- existing progress callbacks push event snapshots into an in-memory bounded queue/history;
- SSE subscribers receive snapshots/events;
- job errors are captured as strings and exposed in status;
- process restart loses only ephemeral job-status history, not actual parquet/SQLite artifacts.

This does not replace the existing single-writer analysis lock; that remains the final protection against concurrent Stockfish writers from separate processes.

## Artifact / state behavior

The UI reads the same paths as CLI settings:
- `data/engine/analysis.parquet`
- `data/processed/features.parquet`
- `data/processed/coaching_report.md`
- `data/training/training.db`
- `models/*`

No separate web database is introduced.

For missing prerequisites, API responses should be actionable, for example:

```text
Puzzle bank not found. Run Features, then Puzzles.
```

Frontend renders the corresponding action rather than a generic error page.

## Security and privacy

V1 is local-only:
- bind default host to `127.0.0.1`, never `0.0.0.0`;
- no authentication because the default process is not remotely reachable;
- do not expose arbitrary filesystem paths through endpoints;
- do not expose raw PGNs unless a future feature explicitly needs them;
- keep personal artifacts gitignored as today.

If remote deployment is added later, authentication, persistent remote storage, engine execution architecture, CSRF/CORS policy, and secret handling must be designed separately rather than assuming local security carries over.

## Frontend structure

Suggested layout:

```text
web/
├── src/
│   ├── api/
│   ├── components/
│   │   ├── AppShell.tsx
│   │   ├── MetricCard.tsx
│   │   ├── ChessPuzzleBoard.tsx
│   │   └── PipelineProgress.tsx
│   ├── pages/
│   │   ├── DashboardPage.tsx
│   │   ├── PracticePage.tsx
│   │   ├── MistakesPage.tsx
│   │   ├── ProgressPage.tsx
│   │   └── PipelinePage.tsx
│   ├── App.tsx
│   └── main.tsx
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts
```

Navigation:
- Dashboard
- Practice
- Mistakes
- Progress
- Pipeline

Use a responsive two-column desktop layout that collapses cleanly on narrow windows. Practice prioritizes the board and keeps metadata secondary.

## CLI integration

Add:

```bash
chess-coach ui
```

Options:
- `--host` default `127.0.0.1`
- `--port` default `8000`
- `--no-open`

Behavior:
- validate that the built frontend exists;
- start uvicorn;
- open `http://127.0.0.1:8000` in the default browser unless disabled.

For frontend development, document:

```bash
# terminal 1
chess-coach ui --no-open

# terminal 2
cd web
npm install
npm run dev
```

Vite proxies `/api` to the FastAPI server.

## Testing strategy

### Python tests

Add tests for:
- health/dashboard endpoints;
- hidden best move in `practice/next`;
- legal/correct/incorrect practice attempts;
- attempts persist through `TrainingStore`;
- skip leaves review history unchanged;
- puzzle filtering/pagination;
- pipeline single-job conflict;
- progress event propagation;
- actionable missing-prerequisite errors;
- `chess-coach ui` host defaults to localhost.

Use FastAPI `TestClient`; no real Stockfish/network calls in API tests.

### Frontend tests

Use Vitest + React Testing Library for:
- dashboard rendering;
- practice move submission/feedback;
- pipeline progress rendering;
- error/prerequisite states.

### Build/CI

Extend CI to:
- run existing Python Ruff/pytest gate;
- install Node dependencies with `npm ci`;
- run frontend tests;
- run `npm run build`.

## Rollout order

1. Extract neutral reusable pipeline-service functions from CLI if needed.
2. FastAPI app + health/dashboard/progress endpoints.
3. Practice API with correctness/state tests.
4. React shell + routing/dashboard.
5. Interactive practice board.
6. Mistakes explorer + progress charts.
7. Pipeline job manager + SSE progress.
8. `chess-coach ui` command and static production serving.
9. README/CI polish.

## Success criteria

The feature is complete when:
- `chess-coach ui` starts the local application;
- the dashboard works with the user's existing artifacts;
- the user can solve a due puzzle by dragging a piece and the same spaced-repetition state is updated as the CLI;
- practice does not reveal the answer before submission;
- the user can browse mistakes and progress without reading raw SQLite/Markdown;
- Sync/Analyze/Features/Puzzles/Train/Report can be launched from the UI with visible progress;
- Analyze cannot be started twice concurrently;
- existing CLI commands remain green and compatible;
- frontend + backend tests/build pass in CI.

## Deferred follow-ups

- remote/cloud deployment;
- authentication/multi-user accounts;
- multi-ply puzzle continuations;
- engine-generated explanations;
- rolling 7/30/90-day in-game improvement charts;
- mobile/PWA packaging;
- richer board annotations/arrows and variation explorer.
