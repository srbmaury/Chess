from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import chess
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ..config import Settings
from ..services import training_db_path
from ..training import TrainingStore
from .pipeline import (
    TERMINAL_STATUSES,
    PipelineBusyError,
    PipelineManager,
    UnknownPipelineStageError,
)
from .schemas import (
    ArtifactState,
    AttemptRequest,
    AttemptResponse,
    DashboardResponse,
    HealthResponse,
    PracticeNextResponse,
    PracticePuzzle,
    ProgressGroupRow,
    ProgressResponse,
    PuzzleItem,
    PuzzleListResponse,
    TrainingSummary,
)
from .storage import daily_reviews, get_puzzle, list_puzzles

APP_VERSION = "0.1.0"


def _require_training_db(settings: Settings) -> Path:
    path = training_db_path(settings)
    if not path.exists():
        raise HTTPException(
            status_code=409,
            detail="Puzzle bank not found. Run `chess-coach features` then `chess-coach puzzles`.",
        )
    return path


def _artifact_state(path: Path, *, parquet_rows: bool = False) -> ArtifactState:
    if not path.exists():
        return ArtifactState(exists=False)
    rows = None
    if parquet_rows:
        try:
            import pyarrow.parquet as pq

            rows = int(pq.ParquetFile(path).metadata.num_rows)
        except (OSError, ValueError):
            rows = None
    return ArtifactState(
        exists=True,
        updated_at=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
        rows=rows,
    )


def _empty_training_summary() -> TrainingSummary:
    return TrainingSummary(
        total_puzzles=0,
        due_puzzles=0,
        reviewed_puzzles=0,
        mastered_puzzles=0,
        total_reviews=0,
        accuracy=None,
    )


def _training_summary(settings: Settings) -> TrainingSummary:
    db_path = training_db_path(settings)
    if not db_path.exists():
        return _empty_training_summary()
    summary = TrainingStore(db_path).progress()
    return TrainingSummary(
        total_puzzles=summary.total_puzzles,
        due_puzzles=summary.due_puzzles,
        reviewed_puzzles=summary.reviewed_puzzles,
        mastered_puzzles=summary.mastered_puzzles,
        total_reviews=summary.total_reviews,
        accuracy=summary.accuracy,
    )


def _public_practice_puzzle(puzzle) -> PracticePuzzle:
    return PracticePuzzle(
        puzzle_id=puzzle.puzzle_id,
        fen=puzzle.fen_before,
        orientation=puzzle.color,
        game=puzzle.game_label,
        move=puzzle.move_label,
        opening=puzzle.opening,
        eco=puzzle.eco,
        phase=puzzle.game_phase,
        motif=puzzle.motif,
        difficulty=puzzle.difficulty,
        source_url=puzzle.source_url or None,
    )


def create_app(
    settings: Settings | None = None,
    *,
    pipeline_manager: PipelineManager | None = None,
) -> FastAPI:
    resolved = settings or Settings()
    manager = pipeline_manager or PipelineManager(resolved)
    app = FastAPI(title="Chess ML Coach", version=APP_VERSION)
    app.state.settings = resolved
    app.state.pipeline_manager = manager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            version=APP_VERSION,
            username=resolved.username,
            data_dir=str(resolved.data_dir),
            model_dir=str(resolved.model_dir),
        )

    @app.get("/api/dashboard", response_model=DashboardResponse)
    def dashboard() -> DashboardResponse:
        analysis_path = resolved.data_dir / "engine" / "analysis.parquet"
        features_path = resolved.data_dir / "processed" / "features.parquet"
        report_path = resolved.data_dir / "processed" / "coaching_report.md"
        db_path = training_db_path(resolved)
        model_path = resolved.model_dir / "mistake_model.joblib"
        analysis_state = _artifact_state(analysis_path, parquet_rows=True)
        return DashboardResponse(
            analyzed_moves=analysis_state.rows or 0,
            training=_training_summary(resolved),
            artifacts={
                "analysis": analysis_state,
                "features": _artifact_state(features_path, parquet_rows=True),
                "report": _artifact_state(report_path),
                "puzzles": _artifact_state(db_path),
                "model": _artifact_state(model_path),
            },
        )

    @app.get("/api/practice/next", response_model=PracticeNextResponse)
    def practice_next() -> PracticeNextResponse:
        db_path = _require_training_db(resolved)
        due = TrainingStore(db_path).due_puzzles(limit=1)
        return PracticeNextResponse(
            puzzle=_public_practice_puzzle(due[0]) if due else None,
        )

    @app.post(
        "/api/practice/{puzzle_id}/attempt",
        response_model=AttemptResponse,
    )
    def practice_attempt(puzzle_id: str, request: AttemptRequest) -> AttemptResponse:
        db_path = _require_training_db(resolved)
        store = TrainingStore(db_path)
        puzzle = store.get_puzzle(puzzle_id)
        if puzzle is None or not puzzle.active:
            raise HTTPException(status_code=404, detail="Puzzle not found")
        try:
            move = chess.Move.from_uci(request.move_uci.lower())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Move must be legal UCI notation") from exc
        board = chess.Board(puzzle.fen_before)
        if move not in board.legal_moves:
            raise HTTPException(status_code=422, detail="Submitted move is not legal in this position")
        normalized = move.uci()
        correct = normalized == puzzle.best_move_uci
        review = store.record_review(
            puzzle_id,
            answer=normalized,
            correct=correct,
        )
        return AttemptResponse(
            correct=correct,
            best_move_san=puzzle.best_move_san,
            best_move_uci=puzzle.best_move_uci,
            your_game_move=puzzle.your_move_san,
            evaluation_loss_pawns=puzzle.eval_loss_pawns,
            next_interval_days=review.next_interval_days,
            next_review_at=review.next_review_at,
            source_url=puzzle.source_url or None,
        )

    @app.post("/api/practice/{puzzle_id}/skip")
    def practice_skip(puzzle_id: str) -> dict[str, bool]:
        db_path = _require_training_db(resolved)
        puzzle = TrainingStore(db_path).get_puzzle(puzzle_id)
        if puzzle is None or not puzzle.active:
            raise HTTPException(status_code=404, detail="Puzzle not found")
        return {"skipped": True}

    @app.get("/api/puzzles", response_model=PuzzleListResponse)
    def puzzles(
        quality: str | None = None,
        motif: str | None = None,
        opening: str | None = None,
        reviewed: bool | None = None,
        mastered: bool | None = None,
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> PuzzleListResponse:
        db_path = _require_training_db(resolved)
        items, total = list_puzzles(
            db_path,
            quality=quality,
            motif=motif,
            opening=opening,
            reviewed=reviewed,
            mastered=mastered,
            limit=limit,
            offset=offset,
        )
        return PuzzleListResponse(
            items=[PuzzleItem.model_validate(item) for item in items],
            total=total,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/puzzles/{puzzle_id}", response_model=PuzzleItem)
    def puzzle_detail(puzzle_id: str) -> PuzzleItem:
        db_path = _require_training_db(resolved)
        item = get_puzzle(db_path, puzzle_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Puzzle not found")
        return PuzzleItem.model_validate(item)

    @app.get("/api/progress", response_model=ProgressResponse)
    def progress() -> ProgressResponse:
        db_path = _require_training_db(resolved)
        summary = TrainingStore(db_path).progress()
        return ProgressResponse(
            total_puzzles=summary.total_puzzles,
            due_puzzles=summary.due_puzzles,
            reviewed_puzzles=summary.reviewed_puzzles,
            mastered_puzzles=summary.mastered_puzzles,
            total_reviews=summary.total_reviews,
            accuracy=summary.accuracy,
            by_motif=[ProgressGroupRow(**row.__dict__) for row in summary.by_motif],
            by_opening=[ProgressGroupRow(**row.__dict__) for row in summary.by_opening],
            daily_reviews=daily_reviews(db_path),
        )

    @app.get("/api/pipeline/status")
    def pipeline_status() -> dict[str, object]:
        return jsonable_encoder(asdict(manager.snapshot()))

    @app.post("/api/pipeline/{stage}", status_code=202)
    def start_pipeline(
        stage: str,
        options: dict[str, object] | None = Body(default=None),
    ) -> dict[str, object]:
        try:
            snapshot = manager.start(stage, options)
        except UnknownPipelineStageError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PipelineBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return jsonable_encoder(asdict(snapshot))

    @app.get("/api/pipeline/events")
    def pipeline_events(after_sequence: int = Query(default=0, ge=0)) -> StreamingResponse:
        def stream():
            sequence = after_sequence
            while True:
                events = manager.events(after_sequence=sequence)
                for event in events:
                    sequence = event.sequence
                    payload = {
                        "sequence": event.sequence,
                        "created_at": event.created_at.isoformat(),
                        **event.payload,
                    }
                    encoded = json.dumps(jsonable_encoder(payload), separators=(",", ":"))
                    yield f"data: {encoded}\n\n"
                snapshot = manager.snapshot()
                if snapshot.status in TERMINAL_STATUSES and not manager.events(
                    after_sequence=sequence
                ):
                    break
                if not events:
                    yield ": heartbeat\n\n"
                    time.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
