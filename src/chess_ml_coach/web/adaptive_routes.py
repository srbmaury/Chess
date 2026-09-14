from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ..adaptive_practice import (
    MAX_USER_DECISIONS,
    AdaptiveEngineError,
    AdaptiveIllegalMove,
    AdaptivePracticeError,
    AdaptivePracticeService,
    AdaptivePuzzleInactive,
    AdaptiveSessionNotFound,
    AdaptiveState,
)
from ..config import Settings
from ..services import training_db_path
from ..training import ReviewResult, TrainingStore
from .schemas import (
    AdaptiveMoveRequest,
    AdaptiveMoveResponse,
    AdaptiveReviewResult,
    AdaptiveSafeStep,
    AdaptiveStartResponse,
)

router = APIRouter()


class AdaptiveServiceRegistry:
    """Owns per-player adaptive services and their in-memory Stockfish processes."""

    def __init__(self, *, engine_factory=None):
        self._engine_factory = engine_factory
        self._services: dict[str, AdaptivePracticeService] = {}

    @staticmethod
    def _key(path: Path) -> str:
        return str(path.resolve())

    def for_settings(self, settings: Settings) -> AdaptivePracticeService:
        db_path = training_db_path(settings)
        key = self._key(db_path)
        service = self._services.get(key)
        if service is None:
            service = AdaptivePracticeService(
                settings,
                db_path,
                engine_factory=self._engine_factory,
            )
            self._services[key] = service
        return service

    def close_all(self) -> None:
        services = list(self._services.values())
        self._services.clear()
        for service in services:
            service.close()


def _registry(request: Request) -> AdaptiveServiceRegistry:
    registry = getattr(request.app.state, "adaptive_services", None)
    if registry is None:
        registry = AdaptiveServiceRegistry()
        request.app.state.adaptive_services = registry
    return registry


def _current_settings(request: Request) -> Settings:
    return request.app.state.settings


def _require_db(request: Request) -> Path:
    path = training_db_path(_current_settings(request))
    if not path.exists():
        raise HTTPException(
            status_code=409,
            detail="Puzzle bank not found. Run Features, then Puzzles from the Pipeline page.",
        )
    return path


def _review_payload(review: ReviewResult | None) -> AdaptiveReviewResult | None:
    if review is None:
        return None
    return AdaptiveReviewResult(
        next_interval_days=review.next_interval_days,
        next_review_at=review.next_review_at,
        consecutive_correct=review.consecutive_correct,
        mastered=review.mastered,
    )


def _start_payload(state: AdaptiveState) -> AdaptiveStartResponse:
    return AdaptiveStartResponse(
        session_id=state.session_id,
        puzzle_id=state.puzzle_id,
        status=state.status,
        current_fen=state.current_fen,
        orientation=state.orientation,
        user_moves_attempted=state.user_moves_attempted,
        user_moves_accepted=state.user_moves_accepted,
        current_ply=state.current_ply,
        max_eval_loss_cp=state.max_eval_loss_cp,
        max_user_decisions=MAX_USER_DECISIONS,
        steps=[
            AdaptiveSafeStep(
                step_index=step.step_index,
                side=step.side,
                move_uci=step.move_uci,
                move_san=step.move_san,
                accepted=step.accepted,
            )
            for step in state.steps
        ],
        review=_review_payload(state.review),
    )


def _raise_domain_error(exc: Exception) -> None:
    if isinstance(exc, AdaptiveSessionNotFound):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, AdaptivePuzzleInactive):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, AdaptiveIllegalMove):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, AdaptiveEngineError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, AdaptivePracticeError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    raise exc


@router.post(
    "/api/practice/{puzzle_id}/adaptive/start",
    response_model=AdaptiveStartResponse,
)
def start_adaptive(request: Request, puzzle_id: str) -> AdaptiveStartResponse:
    db_path = _require_db(request)
    puzzle = TrainingStore(db_path).get_puzzle(puzzle_id)
    if puzzle is None:
        raise HTTPException(status_code=404, detail="Puzzle not found")
    if not puzzle.active:
        raise HTTPException(status_code=409, detail="Puzzle is no longer active")
    try:
        state = _registry(request).for_settings(_current_settings(request)).start(puzzle)
    except Exception as exc:
        _raise_domain_error(exc)
        raise AssertionError("unreachable") from exc
    return _start_payload(state)


@router.post(
    "/api/practice/adaptive/{session_id}/move",
    response_model=AdaptiveMoveResponse,
)
def adaptive_move(
    request: Request,
    session_id: str,
    payload: AdaptiveMoveRequest,
) -> AdaptiveMoveResponse:
    _require_db(request)
    try:
        result = _registry(request).for_settings(_current_settings(request)).submit_move(
            session_id,
            payload.move_uci,
        )
    except Exception as exc:
        _raise_domain_error(exc)
        raise AssertionError("unreachable") from exc
    return AdaptiveMoveResponse(
        session_id=result.session_id,
        puzzle_id=result.puzzle_id,
        status=result.status,
        accepted=result.accepted,
        move_uci=result.move_uci,
        move_san=result.move_san,
        eval_loss_cp=result.eval_loss_cp,
        engine_reply_uci=result.engine_reply_uci,
        engine_reply_san=result.engine_reply_san,
        current_fen=result.current_fen,
        user_moves_attempted=result.user_moves_attempted,
        user_moves_accepted=result.user_moves_accepted,
        current_ply=result.current_ply,
        max_eval_loss_cp=result.max_eval_loss_cp,
        review=_review_payload(result.review),
    )


@router.post(
    "/api/practice/adaptive/{session_id}/abandon",
    response_model=AdaptiveStartResponse,
)
def abandon_adaptive(request: Request, session_id: str) -> AdaptiveStartResponse:
    _require_db(request)
    try:
        state = _registry(request).for_settings(_current_settings(request)).abandon(session_id)
    except Exception as exc:
        _raise_domain_error(exc)
        raise AssertionError("unreachable") from exc
    return _start_payload(state)
