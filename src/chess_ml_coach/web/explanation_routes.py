from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..explanations import ExplanationStore, PuzzleExplanationService
from ..services import training_db_path
from ..training import TrainingStore

router = APIRouter()


@router.get("/api/practice/{puzzle_id}/explanation")
def puzzle_explanation(request: Request, puzzle_id: str) -> dict[str, object]:
    settings = request.app.state.settings
    db_path = training_db_path(settings)
    if not db_path.exists():
        raise HTTPException(status_code=409, detail="Puzzle bank not found")
    puzzle = TrainingStore(db_path).get_puzzle(puzzle_id)
    if puzzle is None or not puzzle.active:
        raise HTTPException(status_code=404, detail="Puzzle not found")
    if puzzle.attempts <= 0:
        raise HTTPException(
            status_code=409,
            detail="Attempt this puzzle before requesting its explanation.",
        )
    return PuzzleExplanationService(settings, ExplanationStore(db_path)).explain(puzzle)
