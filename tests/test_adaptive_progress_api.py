from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from chess_ml_coach.adaptive_store import AdaptiveSessionStore, NewAdaptiveStep
from chess_ml_coach.config import Settings
from chess_ml_coach.puzzles import PuzzleSeed
from chess_ml_coach.services import training_db_path
from chess_ml_coach.training import TrainingStore
from chess_ml_coach.web.app import create_app

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _seed(puzzle_id: str) -> PuzzleSeed:
    return PuzzleSeed(
        puzzle_id=puzzle_id,
        game_id=f"game-{puzzle_id}",
        ply=1,
        fen_before=START_FEN,
        color="white",
        game_label="player vs opponent",
        move_label="1.e4",
        your_move_san="e4",
        your_move_uci="e2e4",
        best_move_san="d4",
        best_move_uci="d2d4",
        cpl=250,
        eval_loss_pawns=2.5,
        quality="blunder",
        opening="Queen's Pawn Opening",
        eco="D00",
        game_phase="opening",
        source_url="",
        motif="positional / calculation",
        difficulty=3,
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        username="player",
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        stockfish_depth=14,
    )


def _finish_session(
    sessions: AdaptiveSessionStore,
    puzzle,
    *,
    succeeded: bool,
    second_accepted: bool,
) -> None:
    session = sessions.start_or_resume(puzzle, depth=14)
    sessions.advance(
        session.session_id,
        expected_current_fen=START_FEN,
        new_current_fen=START_FEN,
        steps=[
            NewAdaptiveStep(
                side="user",
                fen_before=START_FEN,
                move_uci="d2d4",
                move_san="d4",
                accepted=True,
                source="pv",
                eval_cp=100,
                best_eval_cp=100,
                eval_loss_cp=0,
            ),
            NewAdaptiveStep(
                side="engine",
                fen_before=START_FEN,
                move_uci="d7d5",
                move_san="d5",
                accepted=True,
                source="engine",
            ),
            NewAdaptiveStep(
                side="user",
                fen_before=START_FEN,
                move_uci="g1f3",
                move_san="Nf3",
                accepted=second_accepted,
                source="user",
                eval_cp=90 if second_accepted else 20,
                best_eval_cp=100,
                eval_loss_cp=10 if second_accepted else 80,
            ),
        ],
        user_attempted_delta=2,
        user_accepted_delta=2 if second_accepted else 1,
        current_ply=3,
        max_eval_loss_cp=10 if second_accepted else 80,
        previous_best_eval_cp=100,
    )
    sessions.finalize(
        session.session_id,
        succeeded=succeeded,
        answer="d2d4",
    )


def test_progress_includes_adaptive_metrics_without_changing_review_accuracy(tmp_path: Path):
    settings = _settings(tmp_path)
    db_path = training_db_path(settings)
    training = TrainingStore(db_path)
    training.upsert_puzzles([_seed("p1"), _seed("p2")])
    sessions = AdaptiveSessionStore(db_path)

    first = training.get_puzzle("p1")
    second = training.get_puzzle("p2")
    assert first is not None and second is not None
    _finish_session(sessions, first, succeeded=True, second_accepted=True)
    _finish_session(sessions, second, succeeded=False, second_accepted=False)

    response = TestClient(create_app(settings)).get("/api/progress")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_reviews"] == 2
    assert payload["accuracy"] == 0.5
    assert payload["adaptive"] == {
        "sessions_completed": 2,
        "success_rate": 0.5,
        "continuation_accuracy": 0.5,
        "average_accepted_decisions": 1.5,
        "average_calculation_depth_plies": 3.0,
    }
