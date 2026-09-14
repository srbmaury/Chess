from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from chess_ml_coach.config import Settings
from chess_ml_coach.puzzles import PuzzleSeed
from chess_ml_coach.training import TrainingStore
from chess_ml_coach.web.app import create_app

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_practice_feedback_uses_quality_reason_instead_of_fake_mate_pawn_loss(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    store = TrainingStore(settings.data_dir / "training" / "training.db")
    store.upsert_puzzles(
        [
            PuzzleSeed(
                puzzle_id="mate-miss",
                game_id="g1",
                ply=1,
                fen_before=START_FEN,
                color="white",
                game_label="player vs opponent",
                move_label="1.e4",
                your_move_san="e4",
                your_move_uci="e2e4",
                best_move_san="d4",
                best_move_uci="d2d4",
                cpl=99_808,
                eval_loss_pawns=None,
                quality="miss",
                quality_reason="forced mate was available",
                opening="Opening",
                eco="A00",
                game_phase="opening",
                source_url="",
                motif="missed mate",
                difficulty=5,
            )
        ],
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    response = TestClient(create_app(settings)).post(
        "/api/practice/mate-miss/attempt",
        json={"move_uci": "d2d4"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["quality"] == "miss"
    assert payload["quality_reason"] == "forced mate was available"
    assert payload["evaluation_loss_pawns"] is None
