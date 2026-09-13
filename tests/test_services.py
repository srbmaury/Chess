from pathlib import Path

from chess_ml_coach.config import Settings
from chess_ml_coach.services import answer_to_uci, run_puzzles, training_db_path

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_training_db_path_uses_local_training_directory(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")

    assert training_db_path(settings) == tmp_path / "data" / "training" / "training.db"


def test_answer_to_uci_accepts_legal_san_and_uci_and_rejects_illegal_moves():
    assert answer_to_uci(START_FEN, "e4") == "e2e4"
    assert answer_to_uci(START_FEN, "d2d4") == "d2d4"
    assert answer_to_uci(START_FEN, "e5") is None


def test_run_puzzles_requires_feature_dataset(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")

    try:
        run_puzzles(settings)
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("run_puzzles should require features.parquet")

    assert "features.parquet" in message
    assert "chess-coach features" in message
