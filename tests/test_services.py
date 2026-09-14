from pathlib import Path

import pandas as pd
import pytest

from chess_ml_coach.config import Settings
from chess_ml_coach.services import (
    answer_to_uci,
    run_features,
    run_puzzles,
    training_db_path,
)

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
    assert "run features first" in message.lower()
    assert "chess-coach" not in message.lower()


def test_run_features_rejects_partial_analysis_after_stopped_analyze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    processed = settings.data_dir / "processed"
    engine = settings.data_dir / "engine"
    processed.mkdir(parents=True)
    engine.mkdir(parents=True)
    (processed / "games.parquet").touch()
    (processed / "moves.parquet").touch()
    (engine / "analysis.parquet").touch()

    moves = pd.DataFrame(
        [
            {"game_id": "g1", "ply": 1, "is_user_move": True},
            {"game_id": "g1", "ply": 2, "is_user_move": True},
            {"game_id": "g1", "ply": 3, "is_user_move": False},
        ]
    )
    analysis = pd.DataFrame([{"game_id": "g1", "ply": 1}])

    monkeypatch.setattr(
        "chess_ml_coach.pgn.read_normalized",
        lambda _processed: (pd.DataFrame(), moves),
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _path: analysis)

    with pytest.raises(RuntimeError, match="Analysis is incomplete") as exc_info:
        run_features(settings)

    message = str(exc_info.value)
    assert "1/2 user moves analyzed" in message
    assert "resume analyze" in message.lower()
    assert "chess-coach" not in message.lower()
