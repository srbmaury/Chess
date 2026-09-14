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


def _feature_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    games = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "game_date": pd.Timestamp("2026-09-01"),
                "white": "srbmaury",
                "black": "opponent",
                "white_rating": 1500,
                "black_rating": 1510,
                "result": "1-0",
                "time_control": "600+5",
                "rated": True,
                "eco": "C20",
                "opening": "King Pawn",
                "source_url": "url",
            }
        ]
    )
    moves = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "ply": 1,
                "fullmove_number": 1,
                "color": "white",
                "fen_before": START_FEN,
                "fen_after": START_FEN,
                "san": "e4",
                "uci": "e2e4",
                "is_user_move": True,
                "clock_seconds": 600.0,
            },
            {
                "game_id": "g1",
                "ply": 2,
                "fullmove_number": 1,
                "color": "black",
                "fen_before": START_FEN,
                "fen_after": START_FEN,
                "san": "e5",
                "uci": "e7e5",
                "is_user_move": False,
                "clock_seconds": 600.0,
            },
            {
                "game_id": "g1",
                "ply": 3,
                "fullmove_number": 2,
                "color": "white",
                "fen_before": START_FEN,
                "fen_after": START_FEN,
                "san": "Nf3",
                "uci": "g1f3",
                "is_user_move": True,
                "clock_seconds": 595.0,
            },
        ]
    )
    partial_analysis = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "ply": 1,
                "best_move_uci": "e2e4",
                "eval_before_cp": 10,
                "eval_after_cp": -89,
                "cpl": 99,
                "quality": "inaccuracy",
                "engine_config_hash": "x",
            }
        ]
    )
    return games, moves, partial_analysis


def _prepare_feature_paths(settings: Settings) -> None:
    processed = settings.data_dir / "processed"
    engine = settings.data_dir / "engine"
    processed.mkdir(parents=True)
    engine.mkdir(parents=True)
    (processed / "games.parquet").touch()
    (processed / "moves.parquet").touch()
    (engine / "analysis.parquet").touch()


def test_run_features_builds_from_partial_analysis_after_stopped_analyze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    _prepare_feature_paths(settings)
    games, moves, analysis = _feature_inputs()

    monkeypatch.setattr(
        "chess_ml_coach.pgn.read_normalized",
        lambda _processed: (games, moves),
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _path: analysis)

    result = run_features(settings)

    assert result["rows"] == 1
    assert result["analyzed_user_moves"] == 1
    assert result["total_user_moves"] == 2
    assert result["analysis_complete"] is False
    assert Path(result["output"]).exists()


def test_run_features_rejects_when_no_user_moves_have_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    _prepare_feature_paths(settings)
    games, moves, analysis = _feature_inputs()
    empty_analysis = analysis.iloc[0:0]

    monkeypatch.setattr(
        "chess_ml_coach.pgn.read_normalized",
        lambda _processed: (games, moves),
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _path: empty_analysis)

    with pytest.raises(RuntimeError, match="No analyzed user moves"):
        run_features(settings)
