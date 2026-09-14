import pandas as pd

from chess_ml_coach.config import MoveQualityThresholds
from chess_ml_coach.features import (
    build_feature_dataset,
    extract_position_features,
    time_control_category,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_start_position_features_are_stable():
    f = extract_position_features(START_FEN)
    assert f["legal_move_count"] == 20
    assert f["total_non_king_material"] == 78
    assert f["white_castling_rights"] == 2
    assert f["black_castling_rights"] == 2
    assert f["white_queen_present"] == 1
    assert f["black_queen_present"] == 1
    assert f["in_check"] == 0


def test_pawn_structure_counts_doubled_isolated_and_islands():
    f = extract_position_features("8/8/8/8/8/P7/P1P5/4K2k w - - 0 1")
    assert f["white_doubled_pawns"] == 1
    assert f["white_isolated_pawns"] == 3
    assert f["white_pawn_islands"] == 2


def test_daily_time_control_is_not_lumped_into_unknown():
    assert time_control_category("1/86400") == "daily"


def _game() -> pd.DataFrame:
    return pd.DataFrame([
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
    ])


def test_feature_dataset_keeps_only_user_moves_and_builds_binary_target():
    games = _game()
    moves = pd.DataFrame([
        {"game_id": "g1", "ply": 1, "fullmove_number": 1, "color": "white", "fen_before": START_FEN, "fen_after": START_FEN, "san": "e4", "uci": "e2e4", "is_user_move": True, "clock_seconds": 600.0},
        {"game_id": "g1", "ply": 2, "fullmove_number": 1, "color": "black", "fen_before": START_FEN, "fen_after": START_FEN, "san": "e5", "uci": "e7e5", "is_user_move": False, "clock_seconds": 600.0},
        {"game_id": "g1", "ply": 3, "fullmove_number": 2, "color": "white", "fen_before": START_FEN, "fen_after": START_FEN, "san": "Nf3", "uci": "g1f3", "is_user_move": True, "clock_seconds": 595.0},
    ])
    analysis = pd.DataFrame([
        {"game_id": "g1", "ply": 1, "best_move_uci": "e2e4", "eval_before_cp": 10, "eval_after_cp": -89, "cpl": 99, "quality": "inaccuracy", "engine_config_hash": "x"},
        {"game_id": "g1", "ply": 3, "best_move_uci": "g1f3", "eval_before_cp": 20, "eval_after_cp": -80, "cpl": 100, "quality": "mistake", "engine_config_hash": "x"},
    ])
    frame = build_feature_dataset(games, moves, analysis, MoveQualityThresholds())
    assert frame.ply.tolist() == [1, 3]
    assert frame.significant_mistake.tolist() == [0, 1]
    assert frame.time_control_category.tolist() == ["rapid", "rapid"]
    assert frame.rating_difference.tolist() == [-10, -10]


def test_binary_target_uses_quality_not_raw_cpl_after_v3_scoring():
    games = _game()
    moves = pd.DataFrame([
        {"game_id": "g1", "ply": 1, "fullmove_number": 1, "color": "white", "fen_before": START_FEN, "fen_after": START_FEN, "san": "e4", "uci": "e2e4", "is_user_move": True, "clock_seconds": 600.0},
        {"game_id": "g1", "ply": 3, "fullmove_number": 2, "color": "white", "fen_before": START_FEN, "fen_after": START_FEN, "san": "Nf3", "uci": "g1f3", "is_user_move": True, "clock_seconds": 595.0},
    ])
    analysis = pd.DataFrame([
        {"game_id": "g1", "ply": 1, "best_move_uci": "d2d4", "eval_before_cp": 900, "eval_after_cp": 400, "cpl": 500, "quality": "good", "engine_config_hash": "x"},
        {"game_id": "g1", "ply": 3, "best_move_uci": "d2d4", "eval_before_cp": 500, "eval_after_cp": 450, "cpl": 50, "quality": "miss", "engine_config_hash": "x"},
    ])

    frame = build_feature_dataset(games, moves, analysis, MoveQualityThresholds())

    assert frame.significant_mistake.tolist() == [0, 1]
