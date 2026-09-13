from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chess_ml_coach.train import TrainingDataError, chronological_game_split, train_model


def _training_frame() -> pd.DataFrame:
    rows = []
    for game_index in range(12):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=game_index)
        for ply in range(4):
            rows.append(
                {
                    "game_id": f"g{game_index}",
                    "game_date": date,
                    "ply": ply + 1,
                    "color": "white" if ply % 2 == 0 else "black",
                    "game_phase": ["opening", "middlegame", "middlegame", "endgame"][ply],
                    "time_control_category": "rapid" if game_index % 2 == 0 else "blitz",
                    "eco": "C20" if game_index % 2 == 0 else "D00",
                    "opening": "King Pawn" if game_index % 2 == 0 else "Queen Pawn",
                    "user_rating": 1500 + game_index,
                    "opponent_rating": 1490 + game_index,
                    "rating_difference": 10,
                    "legal_move_count": 20 + ply,
                    "material_balance": ply - 1,
                    "total_non_king_material": 78 - game_index,
                    "in_check": int(ply == 3),
                    "own_castling_rights": 2 if ply == 0 else 1,
                    "own_doubled_pawns": game_index % 2,
                    "own_isolated_pawns": ply % 2,
                    "own_pawn_islands": 4 - (ply % 2),
                    "king_ring_attacks": ply,
                    "engine_eval_before_cp": (game_index - 6) * 10 + ply,
                    "clock_seconds": 600 - 10 * ply,
                    "cpl": 120 if (game_index + ply) % 3 == 0 else 25,
                    "quality": "mistake" if (game_index + ply) % 3 == 0 else "good",
                    "significant_mistake": int((game_index + ply) % 3 == 0),
                    "fen_before": "fen",
                    "fen_after": "fen2",
                    "san": "e4",
                    "uci": "e2e4",
                    "best_move_uci": "e2e4",
                    "eval_after_cp": 0,
                }
            )
    return pd.DataFrame(rows)


def test_split_never_leaks_a_game_across_boundaries():
    df = _training_frame()
    train, test = chronological_game_split(df, test_fraction=0.25)
    assert set(train.game_id).isdisjoint(set(test.game_id))
    assert train.game_date.max() <= test.game_date.min()


def test_single_class_training_is_rejected():
    df = _training_frame()
    df["significant_mistake"] = 0
    with pytest.raises(TrainingDataError, match="both target classes"):
        chronological_game_split(df)


def test_train_model_saves_pipeline_and_metrics(tmp_path: Path):
    result = train_model(_training_frame(), tmp_path)
    assert result.model_path.exists()
    assert result.metadata_path.exists()
    assert set(result.metrics) >= {"roc_auc", "pr_auc", "log_loss", "brier_score"}
    assert all(value is None or np.isfinite(value) for value in result.metrics.values())
