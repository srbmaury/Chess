from pathlib import Path

import chess
import chess.engine
import pandas as pd

from chess_ml_coach.config import MoveQualityThresholds, Settings
from chess_ml_coach.engine import (
    analyze_user_moves,
    centipawn_loss,
    normalize_score,
    quality_label,
)


def test_normalize_score_uses_user_perspective():
    score = chess.engine.PovScore(chess.engine.Cp(80), chess.WHITE)
    assert normalize_score(score, chess.WHITE) == 80
    assert normalize_score(score, chess.BLACK) == -80


def test_mate_score_is_bounded_numeric():
    score = chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)
    assert normalize_score(score, chess.WHITE) == 100_000
    assert normalize_score(score, chess.BLACK) == -100_000


def test_centipawn_loss_never_goes_negative():
    assert centipawn_loss(100, 20) == 80
    assert centipawn_loss(20, 100) == 0


def test_quality_threshold_boundaries():
    t = MoveQualityThresholds(50, 100, 200)
    assert quality_label(49, t) == "good"
    assert quality_label(50, t) == "inaccuracy"
    assert quality_label(100, t) == "mistake"
    assert quality_label(200, t) == "blunder"


class FakeEngine:
    def __init__(self):
        self.calls = 0

    def analyse(self, board: chess.Board, depth: int) -> dict:
        self.calls += 1
        cp = 100 if self.calls % 2 == 1 else 20
        move = next(iter(board.legal_moves), None)
        return {
            "score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn),
            "pv": [move] if move is not None else [],
        }

    def close(self) -> None:
        pass


def test_analysis_is_resumable_for_same_engine_config(tmp_path: Path):
    start = chess.Board()
    move = chess.Move.from_uci("e2e4")
    after = start.copy(); after.push(move)
    moves = pd.DataFrame([
        {
            "game_id": "g1",
            "ply": 1,
            "color": "white",
            "fen_before": start.fen(),
            "fen_after": after.fen(),
            "is_user_move": True,
        }
    ])
    output = tmp_path / "analysis.parquet"
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")

    first_engine = FakeEngine()
    first = analyze_user_moves(moves, settings, output, adapter=first_engine)
    assert len(first) == 1
    assert first_engine.calls == 2

    second_engine = FakeEngine()
    second = analyze_user_moves(moves, settings, output, adapter=second_engine)
    assert len(second) == 1
    assert second_engine.calls == 0


def test_changed_engine_config_replaces_stale_analysis(tmp_path: Path):
    start = chess.Board()
    move = chess.Move.from_uci("e2e4")
    after = start.copy(); after.push(move)
    moves = pd.DataFrame([
        {"game_id": "g1", "ply": 1, "color": "white", "fen_before": start.fen(), "fen_after": after.fen(), "is_user_move": True}
    ])
    output = tmp_path / "analysis.parquet"
    first_settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models", stockfish_depth=14)
    second_settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models", stockfish_depth=16)
    first = analyze_user_moves(moves, first_settings, output, adapter=FakeEngine())
    second_engine = FakeEngine()
    second = analyze_user_moves(moves, second_settings, output, adapter=second_engine)
    assert len(first) == 1
    assert len(second) == 1
    assert second_engine.calls == 2
    assert first.iloc[0].engine_config_hash != second.iloc[0].engine_config_hash
