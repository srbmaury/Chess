from __future__ import annotations

from pathlib import Path

import chess
import chess.engine

from chess_ml_coach.adaptive_practice import AdaptivePracticeService
from chess_ml_coach.config import Settings
from chess_ml_coach.puzzles import PuzzleSeed
from chess_ml_coach.training import TrainingStore

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _position(*moves: str) -> chess.Board:
    board = chess.Board(START_FEN)
    for move in moves:
        board.push_uci(move)
    return board


def _info(cp: int, *pv: str) -> dict:
    return {
        "score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE),
        "pv": [chess.Move.from_uci(move) for move in pv],
    }


class ScriptedEngine:
    def __init__(self, script: dict[str, dict]):
        self.script = script
        self.calls: list[str] = []

    def analyse(self, board: chess.Board, depth: int) -> dict:
        assert depth == 14
        fen = board.fen()
        self.calls.append(fen)
        return self.script[fen]

    def close(self) -> None:
        pass


def _puzzle() -> PuzzleSeed:
    return PuzzleSeed(
        puzzle_id="p1",
        game_id="game-p1",
        ply=1,
        fen_before=START_FEN,
        color="white",
        game_label="player vs opponent",
        move_label="1.d4",
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


def test_initial_stored_best_move_remains_exact_even_if_fresh_pv_changes(tmp_path: Path):
    start = _position()
    after_d4 = _position("d2d4")
    after_reply = _position("d2d4", "d7d5")
    engine = ScriptedEngine(
        {
            # A fresh analysis now prefers 1.c4 by enough that 1.d4 would fail
            # the ordinary alternate-move tolerance if treated as a deviation.
            start.fen(): _info(100, "c2c4", "e7e5"),
            after_d4.fen(): _info(60, "d7d5", "g1f3"),
        }
    )
    db_path = tmp_path / "training.db"
    training = TrainingStore(db_path)
    training.upsert_puzzles([_puzzle()])
    puzzle = training.get_puzzle("p1")
    assert puzzle is not None

    service = AdaptivePracticeService(
        Settings(
            data_dir=tmp_path / "data",
            model_dir=tmp_path / "models",
            stockfish_depth=14,
            stockfish_path="unused-in-tests",
        ),
        db_path,
        engine_factory=lambda: engine,
    )
    session = service.start(puzzle)

    result = service.submit_move(session.session_id, puzzle.best_move_uci)

    assert result.accepted is True
    assert result.status == "active"
    assert result.eval_loss_cp == 0
    assert result.engine_reply_uci == "d7d5"
    assert result.current_fen == after_reply.fen()
    assert training.review_count("p1") == 0
