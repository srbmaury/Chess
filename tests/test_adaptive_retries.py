from __future__ import annotations

from pathlib import Path

import chess
import chess.engine

from chess_ml_coach.adaptive_practice import AdaptivePracticeService
from chess_ml_coach.adaptive_store import AdaptiveSessionStore
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

    def analyse(self, board: chess.Board, depth: int) -> dict:
        assert depth == 14
        result = self.script.get(board.fen())
        if result is None:
            raise AssertionError(f"No scripted analysis for {board.fen()}")
        return result

    def close(self) -> None:
        pass


def _seed() -> PuzzleSeed:
    return PuzzleSeed(
        puzzle_id="p1",
        game_id="game-p1",
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


def _service(tmp_path: Path, engine: ScriptedEngine):
    db_path = tmp_path / "training.db"
    training = TrainingStore(db_path)
    training.upsert_puzzles([_seed()])
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
    return service, training, puzzle


def test_rejected_moves_keep_same_position_until_an_accepted_retry(tmp_path: Path):
    start = _position()
    after_c4 = _position("c2c4")
    after_e4 = _position("e2e4")
    after_d4 = _position("d2d4")
    after_reply = _position("d2d4", "d7d5")
    engine = ScriptedEngine(
        {
            start.fen(): _info(100, "d2d4", "d7d5", "g1f3"),
            after_c4.fen(): _info(50, "e7e5"),
            after_e4.fen(): _info(40, "e7e5"),
            after_d4.fen(): _info(100, "d7d5", "g1f3"),
        }
    )
    service, training, puzzle = _service(tmp_path, engine)
    session = service.start(puzzle)

    first = service.submit_move(session.session_id, "c2c4")
    second = service.submit_move(session.session_id, "e2e4")
    corrected = service.submit_move(session.session_id, "d2d4")

    assert first.accepted is False
    assert first.status == "active"
    assert first.current_fen == START_FEN
    assert first.user_moves_attempted == 1
    assert first.user_moves_accepted == 0
    assert first.current_ply == 0
    assert first.review is None

    assert second.accepted is False
    assert second.status == "active"
    assert second.current_fen == START_FEN
    assert second.user_moves_attempted == 2
    assert second.user_moves_accepted == 0
    assert second.current_ply == 0
    assert second.review is None

    assert corrected.accepted is True
    assert corrected.status == "active"
    assert corrected.current_fen == after_reply.fen()
    assert corrected.user_moves_attempted == 3
    assert corrected.user_moves_accepted == 1
    assert corrected.current_ply == 2
    assert training.review_count("p1") == 0

    steps = AdaptiveSessionStore(training.path).steps(session.session_id)
    assert [step.accepted for step in steps] == [False, False, True, True]
    assert [step.fen_before for step in steps[:3]] == [START_FEN, START_FEN, START_FEN]


def test_completed_line_after_a_retry_records_an_incorrect_review(tmp_path: Path):
    start = _position()
    after_c4 = _position("c2c4")
    after_d4 = _position("d2d4")
    after_nf3 = _position("d2d4", "d7d5", "g1f3")
    quiet = _position("d2d4", "d7d5", "g1f3", "g8f6")
    engine = ScriptedEngine(
        {
            start.fen(): _info(100, "d2d4", "d7d5", "g1f3"),
            after_c4.fen(): _info(50, "e7e5"),
            after_d4.fen(): _info(100, "d7d5", "g1f3"),
            after_nf3.fen(): _info(100, "g8f6", "c2c4"),
            quiet.fen(): _info(105, "c2c4"),
        }
    )
    service, training, puzzle = _service(tmp_path, engine)
    session = service.start(puzzle)

    missed = service.submit_move(session.session_id, "c2c4")
    first = service.submit_move(session.session_id, "d2d4")
    finished = service.submit_move(session.session_id, "g1f3")

    assert missed.status == "active"
    assert first.status == "active"
    assert finished.accepted is True
    assert finished.status == "failed"
    assert finished.current_fen == quiet.fen()
    assert finished.user_moves_attempted == 3
    assert finished.user_moves_accepted == 2
    assert finished.review is not None
    assert finished.review.correct is False
    assert finished.review.next_interval_days == 1
    assert training.review_count("p1") == 1
