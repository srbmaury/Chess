from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import chess
import chess.engine
import pytest

from chess_ml_coach.adaptive_practice import (
    AdaptiveEngineError,
    AdaptiveIllegalMove,
    AdaptivePracticeService,
)
from chess_ml_coach.adaptive_store import AdaptiveSessionStore, NewAdaptiveStep
from chess_ml_coach.config import Settings
from chess_ml_coach.puzzles import PuzzleSeed
from chess_ml_coach.training import TrainingStore

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _seed() -> PuzzleSeed:
    return PuzzleSeed(
        puzzle_id="p1",
        game_id="game-p1",
        ply=1,
        fen_before=START_FEN,
        color="white",
        game_label="srbmaury vs opponent",
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
        source_url="https://www.chess.com/game/live/1",
        motif="positional / calculation",
        difficulty=3,
    )


def _info(score: chess.engine.Score, *pv: str):
    return {
        "score": chess.engine.PovScore(score, chess.WHITE),
        "pv": [chess.Move.from_uci(move) for move in pv],
    }


class ScriptedEngine:
    def __init__(self, script: dict[str, dict | Exception]):
        self.script = script
        self.calls: list[str] = []
        self.closed = False

    def analyse(self, board: chess.Board, depth: int):
        assert depth == 14
        fen = board.fen()
        self.calls.append(fen)
        result = self.script.get(fen)
        if isinstance(result, Exception):
            raise result
        if result is None:
            raise AssertionError(f"No scripted analysis for {fen}")
        return result

    def close(self):
        self.closed = True


def _position(*moves: str) -> chess.Board:
    board = chess.Board(START_FEN)
    for uci in moves:
        board.push_uci(uci)
    return board


def _service(tmp_path: Path, engine: ScriptedEngine):
    db_path = tmp_path / "training.db"
    training = TrainingStore(db_path)
    now = datetime(2026, 9, 14, 6, 0, tzinfo=UTC)
    training.upsert_puzzles([_seed()], now=now)
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        stockfish_depth=14,
        stockfish_path="unused-in-tests",
    )
    service = AdaptivePracticeService(
        settings,
        db_path,
        engine_factory=lambda: engine,
    )
    puzzle = training.get_puzzle("p1")
    assert puzzle is not None
    return service, training, puzzle, now


def test_exact_current_pv_move_is_accepted_without_candidate_comparison(tmp_path: Path):
    start = _position()
    after_d4 = _position("d2d4")
    after_reply = _position("d2d4", "d7d5")
    engine = ScriptedEngine(
        {
            start.fen(): _info(chess.engine.Cp(100), "d2d4", "d7d5", "g1f3"),
            after_d4.fen(): _info(chess.engine.Cp(100), "d7d5", "g1f3"),
        }
    )
    service, training, puzzle, _ = _service(tmp_path, engine)
    session = service.start(puzzle)

    result = service.submit_move(session.session_id, "d2d4")

    assert result.accepted is True
    assert result.status == "active"
    assert result.eval_loss_cp == 0
    assert result.engine_reply_uci == "d7d5"
    assert result.current_fen == after_reply.fen()
    assert len(engine.calls) == 2
    assert training.review_count("p1") == 0


def test_alternate_move_within_30_cp_is_accepted_and_line_branches(tmp_path: Path):
    start = _position()
    after_c4 = _position("c2c4")
    after_reply = _position("c2c4", "e7e5")
    engine = ScriptedEngine(
        {
            start.fen(): _info(chess.engine.Cp(100), "d2d4", "d7d5"),
            after_c4.fen(): _info(chess.engine.Cp(80), "e7e5", "g1f3"),
        }
    )
    service, training, puzzle, _ = _service(tmp_path, engine)
    session = service.start(puzzle)

    result = service.submit_move(session.session_id, "c2c4")

    assert result.accepted is True
    assert result.eval_loss_cp == 20
    assert result.engine_reply_uci == "e7e5"
    assert result.current_fen == after_reply.fen()
    assert training.review_count("p1") == 0


def test_alternate_move_beyond_30_cp_keeps_same_position_for_retry(tmp_path: Path):
    start = _position()
    after_c4 = _position("c2c4")
    engine = ScriptedEngine(
        {
            start.fen(): _info(chess.engine.Cp(100), "d2d4"),
            after_c4.fen(): _info(chess.engine.Cp(69), "e7e5"),
        }
    )
    service, training, puzzle, _ = _service(tmp_path, engine)
    session = service.start(puzzle)

    result = service.submit_move(session.session_id, "c2c4")
    repeated = service.submit_move(session.session_id, "c2c4")

    assert result.accepted is False
    assert result.status == "active"
    assert result.eval_loss_cp == 31
    assert result.current_fen == START_FEN
    assert result.current_ply == 0
    assert repeated.status == "active"
    assert repeated.current_fen == START_FEN
    assert repeated.user_moves_attempted == 2
    assert training.review_count("p1") == 0


def test_forced_winning_mate_must_be_preserved_without_ending_retry(tmp_path: Path):
    start = _position()
    after_c4 = _position("c2c4")
    engine = ScriptedEngine(
        {
            start.fen(): _info(chess.engine.Mate(3), "d2d4"),
            after_c4.fen(): _info(chess.engine.Cp(500), "e7e5"),
        }
    )
    service, training, puzzle, _ = _service(tmp_path, engine)
    session = service.start(puzzle)

    result = service.submit_move(session.session_id, "c2c4")

    assert result.accepted is False
    assert result.status == "active"
    assert result.current_fen == START_FEN
    assert result.current_ply == 0
    assert training.review_count("p1") == 0


def test_alternate_move_that_preserves_winning_mate_is_accepted(tmp_path: Path):
    start = _position()
    after_c4 = _position("c2c4")
    after_reply = _position("c2c4", "e7e5")
    engine = ScriptedEngine(
        {
            start.fen(): _info(chess.engine.Mate(3), "d2d4"),
            after_c4.fen(): _info(chess.engine.Mate(4), "e7e5", "g1f3"),
        }
    )
    service, training, puzzle, _ = _service(tmp_path, engine)
    session = service.start(puzzle)

    result = service.submit_move(session.session_id, "c2c4")

    assert result.accepted is True
    assert result.current_fen == after_reply.fen()
    assert training.review_count("p1") == 0


def test_illegal_move_does_not_mutate_session_or_call_engine(tmp_path: Path):
    engine = ScriptedEngine({})
    service, training, puzzle, _ = _service(tmp_path, engine)
    session = service.start(puzzle)

    with pytest.raises(AdaptiveIllegalMove):
        service.submit_move(session.session_id, "e2e5")

    current = AdaptiveSessionStore(training.path).get(session.session_id)
    assert current is not None
    assert current.user_moves_attempted == 0
    assert current.current_fen == START_FEN
    assert engine.calls == []
    assert training.review_count("p1") == 0


def test_quiet_stable_position_after_two_decisions_completes_sequence(tmp_path: Path):
    start = _position()
    after_d4 = _position("d2d4")
    turn_two = _position("d2d4", "d7d5")
    after_nf3 = _position("d2d4", "d7d5", "g1f3")
    quiet = _position("d2d4", "d7d5", "g1f3", "g8f6")
    engine = ScriptedEngine(
        {
            start.fen(): _info(chess.engine.Cp(100), "d2d4", "d7d5", "g1f3"),
            after_d4.fen(): _info(chess.engine.Cp(100), "d7d5", "g1f3"),
            turn_two.fen(): _info(chess.engine.Cp(100), "g1f3", "g8f6", "c2c4"),
            after_nf3.fen(): _info(chess.engine.Cp(100), "g8f6", "c2c4"),
            quiet.fen(): _info(chess.engine.Cp(105), "c2c4"),
        }
    )
    service, training, puzzle, _ = _service(tmp_path, engine)
    session = service.start(puzzle)
    first = service.submit_move(session.session_id, "d2d4")
    assert first.status == "active"

    second = service.submit_move(session.session_id, "g1f3")

    assert second.accepted is True
    assert second.status == "succeeded"
    assert second.current_fen == quiet.fen()
    assert second.review is not None
    assert second.review.next_interval_days == 3
    assert training.review_count("p1") == 1


def test_fourth_accepted_user_decision_finishes_without_engine_reply(tmp_path: Path):
    start = _position()
    after_d4 = _position("d2d4")
    engine = ScriptedEngine(
        {
            start.fen(): _info(chess.engine.Cp(100), "d2d4"),
            after_d4.fen(): _info(chess.engine.Cp(100), "d7d5"),
        }
    )
    service, training, puzzle, now = _service(tmp_path, engine)
    session = service.start(puzzle)
    store = AdaptiveSessionStore(training.path)
    store.advance(
        session.session_id,
        expected_current_fen=START_FEN,
        new_current_fen=START_FEN,
        steps=[
            NewAdaptiveStep(
                side="user",
                fen_before=START_FEN,
                move_uci="a2a3",
                move_san="a3",
                accepted=True,
                source="user",
            ),
            NewAdaptiveStep(
                side="user",
                fen_before=START_FEN,
                move_uci="b2b3",
                move_san="b3",
                accepted=True,
                source="user",
            ),
            NewAdaptiveStep(
                side="user",
                fen_before=START_FEN,
                move_uci="c2c3",
                move_san="c3",
                accepted=True,
                source="user",
            ),
        ],
        user_attempted_delta=3,
        user_accepted_delta=3,
        current_ply=6,
        max_eval_loss_cp=0,
        previous_best_eval_cp=100,
        now=now,
    )

    result = service.submit_move(session.session_id, "d2d4")

    assert result.status == "succeeded"
    assert result.engine_reply_uci is None
    assert result.current_ply == 7
    assert training.review_count("p1") == 1


def test_engine_failure_leaves_session_active_and_unpenalized(tmp_path: Path):
    start = _position()
    engine = ScriptedEngine({start.fen(): RuntimeError("engine unavailable")})
    service, training, puzzle, _ = _service(tmp_path, engine)
    session = service.start(puzzle)

    with pytest.raises(AdaptiveEngineError):
        service.submit_move(session.session_id, "d2d4")

    current = AdaptiveSessionStore(training.path).get(session.session_id)
    assert current is not None
    assert current.status == "active"
    assert current.user_moves_attempted == 0
    assert training.review_count("p1") == 0
