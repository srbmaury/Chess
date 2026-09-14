from __future__ import annotations

from pathlib import Path

import chess

from chess_ml_coach.adaptive_store import AdaptiveSessionStore, NewAdaptiveStep
from chess_ml_coach.puzzles import PuzzleSeed
from chess_ml_coach.training import TrainingStore

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _fen(*moves: str) -> str:
    board = chess.Board(START_FEN)
    for move in moves:
        board.push_uci(move)
    return board.fen()


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


def _step(
    *,
    side: str,
    fen_before: str,
    move_uci: str,
    move_san: str,
    accepted: bool = True,
    source: str = "user",
) -> NewAdaptiveStep:
    return NewAdaptiveStep(
        side=side,
        fen_before=fen_before,
        move_uci=move_uci,
        move_san=move_san,
        accepted=accepted,
        source=source,
    )


def test_continuation_accuracy_excludes_all_retries_on_starting_position(tmp_path: Path):
    db_path = tmp_path / "training.db"
    training = TrainingStore(db_path)
    training.upsert_puzzles([_seed()])
    puzzle = training.get_puzzle("p1")
    assert puzzle is not None

    sessions = AdaptiveSessionStore(db_path)
    session = sessions.start_or_resume(puzzle, depth=14)
    after_d4 = _fen("d2d4")
    after_reply = _fen("d2d4", "d7d5")
    sessions.advance(
        session.session_id,
        expected_current_fen=START_FEN,
        new_current_fen=_fen("d2d4", "d7d5", "g1f3"),
        steps=[
            _step(
                side="user",
                fen_before=START_FEN,
                move_uci="c2c4",
                move_san="c4",
                accepted=False,
            ),
            _step(
                side="user",
                fen_before=START_FEN,
                move_uci="e2e4",
                move_san="e4",
                accepted=False,
            ),
            _step(
                side="user",
                fen_before=START_FEN,
                move_uci="d2d4",
                move_san="d4",
                source="pv",
            ),
            _step(
                side="engine",
                fen_before=after_d4,
                move_uci="d7d5",
                move_san="d5",
                source="engine",
            ),
            _step(
                side="user",
                fen_before=after_reply,
                move_uci="g1f3",
                move_san="Nf3",
            ),
        ],
        user_attempted_delta=4,
        user_accepted_delta=2,
        current_ply=3,
        max_eval_loss_cp=60,
        previous_best_eval_cp=100,
    )
    sessions.finalize(session.session_id, succeeded=False, answer="c2c4")

    metrics = sessions.metrics()

    assert metrics.sessions_completed == 1
    assert metrics.success_rate == 0.0
    assert metrics.continuation_accuracy == 1.0
    assert metrics.average_accepted_decisions == 2.0
    assert metrics.average_calculation_depth_plies == 3.0
