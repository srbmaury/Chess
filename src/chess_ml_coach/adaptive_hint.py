from __future__ import annotations

from dataclasses import dataclass

import chess

from .adaptive_practice import (
    AdaptiveEngineError,
    AdaptivePracticeError,
    AdaptivePracticeService,
    AdaptiveSessionNotFound,
)


@dataclass(frozen=True)
class AdaptiveHintResult:
    session_id: str
    puzzle_id: str
    status: str
    move_uci: str
    move_san: str
    current_fen: str
    user_moves_attempted: int
    user_moves_accepted: int
    current_ply: int


def reveal_next_move(
    service: AdaptivePracticeService,
    session_id: str,
) -> AdaptiveHintResult:
    """Reveal only the best move for the current adaptive position.

    Requesting help marks a previously perfect session as assisted by adding one
    attempted-but-not-accepted decision. Repeated hint requests do not add more
    penalties because the session is already non-perfect.
    """

    lock = service._lock_for(session_id)
    with lock:
        session = service.sessions.get(session_id)
        if session is None:
            raise AdaptiveSessionNotFound(f"Unknown adaptive session: {session_id}")
        if session.status != "active":
            raise AdaptivePracticeError("Adaptive session is not active")

        puzzle = service._puzzle_for(session)
        board = chess.Board(session.current_fen)
        if board.turn != service._user_color(session):
            raise AdaptivePracticeError("Adaptive session is not at the user's turn")

        engine_state = service._engine_for(session, puzzle)
        expected_uci, best_eval_cp, _, _ = service._baseline(
            session,
            puzzle,
            engine_state,
            board,
        )
        if expected_uci is None:
            raise AdaptiveEngineError("Stockfish returned no usable best move")

        try:
            move = chess.Move.from_uci(expected_uci)
        except ValueError as exc:
            raise AdaptiveEngineError("Stockfish returned an invalid best move") from exc
        if move not in board.legal_moves:
            raise AdaptiveEngineError("Stockfish returned an illegal best move")

        updated = session
        if session.user_moves_attempted == session.user_moves_accepted:
            updated = service.sessions.advance(
                session.session_id,
                expected_current_fen=session.current_fen,
                new_current_fen=session.current_fen,
                steps=[],
                user_attempted_delta=1,
                user_accepted_delta=0,
                current_ply=session.current_ply,
                max_eval_loss_cp=session.max_eval_loss_cp,
                previous_best_eval_cp=best_eval_cp,
            )

        return AdaptiveHintResult(
            session_id=updated.session_id,
            puzzle_id=updated.puzzle_id,
            status=updated.status,
            move_uci=move.uci(),
            move_san=board.san(move),
            current_fen=updated.current_fen,
            user_moves_attempted=updated.user_moves_attempted,
            user_moves_accepted=updated.user_moves_accepted,
            current_ply=updated.current_ply,
        )
