from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

from .adaptive_store import AdaptiveSession, AdaptiveSessionStore, AdaptiveStep, NewAdaptiveStep
from .config import Settings
from .engine import StockfishAdapter, _resolve_stockfish, normalize_score
from .training import ReviewResult, StoredPuzzle, TrainingStore

ACCEPTANCE_TOLERANCE_CP = 30
MAX_USER_DECISIONS = 4
MAX_PLIES = 8


class AdaptivePracticeError(RuntimeError):
    pass


class AdaptiveSessionNotFound(AdaptivePracticeError):
    pass


class AdaptivePuzzleInactive(AdaptivePracticeError):
    pass


class AdaptiveIllegalMove(AdaptivePracticeError):
    pass


class AdaptiveEngineError(AdaptivePracticeError):
    pass


@dataclass(frozen=True)
class AdaptiveState:
    session_id: str
    puzzle_id: str
    status: str
    current_fen: str
    orientation: str
    user_moves_attempted: int
    user_moves_accepted: int
    current_ply: int
    max_eval_loss_cp: int
    steps: list[AdaptiveStep]
    review: ReviewResult | None


@dataclass(frozen=True)
class AdaptiveMoveResult:
    session_id: str
    puzzle_id: str
    status: str
    accepted: bool | None
    move_uci: str | None
    move_san: str | None
    eval_loss_cp: int | None
    engine_reply_uci: str | None
    engine_reply_san: str | None
    current_fen: str
    user_moves_attempted: int
    user_moves_accepted: int
    current_ply: int
    max_eval_loss_cp: int
    review: ReviewResult | None


@dataclass
class _EngineState:
    adapter: object
    expected_user_uci: str | None
    expected_best_eval_cp: int | None = None
    expected_winning_mate: bool | None = None


EngineFactory = Callable[[], object]


class AdaptivePracticeService:
    def __init__(
        self,
        settings: Settings,
        db_path: Path,
        *,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        self.settings = settings
        self.db_path = Path(db_path)
        self.training = TrainingStore(self.db_path)
        self.sessions = AdaptiveSessionStore(self.db_path)
        self._engine_factory = engine_factory
        self._engines: dict[str, _EngineState] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock_for(self, session_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(session_id, threading.Lock())

    def _new_adapter(self):
        try:
            if self._engine_factory is not None:
                return self._engine_factory()
            return StockfishAdapter(_resolve_stockfish(self.settings.stockfish_path))
        except Exception as exc:  # Engine setup failures are user-facing practice errors.
            raise AdaptiveEngineError(str(exc)) from exc

    def _engine_for(self, session: AdaptiveSession, puzzle: StoredPuzzle) -> _EngineState:
        with self._guard:
            existing = self._engines.get(session.session_id)
            if existing is not None:
                return existing
            state = _EngineState(
                adapter=self._new_adapter(),
                expected_user_uci=(
                    puzzle.best_move_uci if session.current_ply == 0 else None
                ),
            )
            self._engines[session.session_id] = state
            return state

    def _replace_adapter(self, session_id: str, state: _EngineState) -> None:
        with suppress(chess.engine.EngineTerminatedError, BrokenPipeError, OSError):
            state.adapter.close()
        state.adapter = self._new_adapter()

    def _analyse(
        self,
        session_id: str,
        state: _EngineState,
        board: chess.Board,
    ) -> dict:
        for attempt in range(2):
            try:
                return state.adapter.analyse(board, self.settings.stockfish_depth)
            except chess.engine.EngineTerminatedError as exc:
                if attempt == 1:
                    raise AdaptiveEngineError("Stockfish stopped during adaptive practice") from exc
                self._replace_adapter(session_id, state)
            except Exception as exc:
                raise AdaptiveEngineError(str(exc)) from exc
        raise AdaptiveEngineError("Stockfish analysis failed")

    @staticmethod
    def _user_color(session: AdaptiveSession) -> chess.Color:
        return chess.WHITE if session.user_color == "white" else chess.BLACK

    @staticmethod
    def _pov_score(info: dict, user_color: chess.Color) -> chess.engine.Score:
        raw = info.get("score")
        if not isinstance(raw, chess.engine.PovScore):
            raise AdaptiveEngineError("Stockfish returned no usable score")
        return raw.pov(user_color)

    @classmethod
    def _winning_mate(cls, info: dict, user_color: chess.Color) -> bool:
        mate = cls._pov_score(info, user_color).mate()
        return mate is not None and mate > 0

    @staticmethod
    def _first_pv_move(info: dict, board: chess.Board) -> chess.Move | None:
        pv = list(info.get("pv") or [])
        if not pv:
            return None
        move = pv[0]
        return move if move in board.legal_moves else None

    @staticmethod
    def _next_user_move_from_reply(
        info: dict,
        board_after_reply: chess.Board,
    ) -> str | None:
        pv = list(info.get("pv") or [])
        if len(pv) < 2:
            return None
        move = pv[1]
        return move.uci() if move in board_after_reply.legal_moves else None

    @staticmethod
    def _is_forcing(board: chess.Board, move: chess.Move | None) -> bool:
        if move is None or move not in board.legal_moves:
            return False
        return bool(board.gives_check(move) or board.is_capture(move) or move.promotion)

    @staticmethod
    def _review_from_session(session: AdaptiveSession) -> ReviewResult | None:
        if not session.review_recorded:
            return None
        if (
            session.review_next_interval_days is None
            or session.review_next_review_at is None
            or session.review_consecutive_correct is None
            or session.review_mastered is None
        ):
            raise AdaptivePracticeError("Terminal adaptive session has incomplete review data")
        return ReviewResult(
            puzzle_id=session.puzzle_id,
            correct=session.status == "succeeded",
            next_interval_days=session.review_next_interval_days,
            consecutive_correct=session.review_consecutive_correct,
            next_review_at=session.review_next_review_at,
            mastered=session.review_mastered,
        )

    def _puzzle_for(self, session: AdaptiveSession) -> StoredPuzzle:
        puzzle = self.training.get_puzzle(session.puzzle_id)
        if puzzle is None or not puzzle.active:
            raise AdaptivePuzzleInactive("Puzzle is no longer active")
        return puzzle

    def _state(self, session: AdaptiveSession) -> AdaptiveState:
        return AdaptiveState(
            session_id=session.session_id,
            puzzle_id=session.puzzle_id,
            status=session.status,
            current_fen=session.current_fen,
            orientation=session.user_color,
            user_moves_attempted=session.user_moves_attempted,
            user_moves_accepted=session.user_moves_accepted,
            current_ply=session.current_ply,
            max_eval_loss_cp=session.max_eval_loss_cp,
            steps=self.sessions.steps(session.session_id),
            review=self._review_from_session(session),
        )

    def start(self, puzzle: StoredPuzzle) -> AdaptiveState:
        if not puzzle.active:
            raise AdaptivePuzzleInactive("Puzzle is no longer active")
        session = self.sessions.start_or_resume(
            puzzle,
            depth=self.settings.stockfish_depth,
        )
        return self._state(session)

    def _terminal_move_result(self, session: AdaptiveSession) -> AdaptiveMoveResult:
        return AdaptiveMoveResult(
            session_id=session.session_id,
            puzzle_id=session.puzzle_id,
            status=session.status,
            accepted=None,
            move_uci=None,
            move_san=None,
            eval_loss_cp=None,
            engine_reply_uci=None,
            engine_reply_san=None,
            current_fen=session.current_fen,
            user_moves_attempted=session.user_moves_attempted,
            user_moves_accepted=session.user_moves_accepted,
            current_ply=session.current_ply,
            max_eval_loss_cp=session.max_eval_loss_cp,
            review=self._review_from_session(session),
        )

    def _first_user_answer(self, session_id: str, fallback: str) -> str:
        for step in self.sessions.steps(session_id):
            if step.side == "user":
                return step.move_uci
        return fallback

    def _finish(
        self,
        session: AdaptiveSession,
        *,
        succeeded: bool,
        fallback_answer: str,
    ) -> tuple[AdaptiveSession, ReviewResult]:
        answer = self._first_user_answer(session.session_id, fallback_answer)
        finished, review = self.sessions.finalize(
            session.session_id,
            succeeded=succeeded,
            answer=answer,
        )
        self.close_session(session.session_id)
        return finished, review

    def _baseline(
        self,
        session: AdaptiveSession,
        puzzle: StoredPuzzle,
        state: _EngineState,
        board: chess.Board,
    ) -> tuple[str | None, int, bool, dict | None]:
        expected = state.expected_user_uci
        if expected is not None and state.expected_best_eval_cp is not None:
            try:
                expected_move = chess.Move.from_uci(expected)
            except ValueError:
                expected_move = None
            if expected_move is not None and expected_move in board.legal_moves:
                return (
                    expected,
                    state.expected_best_eval_cp,
                    bool(state.expected_winning_mate),
                    None,
                )

        info = self._analyse(session.session_id, state, board.copy(stack=False))
        user_color = self._user_color(session)
        best_eval_cp = normalize_score(info["score"], user_color)
        best_move = self._first_pv_move(info, board)
        expected = best_move.uci() if best_move is not None else puzzle.best_move_uci
        state.expected_user_uci = expected
        state.expected_best_eval_cp = best_eval_cp
        state.expected_winning_mate = self._winning_mate(info, user_color)
        return expected, best_eval_cp, bool(state.expected_winning_mate), info

    def submit_move(self, session_id: str, move_uci: str) -> AdaptiveMoveResult:
        lock = self._lock_for(session_id)
        with lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise AdaptiveSessionNotFound(f"Unknown adaptive session: {session_id}")
            if session.status != "active":
                return self._terminal_move_result(session)
            puzzle = self._puzzle_for(session)
            board = chess.Board(session.current_fen)
            user_color = self._user_color(session)
            if board.turn != user_color:
                raise AdaptivePracticeError("Adaptive session is not at the user's turn")
            try:
                move = chess.Move.from_uci(move_uci.lower())
            except ValueError as exc:
                raise AdaptiveIllegalMove("Move must be legal UCI notation") from exc
            if move not in board.legal_moves:
                raise AdaptiveIllegalMove("Submitted move is not legal in this position")

            engine_state = self._engine_for(session, puzzle)
            expected_uci, best_eval_cp, best_winning_mate, _ = self._baseline(
                session,
                puzzle,
                engine_state,
                board,
            )
            move_san = board.san(move)
            after_user = board.copy(stack=False)
            after_user.push(move)
            exact = expected_uci == move.uci()
            candidate_info: dict | None = None
            candidate_eval_cp = best_eval_cp
            loss_cp = 0
            accepted = exact

            if not exact:
                candidate_info = self._analyse(
                    session.session_id,
                    engine_state,
                    after_user.copy(stack=False),
                )
                candidate_eval_cp = normalize_score(candidate_info["score"], user_color)
                if best_winning_mate:
                    accepted = self._winning_mate(candidate_info, user_color)
                    loss_cp = 0 if accepted else ACCEPTANCE_TOLERANCE_CP + 1
                else:
                    loss_cp = max(0, best_eval_cp - candidate_eval_cp)
                    accepted = loss_cp <= ACCEPTANCE_TOLERANCE_CP

            user_step = NewAdaptiveStep(
                side="user",
                fen_before=board.fen(),
                move_uci=move.uci(),
                move_san=move_san,
                accepted=accepted,
                source="pv" if exact else "user",
                eval_cp=candidate_eval_cp,
                best_eval_cp=best_eval_cp,
                eval_loss_cp=loss_cp,
            )
            accepted_count = session.user_moves_accepted + int(accepted)
            ply_after_user = session.current_ply + 1

            if not accepted:
                advanced = self.sessions.advance(
                    session.session_id,
                    expected_current_fen=session.current_fen,
                    new_current_fen=after_user.fen(),
                    steps=[user_step],
                    user_attempted_delta=1,
                    user_accepted_delta=0,
                    current_ply=ply_after_user,
                    max_eval_loss_cp=loss_cp,
                    previous_best_eval_cp=best_eval_cp,
                )
                finished, review = self._finish(
                    advanced,
                    succeeded=False,
                    fallback_answer=move.uci(),
                )
                return AdaptiveMoveResult(
                    session_id=finished.session_id,
                    puzzle_id=finished.puzzle_id,
                    status=finished.status,
                    accepted=False,
                    move_uci=move.uci(),
                    move_san=move_san,
                    eval_loss_cp=loss_cp,
                    engine_reply_uci=None,
                    engine_reply_san=None,
                    current_fen=finished.current_fen,
                    user_moves_attempted=finished.user_moves_attempted,
                    user_moves_accepted=finished.user_moves_accepted,
                    current_ply=finished.current_ply,
                    max_eval_loss_cp=finished.max_eval_loss_cp,
                    review=review,
                )

            stop_after_user = (
                after_user.is_game_over()
                or accepted_count >= MAX_USER_DECISIONS
                or ply_after_user >= MAX_PLIES
            )
            if stop_after_user:
                advanced = self.sessions.advance(
                    session.session_id,
                    expected_current_fen=session.current_fen,
                    new_current_fen=after_user.fen(),
                    steps=[user_step],
                    user_attempted_delta=1,
                    user_accepted_delta=1,
                    current_ply=ply_after_user,
                    max_eval_loss_cp=loss_cp,
                    previous_best_eval_cp=best_eval_cp,
                )
                finished, review = self._finish(
                    advanced,
                    succeeded=True,
                    fallback_answer=move.uci(),
                )
                return AdaptiveMoveResult(
                    session_id=finished.session_id,
                    puzzle_id=finished.puzzle_id,
                    status=finished.status,
                    accepted=True,
                    move_uci=move.uci(),
                    move_san=move_san,
                    eval_loss_cp=loss_cp,
                    engine_reply_uci=None,
                    engine_reply_san=None,
                    current_fen=finished.current_fen,
                    user_moves_attempted=finished.user_moves_attempted,
                    user_moves_accepted=finished.user_moves_accepted,
                    current_ply=finished.current_ply,
                    max_eval_loss_cp=finished.max_eval_loss_cp,
                    review=review,
                )

            reply_info = candidate_info or self._analyse(
                session.session_id,
                engine_state,
                after_user.copy(stack=False),
            )
            reply = self._first_pv_move(reply_info, after_user)
            if reply is None:
                advanced = self.sessions.advance(
                    session.session_id,
                    expected_current_fen=session.current_fen,
                    new_current_fen=after_user.fen(),
                    steps=[user_step],
                    user_attempted_delta=1,
                    user_accepted_delta=1,
                    current_ply=ply_after_user,
                    max_eval_loss_cp=loss_cp,
                    previous_best_eval_cp=best_eval_cp,
                )
                finished, review = self._finish(
                    advanced,
                    succeeded=True,
                    fallback_answer=move.uci(),
                )
                return AdaptiveMoveResult(
                    session_id=finished.session_id,
                    puzzle_id=finished.puzzle_id,
                    status=finished.status,
                    accepted=True,
                    move_uci=move.uci(),
                    move_san=move_san,
                    eval_loss_cp=loss_cp,
                    engine_reply_uci=None,
                    engine_reply_san=None,
                    current_fen=finished.current_fen,
                    user_moves_attempted=finished.user_moves_attempted,
                    user_moves_accepted=finished.user_moves_accepted,
                    current_ply=finished.current_ply,
                    max_eval_loss_cp=finished.max_eval_loss_cp,
                    review=review,
                )

            reply_san = after_user.san(reply)
            after_reply = after_user.copy(stack=False)
            after_reply.push(reply)
            reply_eval_cp = normalize_score(reply_info["score"], user_color)
            engine_step = NewAdaptiveStep(
                side="engine",
                fen_before=after_user.fen(),
                move_uci=reply.uci(),
                move_san=reply_san,
                accepted=True,
                source="engine",
                eval_cp=reply_eval_cp,
                best_eval_cp=reply_eval_cp,
                eval_loss_cp=0,
            )
            ply_after_reply = ply_after_user + 1

            engine_state.expected_user_uci = self._next_user_move_from_reply(
                reply_info,
                after_reply,
            )
            engine_state.expected_best_eval_cp = reply_eval_cp
            engine_state.expected_winning_mate = self._winning_mate(reply_info, user_color)

            should_finish = after_reply.is_game_over() or ply_after_reply >= MAX_PLIES
            if not should_finish and accepted_count >= 2:
                current_info = self._analyse(
                    session.session_id,
                    engine_state,
                    after_reply.copy(stack=False),
                )
                current_eval_cp = normalize_score(current_info["score"], user_color)
                current_move = self._first_pv_move(current_info, after_reply)
                current_winning_mate = self._winning_mate(current_info, user_color)
                forcing = self._is_forcing(after_reply, current_move)
                stable = abs(current_eval_cp - best_eval_cp) <= ACCEPTANCE_TOLERANCE_CP
                should_finish = not current_winning_mate and not forcing and stable
                engine_state.expected_user_uci = (
                    current_move.uci() if current_move is not None else None
                )
                engine_state.expected_best_eval_cp = current_eval_cp
                engine_state.expected_winning_mate = current_winning_mate

            advanced = self.sessions.advance(
                session.session_id,
                expected_current_fen=session.current_fen,
                new_current_fen=after_reply.fen(),
                steps=[user_step, engine_step],
                user_attempted_delta=1,
                user_accepted_delta=1,
                current_ply=ply_after_reply,
                max_eval_loss_cp=loss_cp,
                previous_best_eval_cp=best_eval_cp,
            )
            review: ReviewResult | None = None
            if should_finish:
                advanced, review = self._finish(
                    advanced,
                    succeeded=True,
                    fallback_answer=move.uci(),
                )

            return AdaptiveMoveResult(
                session_id=advanced.session_id,
                puzzle_id=advanced.puzzle_id,
                status=advanced.status,
                accepted=True,
                move_uci=move.uci(),
                move_san=move_san,
                eval_loss_cp=loss_cp,
                engine_reply_uci=reply.uci(),
                engine_reply_san=reply_san,
                current_fen=advanced.current_fen,
                user_moves_attempted=advanced.user_moves_attempted,
                user_moves_accepted=advanced.user_moves_accepted,
                current_ply=advanced.current_ply,
                max_eval_loss_cp=advanced.max_eval_loss_cp,
                review=review,
            )

    def abandon(self, session_id: str) -> AdaptiveState:
        lock = self._lock_for(session_id)
        with lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise AdaptiveSessionNotFound(f"Unknown adaptive session: {session_id}")
            abandoned = self.sessions.abandon(session_id)
            self.close_session(session_id)
            return self._state(abandoned)

    def close_session(self, session_id: str) -> None:
        with self._guard:
            state = self._engines.pop(session_id, None)
            self._locks.pop(session_id, None)
        if state is not None:
            with suppress(chess.engine.EngineTerminatedError, BrokenPipeError, OSError):
                state.adapter.close()

    def close(self) -> None:
        with self._guard:
            session_ids = list(self._engines)
        for session_id in session_ids:
            self.close_session(session_id)
