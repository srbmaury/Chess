from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from chess_ml_coach.adaptive_store import AdaptiveSessionStore, NewAdaptiveStep
from chess_ml_coach.puzzles import PuzzleSeed
from chess_ml_coach.training import TrainingStore

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_D4 = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"
AFTER_D4_D5 = "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2"


def _seed(puzzle_id: str) -> PuzzleSeed:
    return PuzzleSeed(
        puzzle_id=puzzle_id,
        game_id=f"game-{puzzle_id}",
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


def _stores(tmp_path: Path, *puzzles: PuzzleSeed):
    path = tmp_path / "training.db"
    training = TrainingStore(path)
    now = datetime(2026, 9, 14, 6, 0, tzinfo=UTC)
    training.upsert_puzzles(list(puzzles), now=now)
    sessions = AdaptiveSessionStore(path)
    return training, sessions, now


def _step(
    *,
    side: str,
    fen_before: str,
    move_uci: str,
    move_san: str,
    accepted: bool = True,
    source: str = "user",
    eval_cp: int | None = 100,
    best_eval_cp: int | None = 100,
    eval_loss_cp: int | None = 0,
) -> NewAdaptiveStep:
    return NewAdaptiveStep(
        side=side,
        fen_before=fen_before,
        move_uci=move_uci,
        move_san=move_san,
        accepted=accepted,
        source=source,
        eval_cp=eval_cp,
        best_eval_cp=best_eval_cp,
        eval_loss_cp=eval_loss_cp,
    )


def test_start_or_resume_keeps_one_active_session_per_puzzle(tmp_path: Path):
    _, sessions, now = _stores(tmp_path, _seed("p1"))
    puzzle = TrainingStore(sessions.path).get_puzzle("p1")
    assert puzzle is not None

    first = sessions.start_or_resume(puzzle, depth=14, now=now)
    second = sessions.start_or_resume(puzzle, depth=18, now=now)

    assert second.session_id == first.session_id
    assert second.current_fen == START_FEN
    assert second.engine_depth == 14
    assert second.status == "active"


def test_schema_upgrade_preserves_existing_reviews(tmp_path: Path):
    path = tmp_path / "training.db"
    training = TrainingStore(path)
    now = datetime(2026, 9, 14, 6, 0, tzinfo=UTC)
    training.upsert_puzzles([_seed("p1")], now=now)
    training.record_review("p1", answer="e2e4", correct=False, now=now)

    AdaptiveSessionStore(path)

    puzzle = training.get_puzzle("p1")
    assert puzzle is not None
    assert puzzle.attempts == 1
    assert training.review_count("p1") == 1


def test_advance_persists_ordered_steps_and_session_counters(tmp_path: Path):
    _, sessions, now = _stores(tmp_path, _seed("p1"))
    puzzle = TrainingStore(sessions.path).get_puzzle("p1")
    assert puzzle is not None
    session = sessions.start_or_resume(puzzle, depth=14, now=now)

    advanced = sessions.advance(
        session.session_id,
        expected_current_fen=START_FEN,
        new_current_fen=AFTER_D4_D5,
        steps=[
            _step(
                side="user",
                fen_before=START_FEN,
                move_uci="d2d4",
                move_san="d4",
                source="pv",
            ),
            _step(
                side="engine",
                fen_before=AFTER_D4,
                move_uci="d7d5",
                move_san="d5",
                source="engine",
            ),
        ],
        user_attempted_delta=1,
        user_accepted_delta=1,
        current_ply=2,
        max_eval_loss_cp=0,
        previous_best_eval_cp=100,
        now=now,
    )

    assert advanced.current_fen == AFTER_D4_D5
    assert advanced.user_moves_attempted == 1
    assert advanced.user_moves_accepted == 1
    assert advanced.current_ply == 2
    stored = sessions.steps(session.session_id)
    assert [step.step_index for step in stored] == [0, 1]
    assert [step.side for step in stored] == ["user", "engine"]
    assert [step.move_uci for step in stored] == ["d2d4", "d7d5"]


def test_finalize_is_transactional_and_idempotent(tmp_path: Path):
    training, sessions, now = _stores(tmp_path, _seed("p1"))
    puzzle = training.get_puzzle("p1")
    assert puzzle is not None
    session = sessions.start_or_resume(puzzle, depth=14, now=now)
    sessions.advance(
        session.session_id,
        expected_current_fen=START_FEN,
        new_current_fen=AFTER_D4,
        steps=[
            _step(
                side="user",
                fen_before=START_FEN,
                move_uci="d2d4",
                move_san="d4",
                source="pv",
            )
        ],
        user_attempted_delta=1,
        user_accepted_delta=1,
        current_ply=1,
        max_eval_loss_cp=0,
        previous_best_eval_cp=100,
        now=now,
    )

    finished, review = sessions.finalize(
        session.session_id,
        succeeded=True,
        answer="d2d4",
        now=now,
    )
    again, again_review = sessions.finalize(
        session.session_id,
        succeeded=True,
        answer="d2d4",
        now=now,
    )

    assert finished.status == "succeeded"
    assert finished.review_recorded is True
    assert finished.review_id is not None
    assert again.review_id == finished.review_id
    assert again_review == review
    assert review.next_interval_days == 3
    assert training.review_count("p1") == 1


def test_abandon_records_no_spaced_repetition_review(tmp_path: Path):
    training, sessions, now = _stores(tmp_path, _seed("p1"))
    puzzle = training.get_puzzle("p1")
    assert puzzle is not None
    session = sessions.start_or_resume(puzzle, depth=14, now=now)

    abandoned = sessions.abandon(session.session_id, now=now)

    assert abandoned.status == "abandoned"
    assert abandoned.review_recorded is False
    assert training.review_count("p1") == 0


def test_metrics_exclude_first_user_move_from_continuation_accuracy(tmp_path: Path):
    training, sessions, now = _stores(tmp_path, _seed("p1"), _seed("p2"))

    for puzzle_id, continuation_correct, succeeded in [
        ("p1", True, True),
        ("p2", False, False),
    ]:
        puzzle = training.get_puzzle(puzzle_id)
        assert puzzle is not None
        session = sessions.start_or_resume(puzzle, depth=14, now=now)
        sessions.advance(
            session.session_id,
            expected_current_fen=START_FEN,
            new_current_fen=AFTER_D4_D5,
            steps=[
                _step(
                    side="user",
                    fen_before=START_FEN,
                    move_uci="d2d4",
                    move_san="d4",
                    source="pv",
                ),
                _step(
                    side="engine",
                    fen_before=AFTER_D4,
                    move_uci="d7d5",
                    move_san="d5",
                    source="engine",
                ),
                _step(
                    side="user",
                    fen_before=AFTER_D4_D5,
                    move_uci="g1f3",
                    move_san="Nf3",
                    accepted=continuation_correct,
                    eval_loss_cp=0 if continuation_correct else 80,
                ),
            ],
            user_attempted_delta=2,
            user_accepted_delta=2 if continuation_correct else 1,
            current_ply=3,
            max_eval_loss_cp=0 if continuation_correct else 80,
            previous_best_eval_cp=100,
            now=now,
        )
        sessions.finalize(
            session.session_id,
            succeeded=succeeded,
            answer="d2d4",
            now=now,
        )

    metrics = sessions.metrics()

    assert metrics.sessions_completed == 2
    assert metrics.success_rate == 0.5
    assert metrics.continuation_accuracy == 0.5
    assert metrics.average_accepted_decisions == 1.5
    assert metrics.average_calculation_depth_plies == 3.0

