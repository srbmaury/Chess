from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from chess_ml_coach.puzzles import PuzzleSeed
from chess_ml_coach.training import TrainingStore


def _seed(
    puzzle_id: str = "p1",
    *,
    motif: str = "fork",
    opening: str = "King's Pawn Opening",
    difficulty: int = 3,
) -> PuzzleSeed:
    return PuzzleSeed(
        puzzle_id=puzzle_id,
        game_id=f"game-{puzzle_id}",
        ply=21,
        fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        color="white",
        game_label="srbmaury vs opponent",
        move_label="11.Nf3",
        your_move_san="Nf3",
        your_move_uci="g1f3",
        best_move_san="d4",
        best_move_uci="d2d4",
        cpl=250,
        eval_loss_pawns=2.5,
        quality="blunder",
        opening=opening,
        eco="C20",
        game_phase="middlegame",
        source_url="https://www.chess.com/game/live/1",
        motif=motif,
        difficulty=difficulty,
    )


def _store(tmp_path: Path) -> TrainingStore:
    return TrainingStore(tmp_path / "training.db")


def test_upsert_is_idempotent_and_preserves_review_history(tmp_path: Path):
    store = _store(tmp_path)
    now = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)

    first = store.upsert_puzzles([_seed()], now=now)
    assert first.inserted == 1
    assert first.updated == 0

    store.record_review("p1", answer="Nf3", correct=False, now=now)
    changed = replace(_seed(), opening="Updated Opening")
    second = store.upsert_puzzles([changed], now=now + timedelta(hours=1))

    assert second.inserted == 0
    assert second.updated == 1
    puzzle = store.get_puzzle("p1")
    assert puzzle is not None
    assert puzzle.opening == "Updated Opening"
    assert puzzle.attempts == 1
    assert puzzle.correct_attempts == 0
    assert store.review_count("p1") == 1


def test_rebuild_deactivates_obsolete_puzzles_without_deleting_history(tmp_path: Path):
    store = _store(tmp_path)
    now = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
    store.upsert_puzzles([_seed("p1"), _seed("p2")], now=now)
    store.record_review("p1", answer="e4", correct=False, now=now)

    rebuild_time = now + timedelta(days=2)
    result = store.upsert_puzzles([_seed("p2")], now=rebuild_time)

    assert result.total == 1
    assert [puzzle.puzzle_id for puzzle in store.due_puzzles(limit=10, now=rebuild_time)] == ["p2"]
    stale = store.get_puzzle("p1")
    assert stale is not None
    assert stale.active is False
    assert stale.attempts == 1
    assert store.review_count("p1") == 1
    assert store.progress(now=rebuild_time).total_puzzles == 1

    store.upsert_puzzles(
        [_seed("p1"), _seed("p2")],
        now=rebuild_time + timedelta(hours=1),
    )
    restored = store.get_puzzle("p1")
    assert restored is not None
    assert restored.active is True
    assert restored.attempts == 1
    assert store.review_count("p1") == 1


def test_spaced_repetition_schedule_and_mastery(tmp_path: Path):
    store = _store(tmp_path)
    now = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
    store.upsert_puzzles([_seed()], now=now)

    wrong = store.record_review("p1", answer="e4", correct=False, now=now)
    assert wrong.next_interval_days == 1
    assert wrong.consecutive_correct == 0
    assert wrong.mastered is False

    first = store.record_review("p1", answer="d4", correct=True, now=wrong.next_review_at)
    assert first.next_interval_days == 3
    assert first.consecutive_correct == 1

    second = store.record_review("p1", answer="d4", correct=True, now=first.next_review_at)
    assert second.next_interval_days == 7
    assert second.consecutive_correct == 2

    third = store.record_review("p1", answer="d4", correct=True, now=second.next_review_at)
    assert third.next_interval_days == 14
    assert third.consecutive_correct == 3
    assert third.mastered is False

    fourth = store.record_review("p1", answer="d4", correct=True, now=third.next_review_at)
    assert fourth.next_interval_days == 30
    assert fourth.consecutive_correct == 4
    assert fourth.mastered is True


def test_due_puzzles_excludes_future_reviews_and_respects_limit(tmp_path: Path):
    store = _store(tmp_path)
    now = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
    store.upsert_puzzles([_seed("p1"), _seed("p2", difficulty=5)], now=now)

    store.record_review("p1", answer="d4", correct=True, now=now)
    due_now = store.due_puzzles(limit=10, now=now)
    assert [puzzle.puzzle_id for puzzle in due_now] == ["p2"]

    future = store.due_puzzles(limit=1, now=now + timedelta(days=4))
    assert len(future) == 1
    assert future[0].puzzle_id in {"p1", "p2"}


def test_progress_aggregates_accuracy_mastery_and_motifs(tmp_path: Path):
    store = _store(tmp_path)
    now = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
    store.upsert_puzzles(
        [
            _seed("p1", motif="fork", opening="Opening A"),
            _seed("p2", motif="missed mate", opening="Opening B"),
        ],
        now=now,
    )
    store.record_review("p1", answer="d4", correct=True, now=now)
    store.record_review("p2", answer="e4", correct=False, now=now)

    summary = store.progress(now=now)

    assert summary.total_puzzles == 2
    assert summary.reviewed_puzzles == 2
    assert summary.mastered_puzzles == 0
    assert summary.total_reviews == 2
    assert summary.accuracy == 0.5
    motif_names = {row.label for row in summary.by_motif}
    assert motif_names == {"fork", "missed mate"}
    assert {row.label for row in summary.by_opening} == {"Opening A", "Opening B"}
