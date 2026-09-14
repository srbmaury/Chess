import chess
import chess.engine

from chess_ml_coach.move_quality import (
    ScoreSnapshot,
    classify_move_quality,
    display_loss_pawns,
    score_snapshot,
)


def snapshot(*, cp: int | None = None, mate: int | None = None, expected: float) -> ScoreSnapshot:
    return ScoreSnapshot(cp=cp, mate=mate, expected_score=expected)


def test_score_snapshot_preserves_mate_direction_for_both_colors():
    white_mates = chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)

    white = score_snapshot(white_mates, chess.WHITE)
    black = score_snapshot(white_mates, chess.BLACK)

    assert white.mate == 3
    assert white.cp is None
    assert white.expected_score > 0.99
    assert black.mate == -3
    assert black.cp is None
    assert black.expected_score < 0.01


def test_score_snapshot_distinguishes_mate_given_from_being_mated():
    mate_given = chess.engine.PovScore(chess.engine.MateGiven, chess.WHITE)
    mated = chess.engine.PovScore(chess.engine.Mate(0), chess.WHITE)

    assert score_snapshot(mate_given, chess.WHITE).mate == 0
    assert score_snapshot(mate_given, chess.WHITE).expected_score == 1.0
    assert score_snapshot(mated, chess.WHITE).mate == 0
    assert score_snapshot(mated, chess.WHITE).expected_score == 0.0


def test_exact_engine_move_is_best():
    assessment = classify_move_quality(
        before=snapshot(cp=35, expected=0.58),
        after=snapshot(cp=35, expected=0.58),
        cpl=0,
        is_best_move=True,
    )

    assert assessment.label == "best"
    assert assessment.reason == "engine top move"


def test_small_near_best_losses_are_excellent_then_good():
    excellent = classify_move_quality(
        before=snapshot(cp=80, expected=0.66),
        after=snapshot(cp=60, expected=0.65),
        cpl=20,
        is_best_move=False,
    )
    good = classify_move_quality(
        before=snapshot(cp=120, expected=0.72),
        after=snapshot(cp=60, expected=0.69),
        cpl=60,
        is_best_move=False,
    )

    assert excellent.label == "excellent"
    assert good.label == "good"


def test_winning_chance_drop_drives_inaccuracy_mistake_and_blunder():
    inaccuracy = classify_move_quality(
        before=snapshot(cp=90, expected=0.64),
        after=snapshot(cp=40, expected=0.56),
        cpl=50,
        is_best_move=False,
    )
    mistake = classify_move_quality(
        before=snapshot(cp=160, expected=0.72),
        after=snapshot(cp=30, expected=0.54),
        cpl=130,
        is_best_move=False,
    )
    blunder = classify_move_quality(
        before=snapshot(cp=250, expected=0.82),
        after=snapshot(cp=-80, expected=0.45),
        cpl=330,
        is_best_move=False,
    )

    assert inaccuracy.label == "inaccuracy"
    assert mistake.label == "mistake"
    assert blunder.label == "blunder"


def test_forced_mate_thrown_away_is_a_miss_not_a_numeric_blunder():
    assessment = classify_move_quality(
        before=snapshot(mate=4, expected=1.0),
        after=snapshot(cp=120, expected=0.72),
        cpl=99_876,
        is_best_move=False,
    )

    assert assessment.label == "miss"
    assert assessment.reason == "forced mate was available"
    assert display_loss_pawns(99_876, assessment.reason) is None


def test_allowing_forced_mate_is_always_a_blunder():
    assessment = classify_move_quality(
        before=snapshot(cp=15, expected=0.52),
        after=snapshot(mate=-3, expected=0.0),
        cpl=99_985,
        is_best_move=False,
    )

    assert assessment.label == "blunder"
    assert assessment.reason == "allows forced mate"
    assert display_loss_pawns(99_985, assessment.reason) is None


def test_throwing_away_a_decisive_non_mate_win_is_a_miss():
    assessment = classify_move_quality(
        before=snapshot(cp=650, expected=0.96),
        after=snapshot(cp=30, expected=0.55),
        cpl=620,
        is_best_move=False,
    )

    assert assessment.label == "miss"
    assert assessment.reason == "decisive winning chance was missed"


def test_brilliant_requires_verified_candidate_and_near_best_quality():
    brilliant = classify_move_quality(
        before=snapshot(cp=220, expected=0.80),
        after=snapshot(cp=205, expected=0.79),
        cpl=15,
        is_best_move=True,
        brilliant_candidate=True,
    )
    ordinary_best = classify_move_quality(
        before=snapshot(cp=220, expected=0.80),
        after=snapshot(cp=205, expected=0.79),
        cpl=15,
        is_best_move=True,
        brilliant_candidate=False,
    )

    assert brilliant.label == "brilliant"
    assert brilliant.reason == "sound sacrifice and uniquely strong move"
    assert ordinary_best.label == "best"


def test_normal_centipawn_loss_is_still_displayed_in_pawns():
    assert display_loss_pawns(123, "") == 1.23
