import math

import pandas as pd

from chess_ml_coach.report import build_coaching_report, render_markdown

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
MATED_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"


def _frame() -> pd.DataFrame:
    rows = []
    for i in range(18):
        cpl = 250 if i % 7 == 0 else (120 if i % 3 == 0 else 20)
        rows.append(
            {
                "game_id": f"g{i // 3}",
                "ply": i + 1,
                "fullmove_number": (i // 2) + 1,
                "color": "white" if i < 12 else "black",
                "game_phase": "middlegame" if i < 15 else "endgame",
                "eco": "C20" if i < 12 else "B01",
                "opening": "Kings Pawn Opening" if i < 12 else "Scandinavian Defense",
                "time_control_category": "rapid",
                "significant_mistake": int(i % 3 == 0),
                "quality": "blunder" if i % 7 == 0 else ("mistake" if i % 3 == 0 else "good"),
                "cpl": cpl,
                "eval_before_cp": 20,
                "eval_after_cp": 20 - cpl,
                "fen_before": START_FEN,
                "fen_after": AFTER_E4,
                "san": "e4",
                "uci": "e2e4",
                "best_move_uci": "d2d4",
                "source_url": f"https://www.chess.com/game/live/{1000 + i // 3}",
                "white": "srbmaury",
                "black": "opponent",
            }
        )
    return pd.DataFrame(rows)


def test_small_groups_are_suppressed():
    report = build_coaching_report(_frame(), min_group_size=10)
    assert "King's Pawn Opening (C20)" in report.by_opening.index
    assert "Scandinavian Defense (B01)" not in report.by_opening.index
    assert "white" in report.by_color.index
    assert "black" not in report.by_color.index


def test_candidate_positions_limit_two_per_game():
    report = build_coaching_report(_frame(), min_group_size=2)
    counts = report.candidate_positions.groupby("game_id").size()
    assert (counts <= 2).all()


def test_mate_swing_does_not_destroy_cpl_summary():
    frame = _frame()
    frame.loc[0, "cpl"] = 99_286
    frame.loc[0, "eval_after_cp"] = -100_000
    frame.loc[0, "quality"] = "blunder"
    frame.loc[0, "significant_mistake"] = 1

    report = build_coaching_report(frame, min_group_size=2)

    assert report.overall["mate_blunders"] == 1
    assert report.overall["median_cpl"] < 1_000
    assert report.overall["mean_cpl_non_mate"] < 1_000


def test_same_move_false_blunder_is_sanitized_and_never_trained():
    frame = _frame()
    original_mistakes = int(frame["significant_mistake"].sum())
    frame.loc[0, "best_move_uci"] = frame.loc[0, "uci"]
    frame.loc[0, "cpl"] = 500
    frame.loc[0, "quality"] = "blunder"
    frame.loc[0, "significant_mistake"] = 1

    report = build_coaching_report(frame, min_group_size=2)

    expected_mistakes = original_mistakes - 1
    assert math.isclose(report.overall["mistake_rate"], expected_mistakes / len(frame))
    assert not (report.candidate_positions["your_move"] == report.candidate_positions["better_move"]).any()


def test_delivering_checkmate_is_never_classified_as_a_blunder():
    frame = _frame().head(1).copy()
    frame.loc[0, "color"] = "black"
    frame.loc[0, "san"] = "Qh4#"
    frame.loc[0, "uci"] = "d8h4"
    frame.loc[0, "best_move_uci"] = "a7a6"
    frame.loc[0, "fen_after"] = MATED_FEN
    frame.loc[0, "eval_after_cp"] = -100_000
    frame.loc[0, "cpl"] = 100_100
    frame.loc[0, "quality"] = "blunder"
    frame.loc[0, "significant_mistake"] = 1

    report = build_coaching_report(frame, min_group_size=1)

    assert report.overall["mistake_rate"] == 0
    assert report.overall["blunder_rate"] == 0
    assert report.candidate_positions.empty


def test_markdown_uses_human_labels_and_hides_internal_ids():
    report = build_coaching_report(
        _frame(),
        min_group_size=2,
        feature_importance=[
            {"feature": "numeric__eval_before_cp", "importance": 12},
            {"feature": "numeric__clock_seconds", "importance": 6},
        ],
    )

    text = render_markdown(report)

    assert "## Your priorities" in text
    assert "King's Pawn Opening (C20)" in text
    assert "Mistake rate" in text
    assert "Blunder rate" in text
    assert "Typical eval loss" in text
    assert "Average eval loss" in text
    assert "Mate mistakes" in text
    assert "Your move" in text
    assert "Best move" in text
    assert "Eval loss" in text
    assert "2.50 pawns" in text
    assert "| Game ID |" not in text
    assert "game_id" not in text
    assert "mistake_rate" not in text
    assert "median_cpl" not in text
    assert "Position evaluation" in text
    assert "Time remaining" in text
    assert "66.7%" in text
    assert "numeric__" not in text
    assert "Feature importance is associative, not causal." in text


def test_markdown_keeps_detailed_coaching_sections():
    report = build_coaching_report(_frame(), min_group_size=2)
    text = render_markdown(report)
    for section in [
        "Overall",
        "By color",
        "By phase",
        "Openings",
        "Time controls",
        "Recurring weakness contexts",
        "Candidate training positions",
    ]:
        assert section in text
