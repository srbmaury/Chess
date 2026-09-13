import pandas as pd

from chess_ml_coach.report import build_coaching_report, render_markdown

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


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
                "opening": "King Pawn" if i < 12 else "Scandinavian",
                "time_control_category": "rapid",
                "significant_mistake": int(i % 3 == 0),
                "quality": "blunder" if i % 7 == 0 else ("mistake" if i % 3 == 0 else "good"),
                "cpl": cpl,
                "eval_before_cp": 20,
                "eval_after_cp": 20 - cpl,
                "fen_before": START_FEN,
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
    assert "C20 — King Pawn" in report.by_opening.index
    assert "B01 — Scandinavian" not in report.by_opening.index
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


def test_markdown_starts_with_actionable_human_summary():
    frame = _frame()
    frame.loc[0, "cpl"] = 99_286
    frame.loc[0, "eval_after_cp"] = -100_000
    report = build_coaching_report(
        frame,
        min_group_size=2,
        feature_importance=[{"feature": "engine_eval_before_cp", "importance": 12}],
    )

    text = render_markdown(report)

    assert "## Your priorities" in text
    assert "Mate-related blunders: 1" in text
    assert "C20 — King Pawn" in text
    assert "https://www.chess.com/game/live/1000" in text
    assert "1.e4" in text
    assert "d4" in text
    assert "99286" not in text
    assert "weakness_score" not in text
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
