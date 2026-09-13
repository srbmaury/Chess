import pandas as pd

from chess_ml_coach.report import build_coaching_report, render_markdown


def _frame() -> pd.DataFrame:
    rows = []
    for i in range(18):
        rows.append(
            {
                "game_id": f"g{i // 3}",
                "ply": i + 1,
                "color": "white" if i < 12 else "black",
                "game_phase": "middlegame" if i < 15 else "endgame",
                "eco": "C20" if i < 12 else "B01",
                "opening": "King Pawn" if i < 12 else "Scandinavian",
                "time_control_category": "rapid",
                "significant_mistake": int(i % 3 == 0),
                "quality": "blunder" if i % 7 == 0 else ("mistake" if i % 3 == 0 else "good"),
                "cpl": 250 if i % 7 == 0 else (120 if i % 3 == 0 else 20),
                "fen_before": f"fen-{i}",
                "uci": "e2e4",
                "best_move_uci": "d2d4",
            }
        )
    return pd.DataFrame(rows)


def test_small_groups_are_suppressed():
    report = build_coaching_report(_frame(), min_group_size=10)
    assert "C20" in report.by_opening.index
    assert "B01" not in report.by_opening.index
    assert "white" in report.by_color.index
    assert "black" not in report.by_color.index


def test_candidate_positions_limit_two_per_game():
    report = build_coaching_report(_frame(), min_group_size=2)
    counts = report.candidate_positions.groupby("game_id").size()
    assert (counts <= 2).all()


def test_markdown_contains_coaching_sections_and_causality_warning():
    report = build_coaching_report(
        _frame(),
        min_group_size=2,
        feature_importance=[{"feature": "engine_eval_before_cp", "importance": 12}],
    )
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
    assert "Feature importance is associative, not causal." in text
