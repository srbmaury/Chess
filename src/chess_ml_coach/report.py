from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CoachingReport:
    overall: dict[str, float | int]
    by_color: pd.DataFrame
    by_phase: pd.DataFrame
    by_opening: pd.DataFrame
    by_time_control: pd.DataFrame
    recurring_contexts: pd.DataFrame
    candidate_positions: pd.DataFrame
    feature_importance: pd.DataFrame


def _aggregate(frame: pd.DataFrame, column: str, min_group_size: int) -> pd.DataFrame:
    if column not in frame.columns:
        return pd.DataFrame(columns=["samples", "mistake_rate", "blunder_rate", "mean_cpl"])
    grouped = (
        frame.assign(_is_blunder=(frame["quality"] == "blunder").astype(int))
        .groupby(column, dropna=False)
        .agg(
            samples=("significant_mistake", "size"),
            mistake_rate=("significant_mistake", "mean"),
            blunder_rate=("_is_blunder", "mean"),
            mean_cpl=("cpl", "mean"),
        )
        .sort_values(["mistake_rate", "samples"], ascending=[False, False])
    )
    return grouped[grouped["samples"] >= min_group_size]


def _recurring_contexts(
    overall_mistake_rate: float,
    tables: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for dimension, table in tables:
        for context, row in table.iterrows():
            score = (float(row.mistake_rate) - overall_mistake_rate) * math.log1p(int(row.samples))
            rows.append(
                {
                    "dimension": dimension,
                    "context": str(context),
                    "samples": int(row.samples),
                    "mistake_rate": float(row.mistake_rate),
                    "blunder_rate": float(row.blunder_rate),
                    "mean_cpl": float(row.mean_cpl),
                    "weakness_score": score,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "dimension",
                "context",
                "samples",
                "mistake_rate",
                "blunder_rate",
                "mean_cpl",
                "weakness_score",
            ]
        )
    return pd.DataFrame(rows).sort_values("weakness_score", ascending=False).reset_index(drop=True)


def _candidate_positions(frame: pd.DataFrame, limit: int = 20, max_per_game: int = 2) -> pd.DataFrame:
    wanted = [
        column
        for column in ["game_id", "ply", "cpl", "quality", "fen_before", "uci", "best_move_uci"]
        if column in frame.columns
    ]
    rows: list[pd.Series] = []
    per_game: dict[str, int] = {}
    for _, row in frame.sort_values("cpl", ascending=False).iterrows():
        game_id = str(row["game_id"])
        if per_game.get(game_id, 0) >= max_per_game:
            continue
        rows.append(row)
        per_game[game_id] = per_game.get(game_id, 0) + 1
        if len(rows) >= limit:
            break
    if not rows:
        return pd.DataFrame(columns=wanted)
    return pd.DataFrame(rows)[wanted].reset_index(drop=True)


def build_coaching_report(
    frame: pd.DataFrame,
    min_group_size: int = 10,
    *,
    feature_importance: list[dict] | None = None,
) -> CoachingReport:
    if frame.empty:
        raise ValueError("Feature dataset is empty")
    overall_mistake_rate = float(frame["significant_mistake"].mean())
    overall = {
        "samples": len(frame),
        "mistake_rate": overall_mistake_rate,
        "blunder_rate": float((frame["quality"] == "blunder").mean()),
        "mean_cpl": float(frame["cpl"].mean()),
    }
    by_color = _aggregate(frame, "color", min_group_size)
    by_phase = _aggregate(frame, "game_phase", min_group_size)
    by_opening = _aggregate(frame, "eco", min_group_size)
    by_time_control = _aggregate(frame, "time_control_category", min_group_size)
    contexts = _recurring_contexts(
        overall_mistake_rate,
        [
            ("color", by_color),
            ("phase", by_phase),
            ("opening", by_opening),
            ("time_control", by_time_control),
        ],
    )
    importance_frame = pd.DataFrame(feature_importance or [], columns=["feature", "importance"])
    return CoachingReport(
        overall=overall,
        by_color=by_color,
        by_phase=by_phase,
        by_opening=by_opening,
        by_time_control=by_time_control,
        recurring_contexts=contexts,
        candidate_positions=_candidate_positions(frame),
        feature_importance=importance_frame,
    )


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if 0 <= value <= 1:
            return f"{value:.1%}"
        return f"{value:.2f}"
    return str(value)


def _table_markdown(frame: pd.DataFrame, *, include_index: bool = True, max_rows: int = 12) -> str:
    limited = frame.head(max_rows).copy()
    if limited.empty:
        return "_Insufficient sample size._"
    if include_index:
        index_name = limited.index.name or "group"
        columns = [index_name, *limited.columns.tolist()]
        data_rows = [[idx, *row.tolist()] for idx, row in limited.iterrows()]
    else:
        columns = limited.columns.tolist()
        data_rows = limited.values.tolist()
    header = "| " + " | ".join(str(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_fmt(value) for value in row) + " |" for row in data_rows]
    return "\n".join([header, separator, *body])


def render_markdown(report: CoachingReport) -> str:
    lines = [
        "# Chess ML Coach Report",
        "",
        "## Overall",
        f"- Analyzed moves: {report.overall['samples']}",
        f"- Significant mistake rate: {_fmt(report.overall['mistake_rate'])}",
        f"- Blunder rate: {_fmt(report.overall['blunder_rate'])}",
        f"- Mean CPL: {_fmt(report.overall['mean_cpl'])}",
        "",
        "## By color",
        _table_markdown(report.by_color),
        "",
        "## By phase",
        _table_markdown(report.by_phase),
        "",
        "## Openings",
        _table_markdown(report.by_opening),
        "",
        "## Time controls",
        _table_markdown(report.by_time_control),
        "",
        "## Recurring weakness contexts",
        _table_markdown(report.recurring_contexts, include_index=False),
        "",
        "## Candidate training positions",
        _table_markdown(report.candidate_positions, include_index=False, max_rows=20),
    ]
    if not report.feature_importance.empty:
        lines.extend(
            [
                "",
                "## Model feature importance",
                "Feature importance is associative, not causal.",
                "",
                _table_markdown(report.feature_importance, include_index=False),
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
