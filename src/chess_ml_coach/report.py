from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import pandas as pd

MATE_THRESHOLD_CP = 50_000


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


def _mate_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    if "eval_before_cp" in frame.columns:
        mask |= pd.to_numeric(frame["eval_before_cp"], errors="coerce").abs() >= MATE_THRESHOLD_CP
    if "eval_after_cp" in frame.columns:
        mask |= pd.to_numeric(frame["eval_after_cp"], errors="coerce").abs() >= MATE_THRESHOLD_CP
    if "cpl" in frame.columns:
        mask |= pd.to_numeric(frame["cpl"], errors="coerce") >= MATE_THRESHOLD_CP
    return mask.fillna(False)


def _summary_stats(frame: pd.DataFrame) -> dict[str, float | int]:
    mate_mask = _mate_mask(frame)
    normal_cpl = pd.to_numeric(frame.loc[~mate_mask, "cpl"], errors="coerce").dropna()
    return {
        "samples": len(frame),
        "mistake_rate": float(frame["significant_mistake"].mean()),
        "blunder_rate": float((frame["quality"] == "blunder").mean()),
        "median_cpl": float(normal_cpl.median()) if not normal_cpl.empty else 0.0,
        "mean_cpl_non_mate": float(normal_cpl.mean()) if not normal_cpl.empty else 0.0,
        "mate_blunders": int((mate_mask & (frame["quality"] == "blunder")).sum()),
    }


def _aggregate(frame: pd.DataFrame, column: str, min_group_size: int) -> pd.DataFrame:
    if column not in frame.columns:
        return pd.DataFrame(
            columns=[
                "samples",
                "mistake_rate",
                "blunder_rate",
                "median_cpl",
                "mean_cpl_non_mate",
                "mate_blunders",
            ]
        )
    rows: list[dict] = []
    labels: list[object] = []
    for label, group in frame.groupby(column, dropna=False):
        stats = _summary_stats(group)
        if int(stats["samples"]) < min_group_size:
            continue
        labels.append(label)
        rows.append(stats)
    result = pd.DataFrame(rows, index=labels)
    result.index.name = column
    if result.empty:
        return result
    return result.sort_values(["mistake_rate", "samples"], ascending=[False, False])


def _opening_label(row: pd.Series) -> str:
    eco = str(row.get("eco") or "unknown")
    opening = row.get("opening")
    if opening is None or pd.isna(opening) or str(opening).strip().lower() in {"", "unknown", "nan"}:
        return eco
    return f"{eco} — {str(opening).strip()}"


def _recurring_contexts(
    overall_mistake_rate: float,
    tables: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for dimension, table in tables:
        for context, row in table.iterrows():
            excess = float(row.mistake_rate) - overall_mistake_rate
            score = excess * math.log1p(int(row.samples))
            rows.append(
                {
                    "dimension": dimension,
                    "context": str(context),
                    "samples": int(row.samples),
                    "mistake_rate": float(row.mistake_rate),
                    "blunder_rate": float(row.blunder_rate),
                    "median_cpl": float(row.median_cpl),
                    "vs_baseline": excess,
                    "_score": score,
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
                "median_cpl",
                "vs_baseline",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("_score", ascending=False)
        .drop(columns=["_score"])
        .reset_index(drop=True)
    )


def _best_move_san(fen: object, best_move_uci: object) -> str:
    if fen is None or best_move_uci is None or pd.isna(fen) or pd.isna(best_move_uci):
        return "unknown"
    try:
        board = chess.Board(str(fen))
        move = chess.Move.from_uci(str(best_move_uci))
        return board.san(move)
    except (ValueError, AssertionError):
        return str(best_move_uci)


def _move_label(row: pd.Series) -> str:
    fullmove = row.get("fullmove_number")
    san = row.get("san")
    if fullmove is None or pd.isna(fullmove):
        return str(san or row.get("uci") or "unknown")
    separator = "." if row.get("color") == "white" else "..."
    return f"{int(fullmove)}{separator}{san or row.get('uci') or 'unknown'}"


def _game_label(row: pd.Series) -> str:
    white = row.get("white")
    black = row.get("black")
    if white is None or black is None or pd.isna(white) or pd.isna(black):
        return str(row.get("game_id", "unknown"))
    return f"{white} vs {black}"


def _candidate_positions(
    frame: pd.DataFrame,
    limit: int = 20,
    max_per_game: int = 2,
) -> pd.DataFrame:
    rows: list[dict] = []
    per_game: dict[str, int] = {}
    mate_mask = _mate_mask(frame)
    ranked = frame.assign(_mate_related=mate_mask).sort_values("cpl", ascending=False)
    for _, row in ranked.iterrows():
        game_id = str(row["game_id"])
        if per_game.get(game_id, 0) >= max_per_game:
            continue
        cpl = float(row["cpl"])
        rows.append(
            {
                "game_id": game_id,
                "game": _game_label(row),
                "move": _move_label(row),
                "your_move": str(row.get("san") or row.get("uci") or "unknown"),
                "better_move": _best_move_san(row.get("fen_before"), row.get("best_move_uci")),
                "loss": "mate swing" if bool(row["_mate_related"]) else f"{int(cpl)} CPL",
                "quality": str(row.get("quality", "unknown")),
                "game_url": str(row.get("source_url") or ""),
            }
        )
        per_game[game_id] = per_game.get(game_id, 0) + 1
        if len(rows) >= limit:
            break
    return pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "game",
            "move",
            "your_move",
            "better_move",
            "loss",
            "quality",
            "game_url",
        ],
    )


def build_coaching_report(
    frame: pd.DataFrame,
    min_group_size: int = 10,
    *,
    feature_importance: list[dict] | None = None,
) -> CoachingReport:
    if frame.empty:
        raise ValueError("Feature dataset is empty")
    overall = _summary_stats(frame)
    overall_mistake_rate = float(overall["mistake_rate"])
    by_color = _aggregate(frame, "color", min_group_size)
    by_phase = _aggregate(frame, "game_phase", min_group_size)

    opening_frame = frame.copy()
    opening_frame["opening_display"] = opening_frame.apply(_opening_label, axis=1)
    by_opening = _aggregate(opening_frame, "opening_display", min_group_size)
    by_opening.index.name = "opening"

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


def _format_cell(column: str, value: object) -> str:
    if pd.isna(value):
        return "n/a"
    if column in {"mistake_rate", "blunder_rate"}:
        return f"{float(value):.1%}"
    if column == "vs_baseline":
        return f"{float(value) * 100:+.1f} pp"
    if column in {"median_cpl", "mean_cpl_non_mate"}:
        return f"{float(value):.1f}"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
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
    body = []
    for row in data_rows:
        body.append(
            "| "
            + " | ".join(_format_cell(column, value) for column, value in zip(columns, row))
            + " |"
        )
    return "\n".join([header, separator, *body])


def _context_title(dimension: str, context: str) -> str:
    if dimension in {"color", "phase", "time_control"}:
        return context.replace("_", " ").title()
    return context


def _priority_lines(report: CoachingReport) -> list[str]:
    positive = report.recurring_contexts[report.recurring_contexts["vs_baseline"] > 0].head(3)
    if positive.empty:
        return ["- No recurring context is clearly worse than your overall baseline yet."]
    lines: list[str] = []
    for _, row in positive.iterrows():
        title = _context_title(str(row["dimension"]), str(row["context"]))
        lines.append(
            f"- **{title}**: {float(row['mistake_rate']):.1%} significant mistakes "
            f"across {int(row['samples'])} moves "
            f"({float(row['vs_baseline']) * 100:+.1f} percentage points vs your baseline)."
        )
    return lines


def render_markdown(report: CoachingReport) -> str:
    lines = [
        "# Chess ML Coach Report",
        "",
        "## Your priorities",
        *_priority_lines(report),
        "",
        "Use these as training priorities, not absolute judgments; larger samples are more reliable.",
        "",
        "## Overall",
        f"- Analyzed moves: {report.overall['samples']}",
        f"- Significant mistake rate: {float(report.overall['mistake_rate']):.1%}",
        f"- Blunder rate: {float(report.overall['blunder_rate']):.1%}",
        f"- Median CPL (excluding mate-score sentinels): {float(report.overall['median_cpl']):.1f}",
        f"- Mean CPL (excluding mate-score sentinels): {float(report.overall['mean_cpl_non_mate']):.1f}",
        f"- Mate-related blunders: {report.overall['mate_blunders']}",
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
        "These are positions from your own games to review first.",
        "",
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
