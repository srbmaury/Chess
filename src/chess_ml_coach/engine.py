from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from hashlib import sha256
from pathlib import Path
from typing import Protocol

import chess
import chess.engine
import pandas as pd

from .config import MoveQualityThresholds, Settings
from .locking import ProfileBusyError, exclusive_profile_lock
from .move_quality import (
    MATE_CP,
    MoveAssessment,
    ScoreSnapshot,
    alternatives_show_uniqueness,
    classify_move_quality,
    display_loss_pawns,
    sacrifices_material_after_reply,
    score_snapshot,
)

SCORING_VERSION = 3
ProgressCallback = Callable[[dict[str, object]], None]
ANALYSIS_COLUMNS = [
    "game_id",
    "ply",
    "best_move_uci",
    "eval_before_cp",
    "eval_after_cp",
    "mate_before",
    "mate_after",
    "expected_score_before",
    "expected_score_after",
    "cpl",
    "quality",
    "quality_reason",
    "engine_config_hash",
    "scoring_version",
]
REQUIRED_LEGACY_COLUMNS = {
    "game_id",
    "ply",
    "best_move_uci",
    "eval_before_cp",
    "eval_after_cp",
    "cpl",
    "quality",
    "engine_config_hash",
}


class EngineConfigurationError(RuntimeError):
    pass


class EngineAdapter(Protocol):
    def analyse(self, board: chess.Board, depth: int) -> dict: ...
    def close(self) -> None: ...


class StockfishAdapter:
    def __init__(self, path: str):
        self.path = path
        self.engine = chess.engine.SimpleEngine.popen_uci(path)

    def analyse(self, board: chess.Board, depth: int) -> dict:
        return self.engine.analyse(board, chess.engine.Limit(depth=depth))

    def analyse_multipv(self, board: chess.Board, depth: int, multipv: int = 2) -> list[dict]:
        result = self.engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            multipv=max(2, int(multipv)),
        )
        return result if isinstance(result, list) else [result]

    def close(self) -> None:
        self.engine.quit()


def normalize_score(score: chess.engine.PovScore, user_color: chess.Color) -> int:
    value = score.pov(user_color)
    converted = value.score(mate_score=MATE_CP)
    return int(converted or 0)


def centipawn_loss(best_eval_cp: int, played_eval_cp: int) -> int:
    return max(0, int(best_eval_cp) - int(played_eval_cp))


def quality_label(cpl: int, thresholds: MoveQualityThresholds) -> str:
    """Legacy CPL-only fallback retained for callers that lack score context."""
    if cpl >= thresholds.blunder:
        return "blunder"
    if cpl >= thresholds.mistake:
        return "mistake"
    if cpl >= thresholds.inaccuracy:
        return "inaccuracy"
    return "good"


def _config_payload(settings: Settings, *, scoring_version: int | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "depth": settings.stockfish_depth,
        "thresholds": {
            "inaccuracy": settings.thresholds.inaccuracy,
            "mistake": settings.thresholds.mistake,
            "blunder": settings.thresholds.blunder,
        },
    }
    if scoring_version is not None:
        payload["scoring_version"] = scoring_version
    return payload


def _hash_payload(payload: dict[str, object]) -> str:
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _config_hash(settings: Settings) -> str:
    return _hash_payload(_config_payload(settings, scoring_version=SCORING_VERSION))


def _resolve_stockfish(path: str | None) -> str:
    if not path:
        raise EngineConfigurationError(
            "Stockfish is not configured. Set STOCKFISH_PATH or pass --stockfish-path."
        )
    resolved = shutil.which(path)
    if resolved:
        return resolved
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate)
    raise EngineConfigurationError(f"Stockfish executable not found: {path}")


@contextmanager
def _analysis_lock(
    output_path: Path,
    profile_lock_path: Path | None = None,
) -> Iterator[None]:
    lock_path = profile_lock_path or output_path.with_suffix(".lock")

    def busy_message(path: Path) -> str:
        return (
            f"Analysis is already running for {output_path}, or another profile "
            f"operation holds {path}. If no operation is running, remove the stale lock."
        )

    try:
        with exclusive_profile_lock(lock_path, error_message=busy_message):
            yield
    except ProfileBusyError as exc:
        raise EngineConfigurationError(str(exc)) from exc


def _write_analysis(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=".parquet",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        frame.sort_values(["game_id", "ply"]).to_parquet(temp_path, index=False)
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _emit_progress(
    progress: ProgressCallback | None,
    *,
    completed: int,
    total: int,
    reused: int,
    analyzed: int,
    game_id: str | None = None,
    ply: int | None = None,
) -> None:
    if progress is None:
        return
    progress(
        {
            "stage": "analyze",
            "completed": completed,
            "total": total,
            "reused": reused,
            "analyzed": analyzed,
            "game_id": game_id,
            "ply": ply,
        }
    )


def _load_reusable_analysis(
    moves: pd.DataFrame,
    settings: Settings,
    output_path: Path,
    *,
    force: bool,
) -> pd.DataFrame:
    del moves
    if force or not output_path.exists():
        return pd.DataFrame(columns=ANALYSIS_COLUMNS)

    raw = pd.read_parquet(output_path)
    missing = REQUIRED_LEGACY_COLUMNS - set(raw.columns)
    if missing:
        raise EngineConfigurationError(
            f"Invalid analysis artifact {output_path}: missing {sorted(missing)}"
        )
    if "scoring_version" not in raw.columns:
        raw["scoring_version"] = 1
    for column in ANALYSIS_COLUMNS:
        if column not in raw.columns:
            raw[column] = None

    config_hash = _config_hash(settings)
    current = raw[
        (raw["engine_config_hash"] == config_hash)
        & (pd.to_numeric(raw["scoring_version"], errors="coerce") == SCORING_VERSION)
    ].copy()

    if current.empty:
        return pd.DataFrame(columns=ANALYSIS_COLUMNS)
    return (
        current[ANALYSIS_COLUMNS]
        .drop_duplicates(["game_id", "ply"], keep="last")
        .sort_values(["game_id", "ply"])
        .reset_index(drop=True)
    )


def _verified_brilliant_candidate(
    *,
    adapter: EngineAdapter,
    before_board: chess.Board,
    after_board: chess.Board,
    after_info: dict,
    user_color: chess.Color,
    depth: int,
) -> bool:
    reply_line = after_info.get("pv") or []
    reply = reply_line[0] if reply_line and isinstance(reply_line[0], chess.Move) else None
    if not sacrifices_material_after_reply(before_board, after_board, user_color, reply):
        return False

    analyse_multipv = getattr(adapter, "analyse_multipv", None)
    if not callable(analyse_multipv):
        return False
    try:
        candidates = analyse_multipv(before_board, depth, 2)
    except (chess.engine.EngineTerminatedError, BrokenPipeError, OSError, TypeError, ValueError):
        return False
    if not isinstance(candidates, list):
        return False
    scores = [item.get("score") for item in candidates if isinstance(item, dict)]
    pov_scores = [score for score in scores if isinstance(score, chess.engine.PovScore)]
    return alternatives_show_uniqueness(pov_scores, user_color)


def _analyze_user_moves_unlocked(
    moves: pd.DataFrame,
    settings: Settings,
    output_path: Path,
    *,
    adapter: EngineAdapter | None,
    force: bool,
    progress: ProgressCallback | None,
) -> pd.DataFrame:
    config_hash = _config_hash(settings)
    existing = _load_reusable_analysis(moves, settings, output_path, force=force)

    keys = {
        (str(row.game_id), int(row.ply), str(row.engine_config_hash))
        for row in existing.itertuples(index=False)
    }
    user_moves = moves[moves["is_user_move"].fillna(False).astype(bool)].copy()
    total = len(user_moves)
    reused = sum(
        1
        for row in user_moves.itertuples(index=False)
        if (str(row.game_id), int(row.ply), config_hash) in keys
    )
    analyzed = 0
    _emit_progress(
        progress,
        completed=reused,
        total=total,
        reused=reused,
        analyzed=analyzed,
    )

    owns_adapter = adapter is None
    resolved_stockfish: str | None = None
    if adapter is None:
        resolved_stockfish = _resolve_stockfish(settings.stockfish_path)
        adapter = StockfishAdapter(resolved_stockfish)

    rows = existing.to_dict("records")
    try:
        for row in user_moves.itertuples(index=False):
            key = (str(row.game_id), int(row.ply), config_hash)
            if key in keys:
                continue
            user_color = chess.WHITE if row.color == "white" else chess.BLACK
            before_board = chess.Board(row.fen_before)
            after_board = chess.Board(row.fen_after)
            for attempt in range(2):
                try:
                    before_info = adapter.analyse(before_board, settings.stockfish_depth)
                    after_info = adapter.analyse(after_board, settings.stockfish_depth)
                    break
                except chess.engine.EngineTerminatedError:
                    if not owns_adapter or attempt == 1 or resolved_stockfish is None:
                        raise
                    with suppress(chess.engine.EngineTerminatedError, BrokenPipeError, OSError):
                        adapter.close()
                    adapter = StockfishAdapter(resolved_stockfish)

            before_eval = normalize_score(before_info["score"], user_color)
            after_eval = normalize_score(after_info["score"], user_color)
            before_snapshot = score_snapshot(before_info["score"], user_color, ply=int(row.ply))
            after_snapshot = score_snapshot(after_info["score"], user_color, ply=int(row.ply) + 1)
            pv = before_info.get("pv") or []
            best_move = pv[0].uci() if pv and isinstance(pv[0], chess.Move) else None
            actual_move = getattr(row, "uci", None)
            delivered_mate = after_board.is_checkmate()
            is_best_move = (
                best_move is not None
                and actual_move is not None
                and best_move == str(actual_move)
            )
            if is_best_move or delivered_mate:
                cpl = 0
            else:
                cpl = centipawn_loss(before_eval, after_eval)

            brilliant_candidate = False
            if is_best_move and not delivered_mate and after_snapshot.expected_score >= 0.70:
                brilliant_candidate = _verified_brilliant_candidate(
                    adapter=adapter,
                    before_board=before_board,
                    after_board=after_board,
                    after_info=after_info,
                    user_color=user_color,
                    depth=settings.stockfish_depth,
                )

            assessment = classify_move_quality(
                before=before_snapshot,
                after=after_snapshot,
                cpl=cpl,
                is_best_move=is_best_move,
                brilliant_candidate=brilliant_candidate,
                delivered_mate=delivered_mate,
            )
            rows.append(
                {
                    "game_id": str(row.game_id),
                    "ply": int(row.ply),
                    "best_move_uci": best_move,
                    "eval_before_cp": before_eval,
                    "eval_after_cp": after_eval,
                    "mate_before": before_snapshot.mate,
                    "mate_after": after_snapshot.mate,
                    "expected_score_before": round(before_snapshot.expected_score, 6),
                    "expected_score_after": round(after_snapshot.expected_score, 6),
                    "cpl": cpl,
                    "quality": assessment.label,
                    "quality_reason": assessment.reason,
                    "engine_config_hash": config_hash,
                    "scoring_version": SCORING_VERSION,
                }
            )
            keys.add(key)
            analyzed += 1
            _write_analysis(pd.DataFrame(rows, columns=ANALYSIS_COLUMNS), output_path)
            _emit_progress(
                progress,
                completed=reused + analyzed,
                total=total,
                reused=reused,
                analyzed=analyzed,
                game_id=str(row.game_id),
                ply=int(row.ply),
            )
    finally:
        if owns_adapter:
            adapter.close()

    return (
        pd.DataFrame(rows, columns=ANALYSIS_COLUMNS)
        .sort_values(["game_id", "ply"])
        .reset_index(drop=True)
    )


def analyze_user_moves(
    moves: pd.DataFrame,
    settings: Settings,
    output_path: Path,
    *,
    adapter: EngineAdapter | None = None,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    with _analysis_lock(output_path, settings.profile_lock_path):
        return _analyze_user_moves_unlocked(
            moves,
            settings,
            output_path,
            adapter=adapter,
            force=force,
            progress=progress,
        )
