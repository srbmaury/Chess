from __future__ import annotations

import json
import os
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

MATE_CP = 100_000
MATE_REANALYZE_THRESHOLD = 50_000
SCORING_VERSION = 2
ProgressCallback = Callable[[dict[str, object]], None]
ANALYSIS_COLUMNS = [
    "game_id",
    "ply",
    "best_move_uci",
    "eval_before_cp",
    "eval_after_cp",
    "cpl",
    "quality",
    "engine_config_hash",
    "scoring_version",
]
LEGACY_ANALYSIS_COLUMNS = [column for column in ANALYSIS_COLUMNS if column != "scoring_version"]


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

    def close(self) -> None:
        self.engine.quit()


def normalize_score(score: chess.engine.PovScore, user_color: chess.Color) -> int:
    value = score.pov(user_color)
    converted = value.score(mate_score=MATE_CP)
    return int(converted or 0)


def centipawn_loss(best_eval_cp: int, played_eval_cp: int) -> int:
    return max(0, int(best_eval_cp) - int(played_eval_cp))


def quality_label(cpl: int, thresholds: MoveQualityThresholds) -> str:
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


def _legacy_config_hash(settings: Settings) -> str:
    return _hash_payload(_config_payload(settings, scoring_version=None))


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
def _analysis_lock(output_path: Path) -> Iterator[None]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_suffix(".lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise EngineConfigurationError(
            f"Analysis is already running for {output_path}. "
            f"If no analyzer is running, remove stale lock file {lock_path}."
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock_path.unlink(missing_ok=True)


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


def _is_mate_sentinel(value: object) -> bool:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return bool(pd.notna(number) and abs(float(number)) >= MATE_REANALYZE_THRESHOLD)


def _legacy_row_is_safe(row: pd.Series, user_move: pd.Series | None) -> bool:
    if any(
        _is_mate_sentinel(row.get(column))
        for column in ("eval_before_cp", "eval_after_cp", "cpl")
    ):
        return False
    if user_move is None:
        return True
    actual_uci = user_move.get("uci")
    best_uci = row.get("best_move_uci")
    if (
        actual_uci is not None
        and best_uci is not None
        and str(actual_uci) == str(best_uci)
        and float(row.get("cpl", 0) or 0) > 0
    ):
        return False
    fen_after = user_move.get("fen_after")
    if fen_after is not None and not pd.isna(fen_after):
        try:
            if chess.Board(str(fen_after)).is_checkmate():
                return False
        except ValueError:
            pass
    return True


def _load_reusable_analysis(
    moves: pd.DataFrame,
    settings: Settings,
    output_path: Path,
    *,
    force: bool,
) -> pd.DataFrame:
    if force or not output_path.exists():
        return pd.DataFrame(columns=ANALYSIS_COLUMNS)

    raw = pd.read_parquet(output_path)
    missing = set(LEGACY_ANALYSIS_COLUMNS) - set(raw.columns)
    if missing:
        raise EngineConfigurationError(
            f"Invalid analysis artifact {output_path}: missing {sorted(missing)}"
        )
    if "scoring_version" not in raw.columns:
        raw["scoring_version"] = 1

    config_hash = _config_hash(settings)
    legacy_hash = _legacy_config_hash(settings)
    current = raw[
        (raw["engine_config_hash"] == config_hash)
        & (pd.to_numeric(raw["scoring_version"], errors="coerce") == SCORING_VERSION)
    ].copy()

    legacy = raw[raw["engine_config_hash"] == legacy_hash].copy()
    if not legacy.empty:
        user_moves = moves[moves["is_user_move"].fillna(False).astype(bool)].copy()
        move_lookup = {
            (str(row.game_id), int(row.ply)): pd.Series(row._asdict())
            for row in user_moves.itertuples(index=False)
        }
        safe_indices = []
        for index, row in legacy.iterrows():
            key = (str(row["game_id"]), int(row["ply"]))
            if _legacy_row_is_safe(row, move_lookup.get(key)):
                safe_indices.append(index)
        migrated = legacy.loc[safe_indices].copy()
        if not migrated.empty:
            migrated["engine_config_hash"] = config_hash
            migrated["scoring_version"] = SCORING_VERSION
            current = pd.concat([current, migrated], ignore_index=True)

    if current.empty:
        return pd.DataFrame(columns=ANALYSIS_COLUMNS)
    return (
        current[ANALYSIS_COLUMNS]
        .drop_duplicates(["game_id", "ply"], keep="last")
        .sort_values(["game_id", "ply"])
        .reset_index(drop=True)
    )


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
            pv = before_info.get("pv") or []
            best_move = pv[0].uci() if pv else None
            actual_move = getattr(row, "uci", None)
            delivered_mate = after_board.is_checkmate()
            if best_move is not None and actual_move is not None and best_move == str(actual_move):
                cpl = 0
            elif delivered_mate:
                cpl = 0
            else:
                cpl = centipawn_loss(before_eval, after_eval)
            rows.append(
                {
                    "game_id": str(row.game_id),
                    "ply": int(row.ply),
                    "best_move_uci": best_move,
                    "eval_before_cp": before_eval,
                    "eval_after_cp": after_eval,
                    "cpl": cpl,
                    "quality": quality_label(cpl, settings.thresholds),
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
    with _analysis_lock(output_path):
        return _analyze_user_moves_unlocked(
            moves,
            settings,
            output_path,
            adapter=adapter,
            force=force,
            progress=progress,
        )
