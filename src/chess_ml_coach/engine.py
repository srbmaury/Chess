from __future__ import annotations

import json
import shutil
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from typing import Protocol

import chess
import chess.engine
import pandas as pd

from .config import MoveQualityThresholds, Settings

MATE_CP = 100_000
ANALYSIS_COLUMNS = [
    "game_id",
    "ply",
    "best_move_uci",
    "eval_before_cp",
    "eval_after_cp",
    "cpl",
    "quality",
    "engine_config_hash",
]


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
    if value.is_mate():
        mate = value.mate()
        if mate is None:
            return 0
        return MATE_CP if mate > 0 else -MATE_CP
    cp = value.score()
    return int(cp or 0)


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


def _config_hash(settings: Settings) -> str:
    payload = {
        "depth": settings.stockfish_depth,
        "thresholds": {
            "inaccuracy": settings.thresholds.inaccuracy,
            "mistake": settings.thresholds.mistake,
            "blunder": settings.thresholds.blunder,
        },
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


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


def _write_analysis(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp.parquet")
    frame.sort_values(["game_id", "ply"]).to_parquet(temp_path, index=False)
    temp_path.replace(output_path)


def analyze_user_moves(
    moves: pd.DataFrame,
    settings: Settings,
    output_path: Path,
    *,
    adapter: EngineAdapter | None = None,
    force: bool = False,
) -> pd.DataFrame:
    config_hash = _config_hash(settings)
    if force or not output_path.exists():
        existing = pd.DataFrame(columns=ANALYSIS_COLUMNS)
    else:
        existing = pd.read_parquet(output_path)
        missing = set(ANALYSIS_COLUMNS) - set(existing.columns)
        if missing:
            raise EngineConfigurationError(
                f"Invalid analysis artifact {output_path}: missing {sorted(missing)}"
            )
        existing = existing[existing["engine_config_hash"] == config_hash].copy()

    keys = {
        (str(row.game_id), int(row.ply), str(row.engine_config_hash))
        for row in existing.itertuples(index=False)
    }

    owns_adapter = adapter is None
    resolved_stockfish: str | None = None
    if adapter is None:
        resolved_stockfish = _resolve_stockfish(settings.stockfish_path)
        adapter = StockfishAdapter(resolved_stockfish)

    rows = existing.to_dict("records")
    try:
        for row in moves[moves["is_user_move"].fillna(False).astype(bool)].itertuples(index=False):
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
            cpl = centipawn_loss(before_eval, after_eval)
            pv = before_info.get("pv") or []
            best_move = pv[0].uci() if pv else None
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
                }
            )
            keys.add(key)
            _write_analysis(pd.DataFrame(rows, columns=ANALYSIS_COLUMNS), output_path)
    finally:
        if owns_adapter:
            adapter.close()

    return pd.DataFrame(rows, columns=ANALYSIS_COLUMNS).sort_values(["game_id", "ply"]).reset_index(drop=True)
