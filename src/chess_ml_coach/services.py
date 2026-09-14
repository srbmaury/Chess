from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from .config import Settings

ProgressCallback = Callable[[dict[str, object]], None]


def run_sync(settings: Settings, progress: ProgressCallback | None = None) -> dict:
    from .chesscom import ChessComClient, sync_games

    client = ChessComClient()
    try:
        result = sync_games(client, settings, progress=progress)
    finally:
        client.http.close()
    return {
        "downloaded": result.downloaded,
        "total": result.total,
        "pgn_path": result.pgn_path,
    }


def run_analyze(settings: Settings, progress: ProgressCallback | None = None) -> dict:
    from .engine import analyze_user_moves
    from .pgn import parse_pgn_file, write_normalized

    raw_pgn = settings.data_dir / "raw" / f"{settings.username}_all_games.pgn"
    if not raw_pgn.exists():
        raise FileNotFoundError(f"Missing {raw_pgn}. Run Sync first.")
    games, moves = parse_pgn_file(raw_pgn, settings.username)
    write_normalized(games, moves, settings.data_dir / "processed")
    output = settings.data_dir / "engine" / "analysis.parquet"
    analysis = analyze_user_moves(moves, settings, output, progress=progress)
    return {"rows": len(analysis), "output": output}


def _ensure_analysis_complete(moves, analysis) -> None:
    user_moves = moves[moves["is_user_move"].fillna(False).astype(bool)]
    expected = {
        (str(row.game_id), int(row.ply))
        for row in user_moves[["game_id", "ply"]].itertuples(index=False)
    }
    analyzed = {
        (str(row.game_id), int(row.ply))
        for row in analysis[["game_id", "ply"]].itertuples(index=False)
    }
    missing = expected - analyzed
    if missing:
        completed = len(expected) - len(missing)
        raise RuntimeError(
            f"Analysis is incomplete: {completed}/{len(expected)} user moves analyzed. "
            "Please resume Analyze before building Features."
        )


def run_features(settings: Settings) -> dict:
    import pandas as pd

    from .features import build_feature_dataset, write_feature_dataset
    from .pgn import read_normalized

    games_path = settings.data_dir / "processed" / "games.parquet"
    moves_path = settings.data_dir / "processed" / "moves.parquet"
    analysis_path = settings.data_dir / "engine" / "analysis.parquet"
    for path in (games_path, moves_path, analysis_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing prerequisite {path}")
    games, moves = read_normalized(settings.data_dir / "processed")
    analysis = pd.read_parquet(analysis_path)
    _ensure_analysis_complete(moves, analysis)
    frame = build_feature_dataset(games, moves, analysis, settings.thresholds)
    output = settings.data_dir / "processed" / "features.parquet"
    write_feature_dataset(frame, output)
    return {"rows": len(frame), "output": output}


def run_train(settings: Settings, progress: ProgressCallback | None = None) -> dict:
    import pandas as pd

    from .train import train_model

    features_path = settings.data_dir / "processed" / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run Features first.")
    frame = pd.read_parquet(features_path)
    result = train_model(frame, settings.model_dir, progress=progress)
    return {
        "model_path": result.model_path,
        "metadata_path": result.metadata_path,
        "metrics": result.metrics,
    }


def run_report(settings: Settings) -> dict:
    import pandas as pd

    from .report import build_coaching_report, render_markdown

    features_path = settings.data_dir / "processed" / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run Features first.")
    frame = pd.read_parquet(features_path)
    metadata_path = settings.model_dir / "mistake_model.metadata.json"
    feature_importance = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        feature_importance = metadata.get("feature_importance")
    report = build_coaching_report(
        frame,
        min_group_size=settings.min_group_size,
        feature_importance=feature_importance,
    )
    output = settings.data_dir / "processed" / "coaching_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(report), encoding="utf-8")
    return {"output": output, "samples": report.overall["samples"]}


def training_db_path(settings: Settings) -> Path:
    return settings.data_dir / "training" / "training.db"


def run_puzzles(settings: Settings, progress: ProgressCallback | None = None) -> dict:
    import pandas as pd

    from .puzzles import extract_puzzles
    from .training import TrainingStore

    features_path = settings.data_dir / "processed" / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run Features first.")
    frame = pd.read_parquet(features_path)
    seeds = extract_puzzles(frame, progress=progress)
    db_path = training_db_path(settings)
    result = TrainingStore(db_path).upsert_puzzles(seeds)
    return {
        "source_rows": len(frame),
        "eligible": len(seeds),
        "skipped": len(frame) - len(seeds),
        "inserted": result.inserted,
        "updated": result.updated,
        "total": result.total,
        "db_path": db_path,
    }


def answer_to_uci(fen: str, answer: str) -> str | None:
    import chess

    board = chess.Board(fen)
    normalized = answer.strip()
    if not normalized:
        return None
    try:
        move = chess.Move.from_uci(normalized.lower())
        if move in board.legal_moves:
            return move.uci()
    except ValueError:
        pass
    try:
        return board.parse_san(normalized).uci()
    except ValueError:
        return None
