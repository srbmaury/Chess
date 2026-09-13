from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, TypeVar

import typer

from .config import Settings, get_settings

app = typer.Typer(no_args_is_help=True)
T = TypeVar("T")


def _execute(action: Callable[[], T]) -> T:
    try:
        return action()
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _run_sync(settings: Settings) -> dict:
    from .chesscom import ChessComClient, sync_games

    client = ChessComClient()
    try:
        result = sync_games(client, settings)
    finally:
        client.http.close()
    return {
        "downloaded": result.downloaded,
        "total": result.total,
        "pgn_path": result.pgn_path,
    }


def _run_analyze(settings: Settings) -> dict:
    from .engine import analyze_user_moves
    from .pgn import parse_pgn_file, write_normalized

    raw_pgn = settings.data_dir / "raw" / f"{settings.username}_all_games.pgn"
    if not raw_pgn.exists():
        raise FileNotFoundError(f"Missing {raw_pgn}. Run `chess-coach sync` first.")
    games, moves = parse_pgn_file(raw_pgn, settings.username)
    write_normalized(games, moves, settings.data_dir / "processed")
    output = settings.data_dir / "engine" / "analysis.parquet"
    analysis = analyze_user_moves(moves, settings, output)
    return {"rows": len(analysis), "output": output}


def _run_features(settings: Settings) -> dict:
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
    frame = build_feature_dataset(games, moves, analysis, settings.thresholds)
    output = settings.data_dir / "processed" / "features.parquet"
    write_feature_dataset(frame, output)
    return {"rows": len(frame), "output": output}


def _run_train(settings: Settings) -> dict:
    import pandas as pd

    from .train import train_model

    features_path = settings.data_dir / "processed" / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run `chess-coach features` first.")
    frame = pd.read_parquet(features_path)
    result = train_model(frame, settings.model_dir)
    return {
        "model_path": result.model_path,
        "metadata_path": result.metadata_path,
        "metrics": result.metrics,
    }


def _run_report(settings: Settings) -> dict:
    import pandas as pd

    from .report import build_coaching_report, render_markdown

    features_path = settings.data_dir / "processed" / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run `chess-coach features` first.")
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


@app.command()
def sync(
    username: Annotated[str | None, typer.Option("--username")] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
) -> None:
    """Download and deduplicate Chess.com games."""
    settings = _execute(lambda: get_settings(username, data_dir=data_dir))
    result = _execute(lambda: _run_sync(settings))
    typer.echo(
        f"Synced {result['total']} games ({result['downloaded']} new games) -> {result['pgn_path']}"
    )


@app.command()
def analyze(
    username: Annotated[str | None, typer.Option("--username")] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    stockfish_path: Annotated[str | None, typer.Option("--stockfish-path")] = None,
    depth: Annotated[int | None, typer.Option("--depth")] = None,
    inaccuracy_cpl: Annotated[int | None, typer.Option("--inaccuracy-cpl")] = None,
    mistake_cpl: Annotated[int | None, typer.Option("--mistake-cpl")] = None,
    blunder_cpl: Annotated[int | None, typer.Option("--blunder-cpl")] = None,
) -> None:
    """Parse PGNs and run resumable Stockfish analysis."""
    settings = _execute(
        lambda: get_settings(
            username,
            data_dir=data_dir,
            stockfish_path=stockfish_path,
            stockfish_depth=depth,
            inaccuracy_cpl=inaccuracy_cpl,
            mistake_cpl=mistake_cpl,
            blunder_cpl=blunder_cpl,
        )
    )
    result = _execute(lambda: _run_analyze(settings))
    typer.echo(f"Analyzed {result['rows']} user moves -> {result['output']}")


@app.command()
def features(
    username: Annotated[str | None, typer.Option("--username")] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    inaccuracy_cpl: Annotated[int | None, typer.Option("--inaccuracy-cpl")] = None,
    mistake_cpl: Annotated[int | None, typer.Option("--mistake-cpl")] = None,
    blunder_cpl: Annotated[int | None, typer.Option("--blunder-cpl")] = None,
) -> None:
    """Create the ML-ready feature dataset."""
    settings = _execute(
        lambda: get_settings(
            username,
            data_dir=data_dir,
            inaccuracy_cpl=inaccuracy_cpl,
            mistake_cpl=mistake_cpl,
            blunder_cpl=blunder_cpl,
        )
    )
    result = _execute(lambda: _run_features(settings))
    typer.echo(f"Built {result['rows']} feature rows -> {result['output']}")


@app.command()
def train(
    username: Annotated[str | None, typer.Option("--username")] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    model_dir: Annotated[Path | None, typer.Option("--model-dir")] = None,
) -> None:
    """Train the personalized LightGBM mistake-risk model."""
    settings = _execute(
        lambda: get_settings(username, data_dir=data_dir, model_dir=model_dir)
    )
    result = _execute(lambda: _run_train(settings))
    typer.echo(f"Saved model -> {result['model_path']}")
    typer.echo(json.dumps(result["metrics"], indent=2, sort_keys=True))


@app.command()
def report(
    username: Annotated[str | None, typer.Option("--username")] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    model_dir: Annotated[Path | None, typer.Option("--model-dir")] = None,
    min_group_size: Annotated[int | None, typer.Option("--min-group-size")] = None,
) -> None:
    """Generate a statistically guarded coaching report."""
    settings = _execute(
        lambda: get_settings(
            username,
            data_dir=data_dir,
            model_dir=model_dir,
            min_group_size=min_group_size,
        )
    )
    result = _execute(lambda: _run_report(settings))
    typer.echo(f"Generated report from {result['samples']} moves -> {result['output']}")


if __name__ == "__main__":
    app()
