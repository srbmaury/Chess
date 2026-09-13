from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, TypeVar

import typer

from .config import Settings, get_settings

app = typer.Typer(no_args_is_help=True)
T = TypeVar("T")
ProgressCallback = Callable[[dict[str, object]], None]


def _execute(action: Callable[[], T]) -> T:
    try:
        return action()
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _print_sync_progress(event: dict[str, object]) -> None:
    archive = str(event.get("archive", ""))
    parts = archive.rstrip("/").split("/")
    month = "/".join(parts[-2:]) if len(parts) >= 2 else archive
    status = " skipped (unavailable)" if event.get("skipped") else ""
    typer.echo(
        f"[sync] {event.get('current')}/{event.get('total')} {month} • "
        f"{event.get('game_count')} games{status}"
    )


def _print_analyze_progress(event: dict[str, object]) -> None:
    completed = int(event.get("completed", 0))
    total = int(event.get("total", 0))
    reused = int(event.get("reused", 0))
    analyzed = int(event.get("analyzed", 0))
    detail = ""
    if event.get("game_id") is not None and event.get("ply") is not None:
        detail = f" • ply {event['ply']}"
    typer.echo(
        f"\r[analyze] {completed}/{total} • {reused} reused • {analyzed} new{detail}",
        nl=False,
    )


def _print_train_progress(event: dict[str, object]) -> None:
    typer.echo(f"[train] {event.get('message', event.get('stage', 'working'))}")


def _print_puzzle_progress(event: dict[str, object]) -> None:
    typer.echo(
        f"\r[puzzles] scanning {event.get('current')}/{event.get('total')} • "
        f"{event.get('eligible')} eligible",
        nl=False,
    )


def _run_sync(settings: Settings, progress: ProgressCallback | None = None) -> dict:
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


def _run_analyze(settings: Settings, progress: ProgressCallback | None = None) -> dict:
    from .engine import analyze_user_moves
    from .pgn import parse_pgn_file, write_normalized

    raw_pgn = settings.data_dir / "raw" / f"{settings.username}_all_games.pgn"
    if not raw_pgn.exists():
        raise FileNotFoundError(f"Missing {raw_pgn}. Run `chess-coach sync` first.")
    games, moves = parse_pgn_file(raw_pgn, settings.username)
    write_normalized(games, moves, settings.data_dir / "processed")
    output = settings.data_dir / "engine" / "analysis.parquet"
    analysis = analyze_user_moves(moves, settings, output, progress=progress)
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


def _run_train(settings: Settings, progress: ProgressCallback | None = None) -> dict:
    import pandas as pd

    from .train import train_model

    features_path = settings.data_dir / "processed" / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run `chess-coach features` first.")
    frame = pd.read_parquet(features_path)
    result = train_model(frame, settings.model_dir, progress=progress)
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


def _training_db_path(settings: Settings) -> Path:
    return settings.data_dir / "training" / "training.db"


def _run_puzzles(settings: Settings, progress: ProgressCallback | None = None) -> dict:
    import pandas as pd

    from .puzzles import extract_puzzles
    from .training import TrainingStore

    features_path = settings.data_dir / "processed" / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run `chess-coach features` first.")
    frame = pd.read_parquet(features_path)
    seeds = extract_puzzles(frame, progress=progress)
    db_path = _training_db_path(settings)
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


def _answer_to_uci(fen: str, answer: str) -> str | None:
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


def _run_practice(settings: Settings, limit: int) -> dict:
    import chess

    from .training import TrainingStore

    if limit <= 0:
        raise ValueError("--limit must be greater than zero")
    db_path = _training_db_path(settings)
    if not db_path.exists():
        raise FileNotFoundError(f"Missing {db_path}. Run `chess-coach puzzles` first.")
    store = TrainingStore(db_path)
    due = store.due_puzzles(limit=limit)
    if not due:
        typer.echo("No puzzles are due right now. Use `chess-coach progress` to see your schedule.")
        return {"reviewed": 0, "correct": 0, "stopped": False}

    reviewed = 0
    correct_count = 0
    for index, puzzle in enumerate(due, start=1):
        board = chess.Board(puzzle.fen_before)
        side = "White" if board.turn == chess.WHITE else "Black"
        typer.echo("")
        typer.echo(f"Puzzle {index}/{len(due)} • {side} to move")
        typer.echo(f"Game: {puzzle.game_label} • {puzzle.move_label}")
        typer.echo(f"Opening: {puzzle.opening} ({puzzle.eco})")
        typer.echo(f"Phase: {puzzle.game_phase}")
        typer.echo(f"Theme (heuristic): {puzzle.motif} • Difficulty: {puzzle.difficulty}/5")
        typer.echo(str(board))
        answer = typer.prompt("Your move (SAN/UCI, q to quit)").strip()
        if answer.lower() in {"q", "quit", "exit"}:
            typer.echo("Practice stopped. No review was recorded for this puzzle.")
            return {"reviewed": reviewed, "correct": correct_count, "stopped": True}

        answer_uci = _answer_to_uci(puzzle.fen_before, answer)
        correct = answer_uci == puzzle.best_move_uci
        review = store.record_review(
            puzzle.puzzle_id,
            answer=answer,
            correct=correct,
        )
        reviewed += 1
        if correct:
            correct_count += 1
            typer.echo(f"Correct! Best move: {puzzle.best_move_san}")
        else:
            typer.echo(f"Not quite. Best move: {puzzle.best_move_san}")
            typer.echo(f"You played in the game: {puzzle.your_move_san}")
            typer.echo(f"Evaluation loss: {puzzle.eval_loss_pawns:.2f} pawns")
        typer.echo(
            f"Next review: {review.next_review_at.date().isoformat()} "
            f"(+{review.next_interval_days} days)"
        )
        if puzzle.source_url:
            typer.echo(f"Game: {puzzle.source_url}")

    return {"reviewed": reviewed, "correct": correct_count, "stopped": False}


def _run_progress(settings: Settings):
    from .training import TrainingStore

    db_path = _training_db_path(settings)
    if not db_path.exists():
        raise FileNotFoundError(f"Missing {db_path}. Run `chess-coach puzzles` first.")
    return TrainingStore(db_path).progress()


def _accuracy_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _print_progress_rows(title: str, rows) -> None:
    typer.echo(title)
    if not rows:
        typer.echo("  No data yet.")
        return
    for row in rows[:8]:
        typer.echo(
            f"  {row.label}: {row.puzzles} puzzles • {row.attempts} reviews • "
            f"{_accuracy_text(row.accuracy)} accuracy"
        )


@app.command()
def sync(
    username: Annotated[str | None, typer.Option("--username")] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
) -> None:
    """Download and deduplicate Chess.com games."""
    settings = _execute(lambda: get_settings(username, data_dir=data_dir))
    result = _execute(lambda: _run_sync(settings, progress=_print_sync_progress))
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
    result = _execute(lambda: _run_analyze(settings, progress=_print_analyze_progress))
    typer.echo()
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
    result = _execute(lambda: _run_train(settings, progress=_print_train_progress))
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


@app.command()
def puzzles(
    username: Annotated[str | None, typer.Option("--username")] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
) -> None:
    """Build or refresh a personal puzzle bank from analyzed mistakes."""
    settings = _execute(lambda: get_settings(username, data_dir=data_dir))
    result = _execute(lambda: _run_puzzles(settings, progress=_print_puzzle_progress))
    typer.echo()
    typer.echo(
        f"Puzzle bank: {result['eligible']} eligible • {result['inserted']} new • "
        f"{result['updated']} refreshed • {result['total']} total -> {result['db_path']}"
    )


@app.command()
def practice(
    username: Annotated[str | None, typer.Option("--username")] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1)] = 10,
) -> None:
    """Practice due positions from your own mistakes."""
    settings = _execute(lambda: get_settings(username, data_dir=data_dir))
    result = _execute(lambda: _run_practice(settings, limit))
    if result["reviewed"]:
        typer.echo(
            f"Session: {result['correct']}/{result['reviewed']} correct "
            f"({_accuracy_text(result['correct'] / result['reviewed'])})"
        )


@app.command()
def progress(
    username: Annotated[str | None, typer.Option("--username")] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
) -> None:
    """Show personal puzzle review progress and recurring themes."""
    settings = _execute(lambda: get_settings(username, data_dir=data_dir))
    summary = _execute(lambda: _run_progress(settings))
    typer.echo("Puzzle training progress")
    typer.echo(f"Total puzzles: {summary.total_puzzles}")
    typer.echo(f"Due now: {summary.due_puzzles}")
    typer.echo(f"Reviewed: {summary.reviewed_puzzles}")
    typer.echo(f"Mastered: {summary.mastered_puzzles}")
    typer.echo(f"Total reviews: {summary.total_reviews}")
    typer.echo(f"Review accuracy: {_accuracy_text(summary.accuracy)}")
    _print_progress_rows("Top heuristic motifs:", summary.by_motif)
    _print_progress_rows("Top openings:", summary.by_opening)


if __name__ == "__main__":
    app()
