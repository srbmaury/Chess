from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from chess_ml_coach.cli import app
from chess_ml_coach.config import MoveQualityThresholds, Settings
from chess_ml_coach.puzzles import PuzzleSeed
from chess_ml_coach.training import TrainingStore

runner = CliRunner()


def _training_seed(puzzle_id: str = "p1") -> PuzzleSeed:
    return PuzzleSeed(
        puzzle_id=puzzle_id,
        game_id=f"game-{puzzle_id}",
        ply=1,
        fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        color="white",
        game_label="srbmaury vs opponent",
        move_label="1.e4",
        your_move_san="e4",
        your_move_uci="e2e4",
        best_move_san="d4",
        best_move_uci="d2d4",
        cpl=250,
        eval_loss_pawns=2.5,
        quality="blunder",
        opening="Queen's Pawn Opening",
        eco="D00",
        game_phase="opening",
        source_url="https://www.chess.com/game/live/1",
        motif="positional / calculation",
        difficulty=3,
    )


def test_default_settings_are_personalized_but_local(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    assert settings.username == "srbmaury"
    assert settings.thresholds == MoveQualityThresholds(50, 100, 200)
    assert settings.stockfish_depth == 14


def test_cli_exposes_all_pipeline_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in [
        "sync",
        "analyze",
        "features",
        "train",
        "report",
        "puzzles",
        "practice",
        "progress",
    ]:
        assert command in result.stdout


def test_sync_command_uses_default_username_and_data_dir(monkeypatch, tmp_path: Path):
    from chess_ml_coach import cli

    seen = {}

    def fake_run(settings, progress=None):
        seen["settings"] = settings
        if progress:
            progress(
                {
                    "stage": "sync",
                    "current": 1,
                    "total": 2,
                    "archive": "https://api.chess.com/pub/player/srbmaury/games/2026/08",
                    "game_count": 10,
                    "skipped": False,
                }
            )
        return {"downloaded": 3, "total": 10, "pgn_path": settings.data_dir / "raw" / "x.pgn"}

    monkeypatch.setattr(cli, "_run_sync", fake_run)
    result = runner.invoke(app, ["sync", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert seen["settings"].username == "srbmaury"
    assert seen["settings"].data_dir == tmp_path
    assert "[sync] 1/2" in result.stdout
    assert "10 games" in result.stdout
    assert "3 new games" in result.stdout


def test_sync_command_accepts_username_override(monkeypatch):
    from chess_ml_coach import cli

    seen = {}

    def fake_run(settings, progress=None):
        seen["username"] = settings.username
        return {"downloaded": 0, "total": 0, "pgn_path": Path("games.pgn")}

    monkeypatch.setattr(cli, "_run_sync", fake_run)
    result = runner.invoke(app, ["sync", "--username", "another-user"])
    assert result.exit_code == 0
    assert seen["username"] == "another-user"


def test_analyze_surfaces_actionable_stockfish_error(monkeypatch):
    from chess_ml_coach import cli

    def fake_run(settings, progress=None):
        raise RuntimeError("Stockfish is not configured. Set STOCKFISH_PATH.")

    monkeypatch.setattr(cli, "_run_analyze", fake_run)
    result = runner.invoke(app, ["analyze"])
    assert result.exit_code == 1
    assert "STOCKFISH_PATH" in result.output


def test_analyze_command_prints_move_progress(monkeypatch):
    from chess_ml_coach import cli

    def fake_run(settings, progress=None):
        assert progress is not None
        progress(
            {
                "stage": "analyze",
                "completed": 43,
                "total": 130,
                "reused": 10,
                "analyzed": 33,
                "game_id": "g1",
                "ply": 46,
            }
        )
        return {"rows": 130, "output": Path("data/engine/analysis.parquet")}

    monkeypatch.setattr(cli, "_run_analyze", fake_run)
    result = runner.invoke(app, ["analyze", "--depth", "1"])

    assert result.exit_code == 0
    assert "[analyze] 43/130" in result.stdout
    assert "10 reused" in result.stdout
    assert "33 new" in result.stdout


def test_train_command_prints_stage_progress(monkeypatch):
    from chess_ml_coach import cli

    def fake_run(settings, progress=None):
        assert progress is not None
        progress({"stage": "train", "message": "Training LightGBM"})
        return {
            "model_path": Path("models/mistake_model.joblib"),
            "metadata_path": Path("models/mistake_model.metadata.json"),
            "metrics": {"roc_auc": 0.75},
        }

    monkeypatch.setattr(cli, "_run_train", fake_run)
    result = runner.invoke(app, ["train"])

    assert result.exit_code == 0
    assert "[train] Training LightGBM" in result.stdout
    assert "Saved model" in result.stdout


def test_puzzles_requires_feature_dataset(tmp_path: Path):
    result = runner.invoke(app, ["puzzles", "--data-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "features" in result.output.lower()
    assert "chess-coach features" in result.output


def test_practice_accepts_correct_san_and_records_review(tmp_path: Path):
    db_path = tmp_path / "training" / "training.db"
    store = TrainingStore(db_path)
    store.upsert_puzzles(
        [_training_seed()],
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    result = runner.invoke(
        app,
        ["practice", "--data-dir", str(tmp_path), "--limit", "1"],
        input="d4\n",
    )

    assert result.exit_code == 0
    assert "Correct" in result.stdout
    assert "Best move: d4" in result.stdout
    assert store.review_count("p1") == 1
    puzzle = store.get_puzzle("p1")
    assert puzzle is not None
    assert puzzle.correct_attempts == 1


def test_practice_q_exits_without_recording_review(tmp_path: Path):
    db_path = tmp_path / "training" / "training.db"
    store = TrainingStore(db_path)
    store.upsert_puzzles(
        [_training_seed()],
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    result = runner.invoke(
        app,
        ["practice", "--data-dir", str(tmp_path), "--limit", "1"],
        input="q\n",
    )

    assert result.exit_code == 0
    assert "Practice stopped" in result.stdout
    assert store.review_count("p1") == 0


def test_progress_prints_human_training_summary(tmp_path: Path):
    db_path = tmp_path / "training" / "training.db"
    store = TrainingStore(db_path)
    seeded_at = datetime(2026, 1, 1, tzinfo=UTC)
    store.upsert_puzzles([_training_seed()], now=seeded_at)
    store.record_review("p1", answer="d4", correct=True, now=seeded_at)

    result = runner.invoke(app, ["progress", "--data-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "Puzzle training progress" in result.stdout
    assert "Total puzzles: 1" in result.stdout
    assert "Reviewed: 1" in result.stdout
    assert "Review accuracy: 100.0%" in result.stdout
    assert "positional / calculation" in result.stdout
