from pathlib import Path

from chess_ml_coach.cli import _resolve_profile_settings
from chess_ml_coach.services import training_db_path


def test_cli_resolution_scopes_same_roots_by_username(tmp_path: Path):
    data_root = tmp_path / "data"
    model_root = tmp_path / "models"

    alpha = _resolve_profile_settings(
        "Alpha_User",
        data_dir=data_root,
        model_dir=model_root,
    )
    beta = _resolve_profile_settings(
        "Beta_User",
        data_dir=data_root,
        model_dir=model_root,
    )

    assert alpha.username == "alpha_user"
    assert alpha.data_dir == data_root / "users" / "alpha_user"
    assert alpha.model_dir == model_root / "users" / "alpha_user"
    assert beta.data_dir == data_root / "users" / "beta_user"
    assert beta.model_dir == model_root / "users" / "beta_user"
    assert training_db_path(alpha) != training_db_path(beta)


def test_cli_resolution_uses_configured_default_profile(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CHESS_COACH_USERNAME", "Default_Player")

    settings = _resolve_profile_settings(
        None,
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
    )

    assert settings.username == "default_player"
    assert settings.data_dir == tmp_path / "data" / "users" / "default_player"
    assert settings.model_dir == tmp_path / "models" / "users" / "default_player"
