from pathlib import Path

import pytest

from chess_ml_coach.config import Settings
from chess_ml_coach.profiles import (
    ProfileManager,
    ProfileMigrationConflictError,
    canonicalize_username,
)


def _root(tmp_path: Path) -> Settings:
    return Settings(
        username="srbmaury",
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
    )


def test_profile_paths_are_isolated_and_case_insensitive(tmp_path: Path):
    manager = ProfileManager(_root(tmp_path))

    upper = manager.settings_for("SrbMaury")
    lower = manager.settings_for("srbmaury")
    other = manager.settings_for("another_user")

    assert upper.username == "srbmaury"
    assert upper.data_dir == lower.data_dir == tmp_path / "data" / "users" / "srbmaury"
    assert upper.model_dir == lower.model_dir == tmp_path / "models" / "users" / "srbmaury"
    assert other.data_dir == tmp_path / "data" / "users" / "another_user"
    assert other.model_dir == tmp_path / "models" / "users" / "another_user"
    assert other.data_dir != upper.data_dir
    assert other.model_dir != upper.model_dir


@pytest.mark.parametrize(
    "username",
    ["", "ab", "12345", "-name", "name-", "name/name", "name\\name", "name space"],
)
def test_unsafe_or_invalid_usernames_are_rejected(username: str):
    with pytest.raises(ValueError):
        canonicalize_username(username)


def test_registry_tracks_profiles_and_active_username(tmp_path: Path):
    manager = ProfileManager(_root(tmp_path))

    first = manager.create_or_activate("SrbMaury")
    second = manager.create_or_activate("Another_User", activate=False)

    assert first.username == "srbmaury"
    assert second.username == "another_user"
    assert manager.active_username() == "srbmaury"
    assert [profile.username for profile in manager.list_profiles()] == [
        "another_user",
        "srbmaury",
    ]

    manager.activate("another_user")
    assert manager.active_username() == "another_user"


def test_legacy_migration_preserves_artifact_bytes_and_is_idempotent(tmp_path: Path):
    root = _root(tmp_path)
    legacy = {
        root.data_dir / "raw" / "games.json": b"games",
        root.data_dir / "processed" / "features.parquet": b"features",
        root.data_dir / "engine" / "analysis.parquet": b"analysis",
        root.data_dir / "training" / "training.db": b"sqlite-review-and-explanation-history",
        root.model_dir / "mistake_model.joblib": b"model",
        root.model_dir / "mistake_model.metadata.json": b"metadata",
    }
    for path, content in legacy.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    manager = ProfileManager(root)
    manager.migrate_legacy("srbmaury")
    scoped = manager.settings_for("srbmaury")

    expected = {
        scoped.data_dir / "raw" / "games.json": b"games",
        scoped.data_dir / "processed" / "features.parquet": b"features",
        scoped.data_dir / "engine" / "analysis.parquet": b"analysis",
        scoped.data_dir / "training" / "training.db": b"sqlite-review-and-explanation-history",
        scoped.model_dir / "mistake_model.joblib": b"model",
        scoped.model_dir / "mistake_model.metadata.json": b"metadata",
    }
    for path, content in expected.items():
        assert path.read_bytes() == content

    manager.migrate_legacy("srbmaury")
    for path, content in expected.items():
        assert path.read_bytes() == content


def test_legacy_migration_refuses_conflicting_destination(tmp_path: Path):
    root = _root(tmp_path)
    legacy_db = root.data_dir / "training" / "training.db"
    legacy_db.parent.mkdir(parents=True, exist_ok=True)
    legacy_db.write_bytes(b"legacy")

    manager = ProfileManager(root)
    scoped_db = manager.settings_for("srbmaury").data_dir / "training" / "training.db"
    scoped_db.parent.mkdir(parents=True, exist_ok=True)
    scoped_db.write_bytes(b"different")

    with pytest.raises(ProfileMigrationConflictError):
        manager.migrate_legacy("srbmaury")

    assert legacy_db.read_bytes() == b"legacy"
    assert scoped_db.read_bytes() == b"different"
