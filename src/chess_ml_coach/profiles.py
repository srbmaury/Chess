from __future__ import annotations

import filecmp
import json
import re
import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings

_USERNAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9])?$")
_REGISTRY_VERSION = 1
_LEGACY_DATA_DIRS = ("raw", "processed", "engine", "training")
_LEGACY_MODEL_FILES = ("mistake_model.joblib", "mistake_model.metadata.json")


class ProfileMigrationConflictError(RuntimeError):
    """Raised when legacy and scoped profile artifacts disagree."""


@dataclass(frozen=True)
class ProfileRecord:
    username: str
    display_username: str
    created_at: str
    last_used_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def canonicalize_username(username: str) -> str:
    value = username.strip()
    if len(value) < 3 or not _USERNAME.fullmatch(value) or value.isdigit():
        raise ValueError(
            "Invalid Chess.com username. Use at least 3 characters, begin and end "
            "with a letter or number, and use only letters, numbers, '_' or '-'."
        )
    return value.lower()


class ProfileManager:
    def __init__(self, root_settings: Settings):
        self.root_settings = root_settings
        self.registry_path = root_settings.data_dir / "profiles.json"
        self.migration_marker = root_settings.data_dir / ".profiles-migrated.json"

    def settings_for(self, username: str) -> Settings:
        key = canonicalize_username(username)
        return replace(
            self.root_settings,
            username=key,
            data_dir=self.root_settings.data_dir / "users" / key,
            model_dir=self.root_settings.model_dir / "users" / key,
        )

    def _empty_registry(self) -> dict[str, object]:
        return {
            "version": _REGISTRY_VERSION,
            "active_username": None,
            "profiles": {},
        }

    def _load_registry(self) -> dict[str, object]:
        if not self.registry_path.exists():
            return self._empty_registry()
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read profile registry: {self.registry_path}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("profiles", {}), dict):
            raise TypeError(f"Invalid profile registry: {self.registry_path}")
        payload.setdefault("version", _REGISTRY_VERSION)
        payload.setdefault("active_username", None)
        payload.setdefault("profiles", {})
        return payload

    def _save_registry(self, payload: dict[str, object]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.registry_path)

    def _record_from_payload(self, key: str, payload: dict[str, object]) -> ProfileRecord:
        return ProfileRecord(
            username=key,
            display_username=str(payload.get("display_username") or key),
            created_at=str(payload.get("created_at") or ""),
            last_used_at=str(payload.get("last_used_at") or ""),
        )

    def list_profiles(self) -> list[ProfileRecord]:
        registry = self._load_registry()
        profiles = dict(registry.get("profiles", {}))
        users_root = self.root_settings.data_dir / "users"
        if users_root.exists():
            for path in users_root.iterdir():
                if not path.is_dir():
                    continue
                try:
                    key = canonicalize_username(path.name)
                except ValueError:
                    continue
                profiles.setdefault(
                    key,
                    {
                        "display_username": key,
                        "created_at": "",
                        "last_used_at": "",
                    },
                )
        return [
            self._record_from_payload(key, dict(profiles[key]))
            for key in sorted(profiles)
        ]

    def active_username(self) -> str | None:
        value = self._load_registry().get("active_username")
        if not value:
            return None
        return canonicalize_username(str(value))

    def create_or_activate(
        self,
        username: str,
        *,
        activate: bool = True,
    ) -> ProfileRecord:
        display = username.strip()
        key = canonicalize_username(display)
        timestamp = _now()
        registry = self._load_registry()
        profiles = dict(registry.get("profiles", {}))
        existing = dict(profiles.get(key, {}))
        created_at = str(existing.get("created_at") or timestamp)
        record_payload = {
            "display_username": str(existing.get("display_username") or display),
            "created_at": created_at,
            "last_used_at": timestamp if activate else str(existing.get("last_used_at") or created_at),
        }
        profiles[key] = record_payload
        registry["profiles"] = profiles
        if activate:
            registry["active_username"] = key
        scoped = self.settings_for(key)
        scoped.data_dir.mkdir(parents=True, exist_ok=True)
        scoped.model_dir.mkdir(parents=True, exist_ok=True)
        self._save_registry(registry)
        return self._record_from_payload(key, record_payload)

    def activate(self, username: str) -> ProfileRecord:
        key = canonicalize_username(username)
        registry = self._load_registry()
        profiles = dict(registry.get("profiles", {}))
        if key not in profiles:
            discovered = self.root_settings.data_dir / "users" / key
            if not discovered.exists():
                raise KeyError(f"Unknown player profile: {username}")
            self.create_or_activate(key, activate=False)
            registry = self._load_registry()
            profiles = dict(registry.get("profiles", {}))
        timestamp = _now()
        payload = dict(profiles[key])
        payload["last_used_at"] = timestamp
        profiles[key] = payload
        registry["profiles"] = profiles
        registry["active_username"] = key
        self._save_registry(registry)
        return self._record_from_payload(key, payload)

    def delete_profile(self, username: str) -> None:
        """Permanently remove one local player's artifacts and registry metadata."""
        key = canonicalize_username(username)
        scoped = self.settings_for(key)
        registry = self._load_registry()
        profiles = dict(registry.get("profiles", {}))
        profile_paths = (scoped.data_dir, scoped.model_dir)
        marker_owner = None

        if self.migration_marker.exists():
            try:
                marker = json.loads(self.migration_marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Cannot read profile migration marker: {self.migration_marker}"
                ) from exc
            if isinstance(marker, dict) and marker.get("owner_username"):
                marker_owner = canonicalize_username(str(marker["owner_username"]))

        known = key in profiles or any(
            path.exists() or path.is_symlink() for path in profile_paths
        )
        if not known:
            raise KeyError(f"Unknown player profile: {username}")

        analysis_lock = scoped.data_dir / "engine" / "analysis.lock"
        if analysis_lock.exists():
            raise RuntimeError(
                f"Cannot delete '{key}' while Stockfish analysis is running. "
                "Stop analysis first."
            )

        for path in profile_paths:
            if path.is_symlink():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)

        profiles.pop(key, None)
        registry["profiles"] = profiles
        if registry.get("active_username") == key:
            registry["active_username"] = None
        self._save_registry(registry)

        if marker_owner == key:
            self.migration_marker.unlink(missing_ok=True)

    def _legacy_files(self, scoped: Settings) -> list[tuple[Path, Path]]:
        pairs: list[tuple[Path, Path]] = []
        for dirname in _LEGACY_DATA_DIRS:
            source_root = self.root_settings.data_dir / dirname
            if not source_root.exists():
                continue
            for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
                pairs.append((source, scoped.data_dir / dirname / source.relative_to(source_root)))
        for filename in _LEGACY_MODEL_FILES:
            source = self.root_settings.model_dir / filename
            if source.is_file():
                pairs.append((source, scoped.model_dir / filename))
        return pairs

    def _move_file_safely(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not destination.is_file() or not filecmp.cmp(source, destination, shallow=False):
                raise ProfileMigrationConflictError(
                    f"Legacy artifact conflicts with profile destination: {source} -> {destination}"
                )
            source.unlink()
            return
        shutil.move(str(source), str(destination))

    def _remove_empty_legacy_dirs(self) -> None:
        for dirname in _LEGACY_DATA_DIRS:
            root = self.root_settings.data_dir / dirname
            if not root.exists():
                continue
            for path in sorted(
                (item for item in root.rglob("*") if item.is_dir()),
                key=lambda item: len(item.parts),
                reverse=True,
            ):
                try:
                    path.rmdir()
                except OSError:
                    pass
            try:
                root.rmdir()
            except OSError:
                pass

    def migrate_legacy(self, owner_username: str) -> None:
        key = canonicalize_username(owner_username)
        scoped = self.settings_for(key)
        pairs = self._legacy_files(scoped)
        if not pairs:
            return

        for source, destination in pairs:
            self._move_file_safely(source, destination)
        self._remove_empty_legacy_dirs()
        self.create_or_activate(owner_username)
        self.migration_marker.parent.mkdir(parents=True, exist_ok=True)
        self.migration_marker.write_text(
            json.dumps(
                {"version": _REGISTRY_VERSION, "owner_username": key, "completed_at": _now()},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
