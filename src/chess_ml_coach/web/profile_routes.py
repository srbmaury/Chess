from __future__ import annotations

from dataclasses import replace

from fastapi import APIRouter, FastAPI, HTTPException, Request

from ..config import Settings
from ..profiles import ProfileManager, canonicalize_username

router = APIRouter()


def _root_settings(settings: Settings) -> Settings:
    key = canonicalize_username(settings.username)
    data_root = settings.data_dir
    model_root = settings.model_dir
    if data_root.name == key and data_root.parent.name == "users":
        data_root = data_root.parent.parent
    if model_root.name == key and model_root.parent.name == "users":
        model_root = model_root.parent.parent
    return replace(settings, data_dir=data_root, model_dir=model_root)


def _apply_active_settings(app: FastAPI, username: str) -> Settings:
    manager: ProfileManager = app.state.profile_manager
    scoped = manager.settings_for(username)
    current: Settings = app.state.settings
    object.__setattr__(current, "username", scoped.username)
    object.__setattr__(current, "data_dir", scoped.data_dir)
    object.__setattr__(current, "model_dir", scoped.model_dir)
    return current


def _profiles_payload(manager: ProfileManager) -> dict[str, object]:
    return {
        "active_username": manager.active_username(),
        "profiles": [
            {
                "username": item.username,
                "display_username": item.display_username,
                "created_at": item.created_at,
                "last_used_at": item.last_used_at,
            }
            for item in manager.list_profiles()
        ],
    }


def _ensure_switch_allowed(request: Request) -> None:
    status = request.app.state.pipeline_manager.snapshot().status
    if status in {"running", "stopping"}:
        raise HTTPException(status_code=409, detail="Wait for the pipeline job to finish")


@router.get("/api/profiles")
def list_profiles(request: Request) -> dict[str, object]:
    return _profiles_payload(request.app.state.profile_manager)


@router.post("/api/profiles", status_code=201)
def create_profile(request: Request, payload: dict[str, object]) -> dict[str, object]:
    username = str(payload.get("username") or "")
    activate = bool(payload.get("activate", True))
    if activate:
        _ensure_switch_allowed(request)
    manager: ProfileManager = request.app.state.profile_manager
    try:
        record = manager.create_or_activate(username, activate=activate)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if activate:
        _apply_active_settings(request.app, record.username)
    return {
        "username": record.username,
        "display_username": record.display_username,
        "active_username": manager.active_username(),
    }


@router.post("/api/profiles/{username}/activate")
def activate_profile(request: Request, username: str) -> dict[str, object]:
    _ensure_switch_allowed(request)
    manager: ProfileManager = request.app.state.profile_manager
    try:
        record = manager.activate(username)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _apply_active_settings(request.app, record.username)
    return _profiles_payload(manager)


def enable_profiles(
    app: FastAPI,
    settings: Settings,
    *,
    manager: ProfileManager | None = None,
) -> ProfileManager:
    roots = _root_settings(settings)
    profiles = manager or ProfileManager(roots)
    if manager is not None:
        # The app's Settings object is mutated when players switch. Keep the
        # profile manager anchored to an independent root Settings snapshot.
        profiles.root_settings = roots
        profiles.registry_path = roots.data_dir / "profiles.json"
        profiles.migration_marker = roots.data_dir / ".profiles-migrated.json"
    profiles.migrate_legacy(roots.username)
    profiles.create_or_activate(settings.username, activate=True)
    app.state.root_settings = roots
    app.state.profile_manager = profiles
    _apply_active_settings(app, settings.username)
    app.include_router(router)
    return profiles
