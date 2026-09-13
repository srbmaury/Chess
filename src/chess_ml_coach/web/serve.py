from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings
from .app import create_app


def default_frontend_dist() -> Path:
    return Path(__file__).resolve().parents[3] / "web" / "dist"


def create_served_app(
    settings: Settings | None = None,
    *,
    static_dir: Path | None = None,
):
    dist = Path(static_dir) if static_dir is not None else default_frontend_dist()
    index_path = dist / "index.html"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing built web UI at {index_path}. Run `cd web && npm install && npm run build` first."
        )

    app = create_app(settings)
    assets = dist / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        return FileResponse(index_path)

    return app
