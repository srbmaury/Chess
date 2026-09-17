from __future__ import annotations

from fastapi import APIRouter, Request

from .auth import require_account

router = APIRouter()


@router.get("/api/hosted/account/me")
def whoami(request: Request) -> dict[str, object]:
    account = require_account(request)
    return {"id": account.id, "email": account.email}
