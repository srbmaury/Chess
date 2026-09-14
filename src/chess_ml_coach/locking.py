from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


class ProfileBusyError(RuntimeError):
    """Raised when an operation already holds a player's exclusive lock."""


@contextmanager
def exclusive_profile_lock(
    lock_path: Path,
    *,
    error_message: Callable[[Path], str],
) -> Iterator[None]:
    """Hold an atomic, cross-process lock outside the deletable profile directory."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ProfileBusyError(error_message(lock_path)) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock_path.unlink(missing_ok=True)
