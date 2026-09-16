from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


class ProfileBusyError(RuntimeError):
    """Raised when an operation already holds a player's exclusive lock."""


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by someone else - treat as alive.
        return True
    return True


def _clear_lock_if_stale(lock_path: Path) -> None:
    """Remove lock_path if the pid it records isn't running anymore.

    A lock left behind by a process that crashed or was force-killed (rather
    than exiting cleanly through the `finally` below) never gets unlinked, so
    every future attempt would otherwise fail forever with a "busy" error.
    """
    try:
        pid = int(lock_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return
    if _process_alive(pid):
        return
    lock_path.unlink(missing_ok=True)


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
    except FileExistsError:
        _clear_lock_if_stale(lock_path)
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
