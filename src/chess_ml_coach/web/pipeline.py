from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from threading import RLock
from typing import Any

from .. import services
from ..config import Settings

PipelineRunner = Callable[[Settings, services.ProgressCallback], dict[str, Any]]
TERMINAL_STATUSES = {"succeeded", "failed"}
PIPELINE_STAGES = ("sync", "analyze", "features", "puzzles", "train", "report")


class PipelineBusyError(RuntimeError):
    """Raised when a second pipeline job is started while another is running."""


class UnknownPipelineStageError(ValueError):
    """Raised for a stage the web pipeline does not expose."""


@dataclass(frozen=True)
class PipelineEvent:
    sequence: int
    payload: dict[str, object]
    created_at: datetime


@dataclass(frozen=True)
class PipelineJobSnapshot:
    stage: str | None
    status: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


def _default_runners() -> dict[str, PipelineRunner]:
    return {
        "sync": lambda settings, progress: services.run_sync(settings, progress=progress),
        "analyze": lambda settings, progress: services.run_analyze(settings, progress=progress),
        "features": lambda settings, progress: services.run_features(settings),
        "puzzles": lambda settings, progress: services.run_puzzles(settings, progress=progress),
        "train": lambda settings, progress: services.run_train(settings, progress=progress),
        "report": lambda settings, progress: services.run_report(settings),
    }


class PipelineManager:
    def __init__(
        self,
        settings: Settings,
        *,
        runners: dict[str, PipelineRunner] | None = None,
        event_history_size: int = 500,
    ) -> None:
        self.settings = settings
        self._runners = _default_runners()
        if runners:
            self._runners.update(runners)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chess-pipeline")
        self._lock = RLock()
        self._events: deque[PipelineEvent] = deque(maxlen=event_history_size)
        self._sequence = 0
        self._snapshot = PipelineJobSnapshot(stage=None, status="idle")

    def snapshot(self) -> PipelineJobSnapshot:
        with self._lock:
            return self._snapshot

    def events(self, *, after_sequence: int = 0) -> list[PipelineEvent]:
        with self._lock:
            return [event for event in self._events if event.sequence > after_sequence]

    def _record_event(self, payload: dict[str, object]) -> PipelineEvent:
        with self._lock:
            self._sequence += 1
            event = PipelineEvent(
                sequence=self._sequence,
                payload=dict(payload),
                created_at=datetime.now(UTC),
            )
            self._events.append(event)
            return event

    def _settings_for(self, stage: str, options: dict[str, object]) -> Settings:
        if stage != "analyze" or options.get("depth") is None:
            return self.settings
        depth = int(options["depth"])
        if depth <= 0:
            raise ValueError("Analyze depth must be greater than zero")
        return replace(self.settings, stockfish_depth=depth)

    def start(
        self,
        stage: str,
        options: dict[str, object] | None = None,
    ) -> PipelineJobSnapshot:
        if stage not in PIPELINE_STAGES or stage not in self._runners:
            raise UnknownPipelineStageError(f"Unknown pipeline stage: {stage}")
        resolved_options = options or {}
        run_settings = self._settings_for(stage, resolved_options)
        with self._lock:
            if self._snapshot.status == "running":
                raise PipelineBusyError(
                    f"Pipeline job '{self._snapshot.stage}' is already running"
                )
            started_at = datetime.now(UTC)
            self._snapshot = PipelineJobSnapshot(
                stage=stage,
                status="running",
                started_at=started_at,
            )
            self._record_event(
                {
                    "stage": stage,
                    "status": "running",
                    "started_at": started_at.isoformat(),
                }
            )
            self._executor.submit(self._run, stage, run_settings)
            return self._snapshot

    def _run(self, stage: str, run_settings: Settings) -> None:
        runner = self._runners[stage]

        def progress(payload: dict[str, object]) -> None:
            self._record_event({"stage": stage, "status": "running", **payload})

        try:
            result = runner(run_settings, progress)
        except Exception as exc:  # noqa: BLE001 - worker boundary must capture job failures.
            finished_at = datetime.now(UTC)
            with self._lock:
                self._snapshot = PipelineJobSnapshot(
                    stage=stage,
                    status="failed",
                    started_at=self._snapshot.started_at,
                    finished_at=finished_at,
                    error=str(exc),
                )
            self._record_event(
                {
                    "stage": stage,
                    "status": "failed",
                    "error": str(exc),
                    "finished_at": finished_at.isoformat(),
                }
            )
            return

        finished_at = datetime.now(UTC)
        with self._lock:
            self._snapshot = PipelineJobSnapshot(
                stage=stage,
                status="succeeded",
                started_at=self._snapshot.started_at,
                finished_at=finished_at,
                result=result,
            )
        self._record_event(
            {
                "stage": stage,
                "status": "succeeded",
                "result": result,
                "finished_at": finished_at.isoformat(),
            }
        )
