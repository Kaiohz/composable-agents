"""Durable cron run domain entity (spec §5.3 CRN-003).

Every accepted execution attempt (scheduled or manual) creates exactly one
run record, linked to the cron thread it produced. A failure after acceptance
retains its run record and thread; skipped overlaps produce a run record with
no thread; manual runs carry no occurrence timestamp.
"""

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field


class CronTrigger(StrEnum):
    """How the run was created: a scheduler occurrence or a manual Run now."""

    SCHEDULED = "scheduled"
    MANUAL = "manual"


class CronRunStatus(StrEnum):
    """Durable cron run states. Failed runs are terminal (no automatic retry)."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class CronRun(BaseModel):
    """A durable record of one cron execution attempt."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    cron_id: str
    trigger: CronTrigger = CronTrigger.SCHEDULED
    status: CronRunStatus = CronRunStatus.QUEUED
    # UTC occurrence time for scheduled runs; ``None`` for manual runs.
    occurrence_at: datetime | None = None
    thread_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    # Safe error summary (redacted: never contains application authentication
    # tokens or other sensitive configuration — SCHED-001).
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Owner — set by the repository from the owner-context contextvar on
    # insert so RLS protects run history per user (SEC-001).
    user_id: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source(self) -> str | None:
        """Thread source ``cron`` when a linked thread exists, else ``None``."""
        return "cron" if self.thread_id else None
