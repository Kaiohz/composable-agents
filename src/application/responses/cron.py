"""API response projections for the cron subsystem (spec §5.3, QA BUG-005).

The domain entities (:class:`~src.domain.entities.cron.Cron`,
:class:`~src.domain.entities.cron_run.CronRun`) carry the ``user_id`` ownership
plumbing the RLS write path needs; these DTOs are the HTTP boundary shapes and
deliberately omit it — CRN-001/CRN-003 enumerate the response field sets and
none of them contains the owner id (consistent with ``ModelMetadata``).
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, computed_field


class CronResponse(BaseModel):
    """Cron definition + scheduling state (CRN-001 field set, no owner id)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    agent_name: str
    message: str
    cron_expression: str
    timezone: str
    enabled: bool
    next_run_at: datetime | None = None
    last_run_status: str | None = None
    last_run_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class CronRunResponse(BaseModel):
    """Durable run record (CRN-003 field set, no owner id)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    cron_id: str
    trigger: str
    status: str
    occurrence_at: datetime | None = None
    thread_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    created_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source(self) -> str | None:
        """Thread source ``cron`` when a linked thread exists, else ``None``."""
        return "cron" if self.thread_id else None
