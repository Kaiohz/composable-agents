"""Cron definition domain entity (spec §5.3 CRN-001).

A cron schedules an owned agent with a standard five-field expression (no
seconds) and an IANA timezone. The original expression and timezone are
retained verbatim; occurrence timestamps are evaluated and persisted in UTC.
"""

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator


class Cron(BaseModel):
    """A scheduled agent invocation owned by a user."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    agent_name: str
    message: str
    cron_expression: str
    timezone: str = "UTC"
    enabled: bool = True
    # UTC timestamp of the next due occurrence; ``None`` for disabled crons.
    next_run_at: datetime | None = None
    # Denormalized last-run summary for list/detail views (updated by the
    # runner when a run reaches a terminal state).
    last_run_status: str | None = None
    last_run_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Owner — set by the repository from the ``current_user_id`` contextvar on
    # create; scheduler operations bind the owner context explicitly (SEC-001).
    user_id: str = ""

    @field_validator("name", "message")
    @classmethod
    def required_fields_non_empty(cls, value: str) -> str:
        """Name and message must be non-empty strings (CRN-001)."""
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("must be a non-empty string")
        return trimmed
