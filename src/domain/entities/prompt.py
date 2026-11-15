"""Local prompt registry domain entities (spec §5.1 PR-002).

Prompts are per-user records stored in local PostgreSQL. Prompt identity is
the UUID, not the mutable display name. Versions are immutable and strictly
sequential (version 1 is created atomically with the prompt).
"""

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator


class Prompt(BaseModel):
    """Prompt metadata (owned by a user; versions held separately)."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str | None = None
    latest_version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Owner of the prompt — set by the repository from the ``current_user_id``
    # contextvar on create. Defaults to ``""`` so existing constructions (no
    # auth context) keep working and the row is invisible under RLS but visible
    # in SQLite tests (no RLS policies).
    user_id: str = ""

    @field_validator("name")
    @classmethod
    def name_trimmed_non_empty(cls, value: str) -> str:
        """Trim whitespace and reject empty names (PR-002)."""
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("name must be a non-empty string")
        return trimmed


class PromptVersion(BaseModel):
    """An immutable prompt content version (PR-002/PR-003).

    ``content`` is stored and returned verbatim, including whitespace. Empty
    string is a valid content value.
    """

    prompt_id: str
    version: int = Field(ge=1)
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    user_id: str = ""
