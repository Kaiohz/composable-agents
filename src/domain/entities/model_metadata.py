"""Metadata-only model catalog domain entity (spec §5.2 MDL-001).

A model configuration carries ONLY feature-owned metadata: UUID, display
name, provider, provider-specific model identifier and an optional base URL.
It contains no API keys, JWTs or any other credential field (MDL-002) — the
app-access API key/JWT is never copied into a Model, and existing agent
runtime credential resolution is outside this feature and unchanged.

Ownership (``user_id``) is not part of the entity: the persistence adapter
derives it from the request context on writes and Row-Level Security filters
reads, so API responses expose exactly the metadata field set from MDL-001.
"""

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator


class ModelMetadata(BaseModel):
    """A reusable, user-owned model metadata configuration."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    provider: str
    provider_model: str
    base_url: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("name", "provider", "provider_model")
    @classmethod
    def required_fields_non_empty(cls, value: str) -> str:
        """Provider, provider model identifier and display name must be non-empty (MDL-003)."""
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("must be a non-empty string")
        return trimmed
