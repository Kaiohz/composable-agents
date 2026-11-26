"""Metadata-only model catalog domain entity (spec §B MODEL-001).

A Model resource carries ONLY its identity: UUID, display name and the
``provider_model`` identifier to execute. The provider connection (``provider``,
``base_url``, ``api_key``) is owned exclusively by the user's LLM Settings — a
Model never duplicates it (MODEL-009: no credential field exists here, and no
Settings connection attribute may override the Settings-resolved endpoint at
runtime).

Ownership (``user_id``) is not part of the entity: the persistence adapter
derives it from the request context on writes and Row-Level Security filters
reads, so API responses expose exactly the identity field set from MODEL-001.
"""

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator


class ModelMetadata(BaseModel):
    """A reusable, user-owned model identity (display name + provider_model)."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    provider_model: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("name", "provider_model")
    @classmethod
    def required_fields_non_empty(cls, value: str) -> str:
        """Display name and provider model identifier must be non-empty (MODEL-002)."""
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("must be a non-empty string")
        return trimmed
