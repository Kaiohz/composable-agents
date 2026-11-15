"""Request DTOs for the metadata-only model catalog (spec §5.2 MDL-001…MDL-003)."""

from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from src.application.requests.guards import NonNullableFieldsGuard


def _trim_non_empty(value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("must be a non-empty string")
    return trimmed


def _serialized_base_url(value: HttpUrl | None) -> str | None:
    """Normalize an HttpUrl to its plain string form for storage."""
    return str(value) if value is not None else None


class CreateModelRequest(BaseModel):
    """POST /api/v1/models body — metadata only (MDL-001)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=500)
    provider: str = Field(..., min_length=1, max_length=200)
    provider_model: str = Field(..., min_length=1, max_length=500)
    # A supplied base URL must pass backend URL validation (MDL-003).
    base_url: HttpUrl | None = None

    @field_validator("name", "provider", "provider_model")
    @classmethod
    def _required(cls, value: str) -> str:
        return _trim_non_empty(value)


class UpdateModelRequest(NonNullableFieldsGuard):
    """PATCH /api/v1/models/{id} body — any editable metadata subset.

    Nullable semantics (QA BUG-009): ``base_url`` is the spec-declared nullable
    field (an explicit ``null`` clears it); ``name`` / ``provider`` /
    ``provider_model`` are required and non-nullable — an explicit ``null`` for
    either is rejected at this boundary with a clean 422.
    """

    model_config = ConfigDict(extra="forbid")

    NON_NULLABLE_FIELDS: ClassVar[frozenset[str]] = frozenset({"name", "provider", "provider_model"})

    name: str | None = Field(default=None, min_length=1, max_length=500)
    provider: str | None = Field(default=None, min_length=1, max_length=200)
    provider_model: str | None = Field(default=None, min_length=1, max_length=500)
    base_url: HttpUrl | None = None

    @field_validator("name", "provider", "provider_model")
    @classmethod
    def _required_when_present(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _trim_non_empty(value)

    @model_validator(mode="after")
    def _at_least_one(self) -> Self:
        if self.model_fields_set.isdisjoint({"name", "provider", "provider_model", "base_url"}):
            raise ValueError("provide at least one editable field")
        return self
