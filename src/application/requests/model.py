"""Request DTOs for the identity-only model catalog (spec §B MODEL-001…006).

A Model accepts ONLY ``name`` and ``provider_model`` — the removed connection
fields (``provider``, ``base_url``) are Settings-owned and any unknown field is
rejected (``extra="forbid"``, MODEL-005).
"""

from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.application.requests.guards import NonNullableFieldsGuard


def _trim_non_empty(value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("must be a non-empty string")
    return trimmed


class CreateModelRequest(BaseModel):
    """POST /api/v1/models body — exactly ``name`` and ``provider_model`` (MODEL-002)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=500)
    provider_model: str = Field(..., min_length=1, max_length=500)

    @field_validator("name", "provider_model")
    @classmethod
    def _required(cls, value: str) -> str:
        return _trim_non_empty(value)


class UpdateModelRequest(NonNullableFieldsGuard):
    """PATCH /api/v1/models/{id} body — a non-empty ``name``/``provider_model`` subset.

    Both fields are required and non-nullable (MODEL-004): an explicit ``null``
    is rejected at this boundary with a clean 422, an omitted field keeps its
    current value (MODEL-006) and an empty PATCH ``{}`` stays rejected.
    """

    model_config = ConfigDict(extra="forbid")

    NON_NULLABLE_FIELDS: ClassVar[frozenset[str]] = frozenset({"name", "provider_model"})

    name: str | None = Field(default=None, min_length=1, max_length=500)
    provider_model: str | None = Field(default=None, min_length=1, max_length=500)

    @field_validator("name", "provider_model")
    @classmethod
    def _required_when_present(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _trim_non_empty(value)

    @model_validator(mode="after")
    def _at_least_one(self) -> Self:
        if self.model_fields_set.isdisjoint({"name", "provider_model"}):
            raise ValueError("provide at least one editable field")
        return self
