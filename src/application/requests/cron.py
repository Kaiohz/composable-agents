"""Request DTOs for the cron subsystem (spec §5.3 CRN-001/CRN-002/CRN-003)."""

from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.application.requests.guards import NonNullableFieldsGuard


def _trim_non_empty(value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("must be a non-empty string")
    return trimmed


class CreateCronRequest(BaseModel):
    """POST /api/v1/crons body (CRN-001 defaults apply)."""

    name: str = Field(..., min_length=1, max_length=200)
    agent_name: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1)
    # Standard five-field expression WITHOUT seconds (§9 rejects six fields).
    cron_expression: str = Field(..., min_length=1, max_length=100)
    timezone: str = Field(default="UTC", min_length=1, max_length=100)
    enabled: bool = True

    @field_validator("name", "message", "cron_expression", "timezone", "agent_name")
    @classmethod
    def _required(cls, value: str) -> str:
        return _trim_non_empty(value)


class UpdateCronRequest(NonNullableFieldsGuard):
    """PATCH /api/v1/crons/{id} body — any editable subset of the definition.

    Nullable semantics (QA BUG-006…BUG-010): every editable field except
    ``timezone`` (which the spec defaults to UTC) is required and non-nullable —
    an explicit ``null`` for any of them is rejected at this boundary with a
    clean 422 (never a NOT NULL 5xx or a ``None``-interpolated 404).
    """

    model_config = ConfigDict(extra="forbid")

    NON_NULLABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"name", "agent_name", "message", "cron_expression", "enabled"}
    )

    name: str | None = Field(default=None, min_length=1, max_length=200)
    agent_name: str | None = Field(default=None, min_length=1, max_length=100)
    message: str | None = None
    cron_expression: str | None = Field(default=None, min_length=1, max_length=100)
    timezone: str | None = Field(default=None, min_length=1, max_length=100)
    enabled: bool | None = None

    @field_validator("name", "agent_name", "cron_expression", "timezone", "message")
    @classmethod
    def _required_when_present(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _trim_non_empty(value)

    @model_validator(mode="after")
    def _at_least_one(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("provide at least one editable field")
        return self
