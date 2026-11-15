"""Request DTOs for the local prompt registry (spec §5.1 PR-004).

Content is required in create/version requests but may be an empty string and
is stored/returned verbatim. ``PATCH`` never accepts content (metadata-only
edits) — the ``content`` field is rejected outright via ``extra="forbid"``.
"""

from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.application.requests.guards import NonNullableFieldsGuard


class CreatePromptRequest(BaseModel):
    """POST /api/v1/prompts body: metadata + version-1 content."""

    name: str = Field(..., min_length=1, max_length=500)
    description: str | None = None
    content: str

    @field_validator("name")
    @classmethod
    def trim_name(cls, value: str) -> str:
        """Trimmed, non-empty name (PR-002)."""
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("name must not be empty or whitespace-only")
        return trimmed


class UpdatePromptRequest(NonNullableFieldsGuard):
    """PATCH /api/v1/prompts/{id} body: name and/or description only.

    ``content`` is NOT accepted here (PR-004) — sending it is a validation
    error, not a silent rename of the immutable versions.

    Nullable semantics (QA BUG-002 + BUG-006): ``description`` is spec-declared
    ``string | null`` so an explicit ``null`` clears it; ``name`` is required,
    non-nullable — an explicit ``null`` is rejected at this boundary.
    """

    model_config = ConfigDict(extra="forbid")

    NON_NULLABLE_FIELDS: ClassVar[frozenset[str]] = frozenset({"name"})

    name: str | None = Field(default=None, min_length=1, max_length=500)
    description: str | None = None

    @field_validator("name")
    @classmethod
    def trim_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("name must not be empty or whitespace-only")
        return trimmed

    @model_validator(mode="after")
    def at_least_one_field(self) -> Self:
        """Require at least one **present** editable field (BUG-002).

        Field presence, not value: an explicit ``"description": null`` is a
        valid single-field PATCH that *clears* the description (PR-004), while
        a genuinely empty body ``{}`` is rejected.
        """
        if self.model_fields_set.isdisjoint({"name", "description"}):
            raise ValueError("provide 'name' and/or 'description'")
        return self


class CreatePromptVersionRequest(BaseModel):
    """POST /api/v1/prompts/{id}/versions body: verbatim new version content."""

    content: str
