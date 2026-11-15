"""API response projections for the local prompt registry (spec §5.1, QA BUG-005).

The domain entities (:class:`~src.domain.entities.prompt.Prompt`,
:class:`~src.domain.entities.prompt.PromptVersion`) carry the ``user_id``
ownership plumbing the RLS write path needs; these DTOs are the HTTP boundary
shapes and deliberately omit it — PR-002 enumerates the response field sets and
none of them contains the owner id (consistent with ``ModelMetadata``).
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from src.domain.entities.prompt_version_summary import PromptVersionSummary as PromptVersionSummaryEntity


class PromptResponse(BaseModel):
    """Prompt metadata (PR-002 field set, no owner id)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str | None = None
    latest_version: int
    created_at: datetime
    updated_at: datetime


class PromptVersionResponse(BaseModel):
    """One immutable prompt version, full verbatim content (PR-002)."""

    model_config = ConfigDict(from_attributes=True)

    prompt_id: str
    version: int
    content: str
    created_at: datetime


class PromptVersionSummary(PromptVersionSummaryEntity):
    """API version-list projection (PR-004, QA BUG-004).

    Reuses the domain read-model fields and only adds the API attribute-source
    configuration; the full content of one version remains available through
    ``GET /api/v1/prompts/{id}/versions/{version}``.
    """

    model_config = ConfigDict(from_attributes=True)
