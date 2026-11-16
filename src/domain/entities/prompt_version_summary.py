"""Prompt version list read-model (spec §5.1 PR-004, QA BUG-004).

A version-list row: version number + creation timestamp only. The full
immutable content of one version is read through
``GET /api/v1/prompts/{id}/versions/{version}``.
"""

from datetime import datetime

from pydantic import BaseModel


class PromptVersionSummary(BaseModel):
    """One entry of a prompt's version history."""

    prompt_id: str
    version: int
    created_at: datetime
