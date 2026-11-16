"""Prompt registry read/list/update/delete use cases."""

import logging
from typing import Any

from src.domain.entities.prompt import Prompt
from src.domain.logging.messages import LogMessage
from src.domain.ports.prompt_repository import PromptRepository

logger = logging.getLogger(__name__)


class GetPromptUseCase:
    """Fetch owned prompt metadata (identity is the UUID, PR-001)."""

    def __init__(self, prompt_repository: PromptRepository) -> None:
        self._prompt_repository = prompt_repository

    async def execute(self, prompt_id: str) -> Prompt:
        """Return the prompt metadata.

        Raises:
            PromptNotFoundError: Missing or foreign prompt (non-enumerating).
        """
        prompt = await self._prompt_repository.get(prompt_id)
        logger.info(LogMessage.PROMPT_REGISTRY_FETCHED, prompt.id)
        return prompt


class ListPromptsUseCase:
    """List all prompt metadata owned by the current user."""

    def __init__(self, prompt_repository: PromptRepository) -> None:
        self._prompt_repository = prompt_repository

    async def execute(self) -> list[Prompt]:
        prompts = await self._prompt_repository.list_all()
        logger.info(LogMessage.AGENT_CONFIG_LISTED, len(prompts))
        return prompts


class UpdatePromptUseCase:
    """Metadata-only edit (name and/or description) — never creates a version.

    Only explicitly-provided fields are applied (``exclude_unset`` semantics),
    so an omitted ``description`` leaves it untouched while an explicit
    ``null`` clears it.
    """

    def __init__(self, prompt_repository: PromptRepository) -> None:
        self._prompt_repository = prompt_repository

    async def execute(self, prompt_id: str, changes: dict[str, Any]) -> Prompt:
        """Apply the metadata edit and return the updated prompt.

        Args:
            prompt_id: Prompt UUID.
            changes: Field→value mapping of editable metadata
                (``name`` and/or ``description``).

        Raises:
            PromptNotFoundError: Missing or foreign prompt.
        """
        current = await self._prompt_repository.get(prompt_id)
        updated = current.model_copy(update=changes)
        saved = await self._prompt_repository.update_metadata(updated)
        logger.info(LogMessage.PROMPT_REGISTRY_METADATA_UPDATED, saved.id)
        return saved


class DeletePromptUseCase:
    """Delete a prompt when unreferenced (409 Conflict otherwise)."""

    def __init__(self, prompt_repository: PromptRepository) -> None:
        self._prompt_repository = prompt_repository

    async def execute(self, prompt_id: str) -> None:
        """Delete the prompt and its versions.

        Raises:
            PromptNotFoundError: Missing or foreign prompt.
            PromptInUseError: Referenced by at least one agent (Latest or pinned).
        """
        await self._prompt_repository.delete(prompt_id)
        logger.info(LogMessage.PROMPT_REGISTRY_DELETED, prompt_id)
