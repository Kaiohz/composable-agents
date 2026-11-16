"""Immutable prompt version use cases (spec §5.1 PR-003)."""

import logging

from src.domain.entities.prompt import PromptVersion
from src.domain.entities.prompt_version_summary import PromptVersionSummary
from src.domain.logging.messages import LogMessage
from src.domain.ports.prompt_repository import PromptRepository

logger = logging.getLogger(__name__)


class AddPromptVersionUseCase:
    """Create the next sequential immutable version (transactional, PR-003)."""

    def __init__(self, prompt_repository: PromptRepository) -> None:
        self._prompt_repository = prompt_repository

    async def execute(self, prompt_id: str, content: str) -> PromptVersion:
        """Append an immutable version; concurrent submissions get distinct
        numbers or an explicit conflict — never an overwrite or duplicate.

        Args:
            prompt_id: Owner prompt id.
            content: Verbatim content (may be empty).

        Returns:
            The created PromptVersion; the prompt's ``latest_version`` is bumped.
        """
        version = await self._prompt_repository.add_version(prompt_id, content)
        logger.info(LogMessage.PROMPT_REGISTRY_VERSION_ADDED, prompt_id, version.version)
        return version


class ListPromptVersionsUseCase:
    """List a prompt's versions ordered numerically ascending (PR-003)."""

    def __init__(self, prompt_repository: PromptRepository) -> None:
        self._prompt_repository = prompt_repository

    async def execute(self, prompt_id: str) -> list[PromptVersionSummary]:
        """Return the version history as summary projections.

        Each entry carries prompt_id, version number and creation timestamp
        only (PR-004 / QA BUG-004); the immutable content is read per version.
        """
        return await self._prompt_repository.list_versions(prompt_id)


class GetPromptVersionUseCase:
    """Fetch one immutable version's full verbatim content."""

    def __init__(self, prompt_repository: PromptRepository) -> None:
        self._prompt_repository = prompt_repository

    async def execute(self, prompt_id: str, version: int) -> PromptVersion:
        """Return the exact immutable version (no Latest fallback, §9).

        Args:
            prompt_id: Owner prompt id.
            version: Pinned version number (> 0).

        Raises:
            PromptNotFoundError: Missing or foreign prompt.
            PromptVersionNotFoundError: The version does not exist.
        """
        return await self._prompt_repository.get_version(prompt_id, version)
