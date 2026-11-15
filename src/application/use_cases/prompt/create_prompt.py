"""CreatePromptUseCase — create prompt metadata plus immutable version 1."""

import logging

from src.domain.entities.prompt import Prompt
from src.domain.logging.messages import LogMessage
from src.domain.ports.prompt_repository import PromptRepository

logger = logging.getLogger(__name__)


class CreatePromptUseCase:
    """Create a prompt and its immutable version 1 atomically (PR-003)."""

    def __init__(self, prompt_repository: PromptRepository) -> None:
        self._prompt_repository = prompt_repository

    async def execute(self, name: str, description: str | None, content: str) -> Prompt:
        """Create the prompt (duplicate display names are permitted, PR-001).

        Args:
            name: Trimmed non-empty display name.
            description: Optional description.
            content: Verbatim version-1 content (may be empty).

        Returns:
            The stored prompt metadata with ``latest_version == 1``.
        """
        prompt = Prompt(name=name, description=description)
        created = await self._prompt_repository.create_with_version(prompt, content)
        logger.info(LogMessage.PROMPT_REGISTRY_CREATED, created.id, created.latest_version)
        return created
