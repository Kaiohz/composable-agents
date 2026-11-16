"""Outbound port: PromptRepository (local prompt registry, spec §5.1).

Implementations are per-user: rows are written with the ``current_user_id``
contextvar value and reads are filtered by it (plus Row-Level Security on
PostgreSQL), so a foreign or missing prompt is indistinguishable (PR-005
non-enumerating behavior).
"""

from abc import ABC, abstractmethod

from src.domain.entities.prompt import Prompt, PromptVersion
from src.domain.entities.prompt_version_summary import PromptVersionSummary


class PromptRepository(ABC):
    """Outbound port for per-user prompt metadata and immutable versions."""

    @abstractmethod
    async def create_with_version(self, prompt: Prompt, content: str) -> Prompt:
        """Create a prompt and its immutable version 1 atomically (PR-003).

        Args:
            prompt: The prompt metadata to persist (id/name/description).
            content: Verbatim version-1 content (may be an empty string).

        Returns:
            The stored prompt with ``latest_version == 1``.
        """
        ...

    @abstractmethod
    async def get(self, prompt_id: str) -> Prompt:
        """Return the prompt metadata for ``prompt_id``.

        Raises:
            PromptNotFoundError: If the prompt is missing or foreign.
        """
        ...

    @abstractmethod
    async def list_all(self) -> list[Prompt]:
        """Return all prompt metadata owned by the current user."""
        ...

    @abstractmethod
    async def update_metadata(self, prompt: Prompt) -> Prompt:
        """Persist metadata-only changes (name/description/updated_at).

        Metadata edits never create versions (PR-003).
        """
        ...

    @abstractmethod
    async def delete(self, prompt_id: str) -> None:
        """Delete the prompt and all its versions.

        Raises:
            PromptNotFoundError: If the prompt is missing or foreign.
            PromptInUseError: If any agent references the prompt (PR-005).
        """
        ...

    @abstractmethod
    async def add_version(self, prompt_id: str, content: str) -> PromptVersion:
        """Allocate and persist the next sequential immutable version.

        Version allocation must be transactional and safe under concurrent
        submissions (PR-003): concurrent calls never reuse a version number
        nor overwrite another version.

        Args:
            prompt_id: Owner prompt id.
            content: Verbatim content for the new version.

        Returns:
            The created PromptVersion.

        Raises:
            PromptNotFoundError: If the prompt is missing or foreign.
        """
        ...

    @abstractmethod
    async def list_versions(self, prompt_id: str) -> list[PromptVersionSummary]:
        """List a prompt's versions, ordered numerically ascending (PR-004).

        Each entry is a summary projection (prompt_id, version, created_at) —
        the full immutable content is fetched through :meth:`get_version`.

        Raises:
            PromptNotFoundError: If the prompt is missing or foreign.
        """
        ...

    @abstractmethod
    async def get_version(self, prompt_id: str, version: int | None = None) -> PromptVersion:
        """Return an immutable version's full verbatim content.

        Args:
            prompt_id: Owner prompt id.
            version: Exact version to fetch; ``None`` resolves Latest (the
                prompt's most recent version).

        Raises:
            PromptNotFoundError: If the prompt is missing or foreign.
            PromptVersionNotFoundError: If the pinned version does not exist.
        """
        ...
