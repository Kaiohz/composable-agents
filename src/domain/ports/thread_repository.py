from abc import ABC, abstractmethod

from src.domain.entities.thread import Thread, ThreadSource


class ThreadRepository(ABC):
    @abstractmethod
    async def create(
        self,
        agent_name: str | Thread,
        source: ThreadSource = ThreadSource.CHAT,
    ) -> Thread:
        """Cree un nouveau thread de conversation.

        Two accepted shapes (additive, SRC-001 trusted-source split):

        * ``create("agent-name")`` / ``create("agent-name", source)`` — the
          legacy name-based creation (chat default; the trusted cron path
          passes ``cron`` and the trusted one-shot MCP path passes ``mcp``);
        * ``create(thread_entity)`` — persist from a pre-built domain Thread:
          its ``agent_name`` and ``source`` are used verbatim so the caller
          decides the immutable origin (repository round-trip contract),
          while the id/timestamps/owner plumbing stays repository-owned.

        The thread source is immutable after insert (SRC-001).
        """
        ...

    @abstractmethod
    async def get(self, thread_id: str) -> Thread:
        """Recupere un thread par son ID."""
        ...

    @abstractmethod
    async def list_all(self) -> list[Thread]:
        """Liste tous les threads."""
        ...

    @abstractmethod
    async def delete(self, thread_id: str) -> None:
        """Supprime un thread."""
        ...
