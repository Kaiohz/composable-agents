from abc import ABC, abstractmethod

from src.domain.entities.thread import Thread, ThreadSource


class ThreadRepository(ABC):
    @abstractmethod
    async def create(
        self,
        agent_name: str,
        source: ThreadSource = ThreadSource.CHAT,
    ) -> Thread:
        """Cree un nouveau thread de conversation.

        The thread source is immutable after insert (SRC-001): public/chat
        creation always receives ``chat``; only the trusted cron execution
        path requests ``cron``.
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
