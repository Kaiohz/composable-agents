import logging

from src.domain.entities.thread import Thread, ThreadSource
from src.domain.errors.agent import AgentNotFoundError
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_registry import AgentRegistry
from src.domain.ports.thread_repository import ThreadRepository

logger = logging.getLogger(__name__)


class CreateThreadUseCase:
    def __init__(self, threads: ThreadRepository, registry: AgentRegistry) -> None:
        self._threads = threads
        self._registry = registry

    async def execute(self, agent_name: str, source: ThreadSource = ThreadSource.CHAT) -> Thread:
        """Create a thread for ``agent_name`` with a TRUSTED immutable source.

        Public REST thread creation never passes ``source`` (always ``chat``);
        the trusted cron execution path passes ``cron`` and the trusted
        one-shot MCP execution path passes ``mcp``. Callers can therefore
        never spoof a source through a client-controlled field (SRC-002).

        Args:
            agent_name: The agent that owns the thread.
            source: Immutable thread origin (trusted internal parameter).

        Returns:
            The newly created Thread.

        Raises:
            AgentNotFoundError: The agent is not visible to the current user.
        """
        if agent_name not in await self._registry.list_agents():
            raise AgentNotFoundError(ErrorMessage.AGENT_NOT_FOUND.format(name=agent_name))
        thread = await self._threads.create(agent_name, source)
        logger.info(LogMessage.THREAD_CREATED, thread.id, agent_name)
        return thread
