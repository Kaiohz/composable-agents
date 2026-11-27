import logging
from dataclasses import dataclass, field

from src.application.use_cases._subagent_ref_utils import invalidate_dependent_agents
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_config_loader import AgentConfigLoader
from src.domain.ports.agent_config_repository import AgentConfigRepository
from src.domain.ports.agent_config_store import AgentConfigStore
from src.domain.ports.agent_registry import AgentRegistry
from src.domain.services.agent_rag_spaces import AgentRagSpaces, safe_reason

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DeletedAgentTopology:
    """The space topology of the agent being deleted (the use case's result)."""

    agent_name: str
    subagent_names: list[str] = field(default_factory=list)


class DeleteAgentConfigUseCase:
    """Delete an agent configuration from persistent storage."""

    def __init__(
        self,
        config_store: AgentConfigStore,
        config_repository: AgentConfigRepository,
        agent_registry: AgentRegistry,
        config_loader: AgentConfigLoader | None = None,
        rag_spaces: AgentRagSpaces | None = None,
    ) -> None:
        self._config_store = config_store
        self._config_repository = config_repository
        self._agent_registry = agent_registry
        self._config_loader = config_loader
        self._rag_spaces = rag_spaces

    async def execute(self, name: str) -> None:
        """Delete from MinIO and PostgreSQL, invalidate cache.

        REQ-018: the deletion CASCADES best-effort — the agent's and every
        subagent's RAG space is deprovisioned (``DELETE
        /api/v1/spaces/{agent_slug}``) and ``/memories/{agent_name}/`` (plus
        each subagent's) is purged from the RUNNING user's store. An
        unavailable spaces API or store NEVER prevents the agent deletion
        (only warnings).

        Args:
            name: Agent name to delete.

        Raises:
            AgentNotFoundError: If no agent with this name exists.
        """
        await self._config_repository.get(name)

        topology = await self._agent_topology(name)
        await self._config_store.delete(name)
        await self._config_repository.delete(name)
        await self._agent_registry.invalidate(name)
        await invalidate_dependent_agents(self._config_store, self._agent_registry, name)

        if self._rag_spaces is not None:
            # REQ-018 + review C-3: fire-and-forget background cascade
            # (deprovision + memories purge). The task captures this
            # request's contextvars, so the store namespace stays the
            # RUNNING user's; the C-2 ownership guard inside the task skips
            # any slug another top-level agent now owns.
            agent, subs = topology.agent_name, topology.subagent_names
            self._rag_spaces.schedule_deletion_cascade(agent, subs)

        logger.info(LogMessage.AGENT_CONFIG_DELETED_UC, name)

    async def _agent_topology(self, name: str) -> _DeletedAgentTopology:
        """Resolve the deleted agent's space topology (itself + subagents).

        Best-effort: the topology is read from the STILL-STORED YAML before
        it is deleted. An unreadable YAML degrades to the agent's own name
        only (the destructive deletes below never depend on it).

        Args:
            name: The agent name.

        Returns:
            The topology (agent name + subagent names).
        """
        if self._config_loader is None:
            return _DeletedAgentTopology(name)
        try:
            yaml_content = await self._config_store.get(name)
            config = self._config_loader.load_from_string(yaml_content)
        except Exception as exc:  # noqa: BLE001
            # The topology basis is best-effort only: an unreadable stored YAML degrades to the agent's own name.
            logger.debug(LogMessage.RAG_SPACE_DIFF_BASIS_UNAVAILABLE, name, safe_reason(exc))
            return _DeletedAgentTopology(name)
        return _DeletedAgentTopology(config.name, [subagent.name for subagent in config.subagents])
