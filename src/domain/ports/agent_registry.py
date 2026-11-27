from abc import ABC, abstractmethod

from src.domain.ports.agent_runner import AgentRunner


class AgentRegistry(ABC):
    """Registre d'agents permettant de recuperer un runner par nom."""

    @abstractmethod
    async def get_runner(self, agent_name: str, reasoning_effort: str | None = None) -> AgentRunner:
        """Retourne le runner pour l'agent donne, le creant si necessaire.

        Raises:
            AgentNotFoundError: Si aucun agent avec ce nom n'existe.
        """
        ...

    @abstractmethod
    async def list_agents(self) -> list[str]:
        """Liste les noms des agents disponibles."""
        ...

    @abstractmethod
    async def invalidate(self, agent_name: str) -> None:
        """Invalide le runner cache pour l'agent donne."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Libere les ressources des agents crees."""
        ...

    def degraded_capabilities(self, agent_name: str) -> list[dict]:  # noqa: ARG002 — default carries no state
        """Return degraded capability information from the LATEST agent build.

        REQ-040: when optional capability sources (external MCP servers) fail
        during runner construction, the failure is surfaced as degraded
        capability information rather than silently swallowed. Implementations
        record the information of their most recent build of ``agent_name``;
        the default implementation assumes no degraded capabilities.

        Args:
            agent_name: The agent whose latest build's degradation is requested.

        Returns:
            A list of JSON-safe ``{"name", "reason"}`` entries (empty when the
            latest build had no degradation).
        """
        return []
