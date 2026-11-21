"""Port for the per-user built-in assistant override store."""

from abc import ABC, abstractmethod

from src.domain.entities.builtin_agent_override import BuiltinAgentOverride


class BuiltinAgentOverrideStore(ABC):
    """User-scoped persistence contract for built-in assistant customizations.

    Implementations MUST scope every operation to the authenticated user
    resolved from the request context. The store is intentionally separate
    from :class:`~src.domain.ports.agent_config_store.AgentConfigStore`: the
    shared agent YAML bucket is globally keyed by agent name and must never
    carry per-user built-in overrides.
    """

    @abstractmethod
    async def get(self, agent_name: str) -> BuiltinAgentOverride | None:
        """Return the current user's override for this agent, or ``None``."""
        ...

    @abstractmethod
    async def put(self, override: BuiltinAgentOverride) -> None:
        """Insert or replace the current user's override snapshot."""
        ...

    @abstractmethod
    async def delete(self, agent_name: str) -> None:
        """Delete the current user's override; absent overrides are a no-op."""
        ...
