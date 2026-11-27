"""Store-backed adapter resolving an agent's declared subagent topology (C-2b).

Implements :class:`~src.domain.ports.agent_topology_resolver.AgentTopologyResolver`
over the persisted agent definitions: the stored YAML (object store) parsed
by the shared ``AgentConfigLoader``.

Boundary contract (corrected per review F-2 — the agent-YAML store is keyed
by agent NAME in one bucket, NOT per user, so this read is name-scoped, not
user-scoped): the adapter therefore makes every failure mode FAIL SAFE by
raising :class:`AgentTopologyUnresolvedError` instead of silently narrowing
the caller's protection set:

* the stored blob exceeds the write-time YAML size cap (a read-side bound on
  the parse surface — NV-1: historical objects predating the write-time
  validation must not reach the parser unbounded);
* the definition cannot be read or parsed;
* the fetched definition's root ``name`` field does not match the requested
  agent (a foreign/mismatched definition must never influence which of the
  caller's spaces are destroyed).

In every case the caller (:meth:`AgentRagSpaces._owned_space_slugs`) treats
the peer topology as unresolvable, which makes the WHOLE ownership set
unresolvable and narrows the cascade to the agent's own slug — an
unreadable, oversized or foreign definition can never broaden the
destructive blast radius.
"""

import logging

from src.domain.constants import MAX_YAML_SIZE
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_config_loader import AgentConfigLoader
from src.domain.ports.agent_config_store import AgentConfigStore
from src.domain.ports.agent_topology_resolver import AgentTopologyResolver, AgentTopologyUnresolvedError
from src.domain.services.agent_rag_spaces import safe_reason

logger = logging.getLogger(__name__)


class StoreAgentTopologyResolver(AgentTopologyResolver):
    """Resolve declared subagent names from the persisted agent YAML."""

    def __init__(self, config_store: AgentConfigStore, config_loader: AgentConfigLoader) -> None:
        """Wire the ports.

        Args:
            config_store: The agent YAML object store (name-keyed reads —
                see the module docstring for the boundary contract).
            config_loader: The shared YAML parser/validator.
        """
        self._config_store = config_store
        self._config_loader = config_loader

    async def subagent_names(self, agent_name: str) -> list[str]:
        """Return the subagent names the agent's stored YAML declares.

        Every failure mode raises :class:`AgentTopologyUnresolvedError` so
        the caller fails safe (review F-1: an individually unreadable peer
        definition must invalidate the WHOLE owned set rather than drop only
        that peer's declared slugs from it).

        Args:
            agent_name: The persisted agent's exact name.

        Returns:
            The declared subagent names (empty when the agent declares none).

        Raises:
            AgentTopologyUnresolvedError: The stored definition exceeds the
                size cap, cannot be read/parsed, or its root ``name`` field
                does not match ``agent_name`` (foreign definition).
        """
        try:
            yaml_content = await self._config_store.get(agent_name)
        except Exception as exc:  # noqa: BLE001
            # Fail safe: raise the sentinel so the caller narrows the whole cascade to the own slug (review F-1).
            logger.warning(LogMessage.AGENT_TOPOLOGY_UNRESOLVED, agent_name, safe_reason(exc))
            raise AgentTopologyUnresolvedError(agent_name) from exc

        if len(yaml_content.encode("utf-8")) > MAX_YAML_SIZE:
            # NV-1: the same cap the write paths enforce (the constant's
            # single owner lives in src.domain.constants) — a historical or
            # foreign oversized object must not reach the parser unbounded.
            logger.warning(LogMessage.AGENT_TOPOLOGY_YAML_OVERSIZED, agent_name, MAX_YAML_SIZE)
            raise AgentTopologyUnresolvedError(agent_name)

        try:
            config = self._config_loader.load_from_string(yaml_content)
        except Exception as exc:  # noqa: BLE001
            # Fail safe: raise the sentinel so the caller narrows the whole cascade to the own slug (review F-1).
            logger.warning(LogMessage.AGENT_TOPOLOGY_UNRESOLVED, agent_name, safe_reason(exc))
            raise AgentTopologyUnresolvedError(agent_name) from exc

        if config.name != agent_name:
            # F-2: the store is name-keyed (global); the fetched definition
            # must self-identify as the requested agent — a mismatched (e.g.
            # foreign) definition never influences the ownership set.
            logger.warning(LogMessage.AGENT_TOPOLOGY_FOREIGN_DEFINITION, agent_name, config.name)
            raise AgentTopologyUnresolvedError(agent_name)
        return [subagent.name for subagent in config.subagents]
