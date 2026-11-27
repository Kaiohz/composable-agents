"""Outbound port for resolving an agent's declared subagent topology (C-2b).

The scope-key namespace is shared between top-level agents and subagents
(REQ-016 per-subagent space — the slug is the bare subagent name). The
RAG-space ownership model therefore needs to know which slugs a peer agent
CLAIMS through its DECLARED subagents, not only through its top-level name —
otherwise deleting agent A whose subagent is named ``X`` would destroy agent
B's subagent space when B declares the same ``X``.

Resolver contract (corrected per review F-2):

* the resolver reads the persisted agent definitions (stored YAML) — a LOCAL
  read. NOTE: the shipped agent-YAML store is keyed by agent NAME only
  (``{name}.yaml`` in one bucket — not per user), so this read is
  name-scoped, NOT user-scoped: with two users owning the same agent name
  the resolver may observe the other user's definition. The adapter
  therefore verifies the fetched definition's root ``name`` field against
  the requested agent and refuses (raises
  :class:`AgentTopologyUnresolvedError`) on any mismatch;
* a name-keyed read is unambiguous ONLY while agent names remain
  platform-unique; a definition that cannot be read, parsed, verified or
  trusted must never silently narrow the caller's protection set;
* FAILURES RAISE :class:`AgentTopologyUnresolvedError` — the caller
  (:meth:`AgentRagSpaces._owned_space_slugs`) treats ANY unresolved peer
  topology as an unresolvable ownership set and narrows the whole cascade
  to the agent's own slug (the fail-safe of :meth:`AgentRagSpaces._cascadable_slugs`),
  so an unreadable, oversized or foreign definition can never broaden the
  destructive blast radius.
"""

from abc import ABC, abstractmethod


class AgentTopologyUnresolvedError(RuntimeError):
    """The agent's declared topology could not be resolved."""


class AgentTopologyResolver(ABC):
    """Abstract resolver for the subagent names a persisted agent declares."""

    @abstractmethod
    async def subagent_names(self, agent_name: str) -> list[str]:
        """Return the RAW subagent names declared by the agent's definition.

        Args:
            agent_name: The persisted agent's exact name (the resolver trusts
                the caller-verified metadata row, then cross-checks the
                fetched YAML's root ``name`` against it).

        Returns:
            The declared subagent names (empty when the agent declares none).

        Raises:
            AgentTopologyUnresolvedError: The stored definition could not be
                read, exceeds the size cap, does not parse, or carries a
                root agent name that does not match the requested one — the
                caller fails safe (the whole ownership set is treated as
                unresolvable) instead of narrowing its protection.
        """
        ...
