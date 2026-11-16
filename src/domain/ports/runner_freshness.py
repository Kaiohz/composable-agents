"""Outbound port: RunnerFreshness (cross-worker dependency-aware freshness).

CACHE-002: after a committed mutation (agent YAML edit, new prompt version,
model metadata edit), every backend worker must rebuild the referencing
runner on its next invocation. Process-local cache eviction alone does not
satisfy this; implementations derive a per-agent fingerprint from committed
database state so every worker observes the same revision.
"""

from abc import ABC, abstractmethod


class RunnerFreshness(ABC):
    """Outbound port providing a cross-worker agent dependency fingerprint."""

    @abstractmethod
    async def fingerprint(self, agent_name: str) -> str:
        """Return the current dependency fingerprint for an agent.

        The fingerprint covers the agent YAML revision plus its registry
        dependencies (prompt Latest version/model metadata). A cached runner
        whose fingerprint no longer matches must be rebuilt. When a referenced
        dependency row no longer exists the implementation raises the matching
        domain error so stale references fail explicitly.

        Args:
            agent_name: Name of the agent (resolved under the current user's
                RLS context).

        Returns:
            A deterministic fingerprint string.
        """
        ...
