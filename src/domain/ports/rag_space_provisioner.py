"""Outbound port for RAG-space provisioning on mcp-raganything (REQ-010/018).

mcp-raganything owns the per-user × per-agent RAG spaces (``POST/DELETE
/api/v1/spaces`` contract). This port lets the agent-lifecycle use cases
provision/deprovision the space of an agent (and of each of its subagents)
with the ORIGINATING user's forwarded credential — never a service account —
and tolerate idempotent replays (a space that already exists answers 200 and
MUST be treated as a success).
"""

from abc import ABC, abstractmethod


class RagSpaceProvisioner(ABC):
    """Abstract client for the mcp-raganything RAG-spaces REST API."""

    @abstractmethod
    async def provision(self, agent_slug: str) -> None:
        """Idempotently ensure the space ``{agent_slug}`` exists (REQ-010).

        Args:
            agent_slug: The space slug (the agent/subagent name).

        Raises:
            McpRegistryUnavailableError: The spaces service is unreachable
                (callers treat provisioning as best-effort).
            McpRegistryRequestError: The spaces service rejected the call
                (status-only detail).
        """
        ...

    @abstractmethod
    async def deprovision(self, agent_slug: str) -> None:
        """Cascade-delete the space ``{agent_slug}`` (REQ-010).

        Strict REST semantics on the server (404 for a missing space, no
        partial deletion); a 404 IS a success here — the target state "no
        space for this slug" is already reached (best-effort lifecycle).

        Args:
            agent_slug: The space slug (the agent/subagent name).

        Raises:
            McpRegistryUnavailableError: The spaces service is unreachable.
            McpRegistryRequestError: Any other non-2xx (status-only detail).
        """
        ...

    async def aclose(self) -> None:  # noqa: B027 — optional lifecycle hook  # NOSONAR python:S7503 — intentional async no-op hook: abstract protocol method; concrete adapters are awaited (httpx aclose)  # noqa: B027
        """Release any owned HTTP resources (composition-root lifecycle hook).

        Concrete adapters with a persistent connection pool override this.
        """
        return None
