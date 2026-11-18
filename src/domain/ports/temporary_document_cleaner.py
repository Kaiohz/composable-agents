"""Outbound port: post-turn temporary document cleanup (REQ-041..047).

Use cases rely on this thin port so the MCP plumbing stays behind the
boundary. Implementations MUST be best-effort and idempotent: a cleanup
failure is logged (ERROR) and retried once, NEVER propagated — it must not
mask an agent response that was already produced (REQ-046).
"""

from abc import ABC, abstractmethod

from src.domain.entities.prepared_chat_turn import TemporaryDocumentRef


class TemporaryDocumentCleaner(ABC):
    """Outbound port deleting temporary chat documents after terminal responses."""

    @abstractmethod
    async def delete_temporary_documents(self, agent_name: str, refs: list[TemporaryDocumentRef]) -> None:
        """Delete the given temporary RustFS objects through the Files MCP server.

        Args:
            agent_name: The turn's agent (resolves the ``/files/mcp`` server).
            refs: Temporary document references built by the backend (never by
                the LLM — REQ-040).

        Raises:
            Nothing: every failure is logged with the thread/agent context and
            either retried once or dropped, per REQ-046.
        """
        ...
