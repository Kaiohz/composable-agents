"""JSON-safe thread projections for the MCP thread tools.

The MCP control plane projects threads to a summary WITHOUT ownership plumbing
(``user_id``) and WITHOUT loading trace events — thread management through MCP
answers identity/source/timestamps; history and trace tools carry the turn
data explicitly.
"""

from datetime import datetime

from pydantic import BaseModel

from src.domain.entities.thread import Thread, ThreadSource


class ThreadSummary(BaseModel):
    """Owner-safe thread projection (no ``user_id``, no trace payload)."""

    id: str
    agent_name: str
    source: ThreadSource
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, thread: Thread) -> "ThreadSummary":
        """Project a domain :class:`Thread` into the summary shape (KISS)."""
        return cls(
            id=thread.id,
            agent_name=thread.agent_name,
            source=thread.source,
            created_at=thread.created_at,
            updated_at=thread.updated_at,
        )
