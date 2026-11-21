"""Per-user override of the virtual built-in Composable AI Assistant."""

from datetime import datetime

from pydantic import BaseModel


class BuiltinAgentOverride(BaseModel):
    """Snapshot of one user's built-in assistant customization.

    The override is keyed by ``(user_id, agent_name)`` and NEVER represents a
    custom agent row/object: the default definition stays in code and the
    override only shadows it for its owning user.
    """

    user_id: str
    agent_name: str
    yaml: str
    created_at: datetime
    updated_at: datetime
