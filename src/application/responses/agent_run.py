"""One-shot agent execution result DTO (REQ-024).

The MCP ``composable_agents_run`` tool (and the agent-facing
``composable_agents_run`` platform tool) return this explicit, JSON-safe
projection: the trusted ``mcp`` thread identity, the executed agent, the final
Message projection and the persisted turn trace.

Additive fields beyond REQ-024's minimum:

* ``trace`` — the bounded trace of the executed turn (exactly one turn for a
  one-shot run); lets MCP clients diagnose what the turn actually did.
* ``degraded`` — REQ-040 degraded-capability information from the runner
  build (unavailable optional MCP servers), never a silent capability loss.
"""

from pydantic import BaseModel

from src.domain.entities.message import Message
from src.domain.entities.thread import ThreadSource
from src.domain.entities.trace_event import TraceEvent


class DegradedCapability(BaseModel):
    """One failed optional capability source (REQ-040)."""

    name: str
    reason: str


class RunAgentOneShotResult(BaseModel):
    """JSON-safe result of one trusted one-shot agent execution."""

    thread_id: str
    agent_name: str
    source: ThreadSource
    message: Message
    trace: list[TraceEvent] = []
    degraded: list[DegradedCapability] = []
