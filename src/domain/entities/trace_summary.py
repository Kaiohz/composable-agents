"""Global trace summary domain entity (spec §5.4 TRC-001).

One summary row per thread: the thread identity and origin, a start timestamp,
and aggregate facts computed with a single bounded SQL query over
threads/trace_events (TRC-002) — never by fetching each thread's history.
"""

from datetime import datetime

from pydantic import BaseModel


class TraceSummary(BaseModel):
    """Aggregate summary of one thread for the global trace list."""

    thread_id: str
    agent_name: str
    source: str
    started_at: datetime
    last_event_at: datetime | None = None
    event_count: int = 0
    turn_count: int = 0
