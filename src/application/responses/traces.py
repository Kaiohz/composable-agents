"""Response DTOs for the global trace list (spec §5.4 TRC-001).

The wrapper exists purely to carry the pagination cursor; row payloads are
the domain :class:`TraceSummary` entities serialized verbatim with exactly the
documented field set.
"""

from pydantic import BaseModel

from src.domain.entities.trace_summary import TraceSummary


class TraceListResponse(BaseModel):
    """One page of the global trace list."""

    items: list[TraceSummary]
    next_cursor: str | None = None
