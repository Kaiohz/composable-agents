"""Outbound port: TraceSummaryRepository (global trace list, spec §5.4).

Implementations build per-thread summaries with a bounded SQL aggregation
(TRC-002): the query count must remain constant as the page size changes and
independent of the number of summarized threads.
"""

from abc import ABC, abstractmethod
from datetime import datetime

from src.domain.entities.trace_summary import TraceSummary


class TraceSummaryRepository(ABC):
    """Outbound port for read-only global trace summaries."""

    @abstractmethod
    async def list_summaries(
        self,
        agent_name: str | None = None,
        limit: int = 50,
        cursor_activity_at: datetime | None = None,
        cursor_thread_id: str | None = None,
    ) -> tuple[list[TraceSummary], bool]:
        """Return one page of trace summaries ordered by latest activity.

        Args:
            agent_name: Optional EXACT (case-sensitive) agent filter.
            limit: Maximum rows to return (caller validates bounds).
            cursor_activity_at: Order-key timestamp of the previous page's
                last row (exclusive), or ``None`` for the first page.
            cursor_thread_id: Deterministic thread-ID tiebreaker of the
                previous page's last row.

        Returns:
            Tuple ``(summaries, has_more)`` where ``has_more`` indicates at
            least one further row exists after the returned page.
        """
        ...
