"""Outbound port: TraceUsageRepository (LLM usage dashboard, spec §4).

Implementations compute the LLM-usage aggregations for the authenticated user
with a single SQL query per endpoint — one statement per aggregation across
``trace_events``, ``threads`` and the global ``llm_prices`` reference table,
never a per-run/per-model fetch loop.

Per-user isolation is enforced two-layered (SEC-001 / BUG-001): the RLS
policies filter rows through the ``app.user_id`` GUC AND the query carries an
explicit application-level ownership predicate, so isolation never depends on
the database role. One run = one turn; runs are classified per-turn.
"""

from abc import ABC, abstractmethod
from datetime import datetime

from src.domain.entities.llm_usage_run import LlmUsageRun
from src.domain.entities.llm_usage_summary import LlmUsageSummary


class TraceUsageRepository(ABC):
    """Outbound port for read-only LLM usage aggregations."""

    @abstractmethod
    async def summarize_usage(
        self,
        *,
        agent_name: str | None = None,
        from_ts: datetime | None = None,
        to_ts: datetime | None = None,
    ) -> LlmUsageSummary:
        """Aggregate tokens / cost / runs / latency for the user's window.

        Args:
            agent_name: Optional EXACT (case-sensitive) agent filter.
            from_ts: Inclusive lower bound on the runs' start timestamp.
            to_ts: Inclusive upper bound on the runs' start timestamp.

        Returns:
            The :class:`LlmUsageSummary` with both series breakdowns (by agent
            and by model).
        """
        ...

    @abstractmethod
    async def top_runs(
        self,
        *,
        limit: int,
        agent_name: str | None = None,
        from_ts: datetime | None = None,
        to_ts: datetime | None = None,
    ) -> list[LlmUsageRun]:
        """Return the most expensive completed runs of the window.

        Args:
            limit: Maximum number of runs to return (caller validates bounds).
            agent_name: Optional EXACT (case-sensitive) agent filter.
            from_ts: Inclusive lower bound on the runs' start timestamp.
            to_ts: Inclusive upper bound on the runs' start timestamp.

        Returns:
            Runs sorted by estimated cost descending, capped to ``limit``.
            Error runs (no terminal AI_MESSAGE) are never included.
        """
        ...
