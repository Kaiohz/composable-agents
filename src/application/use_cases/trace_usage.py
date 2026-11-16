"""TraceUsageUseCase — LLM usage dashboard aggregations (dashboard spec §4-§5).

Two thin application entry points over the
:class:`~src.domain.ports.trace_usage_repository.TraceUsageRepository` port:

* :class:`GetTraceUsageUseCase` — aggregated tokens / cost / runs / latency for
  a window (agent + period filters validated upstream by FastAPI).
* :class:`GetTopUsageRunsUseCase` — the most expensive completed runs.

The use cases are pure orchestrators: the single-query SQL aggregation and the
per-user RLS isolation live behind the port.
"""

import logging
from datetime import datetime

from src.domain.entities.llm_usage_run import LlmUsageRun
from src.domain.entities.llm_usage_summary import LlmUsageSummary
from src.domain.ports.trace_usage_repository import TraceUsageRepository

logger = logging.getLogger(__name__)

# Top-N bounds mirror the trace-list conventions (TRC-001): a bounded default
# that leaves room for the dashboard table and a hard maximum page size.
DEFAULT_TOP_LIMIT = 10
MAX_TOP_LIMIT = 100


class GetTraceUsageUseCase:
    """Aggregate LLM usage for the authenticated user's window."""

    def __init__(self, trace_usage_repository: TraceUsageRepository) -> None:
        self._trace_usage_repository = trace_usage_repository

    async def execute(
        self,
        *,
        agent_name: str | None = None,
        from_ts: datetime | None = None,
        to_ts: datetime | None = None,
    ) -> LlmUsageSummary:
        """Return the usage summary with both series breakdowns.

        Args:
            agent_name: Optional EXACT (case-sensitive) agent filter.
            from_ts: Optional inclusive window lower bound.
            to_ts: Optional inclusive window upper bound.

        Returns:
            The :class:`LlmUsageSummary` for the authenticated user.
        """
        summary = await self._trace_usage_repository.summarize_usage(
            agent_name=agent_name, from_ts=from_ts, to_ts=to_ts
        )
        return summary


class GetTopUsageRunsUseCase:
    """Rank the most expensive completed runs of a window."""

    def __init__(self, trace_usage_repository: TraceUsageRepository) -> None:
        self._trace_usage_repository = trace_usage_repository

    async def execute(
        self,
        *,
        limit: int = DEFAULT_TOP_LIMIT,
        agent_name: str | None = None,
        from_ts: datetime | None = None,
        to_ts: datetime | None = None,
    ) -> list[LlmUsageRun]:
        """Return the top-N runs by estimated cost, cost descending.

        Args:
            limit: Page size — 1..100 (route-validated).
            agent_name: Optional EXACT (case-sensitive) agent filter.
            from_ts: Optional inclusive window lower bound.
            to_ts: Optional inclusive window upper bound.

        Returns:
            The ranked :class:`LlmUsageRun` list (error runs never included).
        """
        return await self._trace_usage_repository.top_runs(
            limit=limit, agent_name=agent_name, from_ts=from_ts, to_ts=to_ts
        )
