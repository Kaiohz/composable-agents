"""Global trace list HTTP route (spec §5.4 TRC-001).

``GET /api/v1/traces?agent_name=<exact>&limit=<n>&cursor=<cursor>`` — read-only
aggregate summaries built with a bounded SQL query (TRC-002). Trace detail
reuses the existing ``GET /api/v1/threads/{thread_id}/history`` endpoint.

The LLM usage dashboard endpoints (spec §4-§5) live on the same router:
``GET /api/v1/traces/usage`` (aggregated tokens/cost/runs/latency) and
``GET /api/v1/traces/usage/top`` (top-N most expensive runs).
"""

import logging
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from src.application.responses.trace_usage import TopRunsResponse, TraceUsageResponse, UsageSeries
from src.application.responses.traces import TraceListResponse
from src.application.use_cases.list_traces import DEFAULT_LIMIT, MAX_LIMIT, ListTracesUseCase
from src.application.use_cases.trace_usage import (
    DEFAULT_TOP_LIMIT,
    MAX_TOP_LIMIT,
    GetTopUsageRunsUseCase,
    GetTraceUsageUseCase,
)
from src.dependencies import (
    get_list_traces_use_case,
    get_top_usage_runs_use_case,
    get_trace_usage_use_case,
)
from src.domain.entities.llm_usage_summary import LlmUsageSummary
from src.domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/traces", tags=["traces"])


def _utc_bound(value: datetime | None) -> datetime | None:
    """Normalize a window bound to UTC (the DB stores ``timestamptz``).

    FastAPI parses the ``from``/``to`` ISO parameters into ``datetime``; a
    naive value (no offset) bound against a ``timestamptz`` column would be
    interpreted in the DATABASE session timezone — silently shifting the
    window by the server's UTC offset. Naive bounds are therefore treated as
    UTC at the boundary (ISO datetimes without a ``Z``/offset mean UTC).

    Args:
        value: The parsed bound (or ``None``).

    Returns:
        The UTC-normalized bound (or ``None`` when unset).
    """
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


@router.get("")
async def list_traces(
    use_case: Annotated[ListTracesUseCase, Depends(get_list_traces_use_case)],
    agent_name: Annotated[str | None, Query(min_length=1, max_length=100)] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
    cursor: Annotated[str | None, Query()] = None,
) -> TraceListResponse:
    """Return the global trace summary list for the authenticated user.

    Args:
        use_case: :class:`ListTracesUseCase` wired at startup.
        agent_name: Optional EXACT (case-sensitive, never partial) agent filter.
        limit: Page size — default 50, maximum 100 (TRC-001).
        cursor: Opaque pagination cursor from the previous page.

    Returns:
        ``{"items": [...], "next_cursor": ...}`` sorted by latest activity
        descending with a deterministic thread-ID tiebreaker.
    """
    from src.application.use_cases.list_traces import TracePage

    page: TracePage = await use_case.execute(agent_name=agent_name, limit=limit, cursor=cursor)
    logger.info(LogMessage.TRACE_SUMMARIES_LISTED, len(page.items), agent_name, bool(cursor))
    return TraceListResponse(items=page.items, next_cursor=page.next_cursor)


def _usage_series(summary: LlmUsageSummary, group_by: Literal["agent", "model"] | None) -> UsageSeries:
    """Build the ``series`` DTO for the requested group-by narrowing.

    Args:
        summary: The aggregated usage summary (both breakdowns populated).
        group_by: ``agent`` / ``model`` narrowing, or ``None`` for both.

    Returns:
        A :class:`UsageSeries` carrying only the requested breakdown; the
        unrequested one stays ``None`` (dropped from the serialized payload).
    """
    match group_by:
        case "agent":
            return UsageSeries(by_agent=summary.series_by_agent, by_model=None)
        case "model":
            return UsageSeries(by_agent=None, by_model=summary.series_by_model)
        case _:
            return UsageSeries(by_agent=summary.series_by_agent, by_model=summary.series_by_model)


@router.get("/usage", response_model_exclude_none=True)
async def get_trace_usage(
    use_case: Annotated[GetTraceUsageUseCase, Depends(get_trace_usage_use_case)],
    agent_name: Annotated[str | None, Query(min_length=1, max_length=100)] = None,
    from_ts: Annotated[datetime | None, Query(alias="from")] = None,
    to_ts: Annotated[datetime | None, Query(alias="to")] = None,
    group_by: Annotated[Literal["agent", "model"] | None, Query()] = None,
) -> TraceUsageResponse:
    """Aggregate LLM usage (tokens, cost, runs, latency) for the current user.

    Args:
        use_case: :class:`GetTraceUsageUseCase` wired at startup.
        agent_name: Optional EXACT (case-sensitive, never partial) agent filter
            (TRC-001 semantics).
        from_ts: Optional inclusive lower bound on the runs' start timestamp
            (parsed from the ``from`` ISO query parameter; malformed bounds are
            rejected as client errors; a naive bound is treated as UTC).
        to_ts: Optional inclusive upper bound (``to``).
        group_by: Optional series narrowing — ``agent`` or ``model``; when
            absent both breakdowns are returned.

    Returns:
        Totals, run/error counts, duration percentiles and the token/cost
        series breakdown.
    """
    summary = await use_case.execute(agent_name=agent_name, from_ts=_utc_bound(from_ts), to_ts=_utc_bound(to_ts))
    logger.info(LogMessage.TRACE_USAGE_LISTED, agent_name, summary.run_count, summary.error_count)
    return TraceUsageResponse(
        total_input_tokens=summary.total_input_tokens,
        total_output_tokens=summary.total_output_tokens,
        total_tokens=summary.total_tokens,
        total_cost_usd=summary.total_cost_usd,
        run_count=summary.run_count,
        error_count=summary.error_count,
        duration_ms_p50=summary.duration_ms_p50,
        duration_ms_p95=summary.duration_ms_p95,
        series=_usage_series(summary, group_by),
    )


@router.get("/usage/top", response_model_exclude_none=True)
async def get_top_usage_runs(
    use_case: Annotated[GetTopUsageRunsUseCase, Depends(get_top_usage_runs_use_case)],
    limit: Annotated[int, Query(ge=1, le=MAX_TOP_LIMIT)] = DEFAULT_TOP_LIMIT,
    from_ts: Annotated[datetime | None, Query(alias="from")] = None,
    to_ts: Annotated[datetime | None, Query(alias="to")] = None,
    agent_name: Annotated[str | None, Query(min_length=1, max_length=100)] = None,
) -> TopRunsResponse:
    """Return the top-N most expensive completed runs for the current user.

    Args:
        use_case: :class:`GetTopUsageRunsUseCase` wired at startup.
        limit: Page size — 1..100, default 10.
        from_ts: Optional inclusive lower bound on the runs' start timestamp
            (a naive bound is treated as UTC).
        to_ts: Optional inclusive upper bound.
        agent_name: Optional EXACT (case-sensitive) agent filter.

    Returns:
        ``{"items": [...]}`` — runs sorted by estimated cost descending; error
        runs (no terminal AI_MESSAGE) never appear.
    """
    runs = await use_case.execute(
        limit=limit, agent_name=agent_name, from_ts=_utc_bound(from_ts), to_ts=_utc_bound(to_ts)
    )
    logger.info(LogMessage.TRACE_USAGE_TOP_LISTED, limit, len(runs))
    return TopRunsResponse(items=runs)
