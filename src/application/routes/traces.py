"""Global trace list HTTP route (spec §5.4 TRC-001).

``GET /api/v1/traces?agent_name=<exact>&limit=<n>&cursor=<cursor>`` — read-only
aggregate summaries built with a bounded SQL query (TRC-002). Trace detail
reuses the existing ``GET /api/v1/threads/{thread_id}/history`` endpoint.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from src.application.responses.traces import TraceListResponse
from src.application.use_cases.list_traces import DEFAULT_LIMIT, MAX_LIMIT, ListTracesUseCase
from src.dependencies import get_list_traces_use_case
from src.domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/traces", tags=["traces"])


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
