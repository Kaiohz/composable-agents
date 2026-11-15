"""ListTracesUseCase — global, agent-filterable trace summaries (spec §5.4)."""

import base64
import json
import logging
from dataclasses import dataclass
from datetime import datetime

from src.domain.entities.trace_summary import TraceSummary
from src.domain.errors.config import ConfigError
from src.domain.errors.messages import ErrorMessage
from src.domain.ports.trace_summary_repository import TraceSummaryRepository

logger = logging.getLogger(__name__)

MAX_LIMIT = 100
DEFAULT_LIMIT = 50
_CURSOR_VERSION = "v1"


@dataclass
class TracePage:
    """One page of the global trace list with its pagination cursor."""

    items: list[TraceSummary]
    next_cursor: str | None


def encode_cursor(activity_at: datetime, thread_id: str) -> str:
    """Encode the (activity, thread_id) order key as an opaque cursor."""
    payload = {
        "v": _CURSOR_VERSION,
        "activity": activity_at.isoformat(),
        "thread_id": thread_id,
    }
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def decode_cursor(cursor: str) -> tuple[datetime, str]:
    """Decode an opaque cursor back into its (activity, thread_id) order key.

    Every malformed shape — bad base64, invalid JSON, a non-object payload, a
    missing/non-string key or a non-ISO / timezone-naive timestamp — raises the
    centralized invalid-cursor domain error (mapped to HTTP 400), never an
    unhandled exception (no 500s on client-supplied query parameters).

    Raises:
        ConfigError: On any malformed cursor.
    """
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
    except ValueError as exc:
        # ``binascii.Error`` and ``UnicodeDecodeError`` both derive from
        # ``ValueError``; catching the base class covers every decode failure
        # (Sonar S5713: no redundant exception classes).
        raise ConfigError(ErrorMessage.TRACE_CURSOR_INVALID) from exc
    if not isinstance(payload, dict):
        raise ConfigError(ErrorMessage.TRACE_CURSOR_INVALID)
    if payload.get("v") != _CURSOR_VERSION or "activity" not in payload or "thread_id" not in payload:
        raise ConfigError(ErrorMessage.TRACE_CURSOR_INVALID)
    activity = payload["activity"]
    thread_id = payload["thread_id"]
    if not isinstance(activity, str) or not isinstance(thread_id, str):
        raise ConfigError(ErrorMessage.TRACE_CURSOR_INVALID)
    try:
        parsed = datetime.fromisoformat(activity)
    except (TypeError, ValueError) as exc:
        raise ConfigError(ErrorMessage.TRACE_CURSOR_INVALID) from exc
    if parsed.tzinfo is None:
        # Comparisons mix aware and naive datetimes otherwise (TypeError deep
        # in the repository) — a cursor without an offset is malformed.
        raise ConfigError(ErrorMessage.TRACE_CURSOR_INVALID)
    return parsed, thread_id


class ListTracesUseCase:
    """Build one page of global trace summaries (bounded aggregate query)."""

    def __init__(self, trace_summary_repository: TraceSummaryRepository) -> None:
        self._trace_summary_repository = trace_summary_repository

    async def execute(
        self,
        agent_name: str | None = None,
        limit: int = DEFAULT_LIMIT,
        cursor: str | None = None,
    ) -> TracePage:
        """Return summaries for the authenticated user's threads.

        Args:
            agent_name: Optional EXACT (case-sensitive) agent-name filter.
            limit: Page size (1..100).

        Returns:
            A :class:`TracePage`; ``next_cursor`` is ``None`` on the last page.
        """
        cursor_activity_at: datetime | None
        cursor_thread_id: str | None
        if cursor is not None:
            cursor_activity_at, cursor_thread_id = decode_cursor(cursor)
        else:
            cursor_activity_at, cursor_thread_id = None, None
        items, has_more = await self._trace_summary_repository.list_summaries(
            agent_name=agent_name,
            limit=limit,
            cursor_activity_at=cursor_activity_at,
            cursor_thread_id=cursor_thread_id,
        )
        next_cursor = (
            encode_cursor(items[-1].last_event_at or items[-1].started_at, items[-1].thread_id)
            if has_more and items
            else None
        )
        return TracePage(items=items, next_cursor=next_cursor)
