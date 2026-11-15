"""PostgreSQL adapter for global trace summaries (spec §5.4 TRC-001/TRC-002).

A single SQL statement aggregates, per thread: the latest durable local
trace-event timestamp, the total event count and the count of distinct
non-null turn identifiers. Threads with zero events surface zero counts and a
null last-event timestamp via the LEFT JOIN. The query count is constant
regardless of the page size or row count (TRC-002) — no per-thread history
fetching.

Authorization is two-layered (SEC-001, BUG-001):

* the RLS policies on ``threads``/``trace_events`` filter rows through the
  ``app.user_id`` GUC set from the request context;
* the query ALSO carries an explicit application-level ownership predicate
  (``threads.user_id == current_user_id``) — identical to every other
  user-scoped adapter — so isolation never depends on the database role.
  A deployment whose application role owns the tables or carries
  ``BYPASSRLS``/superuser bypasses RLS unconditionally; without the predicate
  the trace list would leak other tenants' threads (observed live).
  Without an authenticated principal the adapter returns a bounded empty page
  instead of an unscoped scan (non-enumeration by construction).
"""

import logging

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from src.domain.entities.trace_summary import TraceSummary
from src.domain.errors.messages import ErrorMessage
from src.domain.errors.storage import StorageError
from src.domain.logging.messages import LogMessage
from src.domain.ports.trace_summary_repository import TraceSummaryRepository
from src.infrastructure.database.models.thread import ThreadModel
from src.infrastructure.database.models.trace_event import TraceEventModel
from src.infrastructure.database.rls_context import current_user_id

logger = logging.getLogger(__name__)


class PostgresTraceSummaryRepository(TraceSummaryRepository):
    """Adapter computing bounded per-thread trace summaries in PostgreSQL."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def list_summaries(
        self,
        agent_name: str | None = None,
        limit: int = 50,
        cursor_activity_at=None,
        cursor_thread_id: str | None = None,
    ) -> tuple[list[TraceSummary], bool]:
        """One page of summaries, newest activity first (TRC-001).

        Ordering key: ``COALESCE(last_event_at, started_at) DESC, thread_id
        DESC`` — deterministic through the thread-ID tiebreaker.
        """
        uid = current_user_id.get()
        if uid is None:
            # Defense-in-depth (BUG-001): with no authenticated principal there
            # is no ownership scope to honour. Rather than fall back to an
            # unscoped query (which RLS alone may not contain — superuser/
            # BYPASSRLS/table-owner roles bypass it), return a bounded empty
            # page: non-enumerating by construction.
            logger.warning(LogMessage.TRACE_SUMMARIES_UNSCOPED)
            return [], False

        try:
            aggregates = (
                select(
                    ThreadModel.id.label("thread_id"),
                    ThreadModel.agent_name.label("agent_name"),
                    ThreadModel.source.label("source"),
                    ThreadModel.created_at.label("started_at"),
                    func.max(TraceEventModel.timestamp).label("last_event_at"),
                    func.count(TraceEventModel.id).label("event_count"),
                    func.count(func.distinct(TraceEventModel.turn_id)).label("turn_count"),
                )
                .outerjoin(TraceEventModel, TraceEventModel.thread_id == ThreadModel.id)
                # Explicit ownership predicate (BUG-001): isolation must hold even
                # when the database role bypasses RLS (owner / superuser /
                # BYPASSRLS). Same defense-in-depth pattern as every other
                # user-scoped adapter in the codebase.
                .where(ThreadModel.user_id == uid)
                .group_by(ThreadModel.id, ThreadModel.agent_name, ThreadModel.source, ThreadModel.created_at)
                .subquery()
            )

            activity = func.coalesce(aggregates.c.last_event_at, aggregates.c.started_at)
            stmt = select(
                aggregates.c.thread_id,
                aggregates.c.agent_name,
                aggregates.c.source,
                aggregates.c.started_at,
                aggregates.c.last_event_at,
                aggregates.c.event_count,
                aggregates.c.turn_count,
                activity.label("activity"),
            ).order_by(activity.desc(), aggregates.c.thread_id.desc())

            if agent_name is not None:
                stmt = stmt.where(aggregates.c.agent_name == agent_name)
            if cursor_activity_at is not None and cursor_thread_id is not None:
                # Keyed pagination on (activity, thread_id), exclusive: strictly
                # before the cursor, or tied on activity with a lower thread id.
                stmt = stmt.where(
                    (activity < cursor_activity_at)
                    | ((activity == cursor_activity_at) & (aggregates.c.thread_id < cursor_thread_id))
                )

            stmt = stmt.limit(limit + 1)

            async with self._engine.connect() as conn:
                rows = (await conn.execute(stmt)).all()

            has_more = len(rows) > limit
            rows = rows[:limit]
            summaries = [
                TraceSummary(
                    thread_id=r.thread_id,
                    agent_name=r.agent_name,
                    source=r.source,
                    started_at=r.started_at,
                    last_event_at=r.last_event_at,
                    event_count=r.event_count,
                    turn_count=r.turn_count,
                )
                for r in rows
            ]
            logger.info(LogMessage.TRACE_SUMMARIES_LISTED, len(summaries), agent_name, cursor_thread_id)
            return summaries, has_more
        except SQLAlchemyError as e:
            raise StorageError(ErrorMessage.TRACE_FAILED_LIST.format(thread_id="*", error=e)) from e
