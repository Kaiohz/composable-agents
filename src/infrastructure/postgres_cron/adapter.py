"""PostgreSQL adapters for the cron subsystem (spec §5.3).

``PostgresCronRepository`` persists per-user cron definitions. The scheduler
scans due crons in a system (RLS-bypass) context; per-cron mutations run under
the cron owner's context (SEC-001). ``advance_next_run`` only ever moves the
frontier forward so late in-flight executions never regress the schedule.

``PostgresCronRunRepository`` persists durable run records. Scheduled
occurrence claiming (CRN-006) is a single conditional INSERT against the
``uq_cron_runs_occurrence_claim`` partial unique index — competing workers
never both win, with no in-memory locking involved.
"""

import logging
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import Insert, Update, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.domain.entities.cron import Cron
from src.domain.entities.cron_run import CronRun, CronRunStatus, CronTrigger
from src.domain.errors.cron import CronNotFoundError
from src.domain.errors.messages import ErrorMessage
from src.domain.errors.storage import StorageError
from src.domain.logging.messages import LogMessage
from src.domain.ports.cron_repository import CronRepository, CronRunRepository
from src.infrastructure.database.models.cron import CronModel, CronRunModel
from src.infrastructure.database.rls_context import current_user_id


async def _execute_dml(session: AsyncSession, stmt: Update | Insert) -> CursorResult[Any]:
    """Execute a DML statement, typing the cursor precisely for ``rowcount``.

    The typed ``session.execute`` stub widens core DML cursors to the generic
    ``Result[Any]`` which lacks ``rowcount``; the runtime object is always a
    ``CursorResult`` (Sonar S1192 previously flagged four duplicated cast-literal
    strings — this helper is the single home of that cast).
    """
    return cast("CursorResult[Any]", await session.execute(stmt))


logger = logging.getLogger(__name__)


def _to_cron(model: CronModel) -> Cron:
    """Direct transformation: ORM row → domain entity."""
    return Cron(
        id=str(model.id),
        name=model.name,
        agent_name=model.agent_name,
        message=model.message,
        cron_expression=model.cron_expression,
        timezone=model.timezone,
        enabled=model.enabled,
        next_run_at=model.next_run_at,
        last_run_status=model.last_run_status,
        last_run_at=model.last_run_at,
        created_at=model.created_at,
        updated_at=model.updated_at,
        user_id=model.user_id,
    )


def _to_run(model: CronRunModel) -> CronRun:
    """Direct transformation: run row → domain entity."""
    return CronRun(
        id=str(model.id),
        cron_id=str(model.cron_id),
        trigger=CronTrigger(model.trigger),
        status=CronRunStatus(model.status),
        occurrence_at=model.occurrence_at,
        thread_id=model.thread_id,
        started_at=model.started_at,
        finished_at=model.finished_at,
        error=model.error,
        created_at=model.created_at,
        user_id=model.user_id,
    )


class PostgresCronRepository(CronRepository):
    """Adapter persisting per-user cron definitions in PostgreSQL."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @staticmethod
    def _current_user_id() -> str | None:
        return current_user_id.get()

    async def create(self, cron: Cron, next_run_at: datetime | None) -> Cron:
        uid = self._current_user_id() or ""
        now = datetime.now(UTC)
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                row = CronModel(
                    id=UUID(cron.id),
                    name=cron.name,
                    agent_name=cron.agent_name,
                    message=cron.message,
                    cron_expression=cron.cron_expression,
                    timezone=cron.timezone,
                    enabled=cron.enabled,
                    next_run_at=next_run_at,
                    created_at=now,
                    updated_at=now,
                    user_id=uid,
                )
                session.add(row)
                await session.commit()
                logger.info(LogMessage.CRON_CREATED, row.id, next_run_at)
                return _to_cron(row)
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.CRON_FAILED_CREATE.format(error=e)) from e

    async def get(self, cron_id: str) -> Cron:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                row = await self._get_owned(session, cron_id)
                if row is None:
                    raise CronNotFoundError(ErrorMessage.CRON_NOT_FOUND.format(cron_id=cron_id))
                logger.info(LogMessage.CRON_FETCHED, cron_id)
                return _to_cron(row)
            except CronNotFoundError:
                raise
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.CRON_FAILED_GET.format(cron_id=cron_id, error=e)) from e

    async def list_all(self) -> list[Cron]:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                stmt = select(CronModel).order_by(CronModel.created_at.desc())
                uid = self._current_user_id()
                if uid is not None:
                    stmt = stmt.where(CronModel.user_id == uid)
                rows = (await session.execute(stmt)).scalars().all()
                return [_to_cron(m) for m in rows]
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.CRON_FAILED_LIST.format(error=e)) from e

    async def list_due(self, now: datetime) -> list[Cron]:
        """Global scheduler scan — expects a system (RLS-bypass) context."""
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                rows = (
                    (
                        await session.execute(
                            select(CronModel).where(
                                CronModel.enabled.is_(True),
                                CronModel.next_run_at.is_not(None),
                                CronModel.next_run_at <= now,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                return [_to_cron(m) for m in rows]
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.CRON_FAILED_LIST.format(error=e)) from e

    async def update(self, cron: Cron, next_run_at: datetime | None) -> Cron:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                row = await self._get_owned(session, cron.id)
                if row is None:
                    raise CronNotFoundError(ErrorMessage.CRON_NOT_FOUND.format(cron_id=cron.id))
                row.name = cron.name
                row.agent_name = cron.agent_name
                row.message = cron.message
                row.cron_expression = cron.cron_expression
                row.timezone = cron.timezone
                row.enabled = cron.enabled
                row.next_run_at = next_run_at
                row.updated_at = datetime.now(UTC)
                await session.commit()
                logger.info(LogMessage.CRON_UPDATED, row.id, next_run_at)
                return _to_cron(row)
            except CronNotFoundError:
                raise
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.CRON_FAILED_UPDATE.format(cron_id=cron.id, error=e)) from e

    async def delete(self, cron_id: str) -> None:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                row = await self._get_owned(session, cron_id)
                if row is None:
                    raise CronNotFoundError(ErrorMessage.CRON_NOT_FOUND.format(cron_id=cron_id))
                await session.delete(row)
                await session.commit()
                logger.info(LogMessage.CRON_DELETED, cron_id)
            except CronNotFoundError:
                raise
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.CRON_FAILED_DELETE.format(cron_id=cron_id, error=e)) from e

    async def advance_next_run(self, cron_id: str, next_run_at: datetime) -> None:
        """Guarded forward-only frontier move (never regresses / never touches NULL)."""
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                await session.execute(
                    update(CronModel)
                    .where(
                        CronModel.id == UUID(cron_id),
                        CronModel.enabled.is_(True),
                        CronModel.next_run_at.is_not(None),
                        CronModel.next_run_at < next_run_at,
                    )
                    .values(next_run_at=next_run_at)
                )
                await session.commit()
                logger.info(LogMessage.CRON_RUN_FRONTIER_ADVANCED, cron_id, next_run_at)
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.CRON_FAILED_UPDATE.format(cron_id=cron_id, error=e)) from e

    async def set_last_run(self, cron_id: str, status: str, finished_at: datetime) -> None:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                await session.execute(
                    update(CronModel)
                    .where(CronModel.id == UUID(cron_id))
                    .values(last_run_status=status, last_run_at=finished_at)
                )
                await session.commit()
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.CRON_FAILED_UPDATE.format(cron_id=cron_id, error=e)) from e

    async def _get_owned(self, session: AsyncSession, cron_id: str) -> CronModel | None:
        stmt = select(CronModel).where(CronModel.id == UUID(cron_id))
        uid = self._current_user_id()
        if uid is not None:
            stmt = stmt.where(CronModel.user_id == uid)
        return (await session.execute(stmt)).scalar_one_or_none()


class PostgresCronRunRepository(CronRunRepository):
    """Adapter persisting durable cron run records with atomic claims."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @staticmethod
    def _current_user_id() -> str | None:
        return current_user_id.get()

    async def claim_scheduled(self, run: CronRun, advance_next_run_to: datetime | None = None) -> CronRun | None:
        """Conditional INSERT claiming a scheduled occurrence (CRN-006).

        The INSERT uses an arbiter-less ``ON CONFLICT DO NOTHING`` because two
        unique indexes can reject it: the occurrence-claim index (another
        worker claimed/holds this exact occurrence) and ``uq_cron_runs_single_active``
        (another run owns the cron's single active slot). The caller decides
        what a lost claim means from the post-claim active-run state.

        When ``advance_next_run_to`` is provided, the guarded forward-only
        frontier advance is part of the SAME transaction as the claim, closing
        the stale-snapshot window where a concurrent tick could observe the
        just-claimed occurrence as still-due.
        """
        uid = self._current_user_id() or ""
        now = run.created_at if run.created_at.tzinfo else datetime.now(UTC)
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                stmt = pg_insert(CronRunModel).values(
                    id=UUID(run.id),
                    cron_id=UUID(run.cron_id),
                    trigger=CronTrigger.SCHEDULED.value,
                    status=CronRunStatus.QUEUED.value,
                    occurrence_at=run.occurrence_at,
                    created_at=now,
                    user_id=uid,
                )
                stmt = stmt.on_conflict_do_nothing()
                result = await _execute_dml(session, stmt)
                if advance_next_run_to is not None:
                    await session.execute(
                        update(CronModel)
                        .where(
                            CronModel.id == UUID(run.cron_id),
                            CronModel.enabled.is_(True),
                            CronModel.next_run_at.is_not(None),
                            CronModel.next_run_at < advance_next_run_to,
                        )
                        .values(next_run_at=advance_next_run_to)
                    )
                await session.commit()
                if result.rowcount == 0:
                    logger.info(LogMessage.CRON_TICK_CLAIM_CONFLICT, run.cron_id, run.occurrence_at)
                    return None
                logger.info(LogMessage.CRON_TICK_CLAIMED, run.cron_id, run.occurrence_at)
                return CronRun(
                    id=str(run.id),
                    cron_id=run.cron_id,
                    trigger=CronTrigger.SCHEDULED,
                    status=CronRunStatus.QUEUED,
                    occurrence_at=run.occurrence_at,
                    thread_id=None,
                    started_at=None,
                    finished_at=None,
                    error=None,
                    created_at=now,
                    user_id=uid,
                )
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.CRON_RUN_NOT_CREATED.format(cron_id=run.cron_id, error=e)) from e

    async def create_manual(self, run: CronRun) -> CronRun | None:
        """Insert a manual run, or return ``None`` when another run is active.

        The conditional INSERT targets ``uq_cron_runs_single_active``: exactly
        one caller wins the cron's single active slot — including across
        concurrent Run now requests — every loser receives ``None`` (CRN-005).
        """
        uid = self._current_user_id() or ""
        now = datetime.now(UTC)
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                stmt = pg_insert(CronRunModel).values(
                    id=UUID(run.id),
                    cron_id=UUID(run.cron_id),
                    trigger=CronTrigger.MANUAL.value,
                    status=CronRunStatus.QUEUED.value,
                    occurrence_at=None,
                    created_at=now,
                    user_id=uid,
                )
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=["cron_id"],
                    index_where=text("status IN ('queued', 'running')"),
                )
                result = await _execute_dml(session, stmt)
                await session.commit()
                if result.rowcount == 0:
                    logger.info(LogMessage.CRON_TICK_CLAIM_CONFLICT, run.cron_id, None)
                    return None
                logger.info(LogMessage.CRON_RUN_NOW, run.cron_id)
                return CronRun(
                    id=str(run.id),
                    cron_id=run.cron_id,
                    trigger=CronTrigger.MANUAL,
                    status=CronRunStatus.QUEUED,
                    occurrence_at=None,
                    thread_id=None,
                    started_at=None,
                    finished_at=None,
                    error=None,
                    created_at=now,
                    user_id=uid,
                )
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.CRON_RUN_NOT_CREATED.format(cron_id=run.cron_id, error=e)) from e

    async def record_skipped(self, run: CronRun) -> bool:
        """Insert a durable skipped record; idempotent per (cron, occurrence)."""
        uid = self._current_user_id() or ""
        now = datetime.now(UTC)
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                stmt = pg_insert(CronRunModel).values(
                    id=UUID(run.id),
                    cron_id=UUID(run.cron_id),
                    trigger=CronTrigger.SCHEDULED.value,
                    status=CronRunStatus.SKIPPED.value,
                    occurrence_at=run.occurrence_at,
                    finished_at=now,
                    created_at=now,
                    user_id=uid,
                )
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=["cron_id", "occurrence_at"],
                    index_where=text("status = 'skipped'"),
                )
                result = await _execute_dml(session, stmt)
                await session.commit()
                return bool(result.rowcount)
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.CRON_RUN_NOT_CREATED.format(cron_id=run.cron_id, error=e)) from e

    async def reclaim_stale_active_runs(self, older_than: datetime) -> int:
        """Mark lease-expired active runs as failed (SCHED-001, C2).

        A killed worker leaves a ``queued``/``running`` row that would hold the
        cron's single active slot forever. Rows created before ``older_than``
        that never reached ``started_at`` — or whose ``started_at`` also
        predates ``older_than`` — are marked ``failed`` with a safe error
        summary, freeing the slot for the next occurrence. Runs younger than
        the lease are untouched, so a genuinely active execution is never
        reclaimed mid-flight.
        """
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                stmt = (
                    update(CronRunModel)
                    .where(
                        CronRunModel.status.in_([CronRunStatus.QUEUED.value, CronRunStatus.RUNNING.value]),
                        CronRunModel.created_at < older_than,
                        or_(
                            CronRunModel.started_at.is_(None),
                            CronRunModel.started_at < older_than,
                        ),
                    )
                    .values(
                        status=CronRunStatus.FAILED.value,
                        finished_at=datetime.now(UTC),
                        error=ErrorMessage.CRON_RUN_STALE_RECLAIMED,
                    )
                )
                result = await _execute_dml(session, stmt)
                await session.commit()
                return result.rowcount or 0
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.CRON_RUN_NOT_UPDATED.format(run_id="*", error=e)) from e

    async def get_active_run(self, cron_id: str, exclude_run_id: str | None = None) -> CronRun | None:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                stmt = select(CronRunModel).where(
                    CronRunModel.cron_id == UUID(cron_id),
                    CronRunModel.status.in_([CronRunStatus.QUEUED.value, CronRunStatus.RUNNING.value]),
                )
                if exclude_run_id is not None:
                    stmt = stmt.where(CronRunModel.id != UUID(exclude_run_id))
                stmt = stmt.order_by(CronRunModel.created_at.desc()).limit(1)
                row = (await session.execute(stmt)).scalar_one_or_none()
                return _to_run(row) if row is not None else None
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.CRON_RUN_LIST_FAILED.format(cron_id=cron_id, error=e)) from e

    async def mark_running(self, run_id: str, thread_id: str, started_at: datetime) -> None:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                await session.execute(
                    update(CronRunModel)
                    .where(CronRunModel.id == UUID(run_id))
                    .values(status=CronRunStatus.RUNNING.value, thread_id=thread_id, started_at=started_at)
                )
                await session.commit()
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.CRON_RUN_NOT_UPDATED.format(run_id=run_id, error=e)) from e

    async def mark_finished(
        self,
        run_id: str,
        status: CronRunStatus,
        finished_at: datetime,
        error: str | None = None,
    ) -> None:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                await session.execute(
                    update(CronRunModel)
                    .where(CronRunModel.id == UUID(run_id))
                    .values(
                        status=status.value,
                        finished_at=finished_at,
                        error=error,
                    )
                )
                await session.commit()
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.CRON_RUN_NOT_UPDATED.format(run_id=run_id, error=e)) from e

    async def mark_finished_and_advance(
        self,
        run_id: str,
        status: CronRunStatus,
        finished_at: datetime,
        error: str | None,
        cron_id: str,
        next_run_at: datetime,
    ) -> None:
        """Terminal state + frontier coalescing in ONE transaction (CRN-005).

        Closing the window where a competing tick could claim an elapsed
        (missed) occurrence between the run's terminal update and the frontier
        jump guarantees at most one coalesced execution per downtime span.
        """
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                await session.execute(
                    update(CronRunModel)
                    .where(CronRunModel.id == UUID(run_id))
                    .values(status=status.value, finished_at=finished_at, error=error)
                )
                await session.execute(
                    update(CronModel)
                    .where(
                        CronModel.id == UUID(cron_id),
                        CronModel.enabled.is_(True),
                        CronModel.next_run_at.is_not(None),
                        CronModel.next_run_at < next_run_at,
                    )
                    .values(next_run_at=next_run_at)
                )
                await session.commit()
                logger.info(LogMessage.CRON_RUN_FRONTIER_ADVANCED, cron_id, next_run_at)
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.CRON_RUN_NOT_UPDATED.format(run_id=run_id, error=e)) from e

    async def list_by_cron(self, cron_id: str, limit: int = 50) -> list[CronRun]:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                # Defensive clamp (QA BUG-003): a negative/oversized limit must
                # never reach the SQL LIMIT clause — the route validates the
                # client input, this keeps every internal caller safe too.
                safe_limit = max(1, min(limit, 200))
                rows = (
                    (
                        await session.execute(
                            select(CronRunModel)
                            .where(CronRunModel.cron_id == UUID(cron_id))
                            .order_by(CronRunModel.created_at.desc(), CronRunModel.id.desc())
                            .limit(safe_limit)
                        )
                    )
                    .scalars()
                    .all()
                )
                return [_to_run(m) for m in rows]
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.CRON_RUN_LIST_FAILED.format(cron_id=cron_id, error=e)) from e
