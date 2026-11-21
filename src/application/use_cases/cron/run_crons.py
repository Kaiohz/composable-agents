"""Cron execution: Run now, durable run history and the scheduler tick.

Execution behaviour (spec §5.3 CRN-004…CRN-006):

* Scheduled and manual executions share one invocation path
  (:class:`ExecuteCronRunUseCase`).
* Every ACCEPTED execution attempt creates a new thread with
  ``source="cron"`` before invoking the agent; failures after acceptance keep
  the run record and its thread.
* Overlap protection is DB-backed and atomic: at most one run per cron may be
  ``queued``/``running`` (partial unique index). Concurrent manual Run now
  requests are resolved by the conditional INSERT — exactly one wins, the
  losers surface an explicit conflict; overlapping scheduled occurrences are
  converted into durable, idempotent ``skipped`` records.
* Occurrence claims are atomic database claims (unique partial index) — two
  competing workers produce at most one execution and one cron thread.
* Runs left behind by a terminated worker are reclaimed (lease-based,
  SCHED-001) so a crash can never wedge a cron permanently; reclaimed runs
  are terminal failures and never retried automatically.
* Runs execute under the cron owner's user/RLS context and never borrow an
  interactive user's credentials; scheduler error summaries are redacted
  (SCHED-001).
"""

import asyncio
import logging
import re
import uuid
from datetime import UTC, datetime, timedelta

from src.application.services.cron_unattended import find_user_placeholder_sources
from src.domain.entities.cron import Cron
from src.domain.entities.cron_run import CronRun, CronRunStatus, CronTrigger
from src.domain.entities.thread import ThreadSource
from src.domain.errors.cron import CronRunConflictError
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_config_store import AgentConfigStore
from src.domain.ports.agent_registry import AgentRegistry
from src.domain.ports.cron_repository import CronRepository, CronRunRepository
from src.domain.ports.mcp_registry import McpRegistryClient
from src.domain.ports.thread_repository import ThreadRepository
from src.domain.ports.trace_event_repository import TraceEventRepository
from src.domain.services.cron_schedule import latest_due_occurrence, next_occurrence
from src.infrastructure.database.rls_context import user_rls_context

logger = logging.getLogger(__name__)

# Scheduler error summaries must never carry credentials or other sensitive
# configuration (SCHED-001) — redact common secret shapes defensively.
_REDACT_PATTERN = re.compile(r"(sk-[A-Za-z0-9_-]{4,}|cpk_[A-Za-z0-9_-]+|[Bb]earer\s+[A-Za-z0-9._\-]+)")
_ERROR_SUMMARY_MAX = 500

# Lease after which a queued/running run left by a (terminated) worker is
# reclaimed. Far above the agent invoke/stream timeouts, so a genuinely active
# execution is never reclaimed mid-flight; far below "forever", so a crash can
# never wedge a cron permanently.
_STALE_RUN_LEASE = timedelta(minutes=30)


def _safe_error_summary(exc: Exception) -> str:
    """Build a redacted, bounded error summary for a failed run."""
    summary = f"{type(exc).__name__}: {exc}"
    return _REDACT_PATTERN.sub("[redacted]", summary)[:_ERROR_SUMMARY_MAX]


# Cron executions run as detached background tasks. The event loop keeps only
# a weak reference to tasks they are not strongly referenced — hold them here
# until completion so an in-flight run can never be garbage-collected.
_background_tasks: set[asyncio.Task] = set()


def _spawn_execution(executor, cron: Cron, run_id: str, trigger: CronTrigger) -> None:
    """Start a run execution task and keep a strong reference until it finishes."""
    task = asyncio.create_task(executor.execute(cron, run_id, trigger))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


class ExecuteCronRunUseCase:
    """Invoke a cron's agent once, under the cron owner's context."""

    def __init__(
        self,
        cron_repository: CronRepository,
        run_repository: CronRunRepository,
        thread_repository: ThreadRepository,
        trace_repository: TraceEventRepository,
        agent_registry: AgentRegistry,
        config_store: AgentConfigStore,
        mcp_registry: McpRegistryClient | None = None,
    ) -> None:
        self._cron_repository = cron_repository
        self._run_repository = run_repository
        self._thread_repository = thread_repository
        self._trace_repository = trace_repository
        self._agent_registry = agent_registry
        self._config_store = config_store
        self._mcp_registry = mcp_registry

    async def execute(self, cron: Cron, run_id: str, trigger: CronTrigger) -> None:
        """Run one accepted execution attempt to a terminal state.

        The run record and its thread survive any failure; a scheduled terminal
        state advances the scheduling frontier past every elapsed occurrence in
        the SAME transaction (downtime coalescing without a catch-up burst).
        """
        async with user_rls_context(cron.user_id):
            started_at = datetime.now(UTC)
            logger.info(LogMessage.CRON_RUN_STARTING, run_id, cron.id, trigger.value)
            status = CronRunStatus.SUCCEEDED
            error: str | None = None
            try:
                thread = await self._thread_repository.create(cron.agent_name, ThreadSource.CRON)
                await self._run_repository.mark_running(run_id, thread.id, started_at)
                # Explicit unattended-safety failure BEFORE any LLM call: an
                # agent edited to require ${USER_JWT}/${USER_API_KEY} fails
                # without borrowing interactive credentials (§9).
                yaml_text = await self._config_store.get(cron.agent_name)
                sources = await find_user_placeholder_sources(yaml_text, self._mcp_registry, cron.agent_name)
                if sources:
                    raise RuntimeError(
                        ErrorMessage.CRON_AGENT_UNATTENDED_PLACEHOLDER.format(
                            agent_name=f"{cron.agent_name} ({', '.join(sources)})"
                        )
                    )
                runner = await self._agent_registry.get_runner(cron.agent_name)
                turn_id = str(uuid.uuid4())
                _message, trace = await runner.invoke(thread.id, cron.message, turn_id)
                await self._trace_repository.add_batch(thread.id, trace)
            except Exception as exc:  # noqa: BLE001 — every failure becomes a durable failed run
                status = CronRunStatus.FAILED
                error = _safe_error_summary(exc)
                logger.warning(LogMessage.CRON_RUN_INVOCATION_FAILED, error)
            finished_at = datetime.now(UTC)
            await self._finish(cron, run_id, trigger, status, finished_at, error)
            elapsed = (finished_at - started_at).total_seconds()
            logger.info(LogMessage.CRON_RUN_FINISHED, run_id, status.value, elapsed)

    async def _finish(
        self,
        cron: Cron,
        run_id: str,
        trigger: CronTrigger,
        status: CronRunStatus,
        finished_at: datetime,
        error: str | None,
    ) -> None:
        """Persist the terminal state; scheduled runs coalesce the frontier."""
        if trigger is CronTrigger.SCHEDULED:
            latest = latest_due_occurrence(
                cron.cron_expression, cron.timezone, cron.next_run_at or finished_at, finished_at
            )
            if latest is None:
                latest = cron.next_run_at or finished_at
            frontier = next_occurrence(cron.cron_expression, cron.timezone, latest)
            await self._run_repository.mark_finished_and_advance(
                run_id, status, finished_at, error, cron.id, frontier
            )
            logger.info(LogMessage.CRON_RUN_FRONTIER_ADVANCED, cron.id, frontier)
        else:
            await self._run_repository.mark_finished(run_id, status, finished_at, error)
        try:
            await self._cron_repository.set_last_run(cron.id, status.value, finished_at)
        except Exception:  # noqa: BLE001 — summary update must not mask the run outcome
            logger.exception(LogMessage.DB_QUERY_FAILED, "set_last_run")


class RunCronNowUseCase:
    """Trigger Run now: explicit manual overlap conflict, otherwise execute."""

    def __init__(
        self,
        cron_repository: CronRepository,
        run_repository: CronRunRepository,
        executor: ExecuteCronRunUseCase,
    ) -> None:
        self._cron_repository = cron_repository
        self._run_repository = run_repository
        self._executor = executor

    async def execute(self, cron_id: str) -> CronRun:
        """Accept a manual run (allowed while disabled) and execute it async.

        The insert is authoritative against the cron's single-active-run
        constraint: two concurrent Run now requests cannot both hold the slot
        — the loser observes ``None`` and surfaces the conflict without
        starting a second invocation (CRN-005).

        Raises:
            CronNotFoundError: Missing or foreign cron.
            CronRunConflictError: Another run is already active.
        """
        cron = await self._cron_repository.get(cron_id)
        active = await self._run_repository.get_active_run(cron_id)
        if active is not None:
            raise CronRunConflictError(ErrorMessage.CRON_RUN_CONFLICT.format(cron_id=cron_id))
        run = await self._run_repository.create_manual(
            CronRun(cron_id=cron_id, trigger=CronTrigger.MANUAL)
        )
        if run is None:
            # Lost the race for the single active slot (concurrent Run now).
            raise CronRunConflictError(ErrorMessage.CRON_RUN_CONFLICT.format(cron_id=cron_id))
        logger.info(LogMessage.CRON_RUN_NOW, cron_id)
        _spawn_execution(self._executor, cron, run.id, CronTrigger.MANUAL)
        return run


class ListCronRunsUseCase:
    """Paginated run history for a cron, newest first (CRN-003)."""

    def __init__(self, cron_repository: CronRepository, run_repository: CronRunRepository) -> None:
        self._cron_repository = cron_repository
        self._run_repository = run_repository

    async def execute(self, cron_id: str, limit: int = 50) -> list[CronRun]:
        """Return the run history after confirming cron ownership."""
        await self._cron_repository.get(cron_id)
        return await self._run_repository.list_by_cron(cron_id, limit)


class SchedulerTickUseCase:
    """One scheduler pass over due crons (the production background entry point).

    The pass first reclaims active runs whose lease expired (a terminated
    worker's phantom would otherwise wedge its cron forever, CRN-003/CRN-005),
    then scans due crons in the system scope. For every due cron the tick
    claims the frontier occurrence with an atomic database claim (two
    competing workers produce at most one execution and one cron thread,
    CRN-006) and resolves the overlap policy from the post-claim active-run
    state: an occurrence that cannot be executed while another run is active
    becomes a durable, idempotent ``skipped`` record. The claim transaction
    moves the frontier forward for the claimed occurrence; the terminal state
    then coalesces any missed span.
    """

    def __init__(
        self,
        cron_repository: CronRepository,
        run_repository: CronRunRepository,
        executor: ExecuteCronRunUseCase,
    ) -> None:
        self._cron_repository = cron_repository
        self._run_repository = run_repository
        self._executor = executor

    async def execute(self) -> None:
        """Reclaim stale runs, then scan due crons (system scope) and claim per owner."""
        now = datetime.now(UTC)
        from src.infrastructure.database.rls_context import system_rls_context

        async with system_rls_context():
            reclaimed = await self._run_repository.reclaim_stale_active_runs(now - _STALE_RUN_LEASE)
            if reclaimed:
                logger.warning(LogMessage.CRON_RUN_STALE_RECLAIMED, reclaimed)
            due = await self._cron_repository.list_due(now)
        if not due:
            return
        logger.info(LogMessage.CRON_TICK_STARTED, len(due))
        for cron in due:
            await self._process_due(cron, now)

    async def _process_due(self, cron: Cron, now: datetime) -> None:
        """Claim/execute/skip one due cron under its owner's context (SEC-001).

        The claim targets the stored frontier occurrence exactly, and the claim
        transaction advances the frontier ATOMICALLY — no concurrent tick can
        observe the just-claimed occurrence as still-due from a stale
        ``next_run_at`` snapshot. A tick arriving during the active run then
        loses the single-active slot for its (different) occurrence and records
        it as a durable ``skipped`` record (CRN-005 overlap policy) instead of
        starting a concurrent invocation.
        """
        async with user_rls_context(cron.user_id):
            occurrence = cron.next_run_at
            if occurrence is None or occurrence > now:
                return
            next_frontier = next_occurrence(cron.cron_expression, cron.timezone, occurrence)
            candidate = CronRun(
                cron_id=cron.id,
                trigger=CronTrigger.SCHEDULED,
                occurrence_at=occurrence,
            )
            claimed = await self._run_repository.claim_scheduled(candidate, advance_next_run_to=next_frontier)
            active = await self._run_repository.get_active_run(cron.id)

            if active is not None and claimed is not None and active.id == claimed.id:
                # This worker owns the cron's single active slot.
                logger.info(LogMessage.CRON_RUN_FRONTIER_ADVANCED, cron.id, next_frontier)
                _spawn_execution(self._executor, cron, claimed.id, CronTrigger.SCHEDULED)
                return

            if active is not None and active.occurrence_at != occurrence:
                # Overlap: this occurrence becomes due while another run is
                # active — record it skipped, keep the active run going.
                if claimed is not None:
                    # Defensive: the claim won the occurrence index but not the
                    # single-active slot (the active run finished mid-flight).
                    await self._run_repository.mark_finished(
                        claimed.id, CronRunStatus.SKIPPED, datetime.now(UTC)
                    )
                else:
                    await self._run_repository.record_skipped(candidate)
                logger.info(LogMessage.CRON_TICK_OVERLAP_SKIPPED, cron.id, occurrence)

            # Lost the claim on this exact occurrence: either another worker
            # holds a claim for it (its row is the active one) or a terminal
            # row already owns the outcome. Either way this branch does
            # nothing — no retry, no duplicate record; the frontier advanced
            # atomically as part of the lost claim transaction (guarded
            # forward-only), so a re-claimed stale occurrence can never
            # execute twice. (Falls through — nothing further to do.)
