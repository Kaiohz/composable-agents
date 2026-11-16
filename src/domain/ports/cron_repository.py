"""Outbound ports for the cron subsystem (spec §5.3).

``CronRepository`` holds per-user cron definitions and their scheduling state
(exclusive owner of ``next_run_at`` writes outside validation flows).
``CronRunRepository`` persists durable run records with atomic, cross-worker
safe occurrence claims (CRN-006). Scheduled occurrences have a deterministic
identity ``(cron_id, occurrence_at)`` enforced by a database uniqueness
constraint; in-memory locks are insufficient.
"""

from abc import ABC, abstractmethod
from datetime import datetime

from src.domain.entities.cron import Cron
from src.domain.entities.cron_run import CronRun, CronRunStatus


class CronRepository(ABC):
    """Outbound port for per-user cron definitions."""

    @abstractmethod
    async def create(self, cron: Cron, next_run_at: datetime | None) -> Cron:
        """Insert the cron definition with its computed scheduling state."""
        ...

    @abstractmethod
    async def get(self, cron_id: str) -> Cron:
        """Return the cron definition.

        Raises:
            CronNotFoundError: If missing or foreign (non-enumerating).
        """
        ...

    @abstractmethod
    async def list_all(self) -> list[Cron]:
        """Return all crons owned by the current user."""
        ...

    @abstractmethod
    async def list_due(self, now: datetime) -> list[Cron]:
        """Return every enabled cron whose ``next_run_at`` is due.

        Called from a system (RLS-bypass) context by the scheduler; results
        carry each row's ``user_id`` so the tick can bind the owner context
        (SEC-001).
        """
        ...

    @abstractmethod
    async def update(self, cron: Cron, next_run_at: datetime | None) -> Cron:
        """Persist definition edits and the recalculated scheduling state."""
        ...

    @abstractmethod
    async def delete(self, cron_id: str) -> None:
        """Delete the cron (future scheduling stops).

        Raises:
            CronNotFoundError: If missing or foreign.
        """
        ...

    @abstractmethod
    async def advance_next_run(self, cron_id: str, next_run_at: datetime) -> None:
        """Move the scheduling frontier forward only (never regress it).

        A no-op when the stored frontier is already greater or the cron was
        disabled (its frontier is ``None`` until re-enabled).
        """
        ...

    @abstractmethod
    async def set_last_run(self, cron_id: str, status: str, finished_at: datetime) -> None:
        """Record the last-run summary on the cron (list/detail views)."""
        ...


class CronRunRepository(ABC):
    """Outbound port for durable cron run records and occurrence claims."""

    @abstractmethod
    async def claim_scheduled(self, run: CronRun, advance_next_run_to: datetime | None = None) -> CronRun | None:
        """Atomically claim a scheduled occurrence for this worker.

        The claim relies on database uniqueness constraints over
        ``(cron_id, occurrence_at)`` and over the cron's single active slot;
        exactly one competing worker wins and the losers receive ``None``
        (CRN-006).

        When ``advance_next_run_to`` is provided, the CLAIM and the guarded
        forward-only frontier advance happen in the SAME transaction, so no
        concurrent tick can observe (and re-claim) the just-handled occurrence
        from a stale ``next_run_at`` snapshot.
        """
        ...

    @abstractmethod
    async def create_manual(self, run: CronRun) -> CronRun | None:
        """Insert a manual (Run now) run record, or return ``None``.

        The conditional INSERT targets the partial unique index over the
        single active run per cron: exactly one caller wins the active slot,
        every concurrent loser receives ``None`` and must surface the conflict
        (CRN-005) instead of starting a second invocation.
        """
        ...

    @abstractmethod
    async def record_skipped(self, run: CronRun) -> bool:
        """Durably record an overlap-skipped scheduled occurrence (CRN-005).

        Idempotent per ``(cron_id, occurrence_at)``: returns ``False`` when a
        skipped record for the same occurrence already exists. Skipped runs
        never create threads and never execute.
        """
        ...

    @abstractmethod
    async def reclaim_stale_active_runs(self, older_than: datetime) -> int:
        """Mark active runs whose lease expired as failed (SCHED-001).

        A worker killed between the claim/insert and ``mark_finished`` leaves a
        ``queued``/``running`` row behind; without a reaper it would block
        every later execution of that cron forever (a phantom active run).
        Runs created before ``older_than`` that never updated their
        ``started_at`` — or whose ``started_at`` predates ``older_than`` — are
        marked ``failed`` with a safe error summary. Returns the number of
        reclaimed runs. Does NOT start replacement executions (failed runs are
        terminal — no automatic retry); the reaper only frees the cron's
        single active slot.
        """
        ...

    @abstractmethod
    async def get_active_run(self, cron_id: str, exclude_run_id: str | None = None) -> CronRun | None:
        """Return the active (queued/running) run for a cron, if any."""
        ...

    @abstractmethod
    async def mark_running(self, run_id: str, thread_id: str, started_at: datetime) -> None:
        """Transition a claimed run to ``running`` and link its thread."""
        ...

    @abstractmethod
    async def mark_finished(
        self,
        run_id: str,
        status: CronRunStatus,
        finished_at: datetime,
        error: str | None = None,
    ) -> None:
        """Record a terminal (or skipped) transition; failed runs are final."""
        ...

    @abstractmethod
    async def mark_finished_and_advance(
        self,
        run_id: str,
        status: CronRunStatus,
        finished_at: datetime,
        error: str | None,
        cron_id: str,
        next_run_at: datetime,
    ) -> None:
        """Persist a scheduled run's terminal state AND advance the cron's
        scheduling frontier past every elapsed occurrence in the SAME
        transaction, coalescing downtime without a catch-up burst (CRN-005).
        """
        ...

    @abstractmethod
    async def list_by_cron(self, cron_id: str, limit: int = 50) -> list[CronRun]:
        """Return run history for a cron, newest first (CRN-003)."""
        ...
