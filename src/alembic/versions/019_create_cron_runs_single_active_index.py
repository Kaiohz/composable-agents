"""Enforce at most one active run per cron + idempotent skipped records.

Revision ID: 019
Revises: 018

* ``uq_cron_runs_single_active`` — a partial unique index over ``(cron_id)``
  limited to ``queued``/``running`` rows. The occurrence-claim index cannot
  protect manual runs (their ``occurrence_at`` is NULL and PostgreSQL treats
  NULLs as distinct in unique indexes); this index is the DB-level guarantee
  that at most one execution is active per cron for manual AND scheduled runs
  alike (CRN-005). Concurrent "Run now" requests are resolved atomically by
  the conditional INSERT in ``PostgresCronRunRepository.create_manual``.
* ``uq_cron_runs_skipped`` — deduplicates overlap-skip records so concurrent
  ticks record at most one ``skipped`` row per (cron, occurrence).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "019"
down_revision: str | None = "018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_cron_runs_single_active
        ON cron_runs (cron_id)
        WHERE status IN ('queued', 'running');
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_cron_runs_skipped
        ON cron_runs (cron_id, occurrence_at)
        WHERE status = 'skipped';
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_cron_runs_skipped;")
    op.execute("DROP INDEX IF EXISTS uq_cron_runs_single_active;")
