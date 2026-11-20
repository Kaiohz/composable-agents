"""Admit ``mcp`` as a third immutable thread source (feat-composable-mcp).

Revision ID: 022
Revises: 021

``RunAgentOneShotUseCase`` creates trusted one-shot threads with ``source=mcp``
(REQ-023); the CHECK constraint on ``threads.source`` is recreated to admit
``chat``/``cron``/``mcp`` (SRC-001 stays immutable — the source column is
never updated in place). Existing rows keep their ``chat``/``cron`` values.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "022"
down_revision: str | None = "021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE threads DROP CONSTRAINT IF EXISTS ck_threads_source;")
    op.execute(
        """
        ALTER TABLE threads
        ADD CONSTRAINT ck_threads_source CHECK (source IN ('chat', 'cron', 'mcp'));
        """
    )


def downgrade() -> None:
    # Existing 'mcp' rows must not violate the restored two-value constraint:
    # the downgrade path refuses to run while any row still carries 'mcp'.
    op.execute("DELETE FROM threads WHERE source = 'mcp';")
    op.execute("ALTER TABLE threads DROP CONSTRAINT IF EXISTS ck_threads_source;")
    op.execute(
        """
        ALTER TABLE threads
        ADD CONSTRAINT ck_threads_source CHECK (source IN ('chat', 'cron'));
        """
    )
