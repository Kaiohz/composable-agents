"""Create crons and cron_runs with scheduler-claim guarantees (spec §8.1).

Revision ID: 017
Revises: 016

``crons`` — per-user definitions with their scheduling frontier
(``next_run_at`` NULL for disabled crons); a partial index on due rows backs
the scheduler scan.

``cron_runs`` — durable run records. The partial unique index
``uq_cron_runs_occurrence_claim`` over ``(cron_id, occurrence_at)`` where the
run is NOT skipped gives every scheduled occurrence a deterministic identity
and makes the cross-worker claim atomic (CRN-006): competing workers never
both claim an occurrence, and skipped-overlap records never collide with the
claimed occurrence they document.

Row-Level Security is enabled and FORCED on both tables with per-user
policies (SEC-001); cron runs are protected through owner-scoped policies,
including direct access by id.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "017"
down_revision: str | None = "016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("crons", "cron_runs")


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS crons (
            id              UUID PRIMARY KEY,
            user_id         VARCHAR(255) NOT NULL,
            name            TEXT NOT NULL,
            agent_name      VARCHAR(100) NOT NULL,
            message         TEXT NOT NULL,
            cron_expression VARCHAR(100) NOT NULL,
            timezone        VARCHAR(100) NOT NULL,
            enabled         BOOLEAN NOT NULL DEFAULT TRUE,
            next_run_at     TIMESTAMPTZ NULL,
            last_run_status VARCHAR(20) NULL,
            last_run_at     TIMESTAMPTZ NULL,
            created_at      TIMESTAMPTZ NOT NULL,
            updated_at      TIMESTAMPTZ NOT NULL
        );
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS cron_runs (
            id            UUID PRIMARY KEY,
            cron_id       UUID NOT NULL REFERENCES crons(id) ON DELETE CASCADE,
            user_id       VARCHAR(255) NOT NULL,
            trigger       VARCHAR(20) NOT NULL,
            status        VARCHAR(20) NOT NULL,
            occurrence_at TIMESTAMPTZ NULL,
            thread_id     VARCHAR(36) NULL,
            started_at    TIMESTAMPTZ NULL,
            finished_at   TIMESTAMPTZ NULL,
            error         TEXT NULL,
            created_at    TIMESTAMPTZ NOT NULL,
            CONSTRAINT ck_cron_runs_trigger CHECK (trigger IN ('scheduled', 'manual')),
            CONSTRAINT ck_cron_runs_status CHECK (
                status IN ('queued', 'running', 'succeeded', 'failed', 'skipped')
            )
        );
        """
    )
    # Deterministic occurrence claim (CRN-006): one non-skipped row per
    # (cron, occurrence) across ALL workers.
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_cron_runs_occurrence_claim
        ON cron_runs (cron_id, occurrence_at)
        WHERE status <> 'skipped';
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_cron_runs_history ON cron_runs (cron_id, created_at);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_cron_runs_user_id ON cron_runs (user_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_crons_user_id ON crons (user_id);")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_crons_due_scan ON crons (next_run_at) WHERE enabled IS TRUE;"
    )
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY;")
        op.execute(
            f"""
            CREATE POLICY {table}_user_isolation
            ON {table}
            USING (
                user_id = current_setting('app.user_id', true)
                OR current_setting('app.bypass_rls', true) = 'on'
            )
            WITH CHECK (
                user_id = current_setting('app.user_id', true)
                OR current_setting('app.bypass_rls', true) = 'on'
            );
            """
        )


def downgrade() -> None:
    for table in _TABLES:
        op.execute(f"DROP POLICY IF EXISTS {table}_user_isolation ON {table};")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;")
    op.execute("DROP TABLE IF EXISTS cron_runs;")
    op.execute("DROP TABLE IF EXISTS crons;")
