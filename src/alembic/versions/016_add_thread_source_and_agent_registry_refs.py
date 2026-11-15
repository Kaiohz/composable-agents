"""Add the immutable thread source column and agent registry reference projections.

Revision ID: 016
Revises: 015

* ``threads.source`` (SRC-001): non-null, server-default ``chat`` so existing
  rows are backfilled to ``chat``; a CHECK constraint admits only
  ``chat``/``cron``; indexes support the source pill reads and the global
  trace agent filter.
* ``agent_configs.prompt_ref`` / ``agent_configs.model_ref`` (SEC-002): the
  relational projection of the agent YAML's registry references. The two
  foreign keys (RESTRICT) are the race-safe integrity mechanism: an in-use
  prompt/model cannot be deleted, and a prompt/model deletion racing an agent
  save can never persist a dangling reference. Legacy agents (no refs) remain
  valid untouched (§8.3).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "016"
down_revision: str | None = "015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE threads
        ADD COLUMN IF NOT EXISTS source VARCHAR(10) NOT NULL DEFAULT 'chat';
        """
    )
    # Explicit backfill (SRC-001) — redundant with the server default but
    # keeps the migration self-documenting for pre-existing rows.
    op.execute("UPDATE threads SET source = 'chat' WHERE source IS NULL;")
    op.execute(
        """
        ALTER TABLE threads
        ADD CONSTRAINT ck_threads_source CHECK (source IN ('chat', 'cron'));
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_threads_agent_name ON threads (agent_name);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_threads_source ON threads (source);")
    # The global trace aggregate joins threads → trace_events by thread_id;
    # foreign keys are not auto-indexed in PostgreSQL.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_trace_events_thread_id ON trace_events (thread_id, timestamp);"
    )

    op.execute("ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS prompt_ref UUID NULL;")
    op.execute("ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS model_ref UUID NULL;")
    op.execute(
        """
        ALTER TABLE agent_configs
        ADD CONSTRAINT fk_agent_configs_prompt_ref
        FOREIGN KEY (prompt_ref) REFERENCES prompts(id) ON DELETE RESTRICT;
        """
    )
    op.execute(
        """
        ALTER TABLE agent_configs
        ADD CONSTRAINT fk_agent_configs_model_ref
        FOREIGN KEY (model_ref) REFERENCES models(id) ON DELETE RESTRICT;
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_configs_prompt_ref ON agent_configs (prompt_ref);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_configs_model_ref ON agent_configs (model_ref);")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_trace_events_thread_id;")
    op.execute("DROP INDEX IF EXISTS ix_agent_configs_model_ref;")
    op.execute("DROP INDEX IF EXISTS ix_agent_configs_prompt_ref;")
    op.execute("ALTER TABLE agent_configs DROP CONSTRAINT IF EXISTS fk_agent_configs_model_ref;")
    op.execute("ALTER TABLE agent_configs DROP CONSTRAINT IF EXISTS fk_agent_configs_prompt_ref;")
    op.execute("ALTER TABLE agent_configs DROP COLUMN IF EXISTS model_ref;")
    op.execute("ALTER TABLE agent_configs DROP COLUMN IF EXISTS prompt_ref;")
    op.execute("DROP INDEX IF EXISTS ix_threads_source;")
    op.execute("DROP INDEX IF EXISTS ix_threads_agent_name;")
    op.execute("ALTER TABLE threads DROP CONSTRAINT IF EXISTS ck_threads_source;")
    op.execute("ALTER TABLE threads DROP COLUMN IF EXISTS source;")
