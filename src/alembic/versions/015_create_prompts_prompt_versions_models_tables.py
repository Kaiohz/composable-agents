"""Create local registry tables: prompts, prompt_versions, models (spec §8.1).

Revision ID: 015
Revises: 014

Per-user PostgreSQL registries:

* ``prompts`` — prompt metadata; the UUID is the identity (PR-001).
* ``prompt_versions`` — immutable content versions; the composite primary key
  ``(prompt_id, version)`` enforces version uniqueness at the database level
  (PR-003).
* ``models`` — metadata-only model configurations; NO credential columns
  exist by design (MDL-002).

Row-Level Security is enabled and FORCED on all three tables with a per-user
policy keyed on the transaction-scoped ``app.user_id`` GUC (SEC-001), matching
the pattern established by migration 013.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "015"
down_revision: str | None = "014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = (
    "prompts",
    "prompt_versions",
    "models",
)


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS prompts (
            id             UUID PRIMARY KEY,
            user_id        VARCHAR(255) NOT NULL,
            name           TEXT NOT NULL,
            description    TEXT NULL,
            latest_version INTEGER NOT NULL DEFAULT 1,
            created_at     TIMESTAMPTZ NOT NULL,
            updated_at     TIMESTAMPTZ NOT NULL
        );
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS prompt_versions (
            prompt_id  UUID NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
            version    INTEGER NOT NULL,
            content    TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            user_id    VARCHAR(255) NOT NULL,
            CONSTRAINT pk_prompt_versions PRIMARY KEY (prompt_id, version)
        );
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
            id             UUID PRIMARY KEY,
            user_id        VARCHAR(255) NOT NULL,
            name           TEXT NOT NULL,
            provider       TEXT NOT NULL,
            provider_model TEXT NOT NULL,
            base_url       TEXT NULL,
            created_at     TIMESTAMPTZ NOT NULL,
            updated_at     TIMESTAMPTZ NOT NULL
        );
        """
    )
    # Ownership / lookup indexes (SEC-001, TRC-002 query plans).
    op.execute("CREATE INDEX IF NOT EXISTS ix_prompts_user_id ON prompts (user_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_prompt_versions_user_id ON prompt_versions (user_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_prompt_versions_prompt_id ON prompt_versions (prompt_id, version);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_models_user_id ON models (user_id);")
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
    op.execute("DROP TABLE IF EXISTS prompt_versions;")
    op.execute("DROP TABLE IF EXISTS prompts;")
    op.execute("DROP TABLE IF EXISTS models;")
