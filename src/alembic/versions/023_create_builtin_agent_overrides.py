"""Create per-user built-in assistant customization snapshots.

Revision ID: 023
Revises: 022

``builtin_agent_overrides`` stores at most one full YAML snapshot per
``(user_id, agent_name)``. It is deliberately separate from ``agent_configs``
and from the shared agent YAML bucket: the built-in assistant stays a virtual
overlay, and this table only shadows its default definition for its owning
user. Row-Level Security is enabled and FORCED with the same per-user policy
as the other user-owned registries.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "023"
down_revision: str | None = "022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "builtin_agent_overrides"


def upgrade() -> None:
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_TABLE} (
            user_id    VARCHAR(255) NOT NULL,
            agent_name VARCHAR(100) NOT NULL,
            yaml       TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT pk_builtin_agent_overrides PRIMARY KEY (user_id, agent_name)
        );
        """
    )
    op.execute(f"CREATE INDEX IF NOT EXISTS ix_builtin_agent_overrides_user_id ON {_TABLE} (user_id);")
    op.execute(f"ALTER TABLE {_TABLE} ENABLE ROW LEVEL SECURITY;")
    op.execute(f"ALTER TABLE {_TABLE} FORCE ROW LEVEL SECURITY;")
    op.execute(
        f"""
        CREATE POLICY {_TABLE}_user_isolation
        ON {_TABLE}
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
    op.execute(f"DROP POLICY IF EXISTS {_TABLE}_user_isolation ON {_TABLE};")
    op.execute(f"ALTER TABLE {_TABLE} NO FORCE ROW LEVEL SECURITY;")
    op.execute(f"ALTER TABLE {_TABLE} DISABLE ROW LEVEL SECURITY;")
    op.execute("DROP INDEX IF EXISTS ix_builtin_agent_overrides_user_id;")
    op.execute(f"DROP TABLE IF EXISTS {_TABLE};")
