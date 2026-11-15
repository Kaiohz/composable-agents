"""Make the pre-existing per-user RLS policies bypass-aware (app.bypass_rls).

Revision ID: 018
Revises: 017

Realistic deployments (and testcontainers!) connect with a NON-superuser role
— a plain ``SET row_security = off`` cannot bypass RLS for such roles, which
breaks system/scheduler/auth jobs that legitimately need cross-user access
(bounded to server-side code paths, never request payloads).

This migration rewrites the per-user isolation policies created by 013/014 so
they honor a transaction-scoped ``app.bypass_rls`` GUC that the RLS listener
sets for ``system_rls_context`` operations, IN ADDITION to the existing
``app.user_id`` scoping. The isolation semantics are unchanged for
application traffic: without the bypass GUC, rows are visible only when their
``user_id`` matches the authenticated principal.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "018"
down_revision: str | None = "017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("agent_configs", "threads", "trace_events", "api_keys", "user_llm_settings")


def upgrade() -> None:
    for table in _TABLES:
        op.execute(f"DROP POLICY IF EXISTS {table}_user_isolation ON {table};")
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
        op.execute(
            f"""
            CREATE POLICY {table}_user_isolation
            ON {table}
            USING (user_id = current_setting('app.user_id', true))
            WITH CHECK (user_id = current_setting('app.user_id', true));
            """
        )
