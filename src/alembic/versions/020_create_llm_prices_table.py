"""Create the global llm_prices reference table (LLM usage dashboard, spec §3).

Revision ID: 020
Revises: 019

``llm_prices`` holds one priced row per known model name (``model_name`` primary
key): provider, ``input_price_per_million`` / ``output_price_per_million`` in
USD (per 1M tokens) and an optional ``currency``. It is a GLOBAL reference
table — deliberately NOT user-owned (no ``user_id`` column): every user's
``/traces/usage`` aggregation resolves prices against the same seeded rows.

Row-Level Security is enabled AND forced (house rule for every table):

* READ — worldwide: any authenticated principal sees the shared price rows
  (``USING (true)``); the table carries no tenant data.
* WRITE — system-only: INSERT / UPDATE / DELETE require the transaction-scoped
  ``app.bypass_rls`` GUC the RLS listener sets for system contexts
  (production startup seeder / migrations). A normal authenticated user gets a
  row-level-security violation, even though the application role holds table
  privileges.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "020"
down_revision: str | None = "019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_prices (
            model_name               TEXT PRIMARY KEY,
            provider                 TEXT NOT NULL,
            input_price_per_million  NUMERIC(18, 6) NOT NULL,
            output_price_per_million NUMERIC(18, 6) NOT NULL,
            currency                 TEXT NULL
        );
        """
    )
    op.execute("ALTER TABLE llm_prices ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE llm_prices FORCE ROW LEVEL SECURITY;")
    op.execute(
        """
        CREATE POLICY llm_prices_read_global
        ON llm_prices
        FOR SELECT
        USING (true);
        """
    )
    op.execute(
        """
        CREATE POLICY llm_prices_insert_system
        ON llm_prices
        FOR INSERT
        WITH CHECK (
            current_setting('app.bypass_rls', true) = 'on'
        );
        """
    )
    op.execute(
        """
        CREATE POLICY llm_prices_update_system
        ON llm_prices
        FOR UPDATE
        USING (
            current_setting('app.bypass_rls', true) = 'on'
        )
        WITH CHECK (
            current_setting('app.bypass_rls', true) = 'on'
        );
        """
    )
    op.execute(
        """
        CREATE POLICY llm_prices_delete_system
        ON llm_prices
        FOR DELETE
        USING (
            current_setting('app.bypass_rls', true) = 'on'
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS llm_prices_read_global ON llm_prices;")
    op.execute("DROP POLICY IF EXISTS llm_prices_insert_system ON llm_prices;")
    op.execute("DROP POLICY IF EXISTS llm_prices_update_system ON llm_prices;")
    op.execute("DROP POLICY IF EXISTS llm_prices_delete_system ON llm_prices;")
    op.execute("ALTER TABLE llm_prices NO FORCE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE llm_prices DISABLE ROW LEVEL SECURITY;")
    op.execute("DROP TABLE IF EXISTS llm_prices;")
