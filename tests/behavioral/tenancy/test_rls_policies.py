"""RLS & tenancy integration tests (spec §7 SEC-001, §8 migrations).

These tests exercise the PostgreSQL Row-Level Security layer DIRECTLY — the
documented exception to route-level testing (adapter-specific SQL rules):
RLS policies are database semantics, best proven against a real PostgreSQL
server by manipulating the ``app.user_id`` GUC through the REAL production
engine (with its RLS listener registered by ``init_persistence()``).

Covered:
* every new owned table (prompts, prompt_versions, models, crons, cron_runs)
  exists with RLS enabled AND forced;
* per-user isolation policies exist on each of those tables;
* GUC-scoped SELECT/INSERT isolation (no cross-user reads, no foreign-user
  inserts, no unauthenticated visibility);
* ``threads.source`` is NOT NULL, defaults to 'chat' (legacy-row backfill
  target) and only allows 'chat'/'cron'.
"""

import uuid

import pytest
from sqlalchemy import text

from src.infrastructure.database.rls_context import current_user_id

pytestmark = pytest.mark.asyncio(loop_scope="session")

# New per-user tables created by the feature migrations (spec §8.1).
OWNED_TABLES = ("prompts", "prompt_versions", "models", "crons", "cron_runs")


@pytest.fixture
def engine(stack):
    """The REAL production engine (RLS listener registered)."""
    return stack["engine"]


async def _table_state(engine, table: str):
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                    "JOIN pg_namespace n ON n.oid = relnamespace "
                    "WHERE n.nspname = 'public' AND relname = :t"
                ),
                {"t": table},
            )
        ).first()
    return row


class TestRlsOnNewOwnedTables:
    async def test_every_new_owned_table_has_rls_enabled(self, engine):
        """§8.2: enable and test RLS on every new owned table."""
        for table in OWNED_TABLES:
            state = await _table_state(engine, table)
            assert state is not None, f"Table '{table}' is missing — the feature migration did not run"
            relrowsecurity, relforcerowsecurity = state
            assert relrowsecurity is True, f"RLS not enabled on {table}"
            assert relforcerowsecurity is True, f"RLS not forced on {table}"

    async def test_user_isolation_policies_exist(self, engine):
        async with engine.begin() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT tablename, policyname FROM pg_policies "
                        "WHERE schemaname = 'public' ORDER BY tablename, policyname"
                    )
                )
            ).all()
        by_table: dict[str, list[str]] = {}
        for tablename, policyname in rows:
            by_table.setdefault(tablename, []).append(policyname)
        for table in OWNED_TABLES:
            assert by_table.get(table), f"No RLS policy defined on {table}"


@pytest.fixture
def as_user():
    """Context manager: bind the engine's RLS GUC to a user id (or None)."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx(user_id: str | None):
        token = current_user_id.set(user_id)
        try:
            yield
        finally:
            current_user_id.reset(token)

    return _ctx


class TestUserScopedIsolation:
    async def test_rows_are_invisible_across_users(self, engine, as_user):
        """SEC-001: cross-user reads must return nothing — ownership scope is
        enforced by the database, not only by application queries."""
        table, owner_id = "prompts", "user-a"
        with as_user(owner_id):
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        f"INSERT INTO {table} (id, user_id, name, description, latest_version, created_at, updated_at) "
                        f"VALUES (:id, :user_id, :name, NULL, 1, now(), now())"
                    ),
                    {"id": str(uuid.uuid4()), "user_id": owner_id, "name": "rls probe"},
                )

        with as_user("user-b"):
            async with engine.begin() as conn:
                count_b = (
                    await conn.execute(text(f"SELECT count(*) FROM {table}"))
                ).scalar_one()
        assert count_b == 0, "user-b must not see user-a's prompt rows (RLS)"

        with as_user(owner_id):
            async with engine.begin() as conn:
                count_a = (
                    await conn.execute(text(f"SELECT count(*) FROM {table}"))
                ).scalar_one()
        assert count_a == 1

    async def test_insert_with_foreign_user_id_rejected(self, engine, as_user):
        """RLS WITH CHECK: a user cannot INSERT rows owned by someone else."""
        with as_user("user-b"):
            async with engine.begin() as conn:
                raised = False
                try:
                    await conn.execute(
                        text(
                            "INSERT INTO prompts (id, user_id, name, description, latest_version, created_at, updated_at) "
                            "VALUES (:id, :user_id, 'spoofed owner', NULL, 1, now(), now())"
                        ),
                        {"id": str(uuid.uuid4()), "user_id": "user-a"},
                    )
                except Exception:
                    raised = True
                assert raised, "RLS WITH CHECK must reject inserting rows owned by another user"

    async def test_unauthenticated_sessions_see_nothing(self, engine, as_user):
        """Defensive default: no app.user_id GUC → no rows (current_setting
        NULL policy semantics)."""
        table, owner_id = "models", "user-a"
        with as_user(owner_id):
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        f"INSERT INTO {table} (id, user_id, name, provider, provider_model, base_url, created_at, updated_at) "
                        f"VALUES (:id, :user_id, 'm', 'openai', 'pm', NULL, now(), now())"
                    ),
                    {"id": str(uuid.uuid4()), "user_id": owner_id},
                )

        with as_user(None):
            async with engine.begin() as conn:
                count = (await conn.execute(text(f"SELECT count(*) FROM {table}"))).scalar_one()
        assert count == 0, "Unauthenticated sessions must see zero owned rows"


class TestThreadSourceMigration:
    async def test_source_is_not_null_with_chat_default(self, engine):
        """§8: threads.source non-null, backfilled to 'chat'."""
        row = (
            await _raw_columns(engine, "threads")
        )
        source = next((c for c in row if c["name"] == "source"), None)
        assert source is not None, "threads.source column is missing (migration not applied)"
        assert source["nullable"] is False, "threads.source must be NOT NULL"

    async def test_legacy_thread_rows_backfill_to_chat(self, engine):
        """A row inserted without a source (the legacy-row backfill target)
        must receive 'chat'."""
        thread_id = str(uuid.uuid4())
        # The raw insert mimics how migrations wrote legacy rows (owner
        # privileges — no authenticated principal). The RLS bypass context is
        # the test-suite equivalent of that owner-level connection.
        from src.infrastructure.database.rls_context import system_rls_context

        async with system_rls_context(), engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO threads (id, agent_name, user_id, created_at, updated_at) "
                    "VALUES (:id, 'legacy-agent', 'user-a', now(), now())"
                ),
                {"id": thread_id},
            )
            got = (
                await conn.execute(text("SELECT source FROM threads WHERE id = :id"), {"id": thread_id})
            ).scalar_one()
        assert got == "chat", f"Legacy rows must backfill to 'chat', got {got!r}"

    async def test_source_only_allows_chat_or_cron(self, engine):
        """SRC-001: allowed values are chat and cron — enforce a DB CHECK."""
        async with engine.begin() as conn:
            raised = False
            try:
                await conn.execute(
                    text(
                        "INSERT INTO threads (id, agent_name, user_id, created_at, updated_at, source) "
                        "VALUES (:id, 'bogus', 'user-a', now(), now(), 'bogus')"
                    ),
                    {"id": str(uuid.uuid4())},
                )
            except Exception:
                raised = True
            assert raised, "threads.source must only admit 'chat'/'cron' values"

    async def test_cron_owned_tables_guc_scoped_inserts(self, engine, as_user):
        """SEC-001: cron runs are protected through owner-scoped policies."""
        cron_id = str(uuid.uuid4())
        with as_user("user-a"):
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO crons (id, user_id, name, agent_name, message, cron_expression, timezone, "
                        "enabled, created_at, updated_at) "
                        "VALUES (:id, 'user-a', 'c', 'agent', 'm', '* * * * *', 'UTC', true, now(), now())"
                    ),
                    {"id": cron_id},
                )

        with as_user("user-b"):
            async with engine.begin() as conn:
                count = (await conn.execute(text("SELECT count(*) FROM crons"))).scalar_one()
        assert count == 0, "user-b must not see user-a's crons"


async def _raw_columns(engine, table: str):
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT column_name, is_nullable FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = :t"
                ),
                {"t": table},
            )
        ).all()
    return [{"name": r[0], "nullable": r[1] == "YES"} for r in rows]
