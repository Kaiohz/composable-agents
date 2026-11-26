"""Alembic migration 024 — drop the Settings-owned connection columns (spec §D).

The migration chain is run against a DEDICATED scratch database on the real
testcontainer PostgreSQL server, so the shared behavioral stack never leaves
its ``head`` state:

* DB-002: upgrade 023→024 drops ``models.provider`` and ``models.base_url``.
* DB-003: the other columns, indexes, constraints and RLS state are untouched.
* DB-004: downgrade recreates ``base_url`` (nullable) and ``provider``
  (NOT NULL) with the technical non-empty value ``unknown`` on existing rows.
* DB-005: the historic values are definitively lost — the test asserts the
  downgrade does NOT restore them.
"""

import asyncio
import contextlib
import os
from uuid import uuid4

import asyncpg
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

_HEAD = "024"
_PREVIOUS = "023"
_SCRATCH_DB = "migration_024_scratch"

_MODELS_COLUMNS_AT_HEAD = {"id", "user_id", "name", "provider_model", "created_at", "updated_at"}
_REMOVED_COLUMNS = ("provider", "base_url")


@contextlib.asynccontextmanager
async def _scratch_database(stack: dict):
    """Create a throwaway database on the testcontainer server; drop it after."""
    admin_dsn = stack["sync_dsn"].rsplit("/", 1)[0] + "/postgres"
    scratch_dsn = stack["sync_dsn"].rsplit("/", 1)[0] + f"/{_SCRATCH_DB}"
    admin = await asyncpg.connect(admin_dsn)
    try:
        with contextlib.suppress(asyncpg.PostgresError):
            await admin.execute(f"DROP DATABASE IF EXISTS {_SCRATCH_DB} WITH (FORCE)")  # noqa: S608
        await admin.execute(f"CREATE DATABASE {_SCRATCH_DB}")  # noqa: S608
    finally:
        await admin.close()
    try:
        yield scratch_dsn
    finally:
        admin = await asyncpg.connect(admin_dsn)
        try:
            with contextlib.suppress(asyncpg.PostgresError):
                await admin.execute(f"DROP DATABASE IF EXISTS {_SCRATCH_DB} WITH (FORCE)")  # noqa: S608
        finally:
            await admin.close()


async def _run_migrate(dsn_plain: str, verb: str, target: str) -> None:
    """Run one Alembic command on the scratch database (bootstrap superuser)."""
    from alembic.config import Config

    import src.dependencies as deps
    from alembic import command

    dsn_asyncpg = dsn_plain.replace("postgresql://", "postgresql+asyncpg://", 1)
    previous_url = os.environ["DATABASE_URL"]
    os.environ["DATABASE_URL"] = dsn_asyncpg  # env.py resolves Settings() from the env
    try:
        cfg = Config(str(deps.__file__).replace("dependencies.py", "alembic.ini"))
        cfg.set_main_option("sqlalchemy.url", dsn_asyncpg)
        await asyncio.to_thread(getattr(command, verb), cfg, target)
    finally:
        os.environ["DATABASE_URL"] = previous_url


async def _columns(dsn_plain: str, table: str = "models") -> dict[str, bool]:
    """Column name → nullable map for one table (information_schema)."""
    conn = await asyncpg.connect(dsn_plain)
    try:
        rows = await conn.fetch(
            "SELECT column_name, is_nullable FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = $1 ORDER BY ordinal_position",
            table,
        )
    finally:
        await conn.close()
    return {r["column_name"]: r["is_nullable"] == "YES" for r in rows}


async def _seed_models_at_023(dsn_plain: str) -> None:
    """Insert legacy-row Models at revision 023 (provider set, base_url mixed)."""
    conn = await asyncpg.connect(dsn_plain)
    try:
        await conn.execute(
            "INSERT INTO models (id, user_id, name, provider, provider_model, base_url, created_at, updated_at) "
            "VALUES ($1, 'user-a', 'Legacy A', 'openai', 'gpt-a', 'https://api.openai.example/v1', now(), now()), "
            "($2, 'user-a', 'Legacy B', 'anthropic', 'claude-b', NULL, now(), now())",
            uuid4(),
            uuid4(),
        )
    finally:
        await conn.close()


async def test_migration_024_roundtrip(stack: dict):
    """DB-002…005: upgrade drops, rows survive; downgrade restores only shape."""
    async with _scratch_database(stack) as scratch_plain:
        # --- Build the chain up to revision 023 and seed legacy rows ---
        await _run_migrate(scratch_plain, "upgrade", _PREVIOUS)
        columns = await _columns(scratch_plain)
        assert set(columns) == _MODELS_COLUMNS_AT_HEAD | set(_REMOVED_COLUMNS), columns
        assert columns["provider"] is False  # NOT NULL at 023
        assert columns["base_url"] is True  # nullable at 023
        await _seed_models_at_023(scratch_plain)

        # --- Upgrade to head: the two connection columns are GONE ---
        await _run_migrate(scratch_plain, "upgrade", _HEAD)
        columns = await _columns(scratch_plain)
        assert set(columns) == _MODELS_COLUMNS_AT_HEAD, columns
        for removed in _REMOVED_COLUMNS:
            assert removed not in columns, f"{removed} must be dropped by the upgrade"

        # DB-003: rows are conserved, RLS state and index untouched.
        conn = await asyncpg.connect(scratch_plain)
        try:
            rows = await conn.fetch("SELECT name, provider_model, user_id FROM models ORDER BY name")
            assert [(r["name"], r["provider_model"], r["user_id"]) for r in rows] == [
                ("Legacy A", "gpt-a", "user-a"),
                ("Legacy B", "claude-b", "user-a"),
            ]
            state = await conn.fetchrow(
                "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                "JOIN pg_namespace n ON n.oid = relnamespace "
                "WHERE n.nspname = 'public' AND relname = 'models'"
            )
            assert state["relrowsecurity"] and state["relforcerowsecurity"], "RLS must stay enabled and forced"
            assert await conn.fetchval(
                "SELECT count(*) FROM pg_policies WHERE schemaname = 'public' "
                "AND tablename = 'models' AND policyname = 'models_user_isolation'"
            )
            assert await conn.fetchval(
                "SELECT count(*) FROM pg_indexes WHERE schemaname = 'public' "
                "AND tablename = 'models' AND indexname = 'ix_models_user_id'"
            )
        finally:
            await conn.close()

        # --- Downgrade: shape restored, historic values NOT (DB-004/005) ---
        await _run_migrate(scratch_plain, "downgrade", _PREVIOUS)
        columns = await _columns(scratch_plain)
        assert set(columns) == _MODELS_COLUMNS_AT_HEAD | set(_REMOVED_COLUMNS), columns
        assert columns["provider"] is False, "downgraded provider must be NOT NULL"
        assert columns["base_url"] is True, "downgraded base_url must be nullable"
        conn = await asyncpg.connect(scratch_plain)
        try:
            values = await conn.fetch("SELECT name, provider, base_url FROM models ORDER BY name")
            assert [r["name"] for r in values] == ["Legacy A", "Legacy B"]
            for row in values:
                assert row["provider"] == "unknown", f"technical non-empty placeholder expected: {row!r}"
                assert row["base_url"] is None, "no historic base_url may be presented as recovered"
        finally:
            await conn.close()

        # --- Re-upgrade keeps the chain healthy (idempotent head) ---
        await _run_migrate(scratch_plain, "upgrade", _HEAD)
        assert set(await _columns(scratch_plain)) == _MODELS_COLUMNS_AT_HEAD
