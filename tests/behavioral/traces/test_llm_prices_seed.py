"""Behavioral tests — llm_prices table: migration 020 + startup seed (dashboard §3).

The documented exception to route-level testing applies (adapter/database
semantics): table DDL, RLS flags, seed contents and the global-vs-system
policy are Postgres facts best proven DIRECTLY against the real PostgreSQL
testcontainer through the REAL production engine (same pattern as
test_rls_policies.py). The seeder's startup wiring is proven behaviorally
by the stack fixture, which boots the real production startup path
(``init_persistence()``) before any test runs.

Contracts fixed here for the implementation (Step 2):

* Migration 020 creates ``llm_prices``: ``model_name`` (primary key — a
  single row per model), ``provider``, ``input_price_per_million``,
  ``output_price_per_million`` (+ optional currency) and NO ``user_id``
  column (global reference table, not user-owned).
* RLS enabled AND forced on ``llm_prices`` (the house rule for every table).
* Read policy: available to ANY authenticated user (server-visible global) —
  every user resolves prices against the same rows.
* Write policy: restricted to system contexts (``system_rls_context`` /
  ``app.bypass_rls=on``, the production seeder path). A normal authenticated
  user must get a row-level-security violation on INSERT, even though the
  application role holds table privileges.
* The startup seeder runs at startup (after fixture boot: rows exist) and is
  a single-row-per-model idempotent upsert for the known Anthropic / OpenAI /
  Google / Mistral / OpenRouter models and prefixes.
"""

import uuid

import pytest
from sqlalchemy import text

from src.infrastructure.database.rls_context import current_user_id, system_rls_context

pytestmark = pytest.mark.asyncio(loop_scope="session")

# Known seed models the dashboard relies on for token→USD pricing (spec §3).
KNOWN_SEED_MODELS = (
    # Anthropic
    "claude-sonnet-4-5",
    "claude-haiku-4",
    # OpenAI
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "o3",
    # Google
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    # Mistral
    "mistral-large",
    "mistral-medium",
    "mistral-small",
)


@pytest.fixture
def engine(stack):
    """The REAL production engine (RLS listener registered)."""
    return stack["engine"]


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


async def _one(engine, sql: str, params: dict | None = None):
    async with engine.begin() as conn:
        result = await conn.execute(text(sql), params or {})
        return result.first()


async def _all(engine, sql: str, params: dict | None = None):
    async with engine.begin() as conn:
        result = await conn.execute(text(sql), params or {})
        return result.all()


class TestLlmPricesTable:
    async def test_table_exists_with_rls_enabled_and_forced(self, engine):
        state = await _one(
            engine,
            "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
            "JOIN pg_namespace n ON n.oid = relnamespace "
            "WHERE n.nspname = 'public' AND relname = 'llm_prices'",
        )
        assert state is not None, "llm_prices is missing — migration 020 did not run"
        relrowsecurity, relforcerowsecurity = state
        assert relrowsecurity is True, "RLS not enabled on llm_prices"
        assert relforcerowsecurity is True, "RLS not forced on llm_prices"

    async def test_pricing_columns_and_single_row_per_model(self, engine):
        cols = {
            r[0]
            for r in await _all(
                engine,
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = 'llm_prices'",
            )
        }
        for required in ("model_name", "provider", "input_price_per_million", "output_price_per_million"):
            assert required in cols, f"llm_prices is missing the '{required}' column (got {sorted(cols)})"
        # Global reference table: NOT user-owned.
        assert "user_id" not in cols, f"llm_prices must not carry user_id ownership: {sorted(cols)}"

        pk_cols = {
            r[0]
            for r in await _all(
                engine,
                "SELECT a.attname FROM pg_index i "
                "JOIN pg_class c ON c.oid = i.indrelid "
                "JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(i.indkey) "
                "WHERE c.relname = 'llm_prices' AND i.indisprimary",
            )
        }
        assert "model_name" in pk_cols, "model_name must be the primary key (single row per model)"

    async def test_rls_policies_are_defined(self, engine):
        rows = await _all(engine, "SELECT policyname FROM pg_policies WHERE tablename = 'llm_prices'")
        assert rows, "no RLS policy defined on llm_prices"


class TestLlmPricesSeedRows:
    async def test_known_provider_models_are_seeded_after_startup(self, stack, engine):
        """The startup seeder runs at startup: after the stack boot every
        known model (Anthropic / OpenAI / Google / Mistral) has a priced row."""
        names = {r[0] for r in await _all(engine, "SELECT model_name FROM llm_prices")}
        missing = [m for m in KNOWN_SEED_MODELS if m not in names]
        assert missing == [], f"seed is missing known models: {missing}"

    async def test_seeded_prices_are_positive_with_provider(self, engine):
        rows = await _all(
            engine,
            "SELECT model_name, provider, input_price_per_million, output_price_per_million "
            "FROM llm_prices WHERE model_name = ANY(:names)",
            {"names": list(KNOWN_SEED_MODELS)},
        )
        assert len(rows) >= len(KNOWN_SEED_MODELS)
        for model, provider, in_price, out_price in rows:
            assert provider, f"{model}: seeded row must carry a provider"
            assert in_price is not None and float(in_price) > 0, f"{model}: input price must be positive"
            assert out_price is not None and float(out_price) > 0, f"{model}: output price must be positive"

    async def test_seed_is_single_row_per_model(self, engine):
        dupes = await _all(
            engine,
            "SELECT model_name, count(*) FROM llm_prices GROUP BY model_name HAVING count(*) > 1",
        )
        assert dupes == [], f"seed upsert must keep exactly one row per model: {dupes}"

    async def test_seed_covers_openrouter_style_models(self, engine):
        """OpenRouter names are provider-qualified paths — the seed must
        cover at least one OpenRouter-style row (provider or slash model_id)."""
        rows = await _all(
            engine,
            "SELECT count(*) FROM llm_prices WHERE provider = 'openrouter' OR position('/' in model_name) > 0",
        )
        assert rows[0][0] >= 1, "seed must include OpenRouter-style model rows"


class TestLlmPricesGlobalPolicy:
    async def test_price_rows_are_readable_by_any_authenticated_user(self, stack, engine, as_user):
        """Global read: both user-a and user-b see the same seeded prices."""
        for uid in ("user-a", "user-b"):
            with as_user(uid):
                count = (await _one(engine, "SELECT count(*) FROM llm_prices"))[0]
            assert count >= len(KNOWN_SEED_MODELS), f"{uid} must read the global pricing table (got {count} rows)"

    async def test_plain_user_insert_is_denied_by_rls(self, stack, engine, as_user):
        """Write restricted to system: a normal authenticated user INSERT must
        fail with a row-level-security violation (table privileges alone are
        NOT sufficient)."""
        probe = f"probe-{uuid.uuid4().hex[:8]}"
        with as_user("user-a"), pytest.raises(Exception, match="row-level security"):
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO llm_prices (model_name, provider, "
                        "input_price_per_million, output_price_per_million) "
                        "VALUES (:m, 'test', 1, 2)"
                    ),
                    {"m": probe},
                )
        leftover = await _one(engine, "SELECT count(*) FROM llm_prices WHERE model_name = :m", {"m": probe})
        assert leftover[0] == 0, "denied insert must not leave a row behind"

    async def test_system_context_insert_and_delete_is_allowed(self, stack, engine):
        """The production seeder path (system RLS scope) may write prices."""
        probe = f"probe-{uuid.uuid4().hex[:8]}"
        async with system_rls_context(), engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO llm_prices (model_name, provider, "
                    "input_price_per_million, output_price_per_million) "
                    "VALUES (:m, 'test', 10, 20)"
                ),
                {"m": probe},
            )
        stored = await _one(
            engine,
            "SELECT input_price_per_million, output_price_per_million FROM llm_prices WHERE model_name = :m",
            {"m": probe},
        )
        assert stored is not None and float(stored[0]) == 10.0 and float(stored[1]) == 20.0
        async with system_rls_context(), engine.begin() as conn:
            await conn.execute(text("DELETE FROM llm_prices WHERE model_name = :m"), {"m": probe})
        assert (await _one(engine, "SELECT count(*) FROM llm_prices WHERE model_name = :m", {"m": probe}))[0] == 0
