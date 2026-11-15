"""Behavioral test infrastructure: the REAL FastAPI app over REAL PostgreSQL.

Unlike the legacy SQLite unit fixtures in ``tests/conftest.py``, this suite
boots the production stack end-to-end:

* a real PostgreSQL server (testcontainers) that receives the project's
  Alembic migrations (RLS policies included) and serves the real SQLAlchemy
  engine created by ``init_persistence()``;
* a real MinIO server (testcontainers) backing the agent YAML config store;
* the real ``src.main.app`` — every test sends HTTP requests through the
  production router chain;
* real per-user API-key authentication: keys are seeded through the real
  ``PostgresApiKeyRepository`` + ``ApiKeyHasher`` and then used as
  ``X-API-Key`` headers, driving the production dual-auth path that sets the
  RLS contextvars;
* the external LLM provider API is the ONLY network boundary mocked, via the
  in-process ``StubLlmServer`` (see ``llm_stub.py``).

No SQLite fallback: if Docker is unavailable, fixtures fail fast with a clear
error (required by spec §10 — RLS/concurrency/scheduler tests need real
PostgreSQL).

Loop-scope: all behavioral fixtures and tests run on a single session-scoped
event loop (``loop_scope="session"``) so the asyncpg pool is never used across
loops. Test modules must declare
``pytestmark = pytest.mark.asyncio(loop_scope="session")``.
"""

import os

# ---------------------------------------------------------------------------
# Environment for Settings() — set BEFORE any src.* import happens (pytest
# imports this conftest before collecting test modules).
# ---------------------------------------------------------------------------
# The stub LLM is the only external LLM boundary; point default OpenAI env at
# it (per-user credentials still flow through the real user-LLM-settings
# chain, which the tests configure via PUT /api/v1/settings/llm).
os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{os.environ.get('TEST_STUB_LLM_PORT', '28777')}/v1"
os.environ.setdefault("OPENAI_API_KEY", "sk-behavioral-stub")
# Phoenix endpoint deliberately unreachable: the no-Phoenix-prompt-fallback
# acceptance (spec §8.8 / PR-001) requires prompt CRUD and agent execution to
# work with Phoenix effectively dead. Keep tracing itself disabled.
os.environ.setdefault("TRACING__PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:59999")

import contextlib
import threading
from collections.abc import AsyncGenerator
from typing import Any

import pytest
import pytest_asyncio

# NOTE: external-adapter mock fixtures come from the root plugin
# ``tests.fixtures.external`` (registered in tests/conftest.py).


def _start_stub_llm() -> Any:
    """Import-time start of the stub LLM server (thread-bound, daemon)."""
    from tests.behavioral.llm_stub import StubLlmServer

    server = StubLlmServer()
    server.start()
    return server


_stub_llm: Any = None
_stub_lock = threading.Lock()


def get_stub_llm() -> Any:
    """Return the process-wide stub LLM server, starting it on first use."""
    global _stub_llm
    with _stub_lock:
        if _stub_llm is None:
            _stub_llm = _start_stub_llm()
        return _stub_llm


# ---------------------------------------------------------------------------
# Session-scoped containers + booted app
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def real_app_and_db() -> AsyncGenerator[dict, None]:
    """Boot the REAL application against testcontainer PostgreSQL + MinIO.

    Yields a dict with:
        * ``app``          — the real FastAPI app (src.main.app),
        * ``dsn``          — asyncpg-normalized PostgreSQL DSN,
        * ``sync_dsn``     — libpq/psycopg DSN (used by raw helpers),
        * ``stub_llm``     — the LLM boundary mock,
        * ``settings``     — the (mutated) Settings instance used by
          ``src.dependencies``.
    """
    from testcontainers.community.minio import MinioContainer
    from testcontainers.community.postgres import PostgresContainer

    dsn_plain: str
    try:
        pg = PostgresContainer(image="postgres:16-alpine", username="test", password="test", dbname="test")
        pg.start()
    except Exception as exc:  # noqa: BLE001 — fail fast with an explicit message
        raise RuntimeError(
            "Behavioral tests require Docker (testcontainers) for a REAL PostgreSQL. "
            f"Docker unavailable: {exc!r}"
        ) from exc

    host = pg.get_container_host_ip()
    port = pg.get_exposed_port(5432)
    dsn_plain = f"postgresql://test:test@{host}:{port}/test"
    dsn_asyncpg = f"postgresql+asyncpg://test:test@{host}:{port}/test"

    minio = MinioContainer()
    minio.start()
    minio_config = minio.get_config()
    minio_endpoint = minio_config["endpoint"]  # "host:port"

    # Settings() instances created lazily by factory/alembic read env; the
    # long-lived Settings() instances used at import time get mutated below.
    os.environ["DATABASE_URL"] = dsn_plain
    os.environ["MINIO_ENDPOINT"] = minio_endpoint

    import src.dependencies as deps  # local import keeps unit tests unaffected

    # Mutate the import-time Settings instances used by the production boot path.
    deps.settings.database_url = dsn_asyncpg
    deps.settings.minio_endpoint = minio_endpoint
    deps.settings.minio_access_key = minio_config["access_key"]
    deps.settings.minio_secret_key = minio_config["secret_key"]

    # 1) Apply the project's Alembic migrations (RLS policies included) with
    #    the bootstrap (superuser) DSN — migrations need owner privileges.
    import asyncio

    from alembic.config import Config

    from alembic import command

    alembic_cfg = Config(str(deps.__file__).replace("dependencies.py", "alembic.ini"))
    alembic_cfg.set_main_option(
        "script_location",
        str(os.path.join(os.path.dirname(deps.__file__), "alembic")),
    )
    alembic_cfg.set_main_option("sqlalchemy.url", dsn_asyncpg)
    await asyncio.to_thread(command.upgrade, alembic_cfg, "head")

    # 1b) The application engine connects as a DEDICATED NON-SUPERUSER role —
    #     exactly like a production deployment (apps never run as superuser).
    #     This makes the per-user RLS policies actually enforceable, which the
    #     RLS/tenancy tests verify. Auth lookups and scheduler scans use the
    #     system (bypass-GUC) context; migrations keep the bootstrap superuser.
    import asyncpg

    bootstrap_conn = await asyncpg.connect(dsn_plain)
    try:
        await bootstrap_conn.execute("DROP ROLE IF EXISTS app_rls_test")
        await bootstrap_conn.execute("CREATE ROLE app_rls_test LOGIN PASSWORD 'app_rls_test'")
        await bootstrap_conn.execute("GRANT CONNECT ON DATABASE test TO app_rls_test")
        await bootstrap_conn.execute("GRANT USAGE ON SCHEMA public TO app_rls_test")
        # CREATE on the schema: the LangGraph store/checkpointer creates its own
        # tables under the app role (PG15+ no longer grants it implicitly).
        await bootstrap_conn.execute("GRANT CREATE ON SCHEMA public TO app_rls_test")
        await bootstrap_conn.execute(
            "GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA public TO app_rls_test"
        )
        await bootstrap_conn.execute("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_rls_test")
        await bootstrap_conn.execute(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
            "GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON TABLES TO app_rls_test"
        )
        await bootstrap_conn.execute(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO app_rls_test"
        )
    finally:
        await bootstrap_conn.close()

    app_dsn_asyncpg = "postgresql+asyncpg://app_rls_test:app_rls_test@" f"{host}:{port}/test"

    # 2) Boot the REAL composition root (engine + repos + MinIO store +
    #    PersistentAgentRegistry + auth service wiring + RLS listener) — but
    #    pointing the app ENGINE at the dedicated non-superuser role.
    deps.settings.database_url = app_dsn_asyncpg
    await deps.init_persistence()

    # 3) Seed per-user API keys through the REAL repository + hasher so tests
    #    authenticate via the production dual-auth chain (X-API-Key), which
    #    sets the RLS contextvars exactly like production.
    from src.domain.services.auth.api_key_hasher import ApiKeyHasher
    from src.infrastructure.database.rls_context import system_rls_context
    from src.infrastructure.postgres_api_key.adapter import PostgresApiKeyRepository

    assert deps._root.api_key_repository is not None
    engine = deps._root.async_engine
    key_repo = PostgresApiKeyRepository(engine=engine)
    users: dict[str, dict[str, str]] = {}
    # Alias keys "a"/"b" (test modules index users["a"]["headers"]) keep the
    # REAL user ids ("user-a"/"user-b") for RLS scoping.
    async with system_rls_context():
        for alias, uid in (("a", "user-a"), ("b", "user-b")):
            key = ApiKeyHasher.generate_key()
            await key_repo.create(user_id=uid, name=f"behavioral-{uid}", key_hash=ApiKeyHasher.hash_key(key), key_prefix=key[:10])
            users[alias] = {
                "id": uid,
                "key": key,
                "headers": {"X-API-Key": key},
            }

    stub_llm = get_stub_llm()

    from src.main import app

    yield {
        "app": app,
        "dsn": dsn_asyncpg,
        "sync_dsn": dsn_plain,
        "stub_llm": stub_llm,
        "settings": deps.settings,
        "users": users,
        "engine": engine,
    }

    await deps.close_persistence()
    stub_llm.stop()
    with contextlib.suppress(Exception):
        pg.stop()
    with contextlib.suppress(Exception):
        minio.stop()


@pytest_asyncio.fixture(loop_scope="session")
async def stack(real_app_and_db: dict) -> AsyncGenerator[dict, None]:
    """Per-test view over the booted stack with a clean database + stub state."""
    stub_llm = real_app_and_db["stub_llm"]
    stub_llm.reset()
    await _truncate_user_tables(real_app_and_db["engine"])
    yield real_app_and_db
    stub_llm.reset()


@pytest.fixture
def app(stack: dict):
    """The REAL FastAPI application (production composition)."""
    return stack["app"]


@pytest.fixture
def stub_llm(stack: dict):
    """The external-LLM boundary mock (records requests; configurable mode)."""
    return stack["stub_llm"]


@pytest.fixture
def users(stack: dict) -> dict[str, dict[str, str]]:
    """Per-user ``X-API-Key`` headers seeded through the real auth chain."""
    return stack["users"]


@pytest_asyncio.fixture(loop_scope="session")
async def client(stack: dict) -> AsyncGenerator[Any, None]:
    """An httpx AsyncClient bound to the REAL app via ASGITransport."""
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=stack["app"])
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture(loop_scope="session")
async def client_factory(stack: dict):
    """Create a fresh AsyncClient (context manager) — for concurrent requests."""
    from httpx import ASGITransport, AsyncClient

    created: list[AsyncClient] = []

    async def _make() -> AsyncClient:
        ac = AsyncClient(transport=ASGITransport(app=stack["app"]), base_url="http://test")
        created.append(ac)
        return ac

    try:
        yield _make
    finally:
        for ac in created:
            await ac.aclose()


# ---------------------------------------------------------------------------
# Helpers shared by behavioral test modules
# ---------------------------------------------------------------------------
async def _truncate_user_tables(engine) -> None:
    """Reset per-test state by truncating every user-owned table.

    Preserves ``alembic_version``, ``api_keys`` (auth seeding) and the
    LangGraph store/checkpoint tables. Data-level reset only — schemas and
    migrations stay for the whole session.
    """
    from sqlalchemy import text

    from src.infrastructure.database.rls_context import system_rls_context

    keep_prefixes = ("alembic", "api_keys", "checkpoint", "store")
    async with system_rls_context(), engine.begin() as conn:
        rows = await conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
            )
        )
        names = [r[0] for r in rows if not r[0].startswith(keep_prefixes)]
        if names:
            await conn.execute(text("TRUNCATE TABLE " + ", ".join(sorted(names)) + " CASCADE"))


async def configure_llm_credentials(client, headers: dict[str, str], api_key: str = "sk-stub-user-default") -> None:
    """Give a user real per-user LLM credentials pointing at the stub LLM.

    Goes through the REAL ``PUT /api/v1/settings/llm`` endpoint so the
    production per-user credential resolution chain is exercised. ``api_key``
    is the credential the agent runner later presents to the (stubbed) LLM —
    tests pass deterministic per-user values to assert credential isolation.
    """
    from tests.behavioral.llm_stub import STUB_HOST, TEST_STUB_LLM_PORT

    stub_url = f"http://{STUB_HOST}:{TEST_STUB_LLM_PORT}/v1"

    resp = await client.put(
        "/api/v1/settings/llm",
        headers=headers,
        json={
            "provider": "openai",
            "base_url": stub_url,
            "api_key": api_key,
        },
    )

    if resp.status_code != 200:
        raise RuntimeError(f"configure_llm_credentials failed: {resp.status_code} {resp.text}")


async def create_agent_yaml(client, headers: dict[str, str], yaml_text: str, name: str) -> Any:
    """Create an agent through the REAL multipart YAML create endpoint.

    ``yaml_text`` must already contain ``name: <name>`` (callers embed the
    name themselves to avoid brace-formatting YAML content that legitimately
    contains ``${...}`` placeholders or dict syntax).
    """
    resp = await client.post(
        "/api/v1/agents",
        headers=headers,
        data={"agent_name": name},
        files={"file": ("agent.yaml", yaml_text.encode(), "text/yaml")},
    )
    return name, resp


async def create_thread(client, headers: dict[str, str], agent_name: str, **extra: Any) -> Any:
    """Create a thread through the REAL POST /api/v1/threads endpoint."""
    payload: dict[str, Any] = {"agent_name": agent_name}
    payload.update(extra)
    return await client.post("/api/v1/threads", headers=headers, json=payload)


async def invoke_agent(client, headers: dict[str, str], thread_id: str, message: str = "hello") -> Any:
    """Send a message through the REAL chat endpoint (full invocation chain)."""
    return await client.post(f"/api/v1/chat/{thread_id}", headers=headers, json={"message": message})


async def poll_condition(predicate, timeout: float = 12.0, interval: float = 0.15) -> tuple[bool, Any]:
    """Poll an async predicate until its OK component is truthy or timeout.

    Predicates return ``(ok, data)`` tuples (see the crons/traces helpers).
    The loop continues while ``ok`` is falsy — a falsy ``ok`` inside a truthy
    tuple must not end the poll early.
    """
    import asyncio
    import time

    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = await predicate()
        if isinstance(last, tuple) and len(last) == 2:
            ok, last = last
        else:
            ok = bool(last)
        if ok:
            return True, last
        await asyncio.sleep(interval)
    return False, last


# ---------------------------------------------------------------------------
# Cron subsystem seam — the production scheduler entry point.
#
# The scheduler background loop (Phase 3) must process due occurrences by
# awaiting ``run_scheduler_tick()``. Behavioral tests trigger scheduled
# execution exactly the way production does: by awaiting this entry point
# (this suite's equivalent of a consumer entry point). Concurrency scenarios
# simply await it from two tasks, mirroring two competing workers.
# ---------------------------------------------------------------------------
async def run_scheduler_tick() -> None:
    """Invoke the production scheduler tick (single pass over due crons)."""
    from src.infrastructure.cron_scheduler import run_scheduler_tick  # contract seam for Step 2

    await run_scheduler_tick()


async def run_scheduler_tick_concurrently(worker_count: int = 2) -> None:
    """Simulate N competing scheduler workers over the same due state."""
    import asyncio

    await asyncio.gather(*(run_scheduler_tick() for _ in range(worker_count)))


async def set_cron_next_run(stack: dict, cron_id: str, next_run_at: str | None) -> None:
    """Overwrite ``crons.next_run_at`` directly (simulates elapsed downtime).

    The cron table is created by Step 2's migration; writing the state that
    the scheduler consumes is the test-suite equivalent of time travel.
    ISO strings are parsed because asyncpg binds timestamps as datetime objects.
    """
    from datetime import datetime

    from sqlalchemy import text

    from src.infrastructure.database.rls_context import system_rls_context

    parsed: datetime | None = None
    if next_run_at is not None:
        parsed = datetime.fromisoformat(next_run_at)

    async with system_rls_context(), stack["engine"].begin() as conn:
        await conn.execute(text("UPDATE crons SET next_run_at = :t WHERE id = :id"), {"t": parsed, "id": cron_id})
