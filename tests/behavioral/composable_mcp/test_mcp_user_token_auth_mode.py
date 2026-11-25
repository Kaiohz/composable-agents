"""Behavioral tests — registered-server `auth_mode` + `auth_token` (REQ-004/005).

feat-mcp-oauth2-dual-auth, composable-agents slice. Entry points (all REAL):

* the ``/mcp`` endpoint and its REAL built-in assistant resolution chain
  (``composable_agents_run`` → BuiltinToolsetResolver → registry reveal →
  LangchainMcpToolLoader → langchain-mcp-adapters → the target server);
* the trusted-registry REST boundary is the only outbound external adapter
  mocked here (the existing ``StubRegistryServer``), seeded with the registered
  server entries under test;
* the TARGET registered MCP server is a REAL local streamable-HTTP MCP server
  whose ASGI middleware RECORDS every request's headers (``RecordingMcpServer``
  below) — the observable side effects asserted are the headers the BACKEND
  actually transmitted to the registered server during the tool load.

Delegation contract pinned here:

* REQ-004 — ``auth_mode: "user_token"``: at load time the backend injects the
  connected user's credential (`Authorization: Bearer <jwt>` when the
  originating request was JWT-authenticated; `X-API-Key: <key>` when
  API-key-authenticated); explicit headers are merged and the injected
  credential takes precedence on colliding names; default ``auth_mode``
  (absent → "headers") sends the configured headers as-is.
* REQ-005 — the stored ``auth_token`` is transmitted as
  ``Authorization: Bearer <resolved value>`` (placeholders
  ``${USER_JWT}``/``${USER_API_KEY}``/env resolved), never as the unsupported
  ``auth_token`` connection key (langchain-mcp-adapters 0.3.0 → TypeError /
  `McpToolLoadError` today).

RED phase: none of this exists yet — the injected/converted-header assertions
are expected to fail (no request reaches the target, or it arrives without the
expected header).
"""

import os

import pytest

from tests.behavioral.composable_mcp.conftest import (
    assistant_text,
    assistant_tool_call,
    call_mcp_tool,
)
from tests.behavioral.conftest import configure_llm_credentials

pytestmark = pytest.mark.asyncio(loop_scope="session")

BUILTIN_NAME = "composable-ai-assistant"
SERVER_NAME = "user-token-target"

#: env-placeholder auth_token test token (set/cleaned per test).
ENV_TOKEN_VAR = "COMPOSABLE_PROBE_ENV_TOKEN"
ENV_TOKEN_VALUE = "resolved-env-token-63"


# ---------------------------------------------------------------------------
# Real target server that RECORDS the headers the backend transmits
# ---------------------------------------------------------------------------
class RecordingMcpServer:
    """A REAL streamable-HTTP MCP server recording every request's headers.

    Mirrors ``StubMcpServer`` from ``tests/behavioral/composable_mcp/conftest``
    plus an ASGI middleware that snapshots the raw request headers into
    ``recorded_headers`` (list of lowercase-keyed dicts, one per HTTP request).
    """

    HOST = "127.0.0.1"

    def __init__(self, name: str = SERVER_NAME) -> None:
        from mcp.server.fastmcp import FastMCP
        from starlette.middleware import Middleware

        self.recorded_headers: list[dict[str, str]] = []
        records = self.recorded_headers

        mcp = FastMCP(name)

        @mcp.tool()
        def ping() -> str:
            return "pong-record"

        class _HeaderRecorder:
            def __init__(self, app, sink) -> None:  # noqa: ANN001
                self._app = app
                self._sink = sink

            async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
                if scope["type"] == "http":
                    self._sink.append(
                        {
                            key.decode("latin-1").lower(): value.decode("latin-1")
                            for key, value in scope.get("headers", [])
                        }
                    )
                await self._app(scope, receive, send)

        inner = mcp.streamable_http_app()
        # Inserted BEFORE the first request: Starlette builds its stack lazily,
        # and uvicorn keeps serving the app's own lifespan (session manager).
        inner.user_middleware.insert(0, Middleware(_HeaderRecorder, sink=records))
        self._app = inner
        self._inner = None
        self._thread = None

    def start(self) -> "RecordingMcpServer":
        import threading

        import uvicorn

        server = uvicorn.Server(uvicorn.Config(self._app, host=self.HOST, port=0, log_level="error"))
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self._inner = server

        def _run() -> None:
            server.run()

        self._thread = threading.Thread(target=_run, name="recording-mcp-server", daemon=True)
        self._thread.start()
        self._uvicorn_server = server
        return self

    @property
    def url(self) -> str:
        import time

        assert self._uvicorn_server is not None
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not self._uvicorn_server.started:
            time.sleep(0.05)
        if not self._uvicorn_server.started:
            raise RuntimeError("recording MCP server failed to start")
        port = self._uvicorn_server.servers[0].sockets[0].getsockname()[1]
        return f"http://{self.HOST}:{port}/mcp"

    def stop(self) -> None:
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seed_registry(stub_registry, *, url: str, auth_mode: str | None, headers: dict | None, auth_token: str | None):
    """Seed the registry stub with the single registered server under test."""
    entry: dict = {
        "name": SERVER_NAME,
        "transport": "http",
        "url": url,
        "headers": headers or {},
        "env": {},
        "auth_token": auth_token,
        "tool_count": 1,
        "source_type": "external",
    }
    if auth_mode is not None:
        entry["auth_mode"] = auth_mode
    stub_registry.reset(seed={SERVER_NAME: entry})


async def _run_assistant_with_target_tool(
    client, stub_llm, mcp_base_url, users, auth_headers: dict[str, str]
) -> dict:
    """Run the REAL built-in assistant so it loads/calls the registered server."""
    await configure_llm_credentials(client, users["a"]["headers"])
    stub_llm.script(
        assistant_tool_call("ping", {}),
        assistant_text("used the target tool"),
    )
    return await call_mcp_tool(
        mcp_base_url, auth_headers, "composable_agents_run", {"agent_name": BUILTIN_NAME, "message": "use the target"}
    )


def _assert_at_least_one_request_reached_target(records: list[dict]) -> None:
    assert records, (
        "no request reached the registered MCP server — the tool load failed "
        "(e.g. the unsupported auth_token key/McpToolLoadError or a dropped connection)"
    )


def _headers_on_target(records: list[dict]) -> dict[str, set[str]]:
    """Collect the set of observed values per header name across all requests."""
    observed: dict[str, set[str]] = {}
    for record in records:
        for name, value in record.items():
            observed.setdefault(name, set()).add(value)
    return observed


class TestUserTokenAuthMode:
    async def test_user_token_injects_bearer_jwt_when_authenticated_via_jwt(
        self, mcp_base_url, client, stub_llm, users, stub_registry, stub_jwks
    ):
        """REQ-004: `auth_mode: "user_token"` + JWT-authenticated run → the
        backend transmits `Authorization: Bearer <the user's JWT>` (merged with
        explicit headers)."""
        server = RecordingMcpServer().start()
        try:
            token = stub_jwks.issue_token(sub="user-a")
            _seed_registry(
                stub_registry,
                url=server.url,
                auth_mode="user_token",
                headers={"X-Static-Header": "static-value"},
                auth_token=None,
            )
            await _run_assistant_with_target_tool(
                client, stub_llm, mcp_base_url, users, {"Authorization": f"Bearer {token}"}
            )
            _assert_at_least_one_request_reached_target(server.recorded_headers)
            observed = _headers_on_target(server.recorded_headers)
            assert f"Bearer {token}" in observed.get("authorization", set()), (
                "auth_mode 'user_token' must inject the connected user's JWT as "
                f"Authorization: Bearer; observed authorization values: {observed.get('authorization')!r}"
            )
            assert "static-value" in observed.get("x-static-header", set()), (
                f"explicit configured headers must still be transmitted; observed: {observed}"
            )
        finally:
            server.stop()

    async def test_user_token_injects_api_key_when_authenticated_via_api_key(
        self, mcp_base_url, client, stub_llm, users, stub_registry
    ):
        """REQ-004: `auth_mode: "user_token"` + API-key-authenticated run → the
        backend transmits `X-API-Key: <the user's key>`."""
        server = RecordingMcpServer().start()
        try:
            _seed_registry(
                stub_registry,
                url=server.url,
                auth_mode="user_token",
                headers={},
                auth_token=None,
            )
            await _run_assistant_with_target_tool(client, stub_llm, mcp_base_url, users, users["a"]["headers"])
            _assert_at_least_one_request_reached_target(server.recorded_headers)
            observed = _headers_on_target(server.recorded_headers)
            assert users["a"]["key"] in observed.get("x-api-key", set()), (
                "auth_mode 'user_token' must inject the connected user's API key as "
                f"X-API-Key; observed x-api-key values: {observed.get('x-api-key')!r}"
            )
        finally:
            server.stop()

    async def test_injected_credential_wins_over_colliding_explicit_header(
        self, mcp_base_url, client, stub_llm, users, stub_registry
    ):
        """REQ-004 edge: an explicit header colliding with the injected
        credential's name is REPLACED by the user's credential (no duplicated or
        conflicting header)."""
        colliding_value = "cpk_explicit_colliding_value"
        server = RecordingMcpServer().start()
        try:
            _seed_registry(
                stub_registry,
                url=server.url,
                auth_mode="user_token",
                headers={"X-API-Key": colliding_value},
                auth_token=None,
            )
            await _run_assistant_with_target_tool(client, stub_llm, mcp_base_url, users, users["a"]["headers"])
            _assert_at_least_one_request_reached_target(server.recorded_headers)
            observed = _headers_on_target(server.recorded_headers)
            assert users["a"]["key"] in observed.get("x-api-key", set()), (
                "the injected user credential must take precedence over the colliding "
                f"explicit header; observed x-api-key values: {observed.get('x-api-key')!r}"
            )
            assert colliding_value not in observed.get("x-api-key", set()), (
                f"the colliding explicit value must not be transmitted; observed: {observed.get('x-api-key')!r}"
            )
        finally:
            server.stop()


class TestAuthTokenAsAuthorizationHeader:
    async def test_stored_auth_token_sent_as_bearer_header(self, mcp_base_url, client, stub_llm, users, stub_registry):
        """REQ-005: a stored static `auth_token` is transmitted as
        `Authorization: Bearer <token>` — never as the unsupported
        `auth_token` connection key (TypeError with langchain-mcp-adapters 0.3.0)."""
        static_token = "static-secret-token-42"
        server = RecordingMcpServer().start()
        try:
            _seed_registry(
                stub_registry,
                url=server.url,
                auth_mode="headers",
                headers={},
                auth_token=static_token,
            )
            await _run_assistant_with_target_tool(client, stub_llm, mcp_base_url, users, users["a"]["headers"])
            _assert_at_least_one_request_reached_target(server.recorded_headers)
            observed = _headers_on_target(server.recorded_headers)
            assert f"Bearer {static_token}" in observed.get("authorization", set()), (
                "the stored auth_token must be transmitted as 'Authorization: Bearer <value>'; "
                f"observed authorization values: {observed.get('authorization')!r}"
            )
        finally:
            server.stop()

    async def test_auth_token_user_jwt_placeholder_resolved_in_jwt_context(
        self, mcp_base_url, client, stub_llm, users, stub_registry, stub_jwks
    ):
        """REQ-005: `auth_token: "${USER_JWT}"` resolves to the connected
        user's JWT when the run is JWT-authenticated."""
        server = RecordingMcpServer().start()
        try:
            token = stub_jwks.issue_token(sub="user-a")
            _seed_registry(
                stub_registry,
                url=server.url,
                auth_mode="headers",
                headers={},
                auth_token="${USER_JWT}",
            )
            await _run_assistant_with_target_tool(
                client, stub_llm, mcp_base_url, users, {"Authorization": f"Bearer {token}"}
            )
            _assert_at_least_one_request_reached_target(server.recorded_headers)
            observed = _headers_on_target(server.recorded_headers)
            assert f"Bearer {token}" in observed.get("authorization", set()), (
                f'"${{USER_JWT}}" auth_token must resolve to the user\'s JWT; '
                f"observed authorization values: {observed.get('authorization')!r}"
            )
        finally:
            server.stop()

    async def test_auth_token_env_placeholder_resolved(
        self, mcp_base_url, client, stub_llm, users, stub_registry
    ):
        """REQ-005: env placeholders in `auth_token` are resolved (os.environ)."""
        os.environ[ENV_TOKEN_VAR] = ENV_TOKEN_VALUE
        server = RecordingMcpServer().start()
        try:
            _seed_registry(
                stub_registry,
                url=server.url,
                auth_mode="headers",
                headers={},
                auth_token=f"${{{ENV_TOKEN_VAR}}}",
            )
            await _run_assistant_with_target_tool(client, stub_llm, mcp_base_url, users, users["a"]["headers"])
            _assert_at_least_one_request_reached_target(server.recorded_headers)
            observed = _headers_on_target(server.recorded_headers)
            assert f"Bearer {ENV_TOKEN_VALUE}" in observed.get("authorization", set()), (
                f"the env placeholder must resolve to {ENV_TOKEN_VALUE!r}; "
                f"observed authorization values: {observed.get('authorization')!r}"
            )
        finally:
            server.stop()
            os.environ.pop(ENV_TOKEN_VAR, None)

    async def test_auth_mode_headers_default_keeps_static_headers(
        self, mcp_base_url, client, stub_llm, users, stub_registry
    ):
        """REQ-004 default (absent `auth_mode` = "headers"): the entered
        headers are sent as-is, nothing injected."""
        server = RecordingMcpServer().start()
        try:
            _seed_registry(
                stub_registry,
                url=server.url,
                auth_mode=None,
                headers={"X-API-Key": "cpk-registered-static-value"},
                auth_token=None,
            )
            await _run_assistant_with_target_tool(client, stub_llm, mcp_base_url, users, users["a"]["headers"])
            _assert_at_least_one_request_reached_target(server.recorded_headers)
            observed = _headers_on_target(server.recorded_headers)
            assert "cpk-registered-static-value" in observed.get("x-api-key", set()), (
                f"the static configured header must be transmitted as-is; observed: {observed}"
            )
            assert all(value != users["a"]["key"] for value in observed.get("x-api-key", set())), (
                "headers mode must NOT inject the user's credential"
            )
        finally:
            server.stop()
