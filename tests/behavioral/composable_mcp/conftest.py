"""Behavioral test infrastructure for the Composable MCP control plane.

Extends ``tests/behavioral/conftest.py`` (real app over real PostgreSQL/MinIO)
with the extra boundaries the MCP suite needs:

* ``mcp_base_url``      — session-scoped fixture serving the REAL
  ``src.main.app`` over a real uvicorn socket, so tests talk to the mounted
  streamable-HTTP MCP endpoint with the real ``mcp`` SDK client (no ASGI
  transport shortcuts). The app lifespan still runs; persistence init/close
  and the cron scheduler loop are no-oped because the parent session fixture
  already boots/owns them (double-boot would tear down shared state).
* ``StubJwksServer``    — mock of the outbound Logto JWKS boundary: a local
  RSA JWKS endpoint plus JWT minting, retargeting the production
  ``JwtAdapter``'s JWKS URL (external IdP is the ONLY thing mocked here).
* ``stub_registry``     — mock of the outbound ``mcp-raganything``
  registry REST boundary (masked list/get/create/update/validate + reveal
  exists but must never be exercised via MCP).
* ``StubMcpServer``     — a REAL local streamable-HTTP MCP server used as a
  user-registered MCP server for resolution/collision/degradation tests.
* MCP client helpers    — ``open_mcp_session`` / ``list_mcp_tool_names`` /
  ``call_mcp_tool`` / ``mcp_post_status`` built on the real ``mcp`` SDK.
* Scripted-LLM helpers  — ``assistant_tool_call`` / ``assistant_text`` for the
  ``StubLlmServer.script()`` capability added to the shared stub.

Everything else (routers, use cases, repositories, RLS, store) runs real.
"""

import asyncio
import contextlib
import json
import os
import threading
import time
from collections.abc import AsyncGenerator
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
import pytest_asyncio

JWT_AUDIENCE = os.environ.get("JWT_AUDIENCE", "test-audience")

# Registry base-URL configuration seam. The composable-agents backend must
# obtain the trusted ``mcp-raganything`` base URL from configuration (spec:
# "supplied through deployment configuration rather than hard-coded"). The
# tests publish the stub URL under this env var AND best-effort onto Settings
# attributes — a Green-phase implementation just needs to read one of them.
MCP_REGISTRY_BASE_URL_ENV = "MCP_REGISTRY_BASE_URL"

DELETION_TOOLS = [
    "composable_agents_delete",
    "composable_mcp_servers_delete",
    "composable_skills_delete",
    "composable_memories_delete",
    "composable_prompts_delete",
    "composable_models_delete",
    "composable_crons_delete",
    "composable_threads_delete",
]


# ---------------------------------------------------------------------------
# Outbound Logto JWKS boundary mock (+ JWT minting)
# ---------------------------------------------------------------------------
class StubJwksServer:
    """Thread-bound HTTP server serving a fixed RSA JWKS document.

    Mimics the shape of the production Logto JWKS endpoint so the REAL
    ``JwtAdapter`` can verify tokens minted by :meth:`issue_token`.
    """

    HOST = "127.0.0.1"
    JWKS_PATH = "/oidc/jwks"

    def __init__(self) -> None:
        import jwt
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        self._kid = "behavioral-mcp-kid"
        self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self._private_key.public_key()))
        jwk.update({"kid": self._kid, "alg": "RS256", "use": "sig"})
        self._jwks = {"keys": [jwk]}
        self._private_pem = self._private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        self._server: Any = None
        self._thread: threading.Thread | None = None

    # -- raw ASGI app ----------------------------------------------------
    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] != "http":
            return
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if msg["type"] != "http.request" or not msg.get("more_body"):
                break
        if scope["path"] == self.JWKS_PATH:
            payload = json.dumps(self._jwks).encode()
            await send(
                {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]}
            )
            await send({"type": "http.response.body", "body": payload})
            return
        await send({"type": "http.response.start", "status": 404, "headers": [(b"content-length", b"0")]})
        await send({"type": "http.response.body", "body": b""})

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        import uvicorn

        self._server = uvicorn.Config(self, host=self.HOST, port=0, log_level="error")
        server = uvicorn.Server(self._server)
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self._inner = server

        def _run() -> None:
            server.run()

        self._thread = threading.Thread(target=_run, name="stub-jwks-server", daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        assert self._inner is not None
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not self._inner.started:
            time.sleep(0.05)
        if not self._inner.started:
            raise RuntimeError("stub JWKS server failed to start")
        port = self._inner.servers[0].sockets[0].getsockname()[1]
        return f"http://{self.HOST}:{port}"

    # -- token minting ----------------------------------------------------
    def issue_token(
        self,
        sub: str = "user-a",
        audience: str = JWT_AUDIENCE,
        expires_at: int | None = None,
        issued_at: int | None = None,
    ) -> str:
        import jwt

        now = issued_at if issued_at is not None else int(time.time())
        claims: dict[str, Any] = {
            "sub": sub,
            "email": f"{sub}@example.com",
            "name": f"User {sub}",
            "username": sub,
            "created_at": now,
            "updated_at": now,
            "aud": audience,
            "iss": "http://stub-logto.invalid",
            "iat": now,
            "exp": expires_at if expires_at is not None else now + 3600,
        }
        return jwt.encode(claims, self._private_pem, algorithm="RS256", headers={"kid": self._kid})


_jwks_server: StubJwksServer | None = None
_jwks_lock = threading.Lock()


def get_stub_jwks() -> StubJwksServer:
    """Process-wide stub JWKS server, started on first use (like the LLM stub)."""
    global _jwks_server
    with _jwks_lock:
        if _jwks_server is None:
            _jwks_server = StubJwksServer()
            _jwks_server.start()
        return _jwks_server


@pytest.fixture
def jwt_users() -> dict[str, dict[str, str]]:
    """Per-user ``Authorization: Bearer`` headers backed by real signatures."""
    jwks = get_stub_jwks()
    return {
        "a": {"Authorization": f"Bearer {jwks.issue_token(sub='user-a')}"},
        "b": {"Authorization": f"Bearer {jwks.issue_token(sub='user-b')}"},
    }


@pytest.fixture
def stub_jwks() -> StubJwksServer:
    """The process-wide JWKS stub (issued through the conftest's own singleton).

    Test modules MUST mint tokens through this fixture rather than importing
    ``get_stub_jwks`` directly: with pytest's prepend import mode and the
    package-less ``tests`` tree, an absolute import of this conftest creates a
    SECOND module instance whose module-global singleton would start a SECOND
    stub JWKS server bound to a different RSA key — tokens would then never
    verify against the JWKS retargeted onto the production adapter.
    """
    return get_stub_jwks()


# ---------------------------------------------------------------------------
# Real app over a real socket + MCP SDK client helpers
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def mcp_base_url(real_app_and_db: dict) -> AsyncGenerator[str, None]:
    """Serve the REAL app on a real local socket for the whole session.

    The app lifespan is ALLOWED to run (an integrated MCP mount needs it per
    the spec) but ``init_persistence``/``close_persistence``, the cron
    scheduler loop and the MCP loader teardown are no-oped: the parent
    session fixture owns those resources — a second boot/shutdown would rip
    the shared engine out from under every other behavioral test.
    """
    import httpx
    import uvicorn

    # 1) Retarget the production JwtAdapter at the local JWKS stub — the Logto
    #    IdP is an outbound external system and is the ONLY thing mocked here.
    jwks = get_stub_jwks()
    import src.dependencies as deps

    assert deps._root.jwt_adapter is not None
    adapter_obj = deps._root.jwt_adapter
    adapter_obj._jwks_url = f"{jwks.base_url}{StubJwksServer.JWKS_PATH}"
    old_client = adapter_obj._jwks_http_client
    adapter_obj._jwks_http_client = httpx.AsyncClient(timeout=30.0)
    adapter_obj._jwks_cache.clear()
    if old_client is not None:
        with contextlib.suppress(Exception):
            await old_client.aclose()
    # AuthService holds no copies — it delegates to the port each request.

    # 2) uvicorn with lifespan ON, persistence/scheduler/loader lifecycle no-oped.
    import src.main as src_main

    orig_init = src_main.init_persistence
    orig_close = src_main.close_persistence
    orig_sched = src_main.start_scheduler_loop
    orig_loader = src_main.mcp_tool_loader

    class _NoopClosable:
        async def close(self) -> None:  # noqa: D102
            return None

    src_main.init_persistence = lambda: _noop_awaitable()  # type: ignore[assignment]
    src_main.close_persistence = lambda: _noop_awaitable()  # type: ignore[assignment]
    src_main.start_scheduler_loop = lambda _uc: None  # type: ignore[assignment]
    src_main.mcp_tool_loader = _NoopClosable()  # type: ignore[assignment]

    server = uvicorn.Server(uvicorn.Config(real_app_and_db["app"], host="127.0.0.1", port=0, log_level="warning"))
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    serve_task: Any = None
    try:
        serve_task = asyncio.create_task(server.serve())
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not server.started:
            await asyncio.sleep(0.05)
        if not server.started:
            raise RuntimeError("uvicorn test server failed to start")
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        if serve_task is not None:
            server.should_exit = True
            with contextlib.suppress(Exception):
                await asyncio.wait_for(serve_task, timeout=10)
        src_main.init_persistence = orig_init
        src_main.close_persistence = orig_close
        src_main.start_scheduler_loop = orig_sched
        src_main.mcp_tool_loader = orig_loader


def _noop_awaitable() -> Any:
    async def _noop() -> None:
        return None

    return _noop()


@contextlib.asynccontextmanager
async def open_mcp_session(base_url: str, headers: dict[str, str] | None = None) -> AsyncGenerator[Any, None]:
    """Open a REAL MCP client session against the mounted streamable-HTTP app."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with (
        streamablehttp_client(f"{base_url.rstrip('/')}/mcp", headers=headers or {}) as (
            read,
            write,
            _get_session_id,
        ),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


async def list_mcp_tool_names(base_url: str, headers: dict[str, str] | None = None) -> list[str] | None:
    """Return the sorted tool names from ``tools/list`` or ``None`` when denied.

    ``None`` means the MCP handshake/listing failed (unauthorized or transport
    error) — i.e. NO tool data reached the client.
    """
    try:
        async with open_mcp_session(base_url, headers) as session:
            result = await session.list_tools()
            return sorted(tool.name for tool in result.tools)
    except Exception:  # noqa: BLE001 — denial surfaces as a transport/protocol error
        return None


async def call_mcp_tool(base_url: str, headers: dict[str, str], name: str, arguments: dict | None = None) -> dict:
    """Call an MCP tool and return ``{"ok": bool, "result": <json-safe dict>}``.

    Structured content is preferred; unstructured text is parsed as JSON
    (falling back to a ``{"text": ...}`` envelope).
    """
    async with open_mcp_session(base_url, headers) as session:
        result = await session.call_tool(name, arguments or {})
        payload: Any = None
        if getattr(result, "structuredContent", None) is not None:
            payload = result.structuredContent
        else:
            texts = [c.text for c in result.content if getattr(c, "text", None)]
            if texts:
                try:
                    payload = json.loads(texts[0])
                except json.JSONDecodeError:
                    payload = {"text": "\n".join(texts)}
            else:
                payload = {}
        return {"ok": not getattr(result, "isError", False), "result": payload}


async def mcp_post_status(base_url: str, headers: dict[str, str] | None = None, method: str = "initialize") -> Any:
    """Raw JSON-RPC POST against ``/mcp`` — used to pin HTTP status codes."""
    import httpx

    body: dict[str, Any] = {"jsonrpc": "2.0", "id": 0, "method": method}
    if method == "initialize":
        body["params"] = {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "behavioral-probe", "version": "0.0.0"},
        }
    elif method == "tools/call":
        body["params"] = {"name": "composable_agents_list", "arguments": {}}
    else:
        body["params"] = {}
    async with httpx.AsyncClient(timeout=15.0) as client:
        return await client.post(
            f"{base_url.rstrip('/')}/mcp",
            headers={"Accept": "application/json, text/event-stream", **(headers or {})},
            json=body,
        )


# ---------------------------------------------------------------------------
# Outbound mcp-raganything registry REST boundary mock
# ---------------------------------------------------------------------------
def _masked_entry(entry: dict) -> dict:
    """Mask an entry exactly like the real registry ``_mask`` helper."""
    masked = dict(entry)
    masked["auth_token"] = None
    masked["headers"] = {}
    masked["env"] = {}
    return masked


class StubRegistryApp:
    """Emulates the mcp-raganything MCP-server registry REST API (masked)."""

    PREFIX = "/api/v1/mcp/servers"

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.requests: list[dict[str, Any]] = []  # method, path, authorization, api_key
        self.fail = False
        self._lock = asyncio.Lock()

    def reset(self, fail: bool = False, seed: dict[str, dict] | None = None) -> None:
        import datetime as dt

        self.fail = fail
        self.requests = []
        if seed is None:
            now = dt.datetime.now(dt.UTC).isoformat()
            seed = {
                "alpha": {
                    "name": "alpha",
                    "transport": "http",
                    "url": "http://127.0.0.1:59998/mcp",
                    "headers": {"X-Internal-Header": "alpha-plain-header-secret"},
                    "env": {"ALPHA_TOKEN": "alpha-plain-env-secret"},
                    "auth_token": "alpha-plain-token-secret",
                    "tool_count": 2,
                    "source_type": "external",
                    "openapi_url": None,
                    "created_at": now,
                    "updated_at": now,
                }
            }
        self.entries = dict(seed)

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] != "http":
            return
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if msg["type"] != "http.request" or not msg.get("more_body"):
                break
        raw_headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        await self._record(scope, raw_headers, body)

        if self.fail:
            await self._json(send, 503, {"detail": "registry unavailable"})
            return
        return await self._route(scope, body, send)

    async def _record(self, scope, headers: dict[str, str], body: bytes) -> None:  # noqa: ANN001
        async with self._lock:
            self.requests.append(
                {
                    "method": scope["method"],
                    "path": scope["path"],
                    "authorization": headers.get("authorization"),
                    "api_key": headers.get("x-api-key"),
                    "body": body.decode() if body else "",
                }
            )

    async def _route(self, scope, body: bytes, send) -> None:  # noqa: ANN001
        method = scope["method"]
        parsed = urlparse(scope["path"])
        path = parsed.path
        query = parse_qs(parsed.query)
        _ = query  # registry list endpoint currently takes no filters
        segments = [s for s in path.split("/") if s]
        payload = json.loads(body) if body else {}

        if path == self.PREFIX:
            if method == "GET":
                await self._json(send, 200, [_masked_entry(e) for e in self.entries.values()])
                return
            if method == "POST":
                name = payload.get("name", "")
                if name in self.entries:
                    await self._json(send, 409, {"detail": "already registered"})
                    return
                import datetime as dt

                entry = {
                    **payload,
                    "created_at": dt.datetime.now(dt.UTC).isoformat(),
                    "updated_at": dt.datetime.now(dt.UTC).isoformat(),
                }
                self.entries[name] = entry
                await self._json(send, 201, _masked_entry(entry))
                return

        if path == f"{self.PREFIX}/validate" and method == "POST":
            # The REAL mcp-raganything contract (BUG-001): the deployed
            # registry answers McpServerValidateResponse{tool_count} on a
            # validated candidate — it carries neither `name` nor `valid`.
            await self._json(send, 200, {"tool_count": 1})
            return

        if (
            len(segments) == 6
            and segments[:3] == ["api", "v1", "mcp"]
            and segments[3] == "servers"
            and segments[5] == "reveal"
        ):
            # Canonical reveal shape (mirrors the real registry PATCH routes).
            entry = self.entries.get(segments[4])
            if entry is None:
                await self._json(send, 404, {"detail": "not found"})
                return
            await self._json(send, 200, entry)
            return

        if len(segments) == 5 and segments[:3] == ["api", "v1", "mcp"]:
            name = segments[4]
            entry = self.entries.get(name)
            if segments[3] == "servers":
                if entry is None:
                    await self._json(send, 404, {"detail": "not found"})
                    return
                if method == "GET":
                    await self._json(send, 200, _masked_entry(entry))
                    return
                if method == "PUT":
                    # Emulates the real PARTIAL update contract (BUG-001/002/003
                    # fix): only the provided fields change; unprovided fields
                    # (incl. auth_token/headers/env) are preserved from the
                    # stored entry; tool_count survives an unchanged URL.
                    updated = {**entry, **payload}
                    if "url" in payload and payload["url"] != entry.get("url"):
                        # URL changed → the real registry re-validates and recounts.
                        updated["tool_count"] = payload.get("tool_count", 0)
                    merged_for_coherence = updated
                    if (
                        merged_for_coherence.get("source_type", "external") == "openapi"
                    ) and not merged_for_coherence.get("openapi_url"):
                        await self._json(
                            send,
                            422,
                            {
                                "detail": [
                                    {
                                        "type": "value_error",
                                        "loc": ["body"],
                                        "msg": "Value error, 'openapi_url' is required when source_type='openapi'",
                                    }
                                ]
                            },
                        )
                        return
                    if merged_for_coherence.get(
                        "source_type", "external"
                    ) == "external" and not merged_for_coherence.get("url"):
                        await self._json(
                            send,
                            422,
                            {
                                "detail": [
                                    {
                                        "type": "value_error",
                                        "loc": ["body"],
                                        "msg": "Value error, 'url' is required when source_type='external'",
                                    }
                                ]
                            },
                        )
                        return
                    self.entries[name] = updated
                    await self._json(send, 200, _masked_entry(updated))
                    return
                if method == "DELETE":
                    del self.entries[name]
                    await self._json(send, 204, None)
                    return
            if segments[3] == "reveal" and method == "GET" and entry is not None:
                await self._json(send, 200, entry)
                return

        await self._json(send, 404, {"detail": "not found"})

    @staticmethod
    async def _json(send, status: int, payload: Any) -> None:  # noqa: ANN001
        body = b"" if status == 204 else json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})


class StubRegistryServer:
    """Thread-bound uvicorn server fronting :class:`StubRegistryApp`."""

    HOST = "127.0.0.1"

    def __init__(self) -> None:
        self.app = StubRegistryApp()
        self._thread: threading.Thread | None = None
        self._inner: Any = None

    def start(self) -> None:
        import uvicorn

        server = uvicorn.Server(uvicorn.Config(self.app, host=self.HOST, port=0, log_level="error"))
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self._inner = server

        def _run() -> None:
            server.run()

        self._thread = threading.Thread(target=_run, name="stub-registry-server", daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        assert self._inner is not None
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not self._inner.started:
            time.sleep(0.05)
        if not self._inner.started:
            raise RuntimeError("stub registry server failed to start")
        port = self._inner.servers[0].sockets[0].getsockname()[1]
        return f"http://{self.HOST}:{port}"

    def reset(self, fail: bool = False, seed: dict[str, dict] | None = None) -> None:
        self.app.reset(fail=fail, seed=seed)


_registry_server: StubRegistryServer | None = None
_registry_lock = threading.Lock()


@pytest.fixture
def stub_registry() -> StubRegistryServer:
    """The registry REST boundary mock; publishes its URL to the config seam."""
    global _registry_server
    with _registry_lock:
        if _registry_server is None:
            _registry_server = StubRegistryServer()
            _registry_server.start()
    stub = _registry_server
    assert stub is not None
    stub.reset()

    os.environ[MCP_REGISTRY_BASE_URL_ENV] = stub.base_url
    # Best-effort Settings seam (implementation will settle the attribute name).
    try:
        import src.dependencies as deps

        for attr in (
            "mcp_registry_base_url",
            "mcp_registry_url",
            "mcp_registry_service_url",
            "raganything_base_url",
        ):
            try:
                setattr(deps.settings, attr, stub.base_url)
            except Exception:  # noqa: BLE001 — unknown/frozen settings field
                continue
    except Exception:  # noqa: BLE001 — deps not imported yet
        pass

    yield stub
    stub.reset()


# ---------------------------------------------------------------------------
# Real local MCP servers (external user-registered MCP boundary = REAL here)
# ---------------------------------------------------------------------------
class StubMcpServer:
    """A REAL streamable-HTTP MCP server exposing one small tool."""

    HOST = "127.0.0.1"

    def __init__(self, name: str, tool_name: str = "ping", tool_result: str = "pong") -> None:
        from mcp.server.fastmcp import FastMCP

        self.name = name
        self.tool_name = tool_name
        mcp = FastMCP(name)

        @mcp.tool(name=tool_name)
        def ping() -> str:  # noqa: ANN202 — decorated FastMCP tool
            return tool_result

        self._app = mcp.streamable_http_app()
        self._inner: Any = None
        self._thread: threading.Thread | None = None

    def start(self) -> "StubMcpServer":
        import uvicorn

        server = uvicorn.Server(uvicorn.Config(self._app, host=self.HOST, port=0, log_level="error"))
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self._inner = server

        def _run() -> None:
            server.run()

        self._thread = threading.Thread(target=_run, name=f"stub-mcp-{self.name}", daemon=True)
        self._thread.start()
        return self

    @property
    def url(self) -> str:
        assert self._inner is not None
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not self._inner.started:
            time.sleep(0.05)
        if not self._inner.started:
            raise RuntimeError(f"stub MCP server {self.name} failed to start")
        port = self._inner.servers[0].sockets[0].getsockname()[1]
        return f"http://{self.HOST}:{port}/mcp"

    def stop(self) -> None:
        if self._inner is not None:
            self._inner.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Scripted-LLM helpers (consume StubLlmServer.script())
# ---------------------------------------------------------------------------
def assistant_tool_call(name: str, arguments: dict, text: str | None = None, call_id: str = "call_1") -> dict:
    """An OpenAI assistant message carrying exactly one tool call."""
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        ],
    }


def assistant_text(text: str) -> dict:
    """A plain final assistant message."""
    return {"role": "assistant", "content": text}
