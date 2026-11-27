"""In-process stub of the EXTERNAL mcp-raganything REST boundary.

composable-agents is a client of mcp-raganything for two REST APIs:

* the MCP-server registry (``/api/v1/mcp/servers``, masked projection +
  authenticated ``/reveal``) — established behavior wired through the
  ``HttpMcpRegistryClient`` adapter (pattern of the composable-mcp suite's
  ``StubRegistryApp``);
* the RAG spaces API (``/api/v1/spaces``, REQ-010 contract): idempotent
  ``POST /api/v1/spaces`` (``{"agent_slug": ...}``, 201 on create / 200 when
  the space already exists, 422 on an invalid ``agent_slug`` charset per
  REQ-002), ``GET /api/v1/spaces`` (list), and strict-cascade
  ``DELETE /api/v1/spaces/{agent_slug}`` (204, 404 for a nonexistent space —
  decided semantics, no partial deletion).

Every request is recorded (method, path, forwarded credential headers, body)
so tests can assert PROVISIONING CALLS made by the agent-lifecycle use cases
(REQ-018) through the real agent CRUD routes without the external service.

Spaces state is a plain in-memory set. ``reset(seed_spaces=[...])`` pre-seeds
spaces so tests can exercise the idempotent-replay contract (the adapter must
treat the 200 "already exists" answer as a success). ``fail`` mode 503s every
request (drives the best-effort provisioning failure paths).
"""

import datetime as dt
import json
import re
import threading
import time
from typing import Any
from urllib.parse import urlparse

import uvicorn

#: mcp-raganything's agent_slug charset contract (REQ-002/REQ-015).
AGENT_SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,99}$")

SPACES_PATH = "/api/v1/spaces"
REGISTRY_PREFIX = "/api/v1/mcp/servers"


class RagAnythingStubApp:
    """Raw ASGI app emulating the mcp-raganything REST surfaces used here."""

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.spaces: set[str] = set()
        self.requests: list[dict[str, Any]] = []  # every request (both APIs)
        self.space_responses: list[dict[str, Any]] = []  # spaces-call answers
        self.fail = False
        self.reject_slugs: set[str] = set()  # slugs scripted to answer 422 (REQ-002 reject path)
        self._lock = threading.Lock()

    def reset(
        self,
        *,
        fail: bool = False,
        seed: dict[str, dict] | None = None,
        seed_spaces: list[str] | None = None,
        reject_slugs: list[str] | None = None,
    ) -> None:
        with self._lock:
            self.fail = fail
            self.requests = []
            self.space_responses = []
            self.spaces = set(seed_spaces or [])
            self.entries = dict(seed or {})
            self.reject_slugs = set(reject_slugs or [])

    # -- observation helpers used by the tests -----------------------------
    def space_calls(self) -> list[dict[str, Any]]:
        """Every request recorded against ``/api/v1/spaces`` (parsed)."""
        calls: list[dict[str, Any]] = []
        for request in self.requests:
            path = urlparse(request["path"]).path
            if not path.startswith(SPACES_PATH):
                continue
            slug = path[len(SPACES_PATH) :].strip("/") or ""
            if request["method"] == "POST":
                try:
                    payload = json.loads(request["body"]) if request["body"] else {}
                except json.JSONDecodeError:
                    payload = {}
                slug = str(payload.get("agent_slug") or slug)
            calls.append(
                {
                    "method": request["method"],
                    "path": path,
                    "agent_slug": slug,
                    "api_key": request["api_key"],
                    "authorization": request["authorization"],
                    "body": request["body"],
                }
            )
        return calls

    def call_slugs(self, method: str) -> list[str]:
        return [call["agent_slug"] for call in self.space_calls() if call["method"] == method]

    # -- ASGI ----------------------------------------------------------------
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
            await self._json(send, 503, {"detail": "mcp-raganything unavailable"})
            return
        return await self._route(scope, body, send)

    async def _record(self, scope, headers: dict[str, str], body: bytes) -> None:  # noqa: ANN001
        # Synchronous critical section: a plain threading.Lock suffices.
        with self._lock:
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
        path = urlparse(scope["path"]).path

        if path == SPACES_PATH:
            await self._handle_spaces(method, body, send)
            return
        if path.startswith(SPACES_PATH + "/") and method == "DELETE":
            await self._handle_space_delete(path[len(SPACES_PATH) + 1 :], send)
            return

        return await self._route_registry(method, path, body, send)

    # -- spaces API (REQ-010 contract) --------------------------------------
    async def _handle_spaces(self, method: str, body: bytes, send) -> None:  # noqa: ANN001
        if method == "POST":
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                payload = {}
            slug = str(payload.get("agent_slug") or "")
            if slug in self.reject_slugs:
                # Scripted rejection (e.g. the REQ-002 charset contract): the
                # space is never created and the answer is 422.
                self._track(slug, 422)
                await self._json(send, 422, {"detail": f"declined slug {slug!r}"})
                return
            if not AGENT_SLUG_PATTERN.match(slug):
                self._track(slug, 422)
                await self._json(send, 422, {"detail": f"invalid agent_slug {slug!r}"})
                return
            fresh = slug not in self.spaces
            if fresh:
                self.spaces.add(slug)
            status = 201 if fresh else 200
            self._track(slug, status)
            await self._json(send, status, {"agent_slug": slug, "user_id": "stub", "working_dirs": []})
            return
        if method == "GET":
            await self._json(send, 200, sorted(self.spaces))
            return
        await self._json(send, 405, {"detail": "method not allowed"})

    async def _handle_space_delete(self, slug: str, send) -> None:  # noqa: ANN001
        if slug not in self.spaces:
            self._track(slug, 404)
            # Strict REST semantics (REQ-010): 404 and NO partial deletion.
            await self._json(send, 404, {"detail": "space not found"})
            return
        self.spaces.discard(slug)
        self._track(slug, 204)
        await self._json(send, 204, None)

    def _track(self, slug: str, status: int) -> None:
        with self._lock:
            self.space_responses.append({"agent_slug": slug, "status": status})

    def response_statuses(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.space_responses)

    # -- registry API (masked projection + reveal, minimal) ------------------
    @staticmethod
    def _masked_entry(entry: dict[str, Any]) -> dict[str, Any]:
        masked = dict(entry)
        masked["auth_token"] = None
        masked["headers"] = {}
        masked["env"] = {}
        return masked

    async def _route_registry(self, method: str, path: str, body: bytes, send) -> None:  # noqa: ANN001
        if path == REGISTRY_PREFIX and method == "GET":
            await self._json(send, 200, [self._masked_entry(entry) for entry in self.entries.values()])
            return
        if path == REGISTRY_PREFIX and method == "POST":
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                payload = {}
            name = str(payload.get("name") or "")
            entry = {
                "name": name,
                "transport": str(payload.get("transport") or "http"),
                "url": payload.get("url"),
                "command": payload.get("command"),
                "args": list(payload.get("args") or []),
                "headers": dict(payload.get("headers") or {}),
                "env": dict(payload.get("env") or {}),
                "auth_token": payload.get("auth_token"),
                "auth_mode": str(payload.get("auth_mode") or "headers"),
                "tool_count": 1,
                "source_type": str(payload.get("source_type") or "external"),
                "openapi_url": payload.get("openapi_url"),
                "created_at": dt.datetime.now(dt.UTC).isoformat(),
                "updated_at": dt.datetime.now(dt.UTC).isoformat(),
            }
            self.entries[name] = entry
            await self._json(send, 201, self._masked_entry(entry))
            return
        if path == REGISTRY_PREFIX + "/validate" and method == "POST":
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                payload = {}
            # The deployed registry answers a validated candidate with the
            # McpServerValidationReport shape (name/valid/tool_count).
            await self._json(
                send,
                200,
                {"name": str(payload.get("name") or ""), "valid": True, "tool_count": 1, "errors": []},
            )
            return
        segments = [segment for segment in path.split("/") if segment]
        if len(segments) == 6 and segments[:3] == ["api", "v1", "mcp"] and segments[3] == "servers":
            name = segments[4]
            entry = self.entries.get(name)
            if entry is not None and segments[5] == "reveal" and method == "GET":
                # Revealed (unmasked) execution configuration (REQ-037).
                await self._json(send, 200, entry)
                return
        if len(segments) == 5 and segments[:3] == ["api", "v1", "mcp"] and segments[3] == "servers":
            name = segments[4]
            entry = self.entries.get(name)
            if entry is not None and method == "GET":
                await self._json(send, 200, self._masked_entry(entry))
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


class RagAnythingStubServer:
    """Thread-bound uvicorn server fronting :class:`RagAnythingStubApp`."""

    HOST = "127.0.0.1"

    def __init__(self) -> None:
        self.app = RagAnythingStubApp()
        self._inner: Any = None

    def start(self) -> None:
        server = uvicorn.Server(uvicorn.Config(self.app, host=self.HOST, port=0, log_level="error", loop="asyncio"))
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self._thread = None
        self._inner = server

        def _run() -> None:
            server.run()

        import threading

        self._thread = threading.Thread(target=_run, name="stub-raganything", daemon=True)
        self._thread.start()
        # Do not hand the port out before the socket accepts connections.
        self._wait_until_ready()

    def _wait_until_ready(self, timeout: float = 10.0) -> None:
        import socket

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            servers = getattr(self._inner, "servers", None)
            if not servers:
                # uvicorn binds asynchronously on its server thread.
                time.sleep(0.05)
                continue
            with socket.socket() as sock:
                if sock.connect_ex((self.HOST, self.port)) == 0:
                    return
            time.sleep(0.05)
        raise RuntimeError("stub mcp-raganything server did not start")

    @property
    def port(self) -> int:
        servers = getattr(self._inner, "servers", None)
        assert servers, "stub server not started"
        return servers[0].sockets[0].getsockname()[1]

    @property
    def base_url(self) -> str:
        return f"http://{self.HOST}:{self.port}"

    def reset(
        self,
        *,
        fail: bool = False,
        seed: dict[str, dict] | None = None,
        seed_spaces: list[str] | None = None,
        reject_slugs: list[str] | None = None,
    ) -> None:
        self.app.reset(fail=fail, seed=seed, seed_spaces=seed_spaces, reject_slugs=reject_slugs)
