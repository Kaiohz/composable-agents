"""In-process stub of an EXTERNAL HTTP MCP server that records request headers.

This mock lives at the outbound MCP-server network boundary, which is an
EXTERNAL system from composable-agents' perspective (the tools live in
mcp-raganything / user-registered servers). Everything inside composable-agents
— the chat/one-shot chains, the MCP tool loader/client adapters, the agent
registry, the runner, the repositories — runs REAL.

The stub is a real stateless FastMCP server (the ``mcp`` package ships it)
exposing a trivial ``ping`` tool, and records EVERY inbound HTTP request:

* ``method`` / ``path`` — transport-level classification,
* ``headers`` — lowercased request headers (the injected
  ``X-Composable-Agent`` header, configured YAML headers, credentials),
* ``body`` — the raw request body (used to classify the JSON-RPC phase:
  ``initialize`` = tool-load time, ``tools/call`` = tool-execution time).

It exists to prove REQ-016/REQ-017 behaviorally: the header a platform agent
injects on its outbound MCP connections must be observable at the receiving
server (stub) — on both the tool-load connection and the tool-call connection —
and the built-in assistant must send NO such header (REQ-019 personal scope).

Served over streamable-HTTP on a dedicated uvicorn daemon thread with a
DYNAMIC port (``port=0``), exactly like the user-registered ``StubMcpServer``
of the composable-mcp suite; agent YAML ``mcp_servers[].url`` points at
:http:attr:`url` (``http://127.0.0.1:<port>/mcp``).
"""

import asyncio
import json
import socket
import threading
import time
from typing import Any

import uvicorn


def _json_rpc_methods(body: bytes) -> list[str]:
    """Extract the JSON-RPC ``method`` (or methods, for batched bodies)."""
    if not body:
        return []
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return [str(item.get("method", "")) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        method = payload.get("method")
        return [str(method)] if method else []
    return []


class _Recorder:
    """Thread-safe request recorder (stub thread writes, test thread reads).

    The critical sections are synchronous (no awaits inside the lock), so a
    plain ``threading.Lock`` is sufficient.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def record(self, method: str, path: str, headers: dict[str, str], body: bytes) -> None:
        with self._lock:
            self.requests.append(
                {
                    "method": method,
                    "path": path,
                    "headers": headers,
                    "body": body,
                    "json_methods": _json_rpc_methods(body),
                }
            )

    def reset(self) -> None:
        with self._lock:
            self.requests = []

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.requests)

    # -- observation helpers used by the tests -----------------------------
    def agent_header_values(self) -> set[str]:
        """Every distinct ``X-Composable-Agent`` value observed (case-agnostic)."""
        values: set[str] = set()
        for request in self.snapshot():
            value = request["headers"].get("x-composable-agent")
            if value:
                values.add(value.strip())
        return values

    def requests_with(
        self,
        *,
        json_method: str | None = None,
        header: str | None = None,
        header_value: str | None = None,
    ) -> list[dict[str, Any]]:
        """Requests matching a JSON-RPC method and/or an exact header value.

        ``header`` is matched case-insensitively (HTTP semantics);
        ``header_value`` is compared verbatim.
        """
        matched: list[dict[str, Any]] = []
        for request in self.snapshot():
            if json_method is not None and json_method not in request["json_methods"]:
                continue
            headers = {key.lower(): value for key, value in request["headers"].items()}
            if header is not None and header.lower() not in headers:
                continue
            if header_value is not None and headers.get((header or "").lower()) != header_value:
                continue
            matched.append(request)
        return matched

    def missing_header_requests(self, header: str = "x-composable-agent") -> list[dict[str, Any]]:
        """Requests that arrived WITHOUT the given header at all."""
        return [
            request for request in self.snapshot() if header.lower() not in {key.lower() for key in request["headers"]}
        ]


class _RecordingWrapper:
    """ASGI wrapper recording every HTTP request, then delegating unchanged.

    Request bodies are buffered and replayed to the wrapped (FastMCP streamable)
    app, which keeps reading its own ``receive`` stream transparently.
    Only bodied requests (POST/PUT/DELETE) are buffered; other verbs are passed
    through untouched (headers recorded, body ``b""``).
    """

    def __init__(self, app: Any, recorder: _Recorder) -> None:
        self._app = app
        self._recorder = recorder

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        if scope["method"] in ("POST", "PUT", "PATCH", "DELETE"):
            buffered: list[dict[str, Any]] = []
            while True:
                message = await receive()
                buffered.append(message)
                if message.get("type") == "http.request" and not message.get("more_body"):
                    break
                if message.get("type") == "http.disconnect":
                    break

            headers = {k.decode().lower(): v.decode(errors="replace") for k, v in scope.get("headers", [])}
            body = b"".join(m.get("body", b"") for m in buffered if m.get("type") == "http.request")
            self._recorder.record(scope["method"], scope["path"], headers, body)

            # Replay the buffered request body, then hand the real post-body
            # stream through (the streamable transport's disconnect-listener
            # awaits a genuine ``http.disconnect`` — never synthesize one).
            queue: asyncio.Queue = asyncio.Queue()

            async def _pump() -> None:
                while True:
                    try:
                        message = await receive()
                    except Exception:
                        await queue.put({"type": "http.disconnect"})  # pragma: no cover
                        return
                    await queue.put(message)
                    if message.get("type") == "http.disconnect":
                        return

            pump = asyncio.create_task(_pump())
            index = 0
            try:

                async def replay_receive() -> dict[str, Any]:
                    nonlocal index
                    if index < len(buffered):
                        message = buffered[index]
                        index += 1
                        return message
                    return await queue.get()

                await self._app(scope, replay_receive, send)
            finally:
                pump.cancel()
            return

        # Bodyless verb: record and delegate the raw streams.
        headers = {k.decode().lower(): v.decode(errors="replace") for k, v in scope.get("headers", [])}
        self._recorder.record(scope["method"], scope["path"], headers, b"")
        await self._app(scope, receive, send)


class HeaderRecordingMcpStubServer:
    """Runs the recording stub on a dynamic-port uvicorn daemon thread."""

    HOST = "127.0.0.1"

    def __init__(self) -> None:
        from mcp.server.fastmcp import FastMCP

        self.recorder = _Recorder()
        mcp = FastMCP("HeaderStub", stateless_http=True)

        @mcp.tool()
        def ping() -> str:  # noqa: ANN202 — decorated FastMCP tool
            return "pong"

        self._app = _RecordingWrapper(mcp.streamable_http_app(), self.recorder)
        self._server: Any = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "HeaderRecordingMcpStubServer":
        # loop="asyncio" pins the stub's server loop to the stdlib asyncio
        # policy: uvicorn's "auto" loop factory installs uvloop GLOBALLY
        # (asyncio.set_event_loop_policy) from this server thread, which
        # destabilizes pytest-asyncio's session runner loops (same rationale
        # as the chat-attachments stub).
        self._server = uvicorn.Server(
            uvicorn.Config(self._app, host=self.HOST, port=0, log_level="error", loop="asyncio")
        )

        def _run() -> None:
            self._server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
            self._server.run()

        self._thread = threading.Thread(target=_run, name="stub-mcp-header-recorder", daemon=True)
        self._thread.start()
        self._wait_until_ready()
        return self

    def _wait_until_ready(self, timeout: float = 10.0) -> None:
        """Poll the dynamic port until it accepts connections (or raise)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            servers = getattr(self._server, "servers", None)
            if not servers:
                # uvicorn binds asynchronously on its server thread.
                time.sleep(0.05)
                continue
            with socket.socket() as sock:
                if sock.connect_ex((self.HOST, self.port)) == 0:
                    return
            time.sleep(0.05)
        raise RuntimeError("stub recording MCP server did not start")

    @property
    def port(self) -> int:
        servers = getattr(self._server, "servers", None)
        assert servers, "stub server not started"
        return servers[0].sockets[0].getsockname()[1]

    @property
    def url(self) -> str:
        return f"http://{self.HOST}:{self.port}/mcp"

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)

    # -- state -------------------------------------------------------------
    def reset(self) -> None:
        """Clear the recording state (per-test)."""
        self.recorder.reset()
