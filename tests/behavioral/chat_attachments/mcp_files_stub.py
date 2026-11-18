"""In-process stub of the EXTERNAL Files MCP server (RAGAnything ``/files/mcp``).

This mock lives at the MCP-server network boundary, which is an EXTERNAL
system from composable-agents' perspective (REQ-029 spec: the tools live in
``mcp-raganything``). Everything inside composable-agents — the chat router,
the attachments service, the MCP tool loader/client adapters, the agent
registry, the runner, the repositories — runs REAL.

The stub is a real FastMCP server (the ``mcp`` package ships it) exposing the
same tools as the production Files MCP server:

* ``temporary_upload_file(file_name, content_base64, content_type, prefix)``
  → returns ``{"object_name", "size", "content_type"}`` (REQ-033 contract),
  recording every call (decoded bytes + prefix) for assertions.
* ``temporary_delete_file(object_path)`` — records the delete target.
* ``list_files`` / ``list_folders`` / ``read_file`` — required by the agent
  tool loader (REQ-037: the LLM keeps access to the normal tools).

Modes:
* ``normal`` — behave like the production server.
* ``fail_uploads`` — every ``temporary_upload_file`` raises ``ToolError`` to
  drive MC-007/EC-007 (upload failure → explicit backend error, no graph).

Served over streamable-HTTP on a dedicated uvicorn daemon thread, exactly like
the production ``/files/mcp`` endpoint; the agent YAML points ``mcp_servers``
at ``http://127.0.0.1:<port>/mcp``.
"""

import base64
import os
import threading
from typing import Any

import uvicorn

TEST_MCP_FILES_PORT = int(os.environ.get("TEST_MCP_FILES_PORT", "28878"))
# The stub stands in for the EXTERNAL RAGAnything ``/files/mcp`` server (see
# class docstring). It must be served under a path ENDING with ``/files/mcp``
# so the agent YAML ``mcp_servers[].url`` satisfies the backend document
# policy (REQ-009: an HTTP MCP server whose URL ends with ``/files/mcp``).
MCP_PATH = "/files/mcp"


class _StripPrefix:
    """ASGI wrapper mounting the FastMCP streamable app under ``MCP_PATH``.

    The mcp package's ``FastMCP.streamable_http_app()`` serves its transport at
    the internal route ``/mcp`` while RAGAnything mounts the whole server under
    ``/files/mcp`` in production (main.py). The stub mirrors that with a simple
    path rewrite: ``<MCP_PATH>/...`` → ``/mcp/...``.
    """

    def __init__(self, app, prefix: str) -> None:
        self._app = app
        self._prefix = prefix

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] in ("http", "websocket") and scope.get("path", "").startswith(self._prefix):
            scope = dict(scope)
            remainder = scope["path"][len(self._prefix):]
            scope["path"] = f"/mcp{remainder}"
        await self._app(scope, receive, send)


class McpFilesStubServer:
    """Records every temporary upload/delete the backend performs."""

    def __init__(self) -> None:
        from mcp.server.fastmcp import FastMCP

        self.upload_calls: list[dict[str, Any]] = []
        self.delete_calls: list[str] = []
        self.fail_uploads = False
        self.fail_deletes = False
        self._lock = threading.Lock()

        mcp = FastMCP("Files", stateless_http=True)

        @mcp.tool()
        async def temporary_upload_file(
            file_name: str, content_base64: str, content_type: str, prefix: str
        ) -> dict[str, Any]:
            with self._lock:
                if self.fail_uploads:
                    raise ValueError("stub MCP files server: upload failure")
                self.upload_calls.append(
                    {
                        "file_name": file_name,
                        "content_type": content_type,
                        "prefix": prefix,
                        "content_base64": content_base64,
                        "decoded_bytes": base64.b64decode(content_base64),
                    }
                )
            object_name = f"{prefix.rstrip('/')}/{file_name}"
            return {
                "object_name": object_name,
                "size": len(base64.b64decode(content_base64)),
                "content_type": content_type,
            }

        @mcp.tool()
        async def temporary_delete_file(object_path: str) -> dict[str, Any]:
            with self._lock:
                if self.fail_deletes:
                    raise ValueError("stub MCP files server: delete failure")
                self.delete_calls.append(object_path)
            return {"deleted": object_path, "present": True}

        @mcp.tool()
        async def list_files(
            prefix: str = "",  # noqa: ARG001 — mirrors the production tool signature
            recursive: bool = True,  # noqa: ARG001 — mirrors the production tool signature
        ) -> list[dict[str, Any]]:
            return []

        @mcp.tool()
        async def list_folders(prefix: str = "") -> list[str]:  # noqa: ARG001 — production signature
            return []

        @mcp.tool()
        async def read_file(file_path: str) -> dict[str, Any]:  # noqa: ARG001 — production signature
            return {"content": ["stub file content"], "metadata": {}, "tables": []}

        self._app = mcp.streamable_http_app()

    def reset(self, *, fail_uploads: bool = False, fail_deletes: bool = False) -> None:
        self.upload_calls = []
        self.delete_calls = []
        self.fail_uploads = fail_uploads
        self.fail_deletes = fail_deletes

    def start(self) -> None:
        config = uvicorn.Config(_StripPrefix(self._app, MCP_PATH), host="127.0.0.1", port=TEST_MCP_FILES_PORT, log_level="error")
        self._server = uvicorn.Server(config)

        def _run() -> None:
            self._server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
            self._server.run()

        self._thread = threading.Thread(target=_run, name="stub-mcp-files", daemon=True)
        self._thread.start()
        # I12: wait until the socket accepts connections — the agent builds
        # start loading MCP tools immediately after this and must never race
        # the listener.
        self._wait_until_ready()

    def _wait_until_ready(self, timeout: float = 5.0) -> None:
        """Poll the port until it accepts connections (or raise)."""
        import socket
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", TEST_MCP_FILES_PORT)) == 0:
                    return
            time.sleep(0.05)
        raise RuntimeError(f"stub MCP files server did not start on port {TEST_MCP_FILES_PORT}")

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{TEST_MCP_FILES_PORT}{MCP_PATH}"

    def stop(self) -> None:
        should_exit = getattr(self, "_server", None)
        if should_exit is not None:
            should_exit.should_exit = True
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join(timeout=5)
