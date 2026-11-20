"""Inbound Composable MCP control plane — server composition (REQ-001..005).

A streamable-HTTP MCP server mounted at ``/mcp`` inside the existing FastAPI
application, using the ``mcp`` SDK's FastMCP generation (stateless HTTP, plain
JSON responses) with lifespan integration: the MCP session manager starts and
stops with the FastAPI lifespan (Technical Notes: "lifespan integration so
MCP session resources start and stop with FastAPI").

Authentication (REQ-002/003/004): a pure-ASGI guard in front of the MCP ASGI
app reads the ``Authorization`` and ``X-API-Key`` headers, delegates to the
same dual-auth security used by REST, and returns HTTP 401 for missing or
invalid credentials BEFORE any MCP method runs — discovery AND execution are
guarded, and a presented invalid bearer JWT never falls back to a valid API
key. On success the guard establishes the exact user, raw credential and
authentication context REST uses (the RLS contextvars are set in the request
task so every downstream tool call is RLS-scoped).

FastAPI router dependencies do NOT protect an independently mounted ASGI app
(Technical Notes) — that is why the guard is a raw ASGI wrapper here.
"""

import json
import logging
from collections.abc import Awaitable, Callable

from mcp.server.fastmcp import FastMCP

from src.application.mcp.errors import MCP_INSTRUCTIONS
from src.domain.entities.auth.auth_context import AuthContext
from src.domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)

#: Streamable-HTTP MCP endpoint mounted on the FastAPI app.
MCP_MOUNT_PATH = "/mcp"

Authenticator = Callable[[str | None, str | None], Awaitable[AuthContext | None]]

#: 401 response body — generic, non-enumerating (no user data, no tools).
_UNAUTHORIZED_BODY = json.dumps({"detail": "Invalid or missing credentials"}).encode()


class McpAuthGuard:
    """Pure-ASGI guard authenticating every request to the mounted MCP app.

    HTTP 401 semantics: the raw JSON-RPC probe (initialize / tools/call)
    observes a 401 status the same way REST does (REQ-003), instead of a
    JSON-RPC error frame that protocol clients could misread.
    """

    def __init__(self, app, authenticate: Authenticator) -> None:
        self._app = app
        self._authenticate = authenticate

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        await self._handle_http(scope, receive, send)

    async def _handle_http(self, scope, receive, send) -> None:  # noqa: ANN001
        """Authenticate one HTTP request, then forward it to the MCP app."""
        # Path normalization: the exact "/mcp" request (mount redirect target
        # and direct route hit) is rewritten to "/" so the inner Starlette
        # app's single route always matches — no redirect round-trips and the
        # auth guard runs FIRST on the original request.
        if scope.get("path", "").rstrip("/") == MCP_MOUNT_PATH:
            scope = dict(scope)
            scope["path"] = "/"
        headers = _asgi_headers(scope)
        try:
            ctx = await self._authenticate(headers.get("authorization"), headers.get("x-api-key"))
        except Exception:  # noqa: BLE001 — never leak auth internals over the wire
            logger.exception(LogMessage.MCP_REQUEST_DENIED)
            await self._send_unauthorized(send)
            return
        if ctx is None:
            logger.warning(LogMessage.MCP_REQUEST_DENIED)
            await self._send_unauthorized(send)
            return
        logger.info(LogMessage.AUTH_CREDENTIALS_VALIDATED, ctx.user_id, ctx.method)
        await self._app(scope, receive, send)

    @staticmethod
    async def _send_unauthorized(send) -> None:
        """Emit a plain HTTP 401 with a bounded JSON body."""
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_UNAUTHORIZED_BODY)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})


def _asgi_headers(scope) -> dict[str, str]:  # noqa: ANN001 — raw ASGI scope
    """Decode the raw ASGI header list into a lowercase-keyed dict."""
    return {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}


def build_mcp_asgi_app(authenticate: Authenticator):
    """Compose the guarded streamable-HTTP MCP ASGI app (REQ-001/002).

    Args:
        authenticate: The dual-auth callable reused from the REST security
            (delegates to the wired :class:`AuthService` and sets the RLS
            contextvars exactly like ``verify_credentials``).

    Returns:
        ``(guarded_asgi_app, session_manager)`` — the mounted guard and the
        session manager whose ``run()`` context the FastAPI lifespan enters.
    """
    from src.application.mcp.tools import register_mcp_tools

    mcp = FastMCP(
        "composable-agents",
        instructions=MCP_INSTRUCTIONS,
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
    )
    register_mcp_tools(mcp)
    inner = mcp.streamable_http_app()
    logger.info(LogMessage.MCP_SERVER_MOUNTED)
    return McpAuthGuard(inner, authenticate), mcp.session_manager
