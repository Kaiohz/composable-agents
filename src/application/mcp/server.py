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
task so every downstream tool call is RLS-scoped). The 401's OAuth2 discovery
URL comes from :func:`public_discovery_origin` (configured
``MCP_PUBLIC_BASE_URL`` or an allow-listed request ``Host``) — the raw
client-supplied ``Host`` is never echoed into a discovery document (M-2).

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

#: Public OAuth2 protected-resource metadata document (MCP authorization spec
#: 2025-06-18): the discovery target advertised by the 401 challenge and served
#: by the public route wired in ``src/main.py`` (REQ-002).
WELL_KNOWN_PROTECTED_RESOURCE_PATH = "/.well-known/oauth-protected-resource"

Authenticator = Callable[[str | None, str | None], Awaitable[AuthContext | None]]

#: 401 response body — generic, non-enumerating (no user data, no tools).
_UNAUTHORIZED_BODY = json.dumps({"detail": "Invalid or missing credentials"}).encode()


def public_discovery_origin(
    request_host: str,
    *,
    configured_base_url: str,
    allowed_hosts: list[str],
) -> str | None:
    """Return the canonical PUBLIC base origin used by the discovery documents.

    Single derivation point shared by the 401 ``WWW-Authenticate`` challenge
    and the well-known protected-resource route (M-finding M-2: the request
    ``Host`` header is client-supplied and is never echoed unvalidated).

    Precedence:

    1. ``MCP_PUBLIC_BASE_URL`` when configured (the production configmap sets
       it — always wins, trailing slash stripped);
    2. otherwise the request ``Host`` is used ONLY when it is explicitly
       allow-listed in ``MCP_ALLOWED_HOSTS`` (exact entry or ``host:*``
       wildcard) — TLS terminates at the ingress proxy, so the scheme is
       always ``https``;
    3. otherwise ``None``: no trustworthy origin can be advertised, so the
       callers must serve a challenge/metadata WITHOUT a discovery URL
       instead of echoing a spoofable host.

    Args:
        request_host: The raw request ``Host`` header value (may be empty).
        configured_base_url: The ``MCP_PUBLIC_BASE_URL`` setting value.
        allowed_hosts: The ``MCP_ALLOWED_HOSTS`` allow-list.

    Returns:
        The public origin WITHOUT a trailing slash (``https://<host>``), or
        ``None`` when no trustworthy origin can be derived.
    """
    configured = (configured_base_url or "").strip().rstrip("/")
    if configured:
        return configured
    host = (request_host or "").strip()
    if not host or not _host_allowed(host, allowed_hosts):
        return None
    return f"https://{host}"


def _host_allowed(host: str, allowed_hosts: list[str]) -> bool:
    """Whether ``host`` matches the derivation allow-list shape.

    Exact entries plus ``<host>:*`` wildcard entries (any port).
    """
    for allowed in allowed_hosts:
        if allowed == host:
            return True
        if allowed.endswith(":*") and host.startswith(allowed[:-2] + ":"):
            return True
    return False


def www_authenticate_challenge(public_origin: str) -> str:
    """Build the MCP OAuth2 discovery challenge header value (REQ-001).

    Args:
        public_origin: The canonical public origin from
            :func:`public_discovery_origin` (scheme + host, no trailing
            slash) — never a raw, unvalidated request ``Host``.

    Returns:
        The ``WWW-Authenticate`` challenge pointing at the public
        protected-resource metadata document.
    """
    return f'Bearer resource_metadata="{public_origin}{WELL_KNOWN_PROTECTED_RESOURCE_PATH}"'


class McpAuthGuard:
    """Pure-ASGI guard authenticating every request to the mounted MCP app.

    HTTP 401 semantics: the raw JSON-RPC probe (initialize / tools/call)
    observes a 401 status the same way REST does (REQ-003), instead of a
    JSON-RPC error frame that protocol clients could misread.

    The 401's ``WWW-Authenticate`` discovery URL is derived through
    :func:`public_discovery_origin` — the configured ``MCP_PUBLIC_BASE_URL``
    or an explicitly allow-listed request ``Host`` — never from a raw,
    spoofable ``Host`` header (M-finding M-2).
    """

    def __init__(
        self,
        app,
        authenticate: Authenticator,
        *,
        configured_base_url: str = "",
        allowed_hosts: list[str] | None = None,
    ) -> None:
        self._app = app
        self._authenticate = authenticate
        self._configured_base_url = configured_base_url
        self._allowed_hosts = list(allowed_hosts or [])

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
        challenge = self._discovery_challenge(headers.get("host", ""))
        try:
            ctx = await self._authenticate(headers.get("authorization"), headers.get("x-api-key"))
        except Exception:  # noqa: BLE001 — never leak auth internals over the wire
            logger.exception(LogMessage.MCP_REQUEST_DENIED)
            await self._send_unauthorized(send, challenge)
            return
        if ctx is None:
            logger.warning(LogMessage.MCP_REQUEST_DENIED)
            await self._send_unauthorized(send, challenge)
            return
        logger.info(LogMessage.AUTH_CREDENTIALS_VALIDATED, ctx.user_id, ctx.method)
        await self._app(scope, receive, send)

    def _discovery_challenge(self, request_host: str) -> str | None:
        """Build the ``WWW-Authenticate`` value for one request, or ``None``.

        ``None`` means no trustworthy public origin could be derived (the
        configured ``MCP_PUBLIC_BASE_URL`` is unset and the request ``Host``
        is not allow-listed) — the 401 is then served without a discovery
        pointer instead of echoing a spoofable host.
        """
        origin = public_discovery_origin(
            request_host,
            configured_base_url=self._configured_base_url,
            allowed_hosts=self._allowed_hosts,
        )
        if origin is None:
            logger.warning(LogMessage.MCP_DISCOVERY_ORIGIN_UNAVAILABLE)
            return None
        return www_authenticate_challenge(origin)

    @staticmethod
    async def _send_unauthorized(send, www_authenticate: str | None) -> None:  # noqa: ANN001 — raw ASGI
        """Emit a plain HTTP 401 carrying the OAuth2 discovery challenge.

        The body stays the generic, non-enumerating JSON; the
        ``WWW-Authenticate`` header points OAuth2-enabled MCP clients at the
        public protected-resource metadata document (REQ-001) so they can
        start the authorization flow (401 → metadata → authorization server).
        ``Cache-Control: no-store`` keeps a shared cache from ever replaying
        the (per-deployment) challenge to a different client (M-finding M-2
        hardening). ``www_authenticate`` is ``None`` when no trustworthy
        public origin exists — the 401 is sent WITHOUT the header.
        """
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(_UNAUTHORIZED_BODY)).encode()),
            (b"cache-control", b"no-store"),
        ]
        if www_authenticate is not None:
            headers.append((b"www-authenticate", www_authenticate.encode("latin-1")))
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": headers,
            }
        )
        await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})


def _asgi_headers(scope) -> dict[str, str]:  # noqa: ANN001 — raw ASGI scope
    """Decode the raw ASGI header list into a lowercase-keyed dict."""
    return {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}


def build_mcp_asgi_app(
    authenticate: Authenticator,
    *,
    allowed_hosts: list[str],
    allowed_origins: list[str],
    public_base_url: str = "",
):
    """Compose the guarded streamable-HTTP MCP ASGI app (REQ-001/002).

    Args:
        authenticate: The dual-auth callable reused from the REST security
            (delegates to the wired :class:`AuthService` and sets the RLS
            contextvars exactly like ``verify_credentials``).
        allowed_hosts: MCP transport-security Host allow-list (REQ-013, from
            the ``MCP_ALLOWED_HOSTS`` setting). DNS-rebinding protection stays
            ENABLED — an explicit list (which must contain the public proxy
            host in production) only corrects the SDK's localhost-only default
            that otherwise 421s every proxied request. The same list also
            gates the ``Host``-fallback of the discovery-URL derivation (M-2).
        allowed_origins: MCP transport-security Origin allow-list (from
            ``ALLOWED_ORIGINS``), so browser-origin MCP clients are not 403'd.
        public_base_url: Configured public origin for the OAuth2 discovery
            documents (``MCP_PUBLIC_BASE_URL``). Always wins when set; when
            empty the guard falls back to allow-listed request Hosts only
            (:func:`public_discovery_origin`).

    Returns:
        ``(guarded_asgi_app, session_manager)`` — the mounted guard and the
        session manager whose ``run()`` context the FastAPI lifespan enters.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    from src.application.mcp.tools import register_mcp_tools

    mcp = FastMCP(
        "composable-agents",
        instructions=MCP_INSTRUCTIONS,
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
        # Explicit transport security (REQ-013): FastMCP would otherwise
        # auto-enable a localhost-only Host list (its default host is
        # 127.0.0.1) and 421 every proxy-presented production Host. Protection
        # stays ENABLED; only the host/origin lists become configuration.
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        ),
    )
    register_mcp_tools(mcp)
    inner = mcp.streamable_http_app()
    logger.info(LogMessage.MCP_SERVER_MOUNTED)
    return (
        McpAuthGuard(
            inner,
            authenticate,
            configured_base_url=public_base_url,
            allowed_hosts=allowed_hosts,
        ),
        mcp.session_manager,
    )
