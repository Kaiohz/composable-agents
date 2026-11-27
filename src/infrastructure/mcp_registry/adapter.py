"""HTTP adapter for the trusted mcp-raganything MCP-server registry (REQ-017).

The registry remains the SOLE owner of MCP server persistence, validation,
generation, mounting and rehydration. This adapter only forwards REST calls to
the configured trusted registry base URL, carrying the ORIGINATING credential
(JWT bearer or per-user API key) so the registry applies the exact same
per-user scoping (REQ-004/REQ-025). It never caches, never masks/unmasks
anything itself and never logs secrets (REQ-019).

Mapping rules:

* connection / timeout / DNS failures → :class:`McpRegistryUnavailableError`
  (safe retryable, HTTP 503 semantics);
* a failing status → :class:`McpRegistryRequestError` with ONLY the status
  code surfaced (the registry response body may carry secrets and is never
  reflected);
* 404 → :class:`McpRegistryNotFoundError` (non-enumerating not-found);
* success → the parsed JSON validated into the masked domain entities
  (REQ-018). The adapter NEVER fills secret fields back in.

Trusted execution-config resolution (:meth:`get_execution_config`) uses the
registry's authenticated reveal endpoint (REQ-037) and returns a
:class:`McpServerConfig` for server-side tool loading ONLY.
"""

import logging
from typing import Any

import httpx
from pydantic import ValidationError

from src.domain.entities.mcp_server_config import McpAuthMode, McpServerConfig, McpTransportType
from src.domain.entities.registered_mcp_server import (
    McpServerValidationReport,
    McpServerWrite,
    RegisteredMcpServer,
)
from src.domain.errors.mcp import (
    McpRegistryConflictError,
    McpRegistryNotFoundError,
    McpRegistryRequestError,
    McpRegistryUnavailableError,
)
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage
from src.domain.ports.mcp_registry import McpRegistryClient
from src.infrastructure.database.rls_context import current_auth_context
from src.infrastructure.env_utils import resolve_headers_drop_empty

logger = logging.getLogger(__name__)

#: Default REST prefix of the mcp-raganything registry API.
REGISTRY_PREFIX = "/api/v1/mcp/servers"


def _credential_headers() -> dict[str, str]:
    """Return the forwarding headers carrying the ORIGINATING credential.

    REQ-025: originating credentials may be forwarded only to the configured
    trusted registry base URL. The raw credential comes from the RLS auth
    contextvar populated by the dual-auth verification (exactly as REST).
    """
    ctx = current_auth_context.get()
    if ctx is None or not ctx.raw_credential:
        return {}
    if ctx.method == "jwt":
        return {"Authorization": f"Bearer {ctx.raw_credential}"}
    return {"X-API-Key": ctx.raw_credential}


def _write_payload(payload: McpServerWrite) -> dict[str, Any]:
    """Serialize a write input for the registry (secrets included, REQ-019).

    Deployment ordering (feat-mcp-oauth2 M-finding M-5): ``auth_mode`` is
    carried on EVERY create/update/validate payload (the REQ-004 contract —
    the registry must persist it for ``user_token`` to round-trip). The
    registry side (mcp-ragalong slice) MUST accept and persist the field
    BEFORE this service reaches production:

    * a registry that FORBIDS extras rejects these payloads with 422 — the
      rollout order (registry first) is mandatory;
    * a LENIENT registry silently ignores the field — the degradation is made
      LOUD instead of silent by :meth:`_warn_auth_mode_echo` (create/update
      responses are compared against the requested mode) and by the
      ``MCP_REGISTRY_AUTH_MODE_MISSING`` warning on the reveal/read path
      (:meth:`_to_execution_config`).
    """
    return payload.model_dump(mode="json")


def _warn_auth_mode_echo(action: str, requested: str | None, entry: dict[str, Any]) -> None:
    """Warn LOUDLY when a registry response indicates ``auth_mode`` was dropped.

    feat-mcp-oauth2 M-finding M-5: a registry that ignores the new
    ``auth_mode`` field (pre-deployment registry) silently degrades a
    ``user_token`` server to static ``headers`` — the user's chosen OAuth2
    mode would never be transmitted and nothing would say so. The create /
    update RESPONSE is therefore compared against the requested mode: a
    requested ``user_token`` that is absent (or contradicted) in the echoed
    entry is logged as a structured warning instead of failing a request
    that the deployed registry otherwise handled. Never raises, never leaks
    secret material (the echoed mode is metadata, not a secret).
    """
    echoed = entry.get("auth_mode") if isinstance(entry, dict) else None
    if requested == McpAuthMode.USER_TOKEN and echoed != McpAuthMode.USER_TOKEN:
        logger.warning(LogMessage.MCP_REGISTRY_AUTH_MODE_DROPPED, action, requested, echoed)


def _masked_server(data: dict[str, Any]) -> RegisteredMcpServer:
    """Validate an untrusted registry payload into the masked projection."""
    safe = dict(data)
    # Defensive redaction (REQ-018): even a misbehaving registry cannot leak
    # secret material through this adapter.
    safe["auth_token"] = None
    safe["headers"] = {}
    safe["env"] = {}
    return RegisteredMcpServer.model_validate(safe)


def _reveal_candidates(base: str, prefix: str, name: str) -> tuple[str, ...]:
    """Return the accepted reveal URL shapes (canonical first).

    The canonical mcp-raganything shape is ``{prefix}/{name}/reveal``; the
    deployed alias ``{parent}/reveal/{name}`` covers simplified registry
    deployments. A 404 on one candidate falls back to the other.
    """
    parent = prefix.rsplit("/", 1)[0] if "/" in prefix else prefix
    return (f"{base}{prefix}/{name}/reveal", f"{base}{parent}/reveal/{name}")


def _to_execution_config(entry: dict[str, Any], name: str) -> McpServerConfig:
    """Project a revealed registry entry onto the server-side execution config.

    Placeholder resolution (e.g. ``${USER_JWT}``) happens inside the tool
    loader per call, against the originating user's contextvars. The
    credential-transmission mode (``auth_mode``, feat-mcp-oauth2 REQ-004) is
    carried through so the loader can inject the connected user's credential;
    an ABSENT value means the registry did not (yet) accept the field — the
    degradation is logged (M-finding M-5) and defaults to ``headers``; an
    unknown value is rejected by the entity (the source degrades).
    """
    headers = {str(key): str(value) for key, value in (entry.get("headers") or {}).items()}
    env = {str(key): str(value) for key, value in (entry.get("env") or {}).items()}
    mode = entry.get("auth_mode")
    if mode is None:
        logger.warning(LogMessage.MCP_REGISTRY_AUTH_MODE_MISSING, name)
        mode = McpAuthMode.HEADERS
    return McpServerConfig(
        name=str(entry.get("name") or name),
        transport=McpTransportType(str(entry.get("transport") or "http")),
        command=entry.get("command"),
        args=list(entry.get("args") or []),
        url=entry.get("url"),
        headers=resolve_headers_drop_empty(headers),
        env=resolve_headers_drop_empty(env),
        auth_token=entry.get("auth_token"),
        auth_mode=mode,
    )


class HttpMcpRegistryClient(McpRegistryClient):
    """REST client for the configured trusted mcp-raganything registry.

    One long-lived :class:`httpx.AsyncClient` is shared across requests (and
    across the concurrent per-server resolutions of the built-in assistant —
    httpx clients are concurrency-safe) instead of opening a fresh connection
    pool per call. The lifecycle is owned by the composition root, which
    awaits :meth:`aclose` on shutdown.
    """

    def __init__(
        self,
        base_url_provider,
        timeout: float = 30.0,
        prefix: str = REGISTRY_PREFIX,
        transport: httpx.AsyncBaseTransport | None = None,
        internal_rewrite_provider=None,
    ) -> None:
        """Wire the adapter.

        Args:
            base_url_provider: Zero-arg callable returning the trusted
                registry base URL (deployment configuration, REQ: supplied
                through configuration rather than hard-coded). Read at call
                time so test/di seams can retarget it.
            timeout: Per-request timeout in seconds.
            prefix: REST prefix of the registry server API.
            transport: Optional httpx transport override (used by tests to
                drive the adapter without a real socket).
            internal_rewrite_provider: Optional zero-arg callable returning a
                ``(public_origin, internal_base)`` pair. When set, reveal URLs
                starting with the public origin are rewritten onto the
                internal base for SERVER-SIDE executions — the pod egress to
                the public ingress (DNS/cloudflared) is unreliable while the
                ClusterIP route always works.
        """
        self._base_url_provider = base_url_provider
        self._timeout = timeout
        self._prefix = prefix.rstrip("/")
        self._internal_rewrite_provider = internal_rewrite_provider
        self._client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            transport=transport,
        )

    def _rewrite_to_internal(self, config: McpServerConfig) -> McpServerConfig:
        """Rewrite a public-origin URL onto the internal service base.

        Registered MCP servers store their PUBLIC url; server-side tool loads
        keep failing on flaky pod egress (DNS flakes / unreachable network)
        while the internal ClusterIP route is always reachable. When the
        rewrite pair is configured and the URL starts with the public origin,
        the SCHEME+HOST are swapped and the path rides the internal base.
        """
        if not config.url or self._internal_rewrite_provider is None:
            return config
        pair = self._internal_rewrite_provider() or None
        if not pair:
            return config
        public_origin, internal_base = pair
        if config.url.startswith(public_origin):
            rewritten = internal_base.rstrip("/") + config.url[len(public_origin):]
            return config.model_copy(update={"url": rewritten})
        return config

    async def aclose(self) -> None:
        """Release the shared HTTP client (composition-root lifecycle hook)."""
        await self._client.aclose()

    def _base_url(self) -> str:
        base = (self._base_url_provider() or "").rstrip("/")
        if not base:
            raise McpRegistryUnavailableError(ErrorMessage.MCP_REGISTRY_UNAVAILABLE)
        return base

    async def _request(self, method: str, path: str, *, json_body: dict | None = None) -> Any:
        """Execute one registry call with the originating credential forwarded.

        Args:
            method: HTTP method.
            path: Path relative to ``self._prefix`` (e.g. ``/{name}``).
            json_body: Optional JSON body.

        Raises:
            McpRegistryNotFoundError: Registry answered 404.
            McpRegistryRequestError: Any other non-2xx (status-only detail).
            McpRegistryUnavailableError: Connection/timeout failure.
        """
        return await self._request_url(f"{self._base_url()}{self._prefix}{path}", method, json_body)

    async def _request_url(self, url: str, method: str, json_body: dict | None = None) -> Any:
        """Execute one registry call against an absolute URL (reveal aliases)."""
        headers = _credential_headers()
        logger.info(LogMessage.MCP_REGISTRY_FORWARDING, method, url)
        try:
            response = await self._client.request(method, url, headers=headers, json=json_body)
        # ``TimeoutError`` is deliberately absent: it subclasses ``OSError``
        # (already listed), so listing it again is redunant (S5713).
        except (httpx.HTTPError, OSError) as exc:
            logger.warning(LogMessage.MCP_REGISTRY_FORWARD_FAILED, "network")
            raise McpRegistryUnavailableError(ErrorMessage.MCP_REGISTRY_UNAVAILABLE) from exc
        if response.status_code == 404:
            raise McpRegistryNotFoundError(ErrorMessage.MCP_REGISTRY_SERVER_NOT_FOUND.format(name=url))
        if response.status_code == 409:
            # Registration conflicts surface as explicit, safe errors (REQ-TX-002):
            # the body (secrets) is never echoed, only the conflict semantics.
            logger.warning(LogMessage.MCP_REGISTRY_FORWARD_FAILED, response.status_code)
            raise McpRegistryConflictError(ErrorMessage.MCP_REGISTRY_REQUEST_FAILED.format(status=response.status_code))
        if response.status_code in (502, 503, 504):
            # A registry 5xx IS an availability signal: the safe-retryable
            # semantics apply (REQ-TX-003), never an echoed body.
            logger.warning(LogMessage.MCP_REGISTRY_FORWARD_FAILED, response.status_code)
            raise McpRegistryUnavailableError(ErrorMessage.MCP_REGISTRY_UNAVAILABLE)
        if response.status_code >= 400:
            # REQ-009: the registry body is never echoed (it can carry secrets;
            # only the status code reaches logs/results).
            logger.warning(LogMessage.MCP_REGISTRY_FORWARD_FAILED, response.status_code)
            raise McpRegistryRequestError(ErrorMessage.MCP_REGISTRY_REQUEST_FAILED.format(status=response.status_code))
        if response.status_code == 204:
            return None
        try:
            return response.json()
        except ValueError as exc:  # json.JSONDecodeError ⊂ ValueError
            # A non-JSON registry body is a contract drift, exactly like an
            # unparseable dict payload: controlled error, never a 500.
            logger.warning(LogMessage.MCP_REGISTRY_FORWARD_FAILED, "payload-decode")
            raise McpRegistryRequestError(
                ErrorMessage.MCP_REGISTRY_REQUEST_FAILED.format(status=response.status_code)
            ) from exc

    async def list_servers(self) -> list[RegisteredMcpServer]:
        """List the current user's registered MCP servers (masked, REQ-018)."""
        entries = await self._request("GET", "")
        if not isinstance(entries, list):
            raise McpRegistryRequestError(ErrorMessage.MCP_REGISTRY_REQUEST_FAILED.format(status="shape"))
        return [_masked_server(entry) for entry in entries]

    async def get_server(self, name: str) -> RegisteredMcpServer:
        """Return one registered MCP server (masked, REQ-018)."""
        entry = await self._request("GET", f"/{name}")
        return _masked_server(entry)

    async def create_server(self, payload: McpServerWrite) -> RegisteredMcpServer:
        """Register a server through the owning registry (REQ-017).

        A registration conflict is turned into an explicit, safe error naming
        the candidate server (REQ-TX-002/REQ-TX-004) without echoing the
        registry body (which may carry secrets).
        """
        try:
            entry = await self._request("POST", "", json_body=_write_payload(payload))
        except McpRegistryConflictError as exc:
            raise McpRegistryConflictError(ErrorMessage.AGENT_IMPORT_MCP_CONFLICT.format(name=payload.name)) from exc
        _warn_auth_mode_echo("create", payload.auth_mode, entry)
        return _masked_server(entry)

    async def update_server(self, name: str, changes: dict) -> RegisteredMcpServer:
        """Update a registered MCP server (masked response, REQ-018)."""
        entry = await self._request("PUT", f"/{name}", json_body=changes)
        _warn_auth_mode_echo(f"update:{name}", changes.get("auth_mode"), entry)
        return _masked_server(entry)

    async def delete_server(self, name: str) -> None:
        """Delete a registered MCP server (registry-owned persistence)."""
        await self._request("DELETE", f"/{name}")

    async def validate_server(self, payload: McpServerWrite) -> McpServerValidationReport:
        """Validate a candidate server WITHOUT persisting it (REG contract).

        The REAL registry answers ``McpServerValidateResponse{tool_count: N}``
        on a validated candidate and raises an explicit error (non-2xx) for an
        invalid one; fuller deployments may return the whole report shape. A
        payload that cannot be parsed at ALL is a registry contract drift —
        surfaced as a controlled :class:`McpRegistryRequestError`, never as a
        raw validation error (which would reach the client as a 500).

        Raises:
            McpRegistryRequestError: The registry answered with an unusable
                payload (status-only detail, no body echo).
        """
        report = await self._request("POST", "/validate", json_body=_write_payload(payload))
        if not isinstance(report, dict):
            raise McpRegistryRequestError(ErrorMessage.MCP_REGISTRY_REQUEST_FAILED.format(status="shape"))
        try:
            return McpServerValidationReport.model_validate(report)
        except ValidationError as exc:
            logger.warning(LogMessage.MCP_REGISTRY_FORWARD_FAILED, "validation-payload-shape")
            raise McpRegistryRequestError(
                ErrorMessage.MCP_REGISTRY_REQUEST_FAILED.format(status="validation-shape")
            ) from exc

    async def get_execution_config(self, name: str) -> McpServerConfig:
        """Resolve the decrypted execution configuration (TRUSTED, REQ-037).

        Uses the registry's authenticated reveal endpoint server-side. Two
        absolute path shapes are accepted — the canonical
        ``{prefix}/{name}/reveal`` and its deployed alias
        ``{parent}/reveal/{name}`` — so both real registries and simplified
        deployments resolve; a 404 on one candidate falls back to the other.
        The returned config is consumed ONLY by the runner's tool loader;
        decrypted headers/env/tokens never reach management responses, traces
        or logs.

        Raises:
            McpRegistryNotFoundError: The registry denies the entry on every
                candidate path (non-enumerating not-found).
            McpRegistryRequestError: The registry answered with an unusable
                payload (status-only detail).
        """
        entry = await self.reveal_server(name)
        return self._rewrite_to_internal(_to_execution_config(entry, name))

    async def reveal_server(self, name: str) -> dict:
        """Resolve the RAW reveal payload (TRUSTED, REQ-037 family).

        Same endpoint shapes as :meth:`get_execution_config` but WITHOUT
        placeholder resolution: the raw headers/env/token values stay exactly
        as stored by the registry so server-side checks (like the cron
        unattended-safety user-placeholder scan) match them reliably.
        """
        entry = await self._reveal_entry(name)
        if not isinstance(entry, dict):
            raise McpRegistryRequestError(ErrorMessage.MCP_REGISTRY_REQUEST_FAILED.format(status="shape"))
        return entry

    async def _reveal_entry(self, name: str) -> Any:
        """Fetch the revealed registry entry, trying the accepted URL shapes."""
        last_not_found: McpRegistryNotFoundError | None = None
        for candidate in _reveal_candidates(self._base_url(), self._prefix, name):
            try:
                return await self._request_url(candidate, "GET")
            except McpRegistryNotFoundError as exc:
                last_not_found = exc
                continue
        if last_not_found is not None:
            raise last_not_found
        raise McpRegistryRequestError(ErrorMessage.MCP_REGISTRY_REQUEST_FAILED.format(status="shape"))
