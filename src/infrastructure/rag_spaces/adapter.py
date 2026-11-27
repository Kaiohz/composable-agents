"""HTTP adapter for the mcp-raganything RAG-spaces API (REQ-010/REQ-018).

The spaces REST contract of mcp-raganything (REQ-010):

* ``POST /api/v1/spaces`` with ``{"agent_slug": ...}`` — idempotent create:
  201 on create, 200 when the space already exists (BOTH are successes — the
  agent lifecycle must tolerate an idempotent replay), 422 on an invalid
  agent_slug charset (surfaced status-only, then treated best-effort by the
  lifecycle use cases);
* ``DELETE /api/v1/spaces/{agent_slug}`` — strict cascading delete: 204, and
  404 for a nonexistent space which IS the deprovisioning target state
  (success, no partial deletion).

The adapter forwards the ORIGINATING user's credential (JWT bearer or API
key) exactly like the trusted registry client
(:mod:`src.infrastructure.mcp_registry.adapter`), never a service identity,
and never reflects response bodies (they may carry sensitive details).
"""

import logging

import httpx

from src.domain.errors.mcp import McpRegistryRequestError, McpRegistryUnavailableError
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage
from src.domain.ports.rag_space_provisioner import RagSpaceProvisioner
from src.infrastructure.database.rls_context import current_auth_context

logger = logging.getLogger(__name__)

#: REST prefix of the mcp-raganything spaces API (REQ-010).
SPACES_PREFIX = "/api/v1/spaces"


def _credential_headers() -> dict[str, str]:
    """Return the forwarding headers carrying the ORIGINATING credential.

    Same forwarding contract as the trusted registry client: the raw
    credential comes from the RLS auth contextvar populated by the dual-auth
    verification — ``Authorization: Bearer <jwt>`` or ``X-API-Key: <key>``
    (REQ-025, never a service account).
    """
    ctx = current_auth_context.get()
    if ctx is None or not ctx.raw_credential:
        return {}
    if ctx.method == "jwt":
        return {"Authorization": f"Bearer {ctx.raw_credential}"}
    return {"X-API-Key": ctx.raw_credential}


class HttpRagSpaceProvisioner(RagSpaceProvisioner):
    """REST client for the configured mcp-raganything spaces API.

    One long-lived :class:`httpx.AsyncClient` shared across the agent
    lifecycle calls (same pooling rationale as ``HttpMcpRegistryClient``);
    the base URL is read at CALL time so test/di seams can retarget it
    without rebuilding the client. The lifecycle is owned by the composition
    root, which awaits :meth:`aclose` on shutdown.
    """

    def __init__(
        self,
        base_url_provider,
        timeout: float = 30.0,
        prefix: str = SPACES_PREFIX,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Wire the adapter.

        Args:
            base_url_provider: Zero-arg callable returning the mcp-raganything
                base URL (deployment configuration). Read at call time so
                test/di seams can retarget it.
            timeout: Per-request timeout in seconds.
            prefix: REST prefix of the spaces API.
            transport: Optional httpx transport override (tests).
        """
        self._base_url_provider = base_url_provider
        self._timeout = timeout
        self._prefix = prefix.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            transport=transport,
        )

    async def aclose(self) -> None:
        """Release the shared HTTP client (composition-root lifecycle hook)."""
        await self._client.aclose()

    def _base_url(self) -> str:
        base = (self._base_url_provider() or "").rstrip("/")
        if not base:
            raise McpRegistryUnavailableError(ErrorMessage.MCP_REGISTRY_UNAVAILABLE)
        return base

    async def provision(self, agent_slug: str) -> None:
        """Idempotently ensure the space exists (201 create / 200 exists)."""
        url = f"{self._base_url()}{self._prefix}"
        headers = _credential_headers()
        # Review I-6: log the configured PATH only — the base URL is a
        # deployment value that may embed credentials (codebase convention).
        logger.info(LogMessage.RAG_SPACE_FORWARDING, "POST", self._prefix)
        response = await self._send("POST", url, headers=headers, json_body={"agent_slug": agent_slug})
        # 201 = created, 200 = already exists (idempotent replay): both valid.
        if response.status_code not in (200, 201):
            raise McpRegistryRequestError(ErrorMessage.MCP_REGISTRY_REQUEST_FAILED.format(status=response.status_code))

    async def deprovision(self, agent_slug: str) -> None:
        """Cascade-delete the space (204; 404 already meets the target state)."""
        url = f"{self._base_url()}{self._prefix}/{agent_slug}"
        headers = _credential_headers()
        logger.info(LogMessage.RAG_SPACE_FORWARDING, "DELETE", self._prefix)
        response = await self._send("DELETE", url, headers=headers, json_body=None)
        if response.status_code == 404:
            # Strict REST semantics server-side (no partial deletion); here
            # "space absent" is exactly the deprovisioning goal — success.
            return
        if response.status_code != 204:
            raise McpRegistryRequestError(ErrorMessage.MCP_REGISTRY_REQUEST_FAILED.format(status=response.status_code))

    async def _send(self, method: str, url: str, *, headers: dict[str, str], json_body: dict | None) -> httpx.Response:
        """Execute one spaces call with the shared client, mapping failures safely.

        Args:
            method: HTTP verb.
            url: Absolute URL of the spaces endpoint.
            headers: Forwarded originating-credential headers.
            json_body: Optional JSON body (``None`` for DELETE).

        Returns:
            The raw response (status-only consumption; the response body is
            never reflected).

        Raises:
            McpRegistryUnavailableError: Connection/timeout/DNS failure.
        """
        try:
            return await self._client.request(method, url, headers=headers, json=json_body)
        # ``TimeoutError`` is deliberately absent: it subclasses ``OSError``
        # (already listed), so listing it again is redundant (S5713).
        except (httpx.HTTPError, OSError) as exc:
            logger.warning(LogMessage.RAG_SPACE_FORWARD_FAILED, "network")
            raise McpRegistryUnavailableError(ErrorMessage.MCP_REGISTRY_UNAVAILABLE) from exc
