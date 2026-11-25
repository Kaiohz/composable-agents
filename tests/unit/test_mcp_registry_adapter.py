"""Unit tests for the trusted MCP-registry HTTP adapter (C2/I2 remediation).

Drives :class:`HttpMcpRegistryClient` over an ``httpx.MockTransport`` so the
reveal fallback, status mapping and execution-config projection are covered
without a real socket. Also pins the long-lived client contract (I2): the
adapter reuses ONE ``httpx.AsyncClient`` across calls and closes it via
:meth:`aclose`.
"""

import httpx
import pytest

from src.domain.entities.mcp_server_config import McpTransportType
from src.domain.errors.mcp import (
    McpRegistryNotFoundError,
    McpRegistryRequestError,
    McpRegistryUnavailableError,
)
from src.domain.errors.messages import ErrorMessage
from src.infrastructure.mcp_registry.adapter import (
    REGISTRY_PREFIX,
    HttpMcpRegistryClient,
    _reveal_candidates,
    _to_execution_config,
)


def _respond(payload, status: int = 200):
    """Build a fixed httpx MockTransport handler (ignores the request)."""

    def _handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001 — transport contract
        return httpx.Response(status, json=payload)

    return _handler


_REVEALED_ENTRY = {
    "name": "alpha",
    "transport": "http",
    "url": "http://127.0.0.1:1/mcp",
    "headers": {"X-Internal": "plain-header-secret"},
    "env": {"ALPHA_TOKEN": "plain-env-secret"},
    "auth_token": "plain-token-secret",
}


def _client_for(handler, base_url: str = "http://registry.test") -> HttpMcpRegistryClient:
    """Build the adapter over a MockTransport with a static base URL."""
    return HttpMcpRegistryClient(
        base_url_provider=lambda: base_url,
        timeout=5.0,
        transport=httpx.MockTransport(handler),
    )


def test_canonical_and_alias_candidates_in_order():
    candidates = _reveal_candidates("http://registry.test", REGISTRY_PREFIX, "alpha")
    assert candidates == (
        f"http://registry.test{REGISTRY_PREFIX}/alpha/reveal",
        "http://registry.test/api/v1/mcp/reveal/alpha",
    )


class TestGetExecutionConfig:
    async def test_canonical_reveal_path_resolves(self):
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001 — transport contract
            seen.append(request.url.path)
            return httpx.Response(200, json=_REVEALED_ENTRY)

        client = _client_for(handler)
        try:
            config = await client.get_execution_config("alpha")
        finally:
            await client.aclose()
        assert seen == [f"{REGISTRY_PREFIX}/alpha/reveal"]
        assert config.url == "http://127.0.0.1:1/mcp"
        assert config.transport == McpTransportType.HTTP
        # Decrypted values travel ONLY inside the trusted server-side execution
        # config (REQ-037) — resolved for the tool loader, never exposed through
        # management responses (the masked entities zero them out).
        assert config.headers == {"X-Internal": "plain-header-secret"}
        assert config.env == {"ALPHA_TOKEN": "plain-env-secret"}
        assert config.auth_token == "plain-token-secret"

    async def test_alias_fallback_when_canonical_404s(self):
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001 — transport contract
            seen.append(request.url.path)
            if request.url.path == "/api/v1/mcp/reveal/alpha":
                return httpx.Response(200, json={"name": "alpha", "url": "http://x/mcp"})
            return httpx.Response(404, json={"detail": "not found"})

        client = _client_for(handler)
        try:
            config = await client.get_execution_config("alpha")
        finally:
            await client.aclose()
        assert f"{REGISTRY_PREFIX}/alpha/reveal" in seen
        assert config.url == "http://x/mcp"

    async def test_all_candidates_404_raises_not_found(self):
        client = _client_for(_respond({"detail": "not found"}, 404))
        try:
            with pytest.raises(McpRegistryNotFoundError):
                await client.get_execution_config("ghost")
        finally:
            await client.aclose()

    async def test_non_dict_payload_is_a_request_error(self):
        client = _client_for(_respond(["not", "a", "dict"], 200))
        try:
            with pytest.raises(McpRegistryRequestError):
                await client.get_execution_config("alpha")
        finally:
            await client.aclose()

    async def test_registry_outage_is_safe_and_retryable(self):
        def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001 — transport contract
            # pragma: no cover - transport layer
            raise httpx.ConnectError("connection refused")

        client = _client_for(handler)
        try:
            with pytest.raises(McpRegistryUnavailableError) as exc_info:
                await client.get_execution_config("alpha")
        finally:
            await client.aclose()
        assert str(exc_info.value.detail) == ErrorMessage.MCP_REGISTRY_UNAVAILABLE

    async def test_failing_status_only_surfaces_the_code(self):
        def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001 — transport contract
            return httpx.Response(502, json={"detail": "registry stack trace with secrets"})

        client = _client_for(handler)
        try:
            with pytest.raises(McpRegistryRequestError) as exc_info:
                await client.get_execution_config("alpha")
        finally:
            await client.aclose()
        assert str(exc_info.value.detail) == ErrorMessage.MCP_REGISTRY_REQUEST_FAILED.format(status=502)

    async def test_uses_the_originating_api_key_credential(self):
        """REQ-025: the ORIGINATING credential header rides every registry call."""
        from src.domain.entities.auth.auth_context import AuthContext
        from src.infrastructure.database.rls_context import current_auth_context

        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001 — transport contract
            seen["api_key"] = request.headers.get("x-api-key", "")
            return httpx.Response(200, json=[])

        client = _client_for(handler)
        token = current_auth_context.set(
            AuthContext(user_id="user-a", method="api_key", raw_credential="cpk_originating")
        )
        try:
            await client.list_servers()
        finally:
            current_auth_context.reset(token)
            await client.aclose()
        assert seen["api_key"] == "cpk_originating"


class TestManagementPaths:
    async def test_list_validates_shape_and_masks_entries(self):
        def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001 — transport contract
            return httpx.Response(
                200,
                json=[
                    {
                        "name": "alpha",
                        "transport": "http",
                        "url": "http://x/mcp",
                        "headers": {"X-Internal": "plain-header-secret"},
                        "env": {"ALPHA_TOKEN": "plain-env-secret"},
                        "auth_token": "plain-token-secret",
                    }
                ],
            )

        client = _client_for(handler)
        try:
            servers = await client.list_servers()
        finally:
            await client.aclose()
        assert len(servers) == 1
        masked = servers[0]
        # REQ-018: secrets masked defensively, even against a misbehaving registry.
        assert masked.headers == {}
        assert masked.env == {}
        assert masked.auth_token is None

    async def test_non_list_list_payload_is_a_request_error(self):
        client = _client_for(_respond({"servers": []}, 200))
        try:
            with pytest.raises(McpRegistryRequestError):
                await client.list_servers()
        finally:
            await client.aclose()


class TestClientOwnership:
    def test_adapter_owns_one_long_lived_client(self):
        first = _client_for(_respond([], 200))
        second = _client_for(_respond([], 200))
        # I2: NOT a fresh pool per call — one client per adapter, closed via
        # aclose(); the composition root keeps a single adapter instance.
        assert isinstance(first._client, httpx.AsyncClient)
        assert first._client is not second._client

    async def test_aclose_is_idempotent(self):
        client = _client_for(_respond([], 200))
        await client.aclose()
        await client.aclose()


class TestProjectionHelpers:
    def test_missing_optional_fields_default_safely(self):
        config = _to_execution_config({"url": "http://x/mcp"}, "fallback-name")
        assert config.name == "fallback-name"
        assert config.transport == McpTransportType.HTTP
        assert config.headers == {}
        assert config.env == {}
        assert config.auth_token is None

    def test_structurally_invalid_entry_raises_for_upstream_isolation(self):
        """An entry without a URL for http transport cannot become a config.

        The projection RAISES so the built-in resolution can isolate the
        malformed source as degraded capability info (REQ-040) instead of
        booting a tool with an unusable connection.
        """
        with pytest.raises((ValueError, TypeError)):
            # ``McpServerConfig`` (pydantic model_validator) raises ValueError
            # for the http-transport contract; a non-string transport raises
            # TypeError from the enum conversion. Anything else does not
            # belong to the isolation contract of this projection.
            _to_execution_config({}, "malformed")

    def test_non_string_scalar_values_are_coerced(self):
        config = _to_execution_config(
            {"headers": {"X-Count": 3}, "env": {"RETRIES": 5}, "url": "http://x/mcp"}, "mixed"
        )
        assert config.headers == {"X-Count": "3"}
        assert config.env == {"RETRIES": "5"}
