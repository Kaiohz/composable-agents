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
            return httpx.Response(400, json={"detail": "registry stack trace with secrets"})

        client = _client_for(handler)
        try:
            with pytest.raises(McpRegistryRequestError) as exc_info:
                await client.get_execution_config("alpha")
        finally:
            await client.aclose()
        assert str(exc_info.value.detail) == ErrorMessage.MCP_REGISTRY_REQUEST_FAILED.format(status=400)

    async def test_registry_5xx_answer_is_safe_and_retryable(self):
        """REQ-TX-003: a registry 502/503/504 IS an availability signal."""

        def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001 — transport contract
            return httpx.Response(503, json={"detail": "registry unavailable body"})

        client = _client_for(handler)
        try:
            with pytest.raises(McpRegistryUnavailableError) as exc_info:
                await client.get_execution_config("alpha")
        finally:
            await client.aclose()
        assert str(exc_info.value.detail) == ErrorMessage.MCP_REGISTRY_UNAVAILABLE

    async def test_registration_conflict_surfaces_as_safe_conflict(self):
        """REQ-TX-002: a 409 from a creation surfaces as a conflict error whose
        detail names the candidate server, never the registry body."""
        from src.domain.entities.registered_mcp_server import McpServerWrite
        from src.domain.errors.mcp import McpRegistryConflictError

        def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001 — transport contract
            return httpx.Response(409, json={"detail": "body must never be echoed"})

        client = _client_for(handler)
        try:
            with pytest.raises(McpRegistryConflictError) as exc_info:
                await client.create_server(McpServerWrite(name="taken", transport="http", url="http://x/mcp"))
        finally:
            await client.aclose()
        assert "already registered" in str(exc_info.value.detail)
        assert "body must never be echoed" not in str(exc_info.value.detail)


class TestValidateServer:
    async def test_real_registry_payload_parses_as_a_validated_candidate(self):
        """BUG-001: the deployed mcp-raganything registry answers
        `McpServerValidateResponse{tool_count}` (200) for a VALIDATED
        candidate — it carries neither `name` nor `valid`; the adapter must
        parse it (never raise a raw ValidationError) and treat it as valid."""
        from src.domain.entities.registered_mcp_server import McpServerWrite

        client = _client_for(_respond({"tool_count": 1}))
        try:
            report = await client.validate_server(McpServerWrite(name="s", transport="http", url="http://x/mcp"))
        finally:
            await client.aclose()
        assert report.valid is True, "a 200 answer is a validated candidate"
        assert report.tool_count == 1

    async def test_full_report_payload_still_parses(self):
        """Fuller deployments may answer the whole report shape: the adapter
        still surfaces an EXPLICIT invalid verdict."""
        from src.domain.entities.registered_mcp_server import McpServerWrite

        client = _client_for(_respond({"name": "s", "valid": False, "tool_count": 0, "errors": ["bad url"]}))
        try:
            report = await client.validate_server(McpServerWrite(name="s", transport="http", url="http://x/mcp"))
        finally:
            await client.aclose()
        assert report.valid is False
        assert report.errors == ["bad url"]

    async def test_payload_drift_is_a_controlled_error_never_a_raw_500(self):
        """A registry contract drift (unparseable answer) surfaces as a
        controlled McpRegistryRequestError — never as an escaping pydantic
        ValidationError (which would reach the client as a 500)."""
        from src.domain.entities.registered_mcp_server import McpServerWrite
        from src.domain.errors.mcp import McpRegistryRequestError

        client = _client_for(_respond({"tool_count": "not-an-int"}))
        try:
            with pytest.raises(McpRegistryRequestError):
                await client.validate_server(McpServerWrite(name="s", transport="http", url="http://x/mcp"))
        finally:
            await client.aclose()

    async def test_non_dict_payload_is_an_unusable_answer(self):
        from src.domain.entities.registered_mcp_server import McpServerWrite
        from src.domain.errors.mcp import McpRegistryRequestError

        client = _client_for(_respond(["not", "a", "dict"]))
        try:
            with pytest.raises(McpRegistryRequestError):
                await client.validate_server(McpServerWrite(name="s", transport="http", url="http://x/mcp"))
        finally:
            await client.aclose()

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
        with pytest.raises(ValueError):
            _to_execution_config({}, "malformed")

    def test_non_string_scalar_values_are_coerced(self):
        config = _to_execution_config(
            {"headers": {"X-Count": 3}, "env": {"RETRIES": 5}, "url": "http://x/mcp"}, "mixed"
        )
        assert config.headers == {"X-Count": "3"}
        assert config.env == {"RETRIES": "5"}

    async def test_non_json_body_is_a_controlled_error_never_a_raw_500(self):
        """A non-JSON registry answer is the same contract-drift class as an
        unparseable payload: controlled McpRegistryRequestError (status-only
        detail), never an escaping raw parse error."""
        from src.domain.errors.mcp import McpRegistryRequestError

        def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001 — transport contract
            return httpx.Response(200, text="<html>registry hiccup</html>")

        client = _client_for(handler)
        try:
            with pytest.raises(McpRegistryRequestError):
                await client.list_servers()
        finally:
            await client.aclose()
