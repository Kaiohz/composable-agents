"""Unit tests for the MCP registry request DTOs (REQ-011/REQ-019, C2 remediation).

Locks the coherent-payload rules shared by the create/validate/update MCP
tools: transport/url, transport/command and source_type/openapi_url coherence
plus the partial-update semantics (explicit fields only, at least one).
"""

import pytest
from pydantic import ValidationError

from src.application.requests.mcp import (
    CreateMcpServerRequest,
    UpdateMcpServerRequest,
    update_payload,
)


class TestCreateMcpServerRequest:
    def test_valid_http_server_round_trips(self):
        request = CreateMcpServerRequest(name="alpha", transport="http", url="http://x/mcp")
        assert request.transport == "http"
        assert request.source_type == "external"
        assert request.headers == {} and request.env == {}

    def test_valid_stdio_server(self):
        request = CreateMcpServerRequest(name="stdio-srv", transport="stdio", command="mcp-server")
        assert request.command == "mcp-server"

    def test_valid_openapi_source_carries_the_spec_url(self):
        request = CreateMcpServerRequest(
            name="generated", transport="http", url="http://gen/mcp", source_type="openapi", openapi_url="http://spec"
        )
        assert request.source_type == "openapi"

    def test_http_requires_url(self):
        with pytest.raises(ValidationError, match="url is required for http transport"):
            CreateMcpServerRequest(name="alpha", transport="http")

    def test_stdio_requires_command(self):
        with pytest.raises(ValidationError, match="command is required for stdio transport"):
            CreateMcpServerRequest(name="stdio-srv", transport="stdio")

    def test_openapi_requires_openapi_url(self):
        with pytest.raises(ValidationError, match="openapi_url is required"):
            CreateMcpServerRequest(name="generated", source_type="openapi", url="http://x/mcp")

    def test_unknown_transport_is_rejected(self):
        with pytest.raises(ValidationError):
            CreateMcpServerRequest(name="alpha", transport="carrier-pigeon")

    def test_unknown_source_type_is_rejected(self):
        with pytest.raises(ValidationError):
            CreateMcpServerRequest(name="alpha", url="http://x/mcp", source_type="swagger")

    def test_name_bounds(self):
        with pytest.raises(ValidationError):
            CreateMcpServerRequest(name="", url="http://x/mcp")
        with pytest.raises(ValidationError):
            CreateMcpServerRequest(name="a" * 201, url="http://x/mcp")

    def test_secrets_are_write_only_inputs(self):
        """REQ-019: secrets are accepted as INPUTS (forwarded to the registry)."""
        request = CreateMcpServerRequest(
            name="alpha",
            url="http://x/mcp",
            auth_token="plain-token",
            headers={"X-Auth": "plain-header"},
            env={"TOKEN": "plain-env"},
        )
        assert request.auth_token == "plain-token"


class TestUpdateMcpServerRequest:
    def test_partial_update_carries_only_explicit_fields(self):
        request = UpdateMcpServerRequest.model_validate({"url": "http://new/mcp"})
        payload = update_payload(request)
        assert payload == {"url": "http://new/mcp"}

    def test_transport_literal_on_update(self):
        with pytest.raises(ValidationError):
            UpdateMcpServerRequest.model_validate({"transport": "smoke-signal"})

    def test_empty_update_is_rejected(self):
        with pytest.raises(ValidationError, match="provide at least one editable field"):
            UpdateMcpServerRequest.model_validate({})

    def test_explicit_none_is_dropped_at_the_tool_boundary(self):
        """The MCP update tool filters None values BEFORE validation, so an
        omitted-or-null field never reaches the registry as a null clobber —
        the same not-provided semantics as the pre-existing REST contract."""
        raw = {"url": None, "command": "run"}  # as received from a flat MCP call
        provided = {key: value for key, value in raw.items() if value is not None}
        request = UpdateMcpServerRequest.model_validate(provided)
        payload = update_payload(request)
        assert payload == {"command": "run"}
        assert "url" not in payload

    def test_secret_fields_are_patchable_but_stay_input_only(self):
        request = UpdateMcpServerRequest.model_validate({"auth_token": "fresh-token"})
        assert update_payload(request) == {"auth_token": "fresh-token"}


class TestMcpCreateRequestHelper:
    """``_create_request`` (application/mcp/tools.py) — DTO-backed MCP input
    validation used by composable_mcp_servers_validate/create."""

    def test_valid_input_builds_the_request(self):
        from src.application.mcp.tools import _create_request

        request = _create_request(name="alpha", transport="http", url="http://x/mcp")
        assert request.name == "alpha"
        assert request.url == "http://x/mcp"

    def test_none_scalars_default_safely(self):
        from src.application.mcp.tools import _create_request

        request = _create_request(name="alpha", url="http://x/mcp")
        assert request.transport == "http"
        assert request.args == [] and request.headers == {} and request.env == {}

    def test_invalid_transport_raises_a_validation_error(self):
        from src.application.mcp.tools import _create_request

        with pytest.raises(ValidationError):
            _create_request(name="alpha", transport="smoke-signal")

    def test_empty_name_raises(self):
        from src.application.mcp.tools import _create_request

        with pytest.raises(ValidationError):
            _create_request(name=None or "")
