"""Unit tests for the Files MCP client credential resolution (feat-mcp-oauth2
REQ-004 mirror): ``auth_mode: user_token`` servers MUST receive the
originating user's credential — without it the dual-auth guard denies every
initialize and temporary uploads degrade."""

from src.domain.entities.auth.auth_context import AuthContext
from src.domain.entities.mcp_server_config import McpAuthMode, McpServerConfig
from src.infrastructure.database.rls_context import (
    current_auth_context,
    current_user_id,
)
from src.infrastructure.mcp.files_client import McpStreamableFilesClient


def _server(auth_mode: str) -> McpServerConfig:
    return McpServerConfig(
        name="raganything-files",
        transport="http",
        url="http://raganything.internal:8000/files/mcp",
        auth_mode=McpAuthMode(auth_mode),
    )


def _install_context(method: str, credential: str):
    """Set BOTH auth contextvars; return the tokens for the finally resets."""
    context = AuthContext(user_id="76fnky2ndkzi", method=method, raw_credential=credential)
    context_token = current_auth_context.set(context)
    user_token = current_user_id.set("76fnky2ndkzi")
    return context_token, user_token


async def test_user_token_mode_injects_jwt_bearer():
    client = McpStreamableFilesClient(tool_timeout=5.0)
    context_token, user_token = _install_context("jwt", "eyJhbGciOi.eyJzdWIi.faketoken")
    try:
        headers = client._resolved_headers(_server("user_token"))
        assert headers["Authorization"] == "Bearer eyJhbGciOi.eyJzdWIi.faketoken"
    finally:
        current_auth_context.reset(context_token)
        current_user_id.reset(user_token)


async def test_user_token_mode_injects_api_key():
    client = McpStreamableFilesClient(tool_timeout=5.0)
    context_token, user_token = _install_context("api_key", "cpk_test_key")
    try:
        headers = client._resolved_headers(_server("user_token"))
        assert headers["X-API-Key"] == "cpk_test_key"
    finally:
        current_auth_context.reset(context_token)
        current_user_id.reset(user_token)


async def test_headers_mode_stays_configured_only():
    client = McpStreamableFilesClient(tool_timeout=5.0)
    context_token, user_token = _install_context("jwt", "unused-bearer")
    try:
        server = _server("headers").model_copy(update={"headers": {"X-Static": "1"}})
        headers = client._resolved_headers(server)
        assert headers == {"X-Static": "1"}
    finally:
        current_auth_context.reset(context_token)
        current_user_id.reset(user_token)
