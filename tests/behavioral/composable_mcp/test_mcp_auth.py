"""Behavioral tests — inbound MCP endpoint authentication (REQ-001..REQ-005).

Entry point: the streamable-HTTP MCP server mounted at ``/mcp`` on the REAL
app, reached with the REAL ``mcp`` SDK streamable client over a real socket.

Pinned behavior:
* ``/mcp`` exists and denies unauthenticated discovery/execution (401).
* An INVALID bearer JWT does NOT fall back to a valid API key (REQ-002 /
  "presented invalid bearer JWT fails rather than falling back").
* A valid JWT and a valid per-user API key establish the SAME user context
  (REQ-004): identical tool visibility and identical ``composable_users_me``.
"""

import pytest

from tests.behavioral.composable_mcp.conftest import (
    call_mcp_tool,
    get_stub_jwks,
    list_mcp_tool_names,
    mcp_post_status,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

NO_CREDENTIALS: dict[str, str] = {}


class TestMcpEndpointGuarded:
    async def test_mcp_requires_credentials_to_initialize(self, mcp_base_url):
        """REQ-003: missing credentials → 401, no user data, no tools."""
        resp = await mcp_post_status(mcp_base_url, headers=NO_CREDENTIALS, method="initialize")
        assert resp.status_code == 401, f"expected 401 for unauthenticated /mcp, got {resp.status_code}: {resp.text}"
        body = resp.text.lower()
        assert "composable_agents" not in body and "composable_threads" not in body

    async def test_mcp_requires_credentials_to_call_tools(self, mcp_base_url):
        """REQ-003: unauthenticated ``tools/call`` is denied — no execution."""
        resp = await mcp_post_status(mcp_base_url, headers=NO_CREDENTIALS, method="tools/call")
        assert resp.status_code == 401, (
            f"expected 401 for unauthenticated tools/call, got {resp.status_code}: {resp.text}"
        )

    async def test_unauthenticated_mcp_session_lists_no_tools(self, mcp_base_url):
        """A real unauthenticated MCP client session must observe zero tools."""
        # The endpoint must exist and deny (guards against an accidental vacuous
        # pass on 404): the raw probe pins the HTTP-level denial too.
        resp = await mcp_post_status(mcp_base_url, headers=NO_CREDENTIALS, method="initialize")
        assert resp.status_code == 401, f"unauthenticated POST /mcp must be denied, got {resp.status_code}"
        names = await list_mcp_tool_names(mcp_base_url, headers=NO_CREDENTIALS)
        assert not names, f"unauthenticated MCP session observed tools: {names}"


class TestNoJwtToApiKeyFallback:
    async def test_invalid_jwt_with_valid_api_key_is_rejected(self, mcp_base_url, users):
        """A presented invalid bearer JWT must fail WITHOUT falling back to
        the valid per-user API key sent alongside it."""
        bad_token = get_stub_jwks().issue_token(sub="user-a")[:-4] + "zzzz"  # corrupted signature
        headers = {"Authorization": f"Bearer {bad_token}", "X-API-Key": users["a"]["key"]}
        resp = await mcp_post_status(mcp_base_url, headers=headers, method="initialize")
        assert resp.status_code == 401, (
            f"invalid JWT must not fall back to a valid API key — got {resp.status_code}: {resp.text}"
        )

    async def test_invalid_jwt_session_cannot_list_or_call(self, mcp_base_url, users):
        bad_token = get_stub_jwks().issue_token(sub="user-a")[:-4] + "zzzz"
        headers = {"Authorization": f"Bearer {bad_token}", "X-API-Key": users["a"]["key"]}
        resp = await mcp_post_status(mcp_base_url, headers=headers, method="initialize")
        assert resp.status_code == 401, f"invalid JWT must not fall back to the API key, got {resp.status_code}"
        names = await list_mcp_tool_names(mcp_base_url, headers=headers)
        assert not names, "invalid-JWT session fell back to the API key and listed tools"


class TestDualAuthParity:
    async def _first_platform_tool(self, mcp_base_url, headers):
        names = await list_mcp_tool_names(mcp_base_url, headers=headers)
        assert names, f"valid credentials failed to list tools: {names}"
        return names

    async def test_jwt_and_api_key_see_the_same_toolset(self, mcp_base_url, users, jwt_users):
        names_api_key = await self._first_platform_tool(mcp_base_url, users["a"]["headers"])
        names_jwt = await self._first_platform_tool(mcp_base_url, jwt_users["a"])
        assert names_jwt == names_api_key, "JWT and API-key authentication must expose the same MCP toolset"

    async def test_jwt_and_api_key_resolve_the_same_user(self, mcp_base_url, users, jwt_users):
        """REQ-004: both methods establish the same user identity."""
        me_via_key = await call_mcp_tool(mcp_base_url, users["a"]["headers"], "composable_users_me")
        assert me_via_key["ok"], f"composable_users_me failed via API key: {me_via_key}"
        me_via_jwt = await call_mcp_tool(mcp_base_url, jwt_users["a"], "composable_users_me")
        assert me_via_jwt["ok"], f"composable_users_me failed via JWT: {me_via_jwt}"

        key_user_id = me_via_key["result"].get("user_id")
        jwt_user_id = me_via_jwt["result"].get("user_id")
        assert key_user_id == "user-a", f"API-key path resolved unexpected user: {key_user_id}"
        assert jwt_user_id == "user-a", f"JWT path resolved unexpected user: {jwt_user_id}"

    async def test_jwt_user_sees_only_own_data(self, mcp_base_url, jwt_users):
        """The JWT-established context drives the same user-scoped data."""
        result = await call_mcp_tool(mcp_base_url, jwt_users["a"], "composable_agents_list")
        assert result["ok"], f"composable_agents_list failed via JWT: {result}"
        serialized = str(result["result"])
        assert "user-b" not in serialized, "JWT context must not expose another user's data"


class TestCredentialLeakage:
    async def test_mcp_error_never_echoes_the_api_key(self, mcp_base_url, users):
        """REQ-009: safe structured errors never echo credentials."""
        result = await call_mcp_tool(
            mcp_base_url, users["a"]["headers"], "composable_agents_get", {"agent_name": "does-not-exist"}
        )
        # The call fails (resource not found) but must never leak the key.
        assert users["a"]["key"] not in str(result)
