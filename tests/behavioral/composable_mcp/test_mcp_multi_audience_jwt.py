"""Behavioral tests — multi-audience JWT acceptance at `/mcp` (REQ-003).

feat-mcp-oauth2-dual-auth, composable-agents slice. Entry point: the
streamable-HTTP MCP endpoint mounted at ``/mcp`` on the REAL app (session
uvicorn server), authenticating with REAL signed tokens verified through the
production JWT chain (the Logto JWKS boundary is the only mocked external
system, via the local ``StubJwksServer`` — minted through the ``stub_jwks``
fixture, see its docstring for why a direct import is forbidden here).

Delegation contract pinned here:

* a new ``JWT_AUDIENCES`` setting (comma-separated list) is accepted; a Bearer
  JWT whose ``aud`` is ANY listed audience authenticates (method ``jwt``, RLS
  context established exactly as today — REQ-011);
* the existing ``JWT_AUDIENCE`` single value keeps working (listed as the first
  audience by the test environment);
* a JWT whose ``aud`` is NOT listed is rejected with 401.

The test environment sets ``JWT_AUDIENCES=test-audience,opencode-audience``
(legacy value first). Note: the fully-unset ``JWT_AUDIENCES`` case cannot be
observed in the same process (env is process-wide); the backward-compatibility
property it protects (single ``JWT_AUDIENCE`` keeps working) is pinned by the
first-audience acceptance here and by the existing JWT suite.

RED phase: ``JWT_AUDIENCES`` is not honored yet — the second-audience token is
expected to be rejected (401) until the feature is implemented.
"""

import os

import pytest

from tests.behavioral.composable_mcp.conftest import (
    call_mcp_tool,
    mcp_post_status,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

#: Legacy single audience (backward-compat half of the list).
LEGACY_AUDIENCE = os.environ.get("JWT_AUDIENCE", "test-audience")
#: Second configured audience (the Logto OpenCode app-id slot). Derived from
#: the concrete test env value so a configured override stays coherent.
_RAW_AUDIENCES = os.environ.get("JWT_AUDIENCES", "")
SECOND_AUDIENCE = _RAW_AUDIENCES.split(",")[1].strip() if "," in _RAW_AUDIENCES else "opencode-audience"
UNLISTED_AUDIENCE = "not-in-jwt-audiences"


class TestMultiAudienceAcceptance:
    async def test_token_for_second_listed_audience_authenticates(self, mcp_base_url, stub_jwks):
        """A JWT whose aud is the second listed audience (OpenCode app-id) is
        accepted at /mcp."""
        token = stub_jwks.issue_token(sub="user-a", audience=SECOND_AUDIENCE)
        resp = await mcp_post_status(
            mcp_base_url, headers={"Authorization": f"Bearer {token}"}, method="initialize"
        )
        assert resp.status_code == 200, (
            f"a second-listed JWT audience must authenticate (got {resp.status_code}): {resp.text}"
        )

    async def test_token_for_single_legacy_audience_still_authenticates(self, mcp_base_url, stub_jwks):
        """The legacy single `JWT_AUDIENCE` value keeps working (one-element
        list: identical behavior to today)."""
        token = stub_jwks.issue_token(sub="user-a", audience=LEGACY_AUDIENCE)
        resp = await mcp_post_status(
            mcp_base_url, headers={"Authorization": f"Bearer {token}"}, method="initialize"
        )
        assert resp.status_code == 200, (
            f"the legacy JWT_AUDIENCE value must keep authenticating (got {resp.status_code}): {resp.text}"
        )

    async def test_token_for_unlisted_audience_is_rejected(self, mcp_base_url, stub_jwks):
        """A JWT whose aud is not in the configured audience list is rejected."""
        token = stub_jwks.issue_token(sub="user-a", audience=UNLISTED_AUDIENCE)
        resp = await mcp_post_status(
            mcp_base_url, headers={"Authorization": f"Bearer {token}"}, method="initialize"
        )
        assert resp.status_code == 401, (
            f"an unlisted audience must be rejected (got {resp.status_code}): {resp.text}"
        )


class TestMultiAudienceRls:
    async def test_second_audience_token_establishes_jwt_method_and_rls(self, mcp_base_url, stub_jwks):
        """REQ-011: an accepted second-audience token establishes the exact RLS
        context of today — the resolved user is the sub, method ``jwt``."""
        token = stub_jwks.issue_token(sub="user-a", audience=SECOND_AUDIENCE)
        result = await call_mcp_tool(
            mcp_base_url, {"Authorization": f"Bearer {token}"}, "composable_users_me"
        )
        assert result["ok"], f"composable_users_me failed for a second-audience token: {result}"
        assert result["result"].get("user_id") == "user-a", (
            f"the RLS context must resolve the JWT sub; got {result['result']!r}"
        )

    async def test_second_audience_token_sees_only_own_data(self, mcp_base_url, stub_jwks):
        """The second-audience-established context is user-scoped like any JWT."""
        token = stub_jwks.issue_token(sub="user-a", audience=SECOND_AUDIENCE)
        result = await call_mcp_tool(
            mcp_base_url, {"Authorization": f"Bearer {token}"}, "composable_agents_list"
        )
        assert result["ok"], f"composable_agents_list failed for a second-audience token: {result}"
        assert "user-b" not in str(result["result"]), (
            "the second-audience JWT context must not expose another user's data"
        )
