"""Behavioral tests — MCP 401 `WWW-Authenticate` discovery chain (REQ-001).

feat-mcp-oauth2-dual-auth, composable-agents slice. Entry point: the
streamable-HTTP MCP endpoint mounted at ``/mcp`` on the REAL app, served by the
session uvicorn server and probed with the raw JSON-RPC helper (HTTP status +
headers are the behavior under test).

Delegation contract pinned here:

* every 401 emitted by the ``/mcp`` guard (missing credentials, invalid Bearer
  JWT, unknown/revoked API key) carries
  ``WWW-Authenticate: Bearer resource_metadata="https://<host>/.well-known/oauth-protected-resource"``
  — the host is taken from the request ``Host`` header (production sends the
  proxy Host; the tests override it with the production host), the scheme is
  ``https`` because production terminates TLS at the proxy;
* the 401 body stays the existing generic, non-enumerating JSON;
* REQ-011 contract rides along: an invalid Bearer JWT NEVER falls back to a
  valid ``X-API-Key`` present on the same request; a valid Bearer JWT wins over
  a valid API key; the plain ``X-API-Key`` flow is unchanged (REQ-012).

RED phase: the implementation does not exist yet — the ``WWW-Authenticate``
assertions are expected to fail while status/body/precedence assertions may
already pass (they are regression pins).
"""

import pytest

from tests.behavioral.composable_mcp.conftest import (
    call_mcp_tool,
    mcp_post_status,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

NO_CREDENTIALS: dict[str, str] = {}

#: The production Host the tests present to the guard (the test server itself
#: runs on 127.0.0.1; a Host-header override stands in for the TLS proxy).
PROD_HOST = "composables-agent.soludev.tech"

EXPECTED_WWW_AUTHENTICATE = f'Bearer resource_metadata="https://{PROD_HOST}/.well-known/oauth-protected-resource"'


def _host_probe_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Raw-probe headers presenting the production Host (plus extras)."""
    return {"Host": PROD_HOST, **(extra or {})}


def _assert_unauthorized_with_discovery_challenge(resp) -> None:
    """One assertion block for REQ-001: 401 + challenge + generic body."""
    assert resp.status_code == 401, f"expected 401, got {resp.status_code}: {resp.text}"
    assert resp.headers.get("www-authenticate") == EXPECTED_WWW_AUTHENTICATE, (
        "401 must carry the MCP protected-resource discovery challenge "
        f'("WWW-Authenticate: {EXPECTED_WWW_AUTHENTICATE}"), got: {resp.headers.get("www-authenticate")!r}'
    )
    assert resp.json() == {"detail": "Invalid or missing credentials"}, (
        f"401 body must stay the existing generic, non-enumerating JSON, got: {resp.text}"
    )


class TestWwwAuthenticateChallengeOn401:
    async def test_missing_credentials_401_carries_discovery_challenge(self, mcp_base_url, stub_jwks):
        """Missing credentials → 401 + `WWW-Authenticate` + unchanged body."""
        resp = await mcp_post_status(mcp_base_url, headers=_host_probe_headers(), method="initialize")
        _assert_unauthorized_with_discovery_challenge(resp)

    async def test_invalid_bearer_jwt_401_carries_discovery_challenge(self, mcp_base_url, stub_jwks):
        """An INVALID bearer JWT → 401 + `WWW-Authenticate` + unchanged body."""
        bad_token = stub_jwks.issue_token(sub="user-a")[:-4] + "zzzz"
        resp = await mcp_post_status(
            mcp_base_url,
            headers=_host_probe_headers({"Authorization": f"Bearer {bad_token}"}),
            method="initialize",
        )
        _assert_unauthorized_with_discovery_challenge(resp)

    async def test_unknown_api_key_401_carries_discovery_challenge(self, mcp_base_url, stub_jwks):
        """An UNKNOWN `X-API-Key` → 401 + `WWW-Authenticate` + unchanged body."""
        resp = await mcp_post_status(
            mcp_base_url,
            headers=_host_probe_headers({"X-API-Key": "cpk_unknown_never_issued"}),
            method="initialize",
        )
        _assert_unauthorized_with_discovery_challenge(resp)

    async def test_revoked_api_key_401_carries_discovery_challenge(self, mcp_base_url, client, users):
        """A REVOKED API key (created + revoked through the REAL API-key
        routes) → 401 + `WWW-Authenticate` + unchanged body."""
        created = await client.post("/api/v1/api-keys", headers=users["a"]["headers"], json={"name": "revoked-probe"})
        assert created.status_code == 201, created.text
        body = created.json()
        plaintext, key_id = body["plaintext"], body["id"]
        revoked = await client.delete(f"/api/v1/api-keys/{key_id}", headers=users["a"]["headers"])
        assert revoked.status_code == 204, revoked.text

        resp = await mcp_post_status(
            mcp_base_url,
            headers=_host_probe_headers({"X-API-Key": plaintext}),
            method="initialize",
        )
        _assert_unauthorized_with_discovery_challenge(resp)


class TestNoJwtToApiKeyFallback:
    async def test_invalid_jwt_with_valid_api_key_is_rejected_with_challenge(self, mcp_base_url, users, stub_jwks):
        """An invalid bearer JWT must fail WITH the discovery challenge and
        never fall back to the valid per-user API key sent alongside it."""
        bad_token = stub_jwks.issue_token(sub="user-a")[:-4] + "zzzz"
        resp = await mcp_post_status(
            mcp_base_url,
            headers=_host_probe_headers({"Authorization": f"Bearer {bad_token}", "X-API-Key": users["a"]["key"]}),
            method="initialize",
        )
        _assert_unauthorized_with_discovery_challenge(resp)


class TestAuthPrecedenceRegressionPins:
    async def test_valid_jwt_wins_over_valid_api_key_of_other_user(self, mcp_base_url, users, stub_jwks):
        """Both a valid Bearer JWT (user-a) and a valid API key (user-b) → the
        JWT wins: the established RLS context belongs to the JWT's user."""
        token = stub_jwks.issue_token(sub="user-a")
        headers = {"Authorization": f"Bearer {token}", "X-API-Key": users["b"]["key"]}
        result = await call_mcp_tool(mcp_base_url, headers, "composable_users_me")
        assert result["ok"], f"composable_users_me failed on a dual-credential request: {result}"
        assert result["result"].get("user_id") == "user-a", (
            f"JWT must take precedence over the API key; got user {result['result'].get('user_id')!r}"
        )

    async def test_valid_api_key_flow_unchanged(self, mcp_base_url, users):
        """REQ-012 regression pin: the plain `X-API-Key` flow still
        authenticates and resolves its owner (unchanged by the OAuth work)."""
        result = await call_mcp_tool(mcp_base_url, users["a"]["headers"], "composable_users_me")
        assert result["ok"], f"composable_users_me failed via a valid API key: {result}"
        assert result["result"].get("user_id") == "user-a"


class TestChallengeOriginHardening:
    """Review M-2: the discovery URL is never derived from a raw, spoofable
    request Host. The configured `MCP_PUBLIC_BASE_URL` wins when set (unit
    precedence), else ONLY an allow-listed request Host is used (this suite's
    fallback branch): a FOREIGN Host gets a 401 WITHOUT any
    `WWW-Authenticate` discovery pointer, and every 401 carries
    `Cache-Control: no-store` so no shared cache can replay a challenge."""

    async def test_foreign_host_on_401_never_receives_a_discovery_url(self, mcp_base_url):
        """A spoofed Host is answered 401 (unchanged semantics) but the
        response advertises NO resource_metadata URL for that Host."""
        resp = await mcp_post_status(
            mcp_base_url,
            headers={"Host": "dns-rebind.evil.example", "X-API-Key": "cpk_unknown"},
            method="initialize",
        )
        assert resp.status_code == 401, f"a foreign-Host probe stays 401, got {resp.status_code}"
        challenge = resp.headers.get("www-authenticate")
        assert challenge is None, (
            f"a non-allow-listed request Host must NEVER be echoed into the discovery challenge, got: {challenge!r}"
        )
        assert resp.json() == {"detail": "Invalid or missing credentials"}, (
            f"the 401 body stays the generic JSON regardless of the Host, got: {resp.text}"
        )

    async def test_allowed_host_challenge_is_derived_from_the_allowlist(self, mcp_base_url):
        """The allow-listed production Host IS a valid fallback origin: the
        challenge keeps pointing at the production well-known URL."""
        resp = await mcp_post_status(mcp_base_url, headers={"Host": PROD_HOST}, method="initialize")
        assert resp.status_code == 401
        assert resp.headers.get("www-authenticate") == EXPECTED_WWW_AUTHENTICATE

    async def test_401_challenge_response_is_never_cacheable(self, mcp_base_url):
        """Every 401 carries `Cache-Control: no-store` (challenge responses
        must not be cached and replayed to a different client)."""
        resp = await mcp_post_status(mcp_base_url, headers=_host_probe_headers(), method="initialize")
        assert resp.status_code == 401
        assert resp.headers.get("cache-control") == "no-store", (
            f"401 responses must not be cacheable, got cache-control={resp.headers.get('cache-control')!r}"
        )
