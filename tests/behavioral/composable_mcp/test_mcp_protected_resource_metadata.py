"""Behavioral tests — public `/.well-known/oauth-protected-resource` (REQ-002).

feat-mcp-oauth2-dual-auth, composable-agents slice. Entry point: the REAL
FastAPI app (httpx AsyncClient over ASGITransport), i.e. the production router
chain including the CORS middleware.

Delegation contract pinned here:

* ``GET /.well-known/oauth-protected-resource`` is PUBLIC (no credentials,
  no auth dependency at all) and returns
  ``{"resource": "https://<host>/mcp", "authorization_servers": [<Logto OIDC issuer>]}``
  — ``<host>`` comes from the app's configured request host (the tests present
  the production Host header; scheme is https as in production behind TLS), and
  the authorization-server entry is derived from the configured Logto base URL
  (``<logto_url>/oidc``).
* ``OPTIONS`` preflight on the well-known path is answered by the CORS
  middleware WITHOUT any auth check (the configured
  ``settings.allowed_origins`` origin is echoed back).

RED phase: the route does not exist yet — expected to 404 (and the preflight to
404 as well) until the feature is implemented.
"""

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

WELL_KNOWN = "/.well-known/oauth-protected-resource"
PROD_HOST = "composables-agent.soludev.tech"
EXPECTED_RESOURCE = f"https://{PROD_HOST}/mcp"
CONFIGURED_BROWSER_ORIGIN = "http://localhost:8080"


class TestProtectedResourceMetadata:
    async def test_metadata_is_public_and_shaped_like_the_discovery_spec(self, client):
        """Unauthenticated GET returns the static protected-resource metadata."""
        import src.dependencies as deps

        resp = await client.get(WELL_KNOWN, headers={"Host": PROD_HOST})
        assert resp.status_code == 200, (
            f"the well-known route must be PUBLIC and return metadata, got {resp.status_code}: {resp.text}"
        )
        body = resp.json()
        assert body["resource"] == EXPECTED_RESOURCE, f"unexpected `resource`: {body}"
        expected_as = f"{deps.settings.logto_url.rstrip('/')}/oidc"
        assert body["authorization_servers"] == [expected_as], (
            f"`authorization_servers` must be the configured Logto OIDC issuer "
            f"({expected_as!r}), got: {body.get('authorization_servers')}"
        )

    async def test_metadata_is_public_even_with_garbage_credentials(self, client):
        """The route must be reachable with (invalid) credentials present too —
        it is a static public document, never an authenticated resource."""
        resp = await client.get(
            WELL_KNOWN,
            headers={"Host": PROD_HOST, "Authorization": "Bearer not-a-real-token"},
        )
        assert resp.status_code == 200, (
            f"the well-known route must stay public, got {resp.status_code}: {resp.text}"
        )

    async def test_options_preflight_answered_by_cors_without_auth(self, client):
        """A CORS preflight (OPTIONS) on the well-known path is answered by the
        CORS middleware without any auth check."""
        resp = await client.options(
            WELL_KNOWN,
            headers={
                "Origin": CONFIGURED_BROWSER_ORIGIN,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code in (200, 204), (
            f"preflight must be answered by CORS without auth, got {resp.status_code}: {resp.text}"
        )
        assert resp.headers.get("access-control-allow-origin") == CONFIGURED_BROWSER_ORIGIN, (
            f"preflight must echo the configured browser origin, got: {resp.headers.get('access-control-allow-origin')!r}"
        )

    async def test_metadata_refuses_a_spoofable_foreign_host_without_configuration(self, client):
        """Review M-2: without a configured `MCP_PUBLIC_BASE_URL`, a non-allow-
        listed request Host can NEVER shape the advertised `resource` URL —
        the discovery document is simply not served at such an origin (404,
        no derivation details reflected)."""
        resp = await client.get(
            WELL_KNOWN,
            headers={"Host": "dns-rebind.evil.example"},
        )
        assert resp.status_code == 404, (
            "the metadata document must not be served with an attacker-shaped "
            f"resource URL, got {resp.status_code}: {resp.text}"
        )
        assert "resource" not in resp.text, f"no derived resource URL may be echoed, got: {resp.text}"
