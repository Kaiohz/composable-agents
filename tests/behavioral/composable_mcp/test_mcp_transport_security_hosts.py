"""Behavioral tests — explicit MCP `TransportSecuritySettings` (REQ-013).

feat-mcp-oauth2-dual-auth, composable-agents slice. Entry point: the
streamable-HTTP MCP endpoint mounted at ``/mcp`` on the REAL app (session
uvicorn server), probed with valid credentials so the ONLY variable under test
is the transport-security validation of the Host/Origin headers.

Delegation contract pinned here:

* ``build_mcp_asgi_app`` passes an explicit ``TransportSecuritySettings`` with
  ``enable_dns_rebinding_protection=True``, ``allowed_hosts`` from the new
  ``MCP_ALLOWED_HOSTS`` setting and ``allowed_origins`` from
  ``settings.allowed_origins``;
* a request whose Host is the production host succeeds (NO 421) — the
  production regression (Host `composables-agent.soludev.tech` rejected with
  421 by the SDK's local-only default list) is pinned;
* the default-port Host forms (``host:443`` / ``host:80``, QA BUG-002) are
  accepted for an allow-listed host and rejected for a foreign one;
* a non-allowed Host still gets 421 (DNS-rebinding protection stays ENABLED);
* an Origin from ``settings.allowed_origins`` is accepted, a foreign Origin is
  rejected with 403.

The probes intentionally present NON-localhost Host headers (unlike the
historical conftest probes that hit 127.0.0.1 and masked the bug).

RED phase: FastMCP is constructed without ``transport_security`` today → the
SDK auto-enables a localhost-only Host list and an empty Origin allow-list, so
the production-host and allowed-origin probes are expected to fail.
"""

import pytest

from tests.behavioral.composable_mcp.conftest import mcp_post_status

pytestmark = pytest.mark.asyncio(loop_scope="session")

PROD_HOST = "composables-agent.soludev.tech"
#: A Host that is NOT in the test env's MCP_ALLOWED_HOSTS.
FOREIGN_HOST = "dns-rebind.evil.example"
#: From settings.allowed_origins (production browser origin).
ALLOWED_ORIGIN = "http://localhost:8080"
FOREIGN_ORIGIN = "https://evil.example"


class TestTransportSecurityExplicitSettings:
    async def test_production_host_is_accepted(self, mcp_base_url, users):
        """A valid-credential request with the production Host SUCCEEDS (no
        421) — exactly the request that currently fails in production."""
        resp = await mcp_post_status(
            mcp_base_url,
            headers={"Host": PROD_HOST, **users["a"]["headers"]},
            method="initialize",
        )
        assert resp.status_code != 421, (
            f"Host {PROD_HOST} must be in MCP_ALLOWED_HOSTS — got {resp.status_code}: {resp.text}"
        )
        assert resp.status_code == 200, (
            f"the MCP handshake with the production Host must succeed, got {resp.status_code}: {resp.text}"
        )

    async def test_non_allowed_host_still_rejected_with_421(self, mcp_base_url, users):
        """DNS-rebinding protection stays ENABLED: an unknown Host → 421."""
        resp = await mcp_post_status(
            mcp_base_url,
            headers={"Host": FOREIGN_HOST, **users["a"]["headers"]},
            method="initialize",
        )
        assert resp.status_code == 421, (
            f"a non-allowed Host must still be rejected with 421, got {resp.status_code}: {resp.text}"
        )

    async def test_origin_from_allowed_origins_is_accepted(self, mcp_base_url, users):
        """An Origin from `settings.allowed_origins` must not be rejected with
        403 (browser-origin MCP clients otherwise get 403)."""
        resp = await mcp_post_status(
            mcp_base_url,
            headers={"Host": PROD_HOST, "Origin": ALLOWED_ORIGIN, **users["a"]["headers"]},
            method="initialize",
        )
        assert resp.status_code == 200, (
            f"the configured browser origin must be accepted, got {resp.status_code}: {resp.text}"
        )

    async def test_foreign_origin_is_rejected_with_403(self, mcp_base_url, users):
        """A foreign Origin is rejected with 403 (Origins validation enforced,
        not disabled)."""
        resp = await mcp_post_status(
            mcp_base_url,
            headers={"Host": PROD_HOST, "Origin": FOREIGN_ORIGIN, **users["a"]["headers"]},
            method="initialize",
        )
        assert resp.status_code == 403, (
            f"a foreign Origin must be rejected with 403, got {resp.status_code}: {resp.text}"
        )


class TestDefaultPortHostForms:
    """QA BUG-002: the default-port Host forms must reach the MCP endpoint.

    An ingress may forward the port the client dialed — ``Host:
    composables-agent.soludev.tech:443`` is the SAME allow-listed endpoint
    as the port-less form behind the TLS-terminating proxy. The SDK's exact
    Host matcher used to 421 the port form; only the DEFAULT-port suffixes
    (``:443`` https / ``:80`` http) are normalized, a still-foreign Host
    keeps answering 421 with DNS-rebinding protection ENABLED (REQ-013).
    """

    async def test_allowed_host_with_https_default_port_is_accepted(self, mcp_base_url, users):
        """QA curl [35]: the allow-listed Host with an explicit ``:443`` must
        NOT 421 (initialize succeeds)."""
        resp = await mcp_post_status(
            mcp_base_url,
            headers={"Host": f"{PROD_HOST}:443", **users["a"]["headers"]},
            method="initialize",
        )
        assert resp.status_code == 200, (
            f"Host {PROD_HOST}:443 is the allow-listed endpoint behind the TLS proxy — got "
            f"{resp.status_code}: {resp.text}"
        )

    async def test_allowed_host_with_http_default_port_is_accepted(self, mcp_base_url, users):
        """``:80`` is the plain-HTTP default port — normalized the same way."""
        resp = await mcp_post_status(
            mcp_base_url,
            headers={"Host": f"{PROD_HOST}:80", **users["a"]["headers"]},
            method="initialize",
        )
        assert resp.status_code == 200, (
            f"Host {PROD_HOST}:80 is the same allow-listed endpoint — got {resp.status_code}: {resp.text}"
        )

    async def test_foreign_host_with_default_port_suffix_is_still_rejected(self, mcp_base_url, users):
        """DNS-rebinding protection stays ENABLED (REQ-013): a foreign Host
        with a default-port suffix is NOT smuggled through the normalization.
        """
        resp = await mcp_post_status(
            mcp_base_url,
            headers={"Host": f"{FOREIGN_HOST}:443", **users["a"]["headers"]},
            method="initialize",
        )
        assert resp.status_code == 421, (
            f"a foreign Host stays foreign with an explicit port — got {resp.status_code}: {resp.text}"
        )

    async def test_default_port_host_on_401_carries_the_discovery_challenge(self, mcp_base_url):
        """QA curl [36]: without credentials AND with ``MCP_PUBLIC_BASE_URL``
        unset, the ``Host: <allowed-host>:443`` form must still produce the
        ``WWW-Authenticate`` discovery pointer (derived from the canonical,
        port-less origin) instead of a header-less 401 that silently kills
        the OAuth2 discovery chain."""
        resp = await mcp_post_status(
            mcp_base_url,
            headers={"Host": f"{PROD_HOST}:443"},
            method="initialize",
        )
        assert resp.status_code == 401, f"an unauthenticated request is expected to 401, got {resp.status_code}"
        challenge = resp.headers.get("www-authenticate")
        assert challenge == f'Bearer resource_metadata="https://{PROD_HOST}/.well-known/oauth-protected-resource"', (
            f"the 401 on the default-port Host form must carry the discovery challenge, got: {challenge!r}"
        )
