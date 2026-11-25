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
