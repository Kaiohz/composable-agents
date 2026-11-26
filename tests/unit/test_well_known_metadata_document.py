"""Unit tests — the public protected-resource metadata document (REQ-002).

Pins the advertised JSON EXACTLY, at the composition-root route
(``src.main.oauth_protected_resource``), for the QA BUG-003 contract:

* ``authorization_servers`` comes from the ``MCP_AUTHORIZATION_SERVER``
  setting (an explicit PUBLIC issuer wins over the logto fallback);
* an empty setting advertises an EMPTY list — a public document never
  carries an issuer the deployment cannot vouch for;
* a ``Host: <allowed-host>:443`` request (QA BUG-002) derives the SAME
  canonical, port-less ``resource`` as the port-less Host form.
"""

import json

import pytest
from starlette.requests import Request

import src.main as src_main

PROD_HOST = "composables-agent.soludev.tech"
PROD_META = "https://composables-agent.soludev.tech/mcp"


def _well_known_request(host: str) -> Request:
    """Build a minimal ASGI request carrying only the ``Host`` header."""
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/.well-known/oauth-protected-resource",
            "headers": [(b"host", host.encode("latin-1"))],
        },
    )


class TestAdvertisedDocumentJson:
    async def test_document_json_is_pinned_with_the_configured_authorization_server(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The EXACT public document: resource from the configured origin,
        the authorization server from the explicit setting."""
        settings = src_main.settings
        monkeypatch.setattr(settings, "mcp_public_base_url", "https://composables-agent.soludev.tech")
        monkeypatch.setattr(
            settings,
            "mcp_allowed_hosts",
            ["127.0.0.1:*", "localhost:*", "[::1]:*", PROD_HOST],
        )
        monkeypatch.setattr(settings, "mcp_authorization_server", "https://logto.soludev.tech/oidc")

        response = await src_main.oauth_protected_resource(_well_known_request(PROD_HOST))

        assert response.status_code == 200
        assert json.loads(response.body) == {
            "resource": PROD_META,
            "authorization_servers": ["https://logto.soludev.tech/oidc"],
        }

    async def test_document_advertises_no_server_when_the_setting_is_empty(self, monkeypatch: pytest.MonkeyPatch):
        """QA BUG-003: with neither ``MCP_AUTHORIZATION_SERVER`` nor
        ``LOGTO_URL`` configured the document advertises an EMPTY list —
        never a malformed issuer (the historical ``['/oidc']`` shape)."""
        settings = src_main.settings
        monkeypatch.setattr(settings, "mcp_public_base_url", "https://composables-agent.soludev.tech")
        monkeypatch.setattr(settings, "mcp_authorization_server", "")

        response = await src_main.oauth_protected_resource(_well_known_request(PROD_HOST))

        assert response.status_code == 200
        assert json.loads(response.body) == {
            "resource": PROD_META,
            "authorization_servers": [],
        }

    async def test_document_is_identical_for_the_default_port_host_form(self, monkeypatch: pytest.MonkeyPatch):
        """QA BUG-002: ``Host: <allowed-host>:443`` (the allow-listed Host
        fallback) derives the canonical, port-less origin."""
        settings = src_main.settings
        monkeypatch.setattr(settings, "mcp_public_base_url", "")
        monkeypatch.setattr(
            settings,
            "mcp_allowed_hosts",
            ["127.0.0.1:*", "localhost:*", "[::1]:*", PROD_HOST],
        )
        monkeypatch.setattr(settings, "mcp_authorization_server", "https://logto.soludev.tech/oidc")

        response = await src_main.oauth_protected_resource(_well_known_request(f"{PROD_HOST}:443"))

        assert response.status_code == 200
        assert json.loads(response.body) == {
            "resource": PROD_META,
            "authorization_servers": ["https://logto.soludev.tech/oidc"],
        }
