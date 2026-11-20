"""Behavioral tests — MCP registry PARTIAL update contract (BUG-001/002/003).

Entry point: the REAL mounted MCP server (`tools/call` of
``composable_mcp_servers_update``) against the registry boundary mock; the
stub mirrors the REAL mcp-raganything contract exactly (partial merge,
tool_count preservation on an unchanged URL, coherent-merged-body 422s), and
the trusted ``/reveal`` endpoint of the SAME stub is read directly (same user
credential, mirroring the QA evidence) to observe the preserved secrets.

Pinned behavior:
* a partial update preserves the unspecified ``auth_token`` / ``headers`` /
  ``env`` and reports success without echoing secrets (BUG-001);
* ``tool_count`` of a reachable server is NOT reset to 0 by an update whose
  URL is unchanged (BUG-003);
* an incoherent update (merged spec loses its required url / openapi_url) is
  rejected LOCALLY with the stable, actionable coherence message — never the
  opaque forwarded 422 (BUG-002).
"""

import httpx
import pytest

from tests.behavioral.composable_mcp.conftest import call_mcp_tool

pytestmark = pytest.mark.asyncio(loop_scope="session")

SERVER_NAME = "qa-partial-update"
TOKEN_SECRET = "qa-partial-token-secret"
HEADER_SECRET = "qa-original-header-secret"
ENV_SECRET = "qa-original-env-secret"

_SEEDED_ENTRY = {
    "name": SERVER_NAME,
    "transport": "http",
    "url": "http://127.0.0.1:18999/mcp",
    "headers": {"X-Original": HEADER_SECRET},
    "env": {"QA_ORIGINAL": ENV_SECRET},
    "auth_token": TOKEN_SECRET,
    "tool_count": 3,
    "source_type": "external",
    "openapi_url": None,
    "created_at": "2026-11-20T00:00:00+00:00",
    "updated_at": "2026-11-20T00:00:00+00:00",
}


def _seed_registry(stub_registry) -> None:
    """Seed the registry boundary with a credentialed, reachable server."""
    stub_registry.app.entries[SERVER_NAME] = dict(_SEEDED_ENTRY)


async def _reveal(stub_registry, users) -> dict:
    """Read the trusted reveal endpoint with the SAME user credential."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{stub_registry.base_url}/api/v1/mcp/servers/{SERVER_NAME}/reveal",
            headers=users["a"]["headers"],
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.fixture
async def _registered(stub_registry) -> None:
    _seed_registry(stub_registry)


class TestPartialUpdatePreservesSecrets:
    async def test_unspecified_secrets_survive_a_partial_update(self, mcp_base_url, users, stub_registry, _registered):
        """BUG-001: url + one header provided → auth_token/env preserved."""
        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_mcp_servers_update",
            {
                "name": SERVER_NAME,
                "url": _SEEDED_ENTRY["url"],
                "headers": {"X-Updated": "qa-updated-header-secret"},
            },
        )
        assert result["ok"], f"partial update must succeed: {result}"

        revealed = await _reveal(stub_registry, users)
        assert revealed["auth_token"] == TOKEN_SECRET, "the stored auth token must be preserved"
        assert revealed["env"] == {"QA_ORIGINAL": ENV_SECRET}, "the stored env must be preserved"
        assert revealed["headers"] == {"X-Updated": "qa-updated-header-secret"}

    async def test_partial_update_reports_success_without_leaking_secrets(
        self, mcp_base_url, users, stub_registry, _registered
    ):
        """BUG-001 (no-leak): the masked result never carries the secrets."""
        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_mcp_servers_update",
            {
                "name": SERVER_NAME,
                "url": _SEEDED_ENTRY["url"],
                "headers": {"X-Updated": "qa-updated-header-secret"},
            },
        )
        assert result["ok"]
        blob = str(result["result"])
        for secret in (TOKEN_SECRET, ENV_SECRET, HEADER_SECRET):
            assert secret not in blob, f"secret leaked through the masked update result: {secret}"

    async def test_only_provided_fields_change(self, mcp_base_url, users, stub_registry, _registered):
        """A url-only update changes the url and preserves the rest."""
        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_mcp_servers_update",
            {"name": SERVER_NAME, "url": "http://127.0.0.1:18998/mcp"},
        )
        assert result["ok"], str(result)
        revealed = await _reveal(stub_registry, users)
        assert revealed["url"] == "http://127.0.0.1:18998/mcp"
        assert revealed["auth_token"] == TOKEN_SECRET
        assert revealed["headers"] == {"X-Original": HEADER_SECRET}
        assert revealed["env"] == {"QA_ORIGINAL": ENV_SECRET}

    async def test_providing_auth_token_updates_it(self, mcp_base_url, users, stub_registry, _registered):
        """A provided secret IS applied; the OTHER secrets stay untouched."""
        fresh = "qa-fresh-token-secret"
        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_mcp_servers_update",
            {"name": SERVER_NAME, "auth_token": fresh},
        )
        assert result["ok"], str(result)
        blob = str(result["result"])
        assert fresh not in blob, "the freshly provided token must not be echoed"
        revealed = await _reveal(stub_registry, users)
        assert revealed["auth_token"] == fresh
        assert revealed["env"] == {"QA_ORIGINAL": ENV_SECRET}


class TestToolCountPreserved:
    async def test_tool_count_not_reset_for_unchanged_url(self, mcp_base_url, users, stub_registry, _registered):
        """BUG-003: an unchanged-url update must not reset tool_count to 0."""
        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_mcp_servers_update",
            {
                "name": SERVER_NAME,
                "url": _SEEDED_ENTRY["url"],
                "headers": {"X-Updated": "qa-updated-header-secret"},
            },
        )
        assert result["ok"], str(result)
        payload = result["result"]
        assert payload.get("tool_count") == 3, f"tool_count must be preserved: {payload}"
        revealed = await _reveal(stub_registry, users)
        assert revealed["tool_count"] == 3

    async def test_tool_count_resets_only_when_the_url_changes(self, mcp_base_url, users, stub_registry, _registered):
        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_mcp_servers_update",
            {"name": SERVER_NAME, "url": "http://127.0.0.1:18998/mcp"},
        )
        assert result["ok"], str(result)
        revealed = await _reveal(stub_registry, users)
        assert revealed["tool_count"] == 0, "a changed url re-validates (re-count) with the new url"


class TestActionableValidation:
    async def test_openapi_coherence_is_rejected_locally(self, mcp_base_url, users, stub_registry, _registered):
        """BUG-002: an incoherent merged update fails with the stable,
        actionable message — rejected BEFORE forwarding to the registry."""
        boundary_puts = [r for r in stub_registry.app.requests if r["method"] == "PUT"]
        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_mcp_servers_update",
            {"name": SERVER_NAME, "source_type": "openapi"},
        )
        assert not result["ok"], f"an incoherent update must fail: {result}"
        blob = str(result["result"])
        assert "openapi_url is required" in blob, f"the actionable message must surface: {blob}"
        # No update request was forwarded to the registry boundary.
        assert [r for r in stub_registry.app.requests if r["method"] == "PUT"] == boundary_puts

    async def test_null_only_update_is_rejected_and_clobbers_nothing(
        self, mcp_base_url, users, stub_registry, _registered
    ):
        """BUG-002 companion: a null-only update carries NO editable field —
        rejected with the stable, actionable message; the stored url SURVIVES
        (a null never silently clobbers the reachable server's url)."""
        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_mcp_servers_update",
            {"name": SERVER_NAME, "url": None},
        )
        assert not result["ok"], str(result)
        blob = str(result["result"])
        assert "at least one editable field" in blob, blob
        revealed = await _reveal(stub_registry, users)
        assert revealed["url"] == _SEEDED_ENTRY["url"]
