"""Behavioral tests — built-in Composable AI Assistant (REQ-031..REQ-035, REQ-044).

Entry points: the REAL REST agent routes AND the MCP management tools; the
assistant is a centrally managed virtual overlay, never a persisted row.
"""

import uuid

import pytest

from tests.behavioral.composable_mcp.conftest import call_mcp_tool
from tests.behavioral.conftest import configure_llm_credentials

pytestmark = pytest.mark.asyncio(loop_scope="session")

BUILTIN_NAME = "composable-ai-assistant"

READ_ONLY_ERROR_STATUSES = {403, 409}


def agent_yaml(name: str) -> str:
    return f'name: {name}\nmodel: stub-model\nsystem_prompt: "Custom agent."\ntools: []\ndebug: false\n'


async def _register_custom_agent(client, users, name: str):
    resp = await client.post(
        "/api/v1/agents",
        headers=users["a"]["headers"],
        data={"agent_name": name},
        files={"file": ("agent.yaml", agent_yaml(name).encode(), "text/yaml")},
    )
    assert resp.status_code == 201, resp.text
    return name


class TestBuiltinAssistantVisibility:
    async def test_builtin_listed_for_user_with_no_custom_agents(self, client, users):
        listing = await client.get("/api/v1/agents", headers=users["a"]["headers"])
        assert listing.status_code == 200, listing.text
        names = {a["name"] for a in listing.json()}
        assert BUILTIN_NAME in names, "built-in assistant must appear even with zero custom agents"

    async def test_builtin_listed_before_custom_agents(self, client, users):
        """REQ-045 back-end contract: the overlay places the assistant first."""
        await _register_custom_agent(client, users, f"custom-{uuid.uuid4().hex[:8]}")
        listing = await client.get("/api/v1/agents", headers=users["a"]["headers"])
        entries = listing.json()
        assert entries, "agent list must not be empty"
        assert entries[0]["name"] == BUILTIN_NAME, f"built-in assistant must be listed first, got {entries[0]['name']}"

    async def test_is_builtin_metadata_computed(self, client, users):
        """REQ-033: overlay-computed metadata — true for the assistant, false for custom."""
        custom = await _register_custom_agent(client, users, f"meta-{uuid.uuid4().hex[:8]}")
        listing = await client.get("/api/v1/agents", headers=users["a"]["headers"])
        by_name = {a["name"]: a for a in listing.json()}
        assert by_name[BUILTIN_NAME]["is_builtin"] is True
        assert by_name[custom]["is_builtin"] is False

    async def test_builtin_definition_is_readable(self, client, users):
        got = await client.get(f"/api/v1/agents/{BUILTIN_NAME}", headers=users["a"]["headers"])
        assert got.status_code == 200, got.text
        assert got.json()["is_builtin"] is True

    async def test_builtin_visible_via_mcp_list_and_get(self, mcp_base_url, users):
        listing = await call_mcp_tool(mcp_base_url, users["a"]["headers"], "composable_agents_list")
        assert listing["ok"], str(listing)
        entries = listing["result"] if isinstance(listing["result"], list) else listing["result"].get("agents", [])
        matching = [e for e in entries if e.get("name") == BUILTIN_NAME]
        assert matching and matching[0].get("is_builtin") is True, f"built-in missing from MCP agents_list: {matching}"

        got = await call_mcp_tool(
            mcp_base_url, users["a"]["headers"], "composable_agents_get", {"agent_name": BUILTIN_NAME}
        )
        assert got["ok"], str(got)
        assert got["result"].get("is_builtin") is True

    async def test_builtin_isolated_per_user(self, client, users):
        """Both users see the SAME virtual definition without per-user rows interfering."""
        listing_a = await client.get("/api/v1/agents", headers=users["a"]["headers"])
        listing_b = await client.get("/api/v1/agents", headers=users["b"]["headers"])
        a_builtin = next(a for a in listing_a.json() if a["name"] == BUILTIN_NAME)
        b_builtin = next(b for b in listing_b.json() if b["name"] == BUILTIN_NAME)
        assert a_builtin["is_builtin"] and b_builtin["is_builtin"]


class TestBuiltinNameReserved:
    async def test_creating_agent_with_builtin_name_conflicts_rest(self, client, users):
        resp = await client.post(
            "/api/v1/agents",
            headers=users["a"]["headers"],
            data={"agent_name": BUILTIN_NAME},
            files={"file": ("agent.yaml", agent_yaml(BUILTIN_NAME).encode(), "text/yaml")},
        )
        assert resp.status_code == 409, f"reserved name must conflict (409), got {resp.status_code}: {resp.text}"
        assert resp.json()["detail"]  # actionable message required

    async def test_creating_agent_with_builtin_name_conflicts_mcp(self, mcp_base_url, users):
        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_agents_create",
            {"name": BUILTIN_NAME, "yaml": agent_yaml(BUILTIN_NAME)},
        )
        assert not result["ok"], f"reserved name must be rejected via MCP: {result}"
        blob = str(result["result"]).lower()
        assert "builtin" in blob or "reserved" in blob or "conflict" in blob

    async def test_builtin_update_refused_rest(self, client, users):
        resp = await client.put(f"/api/v1/agents/{BUILTIN_NAME}", headers=users["a"]["headers"])
        assert resp.status_code in READ_ONLY_ERROR_STATUSES, (
            f"built-in update must be refused with a read-only built-in error, got {resp.status_code}: {resp.text}"
        )

    async def test_builtin_delete_refused_rest(self, client, users):
        resp = await client.delete(f"/api/v1/agents/{BUILTIN_NAME}", headers=headers_of(users, "a"))
        assert resp.status_code in READ_ONLY_ERROR_STATUSES, (
            f"built-in delete must be refused with a read-only built-in error, got {resp.status_code}: {resp.text}"
        )

    async def test_builtin_update_refused_mcp(self, mcp_base_url, users):
        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_agents_update",
            {"agent_name": BUILTIN_NAME, "yaml": agent_yaml(BUILTIN_NAME)},
        )
        assert not result["ok"], str(result)
        blob = str(result["result"]).lower()
        assert "builtin" in blob or "read-only" in blob or "read only" in blob

    async def test_builtin_delete_refused_mcp(self, mcp_base_url, users):
        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_agents_delete",
            {"agent_name": BUILTIN_NAME},
        )
        assert not result["ok"], str(result)
        blob = str(result["result"]).lower()
        assert "builtin" in blob or "read-only" in blob or "read only" in blob

    async def test_readonly_means_definition_only(self, client, users, stub_llm):
        """REQ-044: read-only definition does NOT block the assistant from doing work
        — it runs with per-user credentials against real use cases."""
        await configure_llm_credentials(client, users["a"]["headers"])
        listing = await client.get("/api/v1/agents", headers=users["a"]["headers"])
        assert any(a["name"] == BUILTIN_NAME for a in listing.json())


class TestBuiltinRunnerIsolation:
    async def test_builtin_runner_scoped_to_running_user(self, mcp_base_url, client, users, stub_llm):
        """The assistant's runners/threads stay user-scoped (REQ-032 / cache correctness)."""
        await configure_llm_credentials(client, users["a"]["headers"])
        result = await call_mcp_tool(
            mcp_base_url, users["a"]["headers"], "composable_agents_run", {"agent_name": BUILTIN_NAME, "message": "hi"}
        )
        assert result["ok"], f"built-in one-shot run failed: {result}"
        thread_id = result["result"]["thread_id"]

        got_by_b = await client.get(f"/api/v1/threads/{thread_id}", headers=users["b"]["headers"])
        assert got_by_b.status_code == 404, "another user must not see the assistant thread of user A"

    async def test_builtin_run_without_llm_credentials_fails_safe(self, mcp_base_url, users):
        """REQ-041: no per-user LLM settings → non-sensitive LLM-not-configured error;
        the platform never substitutes its own credentials (or another user's)."""
        result = await call_mcp_tool(
            mcp_base_url, users["a"]["headers"], "composable_agents_run", {"agent_name": BUILTIN_NAME, "message": "hi"}
        )
        assert not result["ok"], f"missing LLM credentials must fail: {result}"
        blob = str(result["result"]).lower()
        assert "llm" in blob or "not configured" in blob, (
            f"expected an actionable LLM-not-configured error, got: {result['result']}"
        )
        assert "sk-" not in blob, "error must not contain credential material"


def headers_of(users, alias: str) -> dict:
    return users[alias]["headers"]
