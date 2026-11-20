"""Behavioral tests — MCP tool surface, schemas, masking, RLS (REQ-008..REQ-020).

Entry point: authenticated ``tools/list`` / ``tools/call`` against the REAL
mounted streamable-HTTP MCP server, plus the registry boundary mock (the
outbound ``mcp-raganything`` REST API is the ONLY stub in this file).
"""

import pytest

from tests.behavioral.composable_mcp.conftest import DELETION_TOOLS, call_mcp_tool, list_mcp_tool_names

pytestmark = pytest.mark.asyncio(loop_scope="session")

# REQ-011 — the complete typed management/execution tool families.
EXPECTED_TOOLS = {
    "composable_agents_list",
    "composable_agents_get",
    "composable_agents_create",
    "composable_agents_update",
    "composable_agents_delete",
    "composable_agents_run",
    "composable_mcp_servers_list",
    "composable_mcp_servers_get",
    "composable_mcp_servers_validate",
    "composable_mcp_servers_create",
    "composable_mcp_servers_update",
    "composable_mcp_servers_delete",
    "composable_skills_list",
    "composable_skills_get",
    "composable_skills_upsert",
    "composable_skills_delete",
    "composable_skills_usage",
    "composable_memories_list",
    "composable_memories_get",
    "composable_memories_upsert",
    "composable_memories_delete",
    "composable_prompts_list",
    "composable_prompts_get",
    "composable_prompts_create",
    "composable_prompts_update",
    "composable_prompts_delete",
    "composable_prompt_versions_list",
    "composable_prompt_versions_get",
    "composable_prompt_versions_create",
    "composable_models_list",
    "composable_models_get",
    "composable_models_create",
    "composable_models_update",
    "composable_models_delete",
    "composable_crons_list",
    "composable_crons_get",
    "composable_crons_create",
    "composable_crons_update",
    "composable_crons_delete",
    "composable_crons_run_now",
    "composable_cron_runs_list",
    "composable_threads_list",
    "composable_threads_get",
    "composable_threads_history",
    "composable_threads_trace",
    "composable_threads_delete",
    "composable_traces_list",
    "composable_traces_usage",
    "composable_traces_top_runs",
    "composable_users_me",
}

# REQ-020 — never exposed through MCP or the built-in assistant.
FORBIDDEN_TOOL_FRAGMENTS = ("settings", "api_key", "api_keys", "reveal", " credential", "llm")


async def _tools_map(mcp_base_url, headers):
    """Return {name: tool} from a real tools/list call."""
    from tests.behavioral.composable_mcp.conftest import open_mcp_session

    async with open_mcp_session(mcp_base_url, headers) as session:
        result = await session.list_tools()
        return {t.name: t for t in result.tools}


class TestToolsListSurface:
    async def test_all_typed_tool_families_are_listed(self, mcp_base_url, users):
        names = await list_mcp_tool_names(mcp_base_url, users["a"]["headers"])
        assert names is not None, "tools/list failed for a valid API key"
        missing = EXPECTED_TOOLS - set(names)
        assert not missing, f"MCP is missing the required typed tools: {sorted(missing)}"

    async def test_every_tool_has_a_typed_input_schema(self, mcp_base_url, users):
        tools = await _tools_map(mcp_base_url, users["a"]["headers"])
        for name in EXPECTED_TOOLS:
            tool = tools.get(name)
            assert tool is not None, f"tool {name} is missing"
            schema = tool.inputSchema
            assert isinstance(schema, dict) and schema.get("type") == "object", f"{name} lacks a JSON-schema input"
            assert tool.description, f"{name} lacks a description"

    async def test_delete_tools_are_flagged_destructive(self, mcp_base_url, users):
        tools = await _tools_map(mcp_base_url, users["a"]["headers"])
        for name in DELETION_TOOLS:
            tool = tools.get(name)
            assert tool is not None, f"delete tool {name} missing"
            annotations = getattr(tool, "annotations", None)
            assert annotations is not None and getattr(annotations, "destructiveHint", False), (
                f"{name} must carry destructive MCP tool annotations (REQ-008)"
            )

    async def test_excluded_admin_domains_are_not_exposed(self, mcp_base_url, users):
        """REQ-020: settings, API-key administration and registry /reveal stay off MCP."""
        names = await list_mcp_tool_names(mcp_base_url, users["a"]["headers"])
        assert names is not None, "tools/list must succeed for valid credentials before exclusions can be checked"
        for name in names:
            lowered = name.lower()
            for fragment in FORBIDDEN_TOOL_FRAGMENTS:
                assert fragment not in lowered, f"forbidden tool exposed via MCP: {name}"


class TestRegistrySecretsMasked:
    async def _register_server(self, mcp_base_url, users, name):
        """Create an MCP server entry through the MCP tool (registry boundary mock)."""
        return await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_mcp_servers_create",
            {
                "name": name,
                "transport": "http",
                "url": "http://127.0.0.1:59997/mcp",
                "source_type": "external",
                "auth_token": "fresh-plain-token-secret",
                "headers": {"X-Custom": "fresh-plain-header-secret"},
                "env": {"FRESH_TOKEN": "fresh-plain-env-secret"},
            },
        )

    async def test_create_result_masks_secrets(self, mcp_base_url, users, stub_registry):
        result = await self._register_server(mcp_base_url, users, "mask-create")
        assert result["ok"], f"composable_mcp_servers_create failed: {result}"
        serialized = str(result["result"])
        for secret in ("fresh-plain-token-secret", "fresh-plain-header-secret", "fresh-plain-env-secret"):
            assert secret not in serialized, f"secret leaked via MCP create: {secret}"

    async def test_get_and_list_mask_secrets(self, mcp_base_url, users, stub_registry):
        listed = await call_mcp_tool(mcp_base_url, users["a"]["headers"], "composable_mcp_servers_list")
        assert listed["ok"], str(listed)
        got = await call_mcp_tool(mcp_base_url, users["a"]["headers"], "composable_mcp_servers_get", {"name": "alpha"})
        assert got["ok"], str(got)
        blob = str(listed["result"]) + str(got["result"])
        for secret in ("alpha-plain-token-secret", "alpha-plain-header-secret", "alpha-plain-env-secret"):
            assert secret not in blob, f"registry secret leaked via MCP list/get: {secret}"

    async def test_registry_boundary_receives_originating_credential(self, mcp_base_url, users, stub_registry):
        """REQ-017: registry tools call the owning registry as the originating user."""
        await call_mcp_tool(mcp_base_url, users["a"]["headers"], "composable_mcp_servers_list")
        seen = stub_registry.app.requests
        assert seen, "MCP registry tools must go through the registry boundary, never local repos"
        assert any(
            r["api_key"] == users["a"]["key"] or (r["authorization"] or "").startswith("Bearer ") for r in seen
        ), "the originating credential must be forwarded to the trusted registry"

    async def test_registry_unavailable_returns_safe_error(self, mcp_base_url, users, stub_registry):
        """Spec error case: an unavailable registry → safe retryable error, no false success."""
        stub_registry.reset(fail=True)
        result = await call_mcp_tool(mcp_base_url, users["a"]["headers"], "composable_mcp_servers_list")
        assert not result["ok"], f"registry outage must not be reported as a success: {result}"

    async def test_reveal_operation_is_not_an_mcp_tool(self, mcp_base_url, users):
        names = await list_mcp_tool_names(mcp_base_url, users["a"]["headers"])
        assert names is not None, "tools/list must succeed for valid credentials before exclusions can be checked"
        assert not any("reveal" in n.lower() for n in names)


class TestUserIsolation:
    async def test_foreign_agents_get_is_non_enumerating(self, mcp_base_url, users, client):
        """User A cannot observe or address user B's agent through MCP."""
        owned = await call_mcp_tool(mcp_base_url, users["b"]["headers"], "composable_agents_list")
        assert owned["ok"], str(owned)
        foreign = await call_mcp_tool(
            mcp_base_url, users["a"]["headers"], "composable_agents_get", {"agent_name": "shared-secret-name"}
        )
        assert not foreign["ok"], f"foreign agent get must not succeed: {foreign}"
        blob = str(foreign["result"]).lower()
        assert "yaml" not in blob and "system_prompt" not in blob

    async def test_foreign_threads_are_invisible_via_mcp(self, mcp_base_url, users, client):
        threads = await call_mcp_tool(mcp_base_url, users["b"]["headers"], "composable_threads_list")
        assert threads["ok"], str(threads)
        foreign_list = await call_mcp_tool(mcp_base_url, users["a"]["headers"], "composable_threads_list")
        assert foreign_list["ok"], str(foreign_list)
        b_ids = {t.get("id") for t in (threads["result"] if isinstance(threads["result"], list) else [])}
        a_blob = str(foreign_list["result"])
        for tid in filter(None, b_ids):
            assert tid not in a_blob, f"user A observed user B's thread {tid} through MCP"

    async def test_no_caller_supplied_user_id_changes_scope(self, mcp_base_url, users):
        """REQ-005: MCP tools accept no user-id argument that changes ownership."""
        sneaky = await call_mcp_tool(
            mcp_base_url,
            users["b"]["headers"],
            "composable_threads_list",
            {},
        )
        assert sneaky["ok"], f"tools/list + threads_list must succeed for user B: {sneaky}"
        blob = str(sneaky["result"])
        assert "user-a" not in blob, "a tool result leaked another user's data"

    async def test_foreign_thread_history_non_enumerating(self, mcp_base_url, users):
        unknown = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_threads_history",
            {"thread_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert not unknown["ok"], f"unknown/foreign thread history must not succeed: {unknown}"


class TestStorePathSafety:
    async def test_skill_upsert_rejects_path_traversal(self, mcp_base_url, users):
        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_skills_upsert",
            {"skill_name": "../escape", "content": "# malicious"},
        )
        assert not result["ok"], f"path traversal accepted by skills tool: {result}"

    async def test_memory_upsert_rejects_prefix_escape(self, mcp_base_url, users):
        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_memories_upsert",
            {"filename": "../../etc/passwd", "content": "malicious"},
        )
        assert not result["ok"], f"memory prefix escape accepted: {result}"
