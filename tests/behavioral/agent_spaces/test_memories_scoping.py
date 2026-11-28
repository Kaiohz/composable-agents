"""Behavioral tests — scoped memories: route validation + agent closure
(spec REQ-015 + REQ-020, plus REQ-019's personal-scope builtin guard).

Two REAL entry points:

* the REAL mounted composable MCP server (real streamable-HTTP MCP session,
  API-key authenticated): the ``composable_memories_list/get/upsert/delete``
  tools — where the nested agent memory path validation (REQ-015) and the
  external-session PERSONAL fallback (REQ-020) are observable through the
  tool results and the user-namespaced store side effects;
* the REAL one-shot agent execution (``composable_agents_run`` → runner
  build → agent-facing platform tools): the memories tools are bound at
  TOOLSET BUILD time to the calling agent's space (closure over the agent
  name) — the agent operates on ``/memories/{agent_name}/...`` without any
  scope parameter (REQ-020).

The LLM provider API is scripted via ``StubLlmServer``; stores/repositories
run REAL (Testcontainers PostgreSQL-backed shared store).
"""

import json
import uuid

import pytest

from tests.behavioral.agent_spaces.conftest import (
    assistant_tool_call,
    call_mcp_tool,
    create_agent_ok,
    one_shot_run,
    open_mcp_session,
    platform_tools_yaml,
    store_get,
    store_list,
    store_put,
)
from tests.behavioral.conftest import configure_llm_credentials

pytestmark = pytest.mark.asyncio(loop_scope="session")

BUILTIN_NAME = "composable-ai-assistant"

#: Memory tool input schemas: NO scope/agent parameter may ever appear
#: (REQ-020: the scope is bound server-side, not chosen by the caller).
_EXPECTED_MEMORY_TOOL_PARAMS = {
    "composable_memories_list": set(),
    "composable_memories_get": {"filename"},
    "composable_memories_upsert": {"filename", "content"},
    "composable_memories_delete": {"filename"},
}


async def _memory_tool_schemas(mcp_base_url, headers) -> dict:
    async with open_mcp_session(mcp_base_url, headers) as session:
        result = await session.list_tools()
        return {tool.name: tool for tool in result.tools}


class TestNestedAgentMemoryPathValidation:
    """REQ-015 through the real composable memories MCP tools."""

    async def test_upsert_accepts_nested_agent_memory_path(self, mcp_base_url, client, users):
        """The nested shape ``/memories/{agent_slug}/{name}.md`` is ACCEPTED
        (agent segment validated) and the file lands under the user's store
        namespace at exactly that path."""
        slug = f"agent-{uuid.uuid4().hex[:8]}"
        name = "notes.md"
        expected_path = f"/memories/{slug}/{name}"
        content = f"SCOPED-{slug}-CONTENT"

        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_memories_upsert",
            {"filename": f"{slug}/{name}", "content": content},
        )
        assert result["ok"], f"the nested agent memory path must be accepted (REQ-015): {result['result']}"
        assert result["result"].get("path") == expected_path, (
            f"the tool must address /memories/{slug}/{name}: {result['result']}"
        )
        stored = await store_get(client, users["a"]["headers"], expected_path)
        assert stored.status_code == 200, f"the scoped memory must exist in the store: {stored.text}"
        assert stored.json()["content"] == content, "the stored content must round-trip"

    async def test_get_reads_nested_agent_memory_path(self, mcp_base_url, client, users):
        slug = f"agent-{uuid.uuid4().hex[:8]}"
        path = f"/memories/{slug}/prefs.md"
        content = "PREFS-CALLER-CONTENT"
        seeded = await store_put(client, users["a"]["headers"], path, content)
        assert seeded.status_code == 200, seeded.text

        result = await call_mcp_tool(
            mcp_base_url, users["a"]["headers"], "composable_memories_get", {"filename": f"{slug}/prefs.md"}
        )
        assert result["ok"], f"reading an existing nested memory path must succeed (REQ-015): {result['result']}"
        assert result["result"].get("path") == path, f"unexpected addressed path: {result['result']}"
        assert result["result"].get("content") == content, f"unexpected content: {result['result']}"

    async def test_delete_removes_nested_agent_memory_path(self, mcp_base_url, client, users):
        slug = f"agent-{uuid.uuid4().hex[:8]}"
        path = f"/memories/{slug}/obsolete.md"
        seeded = await store_put(client, users["a"]["headers"], path, "OBSOLETE")
        assert seeded.status_code == 200, seeded.text

        result = await call_mcp_tool(
            mcp_base_url, users["a"]["headers"], "composable_memories_delete", {"filename": f"{slug}/obsolete.md"}
        )
        assert result["ok"], f"deleting an existing nested memory path must succeed (REQ-015): {result['result']}"
        gone = await store_get(client, users["a"]["headers"], path)
        assert gone.status_code == 404, "the deleted scoped memory must be gone from the store"

    async def test_deeper_nesting_is_still_rejected(self, mcp_base_url, users):
        """REQ-015/REQ-012: exactly ONE controlled agent segment — deeper
        nesting stays rejected on the memories tools."""
        for filename in (f"agent-{uuid.uuid4().hex[:6]}/deep/notes.md", "a/b/c.md"):
            result = await call_mcp_tool(
                mcp_base_url,
                users["a"]["headers"],
                "composable_memories_upsert",
                {"filename": filename, "content": "should not persist"},
            )
            assert not result["ok"], f"deeper nesting must be rejected: {filename!r} → {result}"

    async def test_invalid_agent_segments_are_still_rejected(self, mcp_base_url, users):
        """REQ-015: the agent segment must match ``^[a-z0-9][a-z0-9-]{0,99}$``
        (REQ-002 charset) — invalid segments stay rejected."""
        for filename in (
            f"Bad_Agent-{uuid.uuid4().hex[:6]}/notes.md",
            f"SubAgent-{uuid.uuid4().hex[:6]}/notes.md",
            f"../escape-{uuid.uuid4().hex[:6]}/notes.md",
        ):
            result = await call_mcp_tool(
                mcp_base_url,
                users["a"]["headers"],
                "composable_memories_upsert",
                {"filename": filename, "content": "should not persist"},
            )
            assert not result["ok"], f"invalid agent segment must be rejected: {filename!r} → {result}"

    async def test_flat_personal_shape_is_rejected(self, mcp_base_url, client, users):
        """User × agent directive: a FLAT (personal) ``{name}.md`` filename is
        REFUSED on the composable memories tools — no store side effect."""
        name = f"personal-{uuid.uuid4().hex[:8]}.md"
        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_memories_upsert",
            {"filename": name, "content": "PERSONAL"},
        )
        assert not result["ok"], f"the flat personal shape must be refused: {result['result']}"
        assert "agent_slug" in json.dumps(result["result"]), result["result"]
        stored = await store_get(client, users["a"]["headers"], f"/memories/{name}")
        assert stored.status_code == 404, "a refused memory path must never have been stored"

    async def test_agent_scopes_coexist_independently(self, mcp_base_url, client, users):
        """Spec edge: the same memory basename in TWO agent scopes coexist
        independently (the personal namespace is no longer one of them)."""
        slug_a = f"agent-{uuid.uuid4().hex[:8]}"
        slug_b = f"agent-{uuid.uuid4().hex[:8]}"
        name = f"prefs-{uuid.uuid4().hex[:6]}.md"

        first = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_memories_upsert",
            {"filename": f"{slug_a}/{name}", "content": "AGENT-A VERSION"},
        )
        assert first["ok"], f"the first agent scope upsert must succeed: {first}"

        second = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_memories_upsert",
            {"filename": f"{slug_b}/{name}", "content": "AGENT-B VERSION"},
        )
        assert second["ok"], f"the second agent scope upsert must be accepted (REQ-015): {second}"

        first_content = await store_get(client, users["a"]["headers"], f"/memories/{slug_a}/{name}")
        assert first_content.status_code == 200 and first_content.json()["content"] == "AGENT-A VERSION", (
            first_content.text
        )
        second_content = await store_get(client, users["a"]["headers"], f"/memories/{slug_b}/{name}")
        assert second_content.status_code == 200 and second_content.json()["content"] == "AGENT-B VERSION", (
            second_content.text
        )
        listed = await store_list(client, users["a"]["headers"], "/memories/")
        assert f"/memories/{slug_a}/{name}" in listed and f"/memories/{slug_b}/{name}" in listed, (
            f"both independently-named agent spaces must coexist in the listing: {listed}"
        )


class TestRestStoreRouteMemoryShapeParity:
    """QA BUG-005 + user × agent directive: the REST store-file route enforces
    REQ-015 exactly like the MCP memories boundary — the agent segment is
    charset-validated and exactly one nesting segment is allowed (deeper
    nesting rejected). The validation is wired through
    ``route_validation.validate_memories_store_path`` on the GET/PUT/DELETE
    ``/api/v1/store/files/{path}`` routes; the single-segment agent shape
    stays accepted on BOTH boundaries and the FLAT personal shape is refused
    on BOTH (memories are scoped user × agent — the personal namespace is a
    system-internal fallback)."""

    async def test_rest_rejects_invalid_agent_segment(self, client, users):
        slug = f"Bad_Slug!-{uuid.uuid4().hex[:8]}"
        headers = users["a"]["headers"]
        put = await client.put(
            f"/api/v1/store/files/memories/{slug}/qa-bad.md", headers=headers, json={"content": "bad slug"}
        )
        got = await client.get(f"/api/v1/store/files/memories/{slug}/qa-bad.md", headers=headers)
        removed = await client.delete(f"/api/v1/store/files/memories/{slug}/qa-bad.md", headers=headers)
        assert put.status_code == 404, f"the invalid agent segment must be rejected: {put.text}"
        assert got.status_code == 404, f"the invalid agent segment must not be readable: {got.text}"
        assert removed.status_code == 404, f"the invalid agent segment must not be destroyable: {removed.text}"

        listed = await store_list(client, headers, f"/memories/{slug}/")
        assert listed == [], "a rejected memory path must never have been stored"

    async def test_rest_rejects_deeper_nesting(self, client, users):
        slug = f"agent-{uuid.uuid4().hex[:8]}"
        headers = users["a"]["headers"]
        put = await client.put(
            f"/api/v1/store/files/memories/{slug}/deeper/qa-too-deep.md",
            headers=headers,
            json={"content": "too deep"},
        )
        assert put.status_code == 404, f"deeper nesting must be rejected on the REST boundary: {put.text}"

        listed = await store_list(client, headers, f"/memories/{slug}/deeper/")
        assert listed == [], "a too-deep memory path must never have been stored"

    async def test_rest_refuses_flat_and_accepts_single_segment_agent_shapes(self, client, users):
        """User × agent directive parity: the ONLY accepted shape on BOTH
        boundaries is the single-segment agent shape — the flat personal
        shape is refused (no store side effect)."""
        headers = users["a"]["headers"]
        name = f"rest-{uuid.uuid4().hex[:8]}.md"
        slug = f"agent-{uuid.uuid4().hex[:8]}"

        flat = await client.put(f"/api/v1/store/files/memories/{name}", headers=headers, json={"content": "FLAT"})
        assert flat.status_code == 404, f"the flat personal shape must be refused on REST too: {flat.text}"
        flat_read = await store_get(client, headers, f"/memories/{name}")
        assert flat_read.status_code == 404, "a refused memory path must never have been stored"

        nested = await client.put(
            f"/api/v1/store/files/memories/{slug}/{name}", headers=headers, json={"content": "NESTED"}
        )
        assert nested.status_code == 200, nested.text
        nested_read = await store_get(client, headers, f"/memories/{slug}/{name}")
        assert nested_read.status_code == 200 and nested_read.json()["content"] == "NESTED", nested_read.text


class TestAgentFacingMemoriesClosure:
    """REQ-020: agent-facing platform memories tools are bound to the agent's
    space at toolset-build time (no scope parameter at call time)."""

    async def test_agent_upserts_land_under_agent_scope(self, mcp_base_url, client, users, stub_llm):
        slug = f"agent-{uuid.uuid4().hex[:8]}"
        await create_agent_ok(client, users["a"]["headers"], platform_tools_yaml(slug), slug)
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(
            assistant_tool_call("composable_memories_upsert", {"filename": "notes.md", "content": "AGENT MEMORY"}),
            {"role": "assistant", "content": "saved the memory"},
        )

        result = await one_shot_run(mcp_base_url, users["a"]["headers"], slug)
        assert result["ok"], f"the one-shot run must succeed: {result}"

        scoped = await store_get(client, users["a"]["headers"], f"/memories/{slug}/notes.md")
        assert scoped.status_code == 200, (
            f"an agent's platform memories_upsert MUST land under its own /memories/{slug}/ scope "
            f"(REQ-020 closure), got {scoped.status_code}: {scoped.text}"
        )
        assert scoped.json()["content"] == "AGENT MEMORY", scoped.text
        flat = await store_get(client, users["a"]["headers"], "/memories/notes.md")
        assert flat.status_code == 404, (
            f"the memory must NOT land in the flat personal namespace when the agent is scoped: {flat.text}"
        )

    async def test_agent_get_reads_agent_scope_content(self, mcp_base_url, client, users, stub_llm):
        """The agent's memories_get resolves against its OWN scope: the tool
        result (carried into the next LLM request, observed at the stub LLM)
        contains the file seeded under ``/memories/{agent}/``."""
        slug = f"agent-{uuid.uuid4().hex[:8]}"
        marker = f"AGENT-GET-MARKER-{uuid.uuid4().hex[:8]}"
        seeded = await store_put(client, users["a"]["headers"], f"/memories/{slug}/notes.md", marker)
        assert seeded.status_code == 200, seeded.text

        await create_agent_ok(client, users["a"]["headers"], platform_tools_yaml(slug), slug)
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(
            assistant_tool_call("composable_memories_get", {"filename": "notes.md"}),
            {"role": "assistant", "content": "read the memory"},
        )
        result = await one_shot_run(mcp_base_url, users["a"]["headers"], slug)
        assert result["ok"], f"the one-shot run must succeed: {result}"

        tool_results = [
            json.dumps(request.body)
            for request in stub_llm.app.requests
            if any(message.get("role") == "tool" for message in request.body.get("messages", []))
        ]
        assert tool_results, "the scripted tool call must have executed and produced a tool result"
        assert any(marker in blob for blob in tool_results), (
            f"the agent's memories_get must read ITS OWN scope (REQ-020 closure): seeded {marker!r} under "
            f"/memories/{slug}/notes.md but no tool result carried it: {[blob[:400] for blob in tool_results]!r}"
        )

    async def test_agent_delete_targets_agent_scope_not_personal(self, mcp_base_url, client, users, stub_llm):
        slug = f"agent-{uuid.uuid4().hex[:8]}"
        seed_a = await store_put(client, users["a"]["headers"], f"/memories/{slug}/obsolete.md", "OBSOLETE")
        assert seed_a.status_code == 200, seed_a.text
        decoy_slug = f"agent-{uuid.uuid4().hex[:8]}"
        decoy_seeded = await store_put(
            client, users["a"]["headers"], f"/memories/{decoy_slug}/decoy.md", "DECOY KEEP"
        )
        assert decoy_seeded.status_code == 200, decoy_seeded.text

        await create_agent_ok(client, users["a"]["headers"], platform_tools_yaml(slug), slug)
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(
            assistant_tool_call("composable_memories_delete", {"filename": "obsolete.md"}),
            {"role": "assistant", "content": "deleted the memory"},
        )
        result = await one_shot_run(mcp_base_url, users["a"]["headers"], slug)
        assert result["ok"], f"the one-shot run must succeed: {result}"

        scoped_gone = await store_get(client, users["a"]["headers"], f"/memories/{slug}/obsolete.md")
        assert scoped_gone.status_code == 404, (
            f"the agent's memories_delete MUST target its own /memories/{slug}/ scope (REQ-020 closure): "
            f"{scoped_gone.text}"
        )
        decoy = await store_get(client, users["a"]["headers"], f"/memories/{decoy_slug}/decoy.md")
        assert decoy.status_code == 200, "another agent's memory must never be touched by an agent-scoped delete"


class TestMemoryToolsExposeNoScopeParameter:
    async def test_memories_tool_schemas_carry_no_scope_or_agent_parameter(self, mcp_base_url, users):
        """REQ-020: the scope is bound SERVER-side (closure / build time);
        the memory tools expose NO agent/scope parameter on their input
        schemas — and their expected parameters stay exactly the same."""
        tools = await _memory_tool_schemas(mcp_base_url, users["a"]["headers"])
        for name, expected_params in _EXPECTED_MEMORY_TOOL_PARAMS.items():
            tool = tools.get(name)
            assert tool is not None, f"{name} must be listed on the composable MCP server"
            properties = tool.inputSchema.get("properties", {})
            assert set(properties) == expected_params, (
                f"{name} input schema must expose exactly {sorted(expected_params)}, got {sorted(properties)} — "
                "no scope/agent parameter is allowed (the scope is bound by the platform)"
            )


class TestExternalSessionsDefaultToPersonalScope:
    async def test_mcp_session_without_agent_context_cannot_touch_the_personal_namespace(
        self, mcp_base_url, client, users
    ):
        """User × agent directive: an external MCP session (no agent context)
        MUST use the agent-scoped shape — a bare filename is REFUSED (the
        personal namespace is a system-internal fallback), the nested shape
        works exactly as validated."""
        name = f"external-{uuid.uuid4().hex[:8]}.md"
        slug = f"agent-{uuid.uuid4().hex[:8]}"

        flat = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_memories_upsert",
            {"filename": name, "content": "EXTERNAL SESSION"},
        )
        assert not flat["ok"], f"the flat personal shape must be refused: {flat['result']}"
        personal = await store_get(client, users["a"]["headers"], f"/memories/{name}")
        assert personal.status_code == 404, "no personal-namespace side effect for a scope-less session"

        scoped = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_memories_upsert",
            {"filename": f"{slug}/{name}", "content": "EXTERNAL SESSION"},
        )
        assert scoped["ok"], f"the agent-scoped shape must be accepted: {scoped['result']}"
        landed = await store_get(client, users["a"]["headers"], f"/memories/{slug}/{name}")
        assert landed.status_code == 200, "the scoped shape lands exactly under its agent namespace"

    async def test_builtin_assistant_memories_bind_to_its_agent_scope(
        self, mcp_base_url, client, users, stub_llm
    ):
        """User × agent directive: the built-in assistant is a platform agent
        like any other — its memories tools bind to ITS canonical slug scope
        (``/memories/composable-ai-assistant/``), never the personal
        namespace."""
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(
            assistant_tool_call("composable_memories_upsert", {"filename": "assistant-note.md", "content": "BUILTIN"}),
            {"role": "assistant", "content": "saved"},
        )
        result = await one_shot_run(mcp_base_url, users["a"]["headers"], BUILTIN_NAME)
        assert result["ok"], f"the builtin assistant run must succeed: {result}"

        scoped = await store_get(client, users["a"]["headers"], f"/memories/{BUILTIN_NAME}/assistant-note.md")
        assert scoped.status_code == 200 and scoped.json()["content"] == "BUILTIN", (
            f"the built-in assistant's memories bind to its agent scope: {scoped.text}"
        )
        flat = await store_get(client, users["a"]["headers"], "/memories/assistant-note.md")
        assert flat.status_code == 404, "the personal namespace stays unreachable (system-internal fallback)"
