"""Behavioral tests — built-in assistant dynamic capability resolution
(REQ-036..REQ-041).

The assistant's toolset is resolved fresh per turn in the RUNNING user's
context: built-in MCP mounts, every MCP server that user registered (real
local streamable-HTTP MCP servers here), platform tools and ``create_pdf``
— while settings/API-key/reveal and self-invocation stay excluded.

The LLM boundary is scripted (``stub_llm.script``) to make the assistant
actually CALL a specific tool, so resolution is observed through real
server-side tool execution, not unit introspection.
"""

import pytest

from tests.behavioral.composable_mcp.conftest import (
    StubMcpServer,
    assistant_text,
    assistant_tool_call,
    call_mcp_tool,
)
from tests.behavioral.conftest import configure_llm_credentials

pytestmark = pytest.mark.asyncio(loop_scope="session")

BUILTIN_NAME = "composable-ai-assistant"

# MODEL-001 — the identity field set of a Model resource; the Settings-owned
# connection fields ``provider``/``base_url`` are removed from the contract.
MODEL_IDENTITY_FIELDS = {"id", "name", "provider_model", "created_at", "updated_at"}
REMOVED_MODEL_FIELDS = ("provider", "base_url")


async def _run_assistant(mcp_base_url, users, message: str = "do work") -> dict:
    return await call_mcp_tool(
        mcp_base_url,
        users["a"]["headers"],
        "composable_agents_run",
        {"agent_name": BUILTIN_NAME, "message": message},
    )


async def _register_user_server(mcp_base_url, users, stub_registry, name: str, url: str) -> dict:  # noqa: ARG001
    """Register a user MCP server through the MCP registry tool."""
    result = await call_mcp_tool(
        mcp_base_url,
        users["a"]["headers"],
        "composable_mcp_servers_create",
        {"name": name, "transport": "http", "url": url, "source_type": "external"},
    )
    assert result["ok"], f"registering MCP server {name} failed: {result}"
    return result


class TestDynamicResolution:
    async def test_user_registered_mcp_server_tools_are_callable(
        self, mcp_base_url, client, users, stub_llm, stub_registry
    ):
        """REQ-036: every supported MCP server registered by the current user is
        available to the assistant in that user's context."""
        await configure_llm_credentials(client, users["a"]["headers"])
        server = StubMcpServer("alpha-tools", tool_name="ping", tool_result="pong-alpha").start()
        try:
            await _register_user_server(mcp_base_url, users, stub_registry, "alpha-tools", server.url)
            stub_llm.script(
                assistant_tool_call("ping", {}),
                assistant_text("used the registered server"),
            )
            result = await _run_assistant(mcp_base_url, users)
            assert result["ok"], f"assistant run failed: {result}"
            blob = str(result["result"])
            assert "pong-alpha" in blob, (
                f"the assistant failed to execute a tool from the user-registered MCP server: {blob[:400]}"
            )
        finally:
            server.stop()

    async def test_platform_tools_available_to_assistant(self, mcp_base_url, client, users, stub_llm, stub_registry):
        """REQ-036: the platform management tools are part of the assistant's toolset."""
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(
            assistant_tool_call("composable_skills_list", {}),
            assistant_text("listed skills"),
        )
        result = await _run_assistant(mcp_base_url, users)
        assert result["ok"], f"assistant run failed: {result}"
        blob = str(result["result"]).lower()
        assert "skills" in blob, f"platform tool did not execute: {blob[:400]}"

    async def test_platform_tools_reflect_the_user_customization(
        self, mcp_base_url, client, users, stub_llm, stub_registry
    ):
        """The in-agent platform tools mirror REST/MCP: the running user's
        effective built-in configuration is returned, while the dynamic
        toolset keeps working."""
        updated = await client.put(
            f"/api/v1/agents/{BUILTIN_NAME}",
            headers=users["a"]["headers"],
            files={
                "file": (
                    "builtin.yaml",
                    b"name: composable-ai-assistant\nmodel: custom-platform-model\n"
                    b'system_prompt: "CUSTOM_PLATFORM_PROMPT"\ntools: []\ndebug: false\n',
                )
            },
        )
        assert updated.status_code == 200, updated.text

        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(
            assistant_tool_call("composable_agents_get", {"agent_name": BUILTIN_NAME}),
            assistant_text("read the customized assistant"),
        )
        result = await _run_assistant(mcp_base_url, users)
        assert result["ok"], f"the platform tool must execute against the customized built-in: {result}"
        blob = str(result["result"])
        assert "CUSTOM_PLATFORM_PROMPT" in blob, (
            f"the in-agent platform tool must mirror the user's effective config: {blob[:400]}"
        )

    async def test_excluded_capabilities_absent_from_assistant_resolution(
        self, mcp_base_url, client, users, stub_llm, stub_registry
    ):
        """REQ-020/REQ-030: settings/API-key/reveal tools and self-invocation are
        NEVER passed to the built-in assistant."""
        await configure_llm_credentials(client, users["a"]["headers"])
        # The scripted call for an excluded tool must fail server-side instead of
        # silently succeeding.
        stub_llm.script(
            assistant_tool_call("composable_settings_llm_get", {}),
            assistant_text("never reached"),
        )
        result = await _run_assistant(mcp_base_url, users)
        assert not result["ok"] or "excluded" in str(result["result"]).lower(), (
            "excluded settings capability must not resolve for the assistant"
        )

    async def test_create_pdf_tool_available(self, mcp_base_url, client, users, stub_llm, stub_registry):
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(
            assistant_tool_call("create_pdf", {"filename": "report", "markdown": "# hi"}),
            assistant_text("created the pdf"),
        )
        result = await _run_assistant(mcp_base_url, users)
        assert result["ok"], f"assistant run failed: {result}"


class TestDegradationIsolation:
    async def test_dead_optional_server_keeps_platform_tools_working(
        self, mcp_base_url, client, users, stub_llm, stub_registry
    ):
        """REQ-040: one unavailable optional MCP server must not remove platform
        tools or other healthy tools; the failure is surfaced, not a false success."""
        await configure_llm_credentials(client, users["a"]["headers"])
        healthy = StubMcpServer("healthy", tool_name="ping", tool_result="pong-healthy").start()
        try:
            await _register_user_server(mcp_base_url, users, stub_registry, "healthy", healthy.url)
            # "dead" points at a closed port — resolution of THIS server fails.
            await _register_user_server(mcp_base_url, users, stub_registry, "dead", "http://127.0.0.1:59996/mcp")

            stub_llm.script(
                assistant_tool_call("composable_skills_list", {}),
                assistant_text("platform tools still work"),
            )
            result = await _run_assistant(mcp_base_url, users)
            assert result["ok"], f"platform tools must survive an optional MCP server outage: {result}"

            blob = str(result["result"]).lower()
            assert "dead" in blob or "degraded" in blob or "unavailable" in blob, (
                "the outage must be surfaced as degraded capability information, not silently swallowed"
            )
        finally:
            healthy.stop()


class TestCollisionSafety:
    async def test_same_tool_name_on_two_servers_is_namespaced_deterministically(
        self, mcp_base_url, client, users, stub_llm, stub_registry
    ):
        """REQ-039: two MCP servers exposing the same original tool name must
        produce distinct, stable agent-facing tool names — a call reaches
        exactly one server, and the resolution is deterministic across turns."""
        await configure_llm_credentials(client, users["a"]["headers"])
        first = StubMcpServer("first", tool_name="ping", tool_result="pong-first").start()
        second = StubMcpServer("second", tool_name="ping", tool_result="pong-second").start()
        try:
            await _register_user_server(mcp_base_url, users, stub_registry, "first", first.url)
            await _register_user_server(mcp_base_url, users, stub_registry, "second", second.url)

            stub_llm.script(assistant_tool_call("ping", {}), assistant_text("done"))
            run_one = await _run_assistant(mcp_base_url, users)

            stub_llm.script(assistant_tool_call("ping", {}), assistant_text("done"))
            run_two = await _run_assistant(mcp_base_url, users)

            assert run_one["ok"] and run_two["ok"], f"collision must never break resolution: {run_one} / {run_two}"
            blob_one = str(run_one["result"])
            blob_two = str(run_two["result"])
            # Deterministic: each turn reaches the SAME server, never a silent
            # replacement by whichever server happened to win.
            assert ("pong-first" in blob_one) != ("pong-second" in blob_one), (
                "the collided 'ping' must resolve to exactly one deterministically namespaced tool"
            )
            assert blob_one.__contains__("pong-first") == blob_two.__contains__("pong-first"), (
                "resolution must be deterministic across turns (silent replacement varies)"
            )
        finally:
            first.stop()
            second.stop()


class TestRegistryFreshness:
    async def test_deleted_server_tool_not_stale_on_next_turn(
        self, mcp_base_url, client, users, stub_llm, stub_registry
    ):
        """REQ-038: deleting a registered MCP server removes its tools from the
        assistant's next turn without a restart."""
        await configure_llm_credentials(client, users["a"]["headers"])
        server = StubMcpServer("ephemeral", tool_name="ping", tool_result="pong-ephemeral").start()
        try:
            await _register_user_server(mcp_base_url, users, stub_registry, "ephemeral", server.url)

            stub_llm.script(assistant_tool_call("ping", {}), assistant_text("used ephemeral"))
            before = await _run_assistant(mcp_base_url, users)
            assert before["ok"] and "pong-ephemeral" in str(before["result"]), str(before)

            deleted = await call_mcp_tool(
                mcp_base_url, users["a"]["headers"], "composable_mcp_servers_delete", {"name": "ephemeral"}
            )
            assert deleted["ok"], str(deleted)

            stub_llm.script(assistant_tool_call("ping", {}), assistant_text("stale!"))
            after = await _run_assistant(mcp_base_url, users)
            assert "pong-ephemeral" not in str(after["result"]), (
                "a deleted MCP server must not keep resolving on the assistant's next turn"
            )
        finally:
            server.stop()


def _target_yaml(name: str) -> str:
    return f'name: {name}\nmodel: stub-model\nsystem_prompt: "One-shot target agent."\ntools: []\ndebug: false\n'


class TestAssistantRunsAgents:
    """REQ-030/REQ-036: the assistant may run OTHER user agents as one-shot
    tasks; only self-invocation is forbidden (C1 remediation)."""

    async def _create_agent(self, client, users, name: str) -> str:
        resp = await client.post(
            "/api/v1/agents",
            headers=users["a"]["headers"],
            data={"agent_name": name},
            files={"file": ("agent.yaml", _target_yaml(name).encode(), "text/yaml")},
        )
        assert resp.status_code == 201, resp.text
        return name

    async def test_assistant_runs_another_user_agent(self, mcp_base_url, client, users, stub_llm, stub_registry):
        """REQ-036: the assistant runs ANOTHER user agent; the nested one-shot
        rides the same invocation chain and target identity."""
        target = await self._create_agent(client, users, f"assistant-target-{__import__('uuid').uuid4().hex[:8]}")
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(
            assistant_tool_call("composable_agents_run", {"agent_name": target, "message": "go"}),
            assistant_text("target agent completed"),
        )
        result = await _run_assistant(mcp_base_url, users, "run the target agent")
        assert result["ok"], f"the assistant must be able to run another user agent: {result}"
        blob = str(result["result"])
        assert target in blob, "the nested one-shot execution must be traceable in the run result"

    async def test_assistant_cannot_invoke_itself(self, mcp_base_url, client, users, stub_llm, stub_registry):
        """REQ-030: self-invocation is refused with a safe error, not a silent success."""
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(
            assistant_tool_call("composable_agents_run", {"agent_name": BUILTIN_NAME, "message": "again"}),
            assistant_text("never reached"),
        )
        result = await _run_assistant(mcp_base_url, users, "call yourself")
        blob = str(result["result"]).lower()
        assert not result["ok"] or "reject" in blob, (
            f"the built-in assistant must not be able to invoke itself: {result}"
        )
        assert "completed" not in blob.split('"status": "')[1][:40] if '"status": "' in blob else True, (
            "a refused self-invocation must not be reported as the final completed state"
        )


class TestPlatformModelTools:
    """MODEL-007 through a REAL agent run (the agent-facing platform toolset):
    the LLM-driven Model create/update tools follow the REST two-field contract
    — same fields, removed (Settings-owned) fields rejected/exposed nowhere and
    the same trim/non-empty rules (review C1: this is the surface that could
    previously persist a blank ``name`` through ``model_copy``)."""

    async def _create_model_via_mcp(
        self, mcp_base_url, users, name: str = "Toolset Base", provider_model: str = "toolset-base-x"
    ) -> dict:
        created = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_models_create",
            {"args": {"name": name, "provider_model": provider_model}},
        )
        assert created["ok"], f"setting up the model through MCP failed: {created}"
        return created["result"]

    async def test_bound_update_schema_carries_the_rest_contract(
        self, mcp_base_url, client, users, stub_llm, stub_registry
    ):
        """What the LLM actually receives: ``composable_models_update``'s bound
        schema must carry the REST fields and never the removed ones."""
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(assistant_text("no tool"))
        run = await _run_assistant(mcp_base_url, users)
        assert run["ok"], f"assistant run failed: {run}"
        tools = stub_llm.app.requests[0].body.get("tools", [])
        binding = next((t for t in tools if t.get("function", {}).get("name") == "composable_models_update"), None)
        assert binding is not None, "the platform toolset must expose composable_models_update to the assistant"
        props = set(binding["function"]["parameters"].get("properties", {}))
        assert props == {"model_id", "name", "provider_model"}, f"bound update schema fields: {sorted(props)}"
        for removed in REMOVED_MODEL_FIELDS:
            assert removed not in props, f"removed field '{removed}' is bound to the LLM-visible tool schema"

    async def test_agent_facing_update_rejects_a_blank_name(self, mcp_base_url, client, users, stub_llm, stub_registry):
        """Review C1: a whitespace-only ``name`` issued by the LLM must be
        rejected (DTO trim rule) and must NOT be persisted."""
        model = await self._create_model_via_mcp(mcp_base_url, users)
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(
            assistant_tool_call("composable_models_update", {"model_id": model["id"], "name": "   "}),
            assistant_text("self-corrected"),
        )
        result = await _run_assistant(mcp_base_url, users, "rename the model")
        assert len(stub_llm.app.requests) >= 2, (
            f"the tool call round must execute and surface the validation error: {result}"
        )
        got = await client.get(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"])
        assert got.status_code == 200, got.text
        assert got.json()["name"] == model["name"], "the blank name must not be persisted"
        assert got.json()["provider_model"] == model["provider_model"]

    async def test_agent_facing_update_rejects_a_removed_field_argument(
        self, mcp_base_url, client, users, stub_llm, stub_registry
    ):
        """A Settings-owned field in the tool call is an unknown field: rejected
        by ``extra="forbid"``, persisted nowhere."""
        model = await self._create_model_via_mcp(mcp_base_url, users)
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(
            assistant_tool_call(
                "composable_models_update",
                {"model_id": model["id"], "name": "Hijack", "provider": "openai"},
            ),
            assistant_text("self-corrected"),
        )
        result = await _run_assistant(mcp_base_url, users, "update the model")
        assert len(stub_llm.app.requests) >= 2, f"the tool call round must execute: {result}"
        got = await client.get(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"])
        assert got.status_code == 200, got.text
        assert got.json()["name"] == model["name"], "the invalid update must not be applied"
        assert got.json()["provider_model"] == model["provider_model"]

    async def test_agent_facing_partial_update_is_trimmed_and_keeps_the_omitted_field(
        self, mcp_base_url, client, users, stub_llm, stub_registry
    ):
        """The DTO-derived schema applies the REST semantics through the real
        agent run: presence-based subset, trimmed values, omitted field kept."""
        model = await self._create_model_via_mcp(mcp_base_url, users)
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(
            assistant_tool_call("composable_models_update", {"model_id": model["id"], "name": "  Renamed  "}),
            assistant_text("renamed"),
        )
        result = await _run_assistant(mcp_base_url, users, "rename the model")
        assert result["ok"], f"the valid update must succeed: {result}"
        got = await client.get(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"])
        assert got.status_code == 200, got.text
        assert got.json()["name"] == "Renamed", "the trimmed name must be persisted"
        assert got.json()["provider_model"] == model["provider_model"], "the omitted field keeps its value"

    async def test_agent_facing_create_is_identity_only(self, mcp_base_url, client, users, stub_llm, stub_registry):
        """A valid two-field create through the toolset persists an identity-only
        resource; a removed field cannot join the payload."""
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(
            assistant_tool_call(
                "composable_models_create", {"name": "  Agent Tool  ", "provider_model": "  agent-create-x  "}
            ),
            assistant_text("created"),
        )
        result = await _run_assistant(mcp_base_url, users, "create a model")
        assert result["ok"], f"the valid create must succeed: {result}"
        listed = await client.get("/api/v1/models", headers=users["a"]["headers"])
        assert listed.status_code == 200, listed.text
        items = listed.json()
        assert len(items) == 1, f"exactly the tool-created model expected: {items}"
        body = items[0]
        assert set(body) == MODEL_IDENTITY_FIELDS, f"identity-only resource expected, got: {sorted(body)}"
        for removed in REMOVED_MODEL_FIELDS:
            assert removed not in body, f"removed field '{removed}' must not be persisted by the tool surface"
        assert body["name"] == "Agent Tool" and body["provider_model"] == "agent-create-x"

    async def test_agent_facing_create_rejects_a_removed_field_argument(
        self, mcp_base_url, client, users, stub_llm, stub_registry
    ):
        """A Settings-owned field in a tool create call must persist nothing."""
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(
            assistant_tool_call(
                "composable_models_create", {"name": "Smuggled", "provider_model": "smuggled-x", "provider": "openai"}
            ),
            assistant_text("self-corrected"),
        )
        result = await _run_assistant(mcp_base_url, users, "create a model")
        assert len(stub_llm.app.requests) >= 2, f"the tool call round must execute: {result}"
        listed = await client.get("/api/v1/models", headers=users["a"]["headers"])
        assert listed.status_code == 200 and listed.json() == [], (
            f"the rejected create must persist nothing: {listed.json()}"
        )
