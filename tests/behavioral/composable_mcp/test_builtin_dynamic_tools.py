"""Behavioral tests — built-in assistant dynamic capability resolution
(REQ-036..REQ-041 + REQ-BUILTIN-003 selective mounting).

The assistant's toolset is resolved fresh per turn in the RUNNING user's
context: built-in MCP mounts, ONLY the MCP servers that user EXPLICITLY
SELECTED via the ``mcp_refs`` customization (real local streamable-HTTP MCP
servers here), platform tools and ``create_pdf`` — while
settings/API-key/reveal and self-invocation stay excluded.

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


async def _select_builtin_refs(client, users, refs: list[str]) -> None:
    """Personalize the built-in with an explicit MCP selection (REQ-BUILTIN-002).

    Selected mounting replaces the legacy auto-mounted-user-registry behavior:
    a registered server is reachable only when its identity is selected.
    """
    lines = [
        f"name: {BUILTIN_NAME}",
        "model: personalized-model",
        'description: "Personalized built-in assistant."',
        'system_prompt: "Personalized built-in assistant."',
        "tools: []",
        "debug: false",
    ]
    if refs is not None:
        lines.append("mcp_refs:")
        lines.extend(f"  - {ref}" for ref in refs)
    updated = await client.put(
        f"/api/v1/agents/{BUILTIN_NAME}",
        headers=users["a"]["headers"],
        files={"file": ("builtin.yaml", "\n".join(lines).encode(), "text/yaml")},
    )
    assert updated.status_code == 200, updated.text


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
        """REQ-036 + REQ-BUILTIN-003: an explicitly SELECTED registered MCP server
        is available to the assistant in that user's context."""
        await configure_llm_credentials(client, users["a"]["headers"])
        server = StubMcpServer("alpha-tools", tool_name="ping", tool_result="pong-alpha").start()
        try:
            await _register_user_server(mcp_base_url, users, stub_registry, "alpha-tools", server.url)
            await _select_builtin_refs(client, users, ["alpha-tools"])
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
            await _select_builtin_refs(client, users, ["healthy", "dead"])

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
            await _select_builtin_refs(client, users, ["first", "second"])

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
            await _select_builtin_refs(client, users, ["ephemeral"])

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
