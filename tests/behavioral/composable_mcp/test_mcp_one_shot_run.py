"""Behavioral tests — one-shot agent execution via MCP (REQ-022..REQ-030).

Entry points:
* ``composable_agents_run`` / ``tools/call`` on the REAL mounted MCP server.
* The REAL REST thread endpoints observing the persisted side effects.
* Scripted LLM messages (stub LLM boundary) for agent-to-agent invocations.
"""

import uuid

import pytest

from tests.behavioral.composable_mcp.conftest import assistant_text, assistant_tool_call, call_mcp_tool
from tests.behavioral.conftest import configure_llm_credentials, create_thread

pytestmark = pytest.mark.asyncio(loop_scope="session")


def agent_yaml(name: str, tools: str = "[]") -> str:
    return f'name: {name}\nmodel: stub-model\nsystem_prompt: "One-shot test agent."\ntools: {tools}\ndebug: false\n'


async def _create_agent(client, users, name: str, tools: str = "[]"):
    resp = await client.post(
        "/api/v1/agents",
        headers=users["a"]["headers"],
        data={"agent_name": name},
        files={"file": ("agent.yaml", agent_yaml(name, tools).encode(), "text/yaml")},
    )
    assert resp.status_code == 201, resp.text
    return name


class TestOneShotExecution:
    async def test_run_creates_mcp_thread_and_returns_final_message(self, mcp_base_url, client, users, stub_llm):
        name = await _create_agent(client, users, f"oneshot-{uuid.uuid4().hex[:8]}")
        await configure_llm_credentials(client, users["a"]["headers"])

        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_agents_run",
            {"agent_name": name, "message": "Summarise my agents"},
        )
        assert result["ok"], f"composable_agents_run failed: {result}"
        payload = result["result"]

        for key in ("thread_id", "agent_name", "source", "message"):
            assert key in payload, f"one-shot result missing '{key}': {payload}"
        assert payload["agent_name"] == name
        assert payload["source"] == "mcp"
        final = payload["message"]
        assert final.get("content") == "stub-ok", f"unexpected final message content: {final}"

        thread_id = payload["thread_id"]
        assert thread_id, "one-shot result must include a persisted thread id"

    async def test_run_persists_source_mcp_thread(self, mcp_base_url, client, users, stub_llm):
        name = await _create_agent(client, users, f"persist-{uuid.uuid4().hex[:8]}")
        await configure_llm_credentials(client, users["a"]["headers"])
        result = await call_mcp_tool(
            mcp_base_url, users["a"]["headers"], "composable_agents_run", {"agent_name": name, "message": "hi"}
        )
        thread_id = result["result"]["thread_id"]

        got = await client.get(f"/api/v1/threads/{thread_id}", headers=users["a"]["headers"])
        assert got.status_code == 200, got.text
        assert got.json()["source"] == "mcp", "one-shot MCP execution must persist a source=mcp thread"

    async def test_run_persists_exactly_one_turn(self, mcp_base_url, client, users, stub_llm):
        """REQ-023: exactly ONE human turn executes through SendMessageUseCase."""
        name = await _create_agent(client, users, f"turns-{uuid.uuid4().hex[:8]}")
        await configure_llm_credentials(client, users["a"]["headers"])
        result = await call_mcp_tool(
            mcp_base_url, users["a"]["headers"], "composable_agents_run", {"agent_name": name, "message": "hello"}
        )
        thread_id = result["result"]["thread_id"]

        history = await client.get(f"/api/v1/threads/{thread_id}/history", headers=users["a"]["headers"])
        assert history.status_code == 200, history.text
        turns = history.json().get("history") or history.json().get("turns") or []
        assert len(turns) == 1, f"one-shot execution must persist exactly one turn, got {len(turns)}"

    async def test_run_persists_the_standard_trace(self, mcp_base_url, client, users, stub_llm):
        name = await _create_agent(client, users, f"trace-{uuid.uuid4().hex[:8]}")
        await configure_llm_credentials(client, users["a"]["headers"])
        result = await call_mcp_tool(
            mcp_base_url, users["a"]["headers"], "composable_agents_run", {"agent_name": name, "message": "hello"}
        )
        thread_id = result["result"]["thread_id"]

        trace = await client.get(f"/api/v1/threads/{thread_id}/trace", headers=users["a"]["headers"])
        assert trace.status_code == 200, trace.text
        assert trace.json(), "one-shot execution must persist the standard trace"

    async def test_run_unknown_agent_is_safe_not_found(self, mcp_base_url, users):
        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_agents_run",
            {"agent_name": "no-such-agent", "message": "hi"},
        )
        assert not result["ok"], f"unknown agent run must fail safely: {result}"
        assert "does-not" not in str(result["result"]).lower()

    async def test_run_rejects_empty_message(self, mcp_base_url, users):
        result = await call_mcp_tool(
            mcp_base_url, users["a"]["headers"], "composable_agents_run", {"agent_name": "any", "message": ""}
        )
        assert not result["ok"], "empty messages must be rejected (REQ-022)"


class TestPublicRestCannotSpoofMcpSource:
    async def test_thread_creation_ignores_client_supplied_mcp_source(self, client, users):
        name = f"spoof-{uuid.uuid4().hex[:8]}"
        await _create_agent(client, users, name)
        resp = await create_thread(client, users["a"]["headers"], name, source="mcp")
        if resp.status_code == 201:
            got = await client.get(f"/api/v1/threads/{resp.json()['id']}", headers=users["a"]["headers"])
            assert got.json()["source"] == "chat", "public REST thread creation must not accept source=mcp"
        else:
            assert resp.status_code in {400, 422}, resp.text


class TestAgentChainsGuarded:
    """REQ-029/REQ-030 — self invocation, cycles and depth are rejected.

    These drive real agent runners whose LLM boundary is scripted to issue
    the platform ``composable_agents_run`` tool call, so the rejection comes
    from the real execution chain. The exact YAML selector that wires the
    agent-facing platform tools into a runner is implementation-defined
    (REQ-028); ``tools: ["mcp"]`` is the declarative opt-in attempted here
    and may be adjusted during the green phase.
    """

    PLATFORM_TOOLS = '["mcp"]'

    async def _run(self, mcp_base_url, users, agent, message):
        return await call_mcp_tool(
            mcp_base_url, users["a"]["headers"], "composable_agents_run", {"agent_name": agent, "message": message}
        )

    async def test_direct_self_invocation_is_rejected(self, mcp_base_url, client, users, stub_llm):
        name = f"self-loop-{uuid.uuid4().hex[:8]}"
        await _create_agent(client, users, name, tools=self.PLATFORM_TOOLS)
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(
            assistant_tool_call("composable_agents_run", {"agent_name": name, "message": "again"}),
            assistant_text("should never happen"),
        )
        result = await self._run(mcp_base_url, users, name, "call yourself")
        assert not result["ok"], f"direct self-invocation must be rejected: {result}"

    async def test_two_agent_cycle_is_rejected(self, mcp_base_url, client, users, stub_llm):
        a_name = f"cycle-a-{uuid.uuid4().hex[:6]}"
        b_name = f"cycle-b-{uuid.uuid4().hex[:6]}"
        await _create_agent(client, users, a_name, tools=self.PLATFORM_TOOLS)
        await _create_agent(client, users, b_name, tools=self.PLATFORM_TOOLS)
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(
            assistant_tool_call("composable_agents_run", {"agent_name": b_name, "message": "go"}),
            assistant_tool_call("composable_agents_run", {"agent_name": a_name, "message": "cycle back"}),
            assistant_text("should never happen"),
        )
        result = await self._run(mcp_base_url, users, a_name, "start the cycle")
        assert not result["ok"], f"A→B→A cycle must be rejected: {result}"

    async def test_depth_beyond_four_is_rejected(self, mcp_base_url, client, users, stub_llm):
        names = [f"depth-{i}-{uuid.uuid4().hex[:6]}" for i in range(6)]
        for n in names:
            await _create_agent(client, users, n, tools=self.PLATFORM_TOOLS)
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(
            *(
                assistant_tool_call("composable_agents_run", {"agent_name": nxt, "message": "deeper"})
                for nxt in names[1:]
            ),
            assistant_text("should never happen"),
        )
        result = await self._run(mcp_base_url, users, names[0], "go deep")
        assert not result["ok"], f"invocations deeper than four must be rejected: {result}"
