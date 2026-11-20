"""Behavioral tests — built-in assistant HITL rules for destructive deletes
(REQ-042/REQ-043, REQ-027).

Pinned behavior:
* The built-in assistant's definition configures HITL interrupt rules for
  EVERY delete tool of REQ-011.
* A delete proposed by the assistant (LLM tool call) interrupts into
  ``awaiting_hitl``; reject keeps the resource; approve deletes it exactly once.
* A DIRECT authenticated MCP ``tools/call`` of a delete tool is an explicit
  client command: it executes immediately (no assistant HITL), subject to
  normal validation/authorization.
"""

import uuid

import pytest

from tests.behavioral.composable_mcp.conftest import (
    DELETION_TOOLS,
    assistant_tool_call,
    call_mcp_tool,
)
from tests.behavioral.conftest import configure_llm_credentials

pytestmark = pytest.mark.asyncio(loop_scope="session")

BUILTIN_NAME = "composable-ai-assistant"


def agent_yaml(name: str) -> str:
    return f'name: {name}\nmodel: stub-model\nsystem_prompt: "Doomed agent."\ntools: []\ndebug: false\n'


async def _create_doomed_agent(client, users) -> str:
    name = f"doomed-{uuid.uuid4().hex[:8]}"
    resp = await client.post(
        "/api/v1/agents",
        headers=users["a"]["headers"],
        data={"agent_name": name},
        files={"file": ("agent.yaml", agent_yaml(name).encode(), "text/yaml")},
    )
    assert resp.status_code == 201, resp.text
    return name


def _rule_interrupts(value) -> bool:
    """A HITL rule entry of ``True`` or an InterruptRule dict means interrupt-on."""
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return value.get("interrupt", True)
    return False


class TestHitlRulesConfigured:
    async def test_builtin_definition_interrupts_on_every_delete_tool(self, client, users):
        """REQ-042: every destructive delete tool is configured with an interrupt rule."""
        got = await client.get(f"/api/v1/agents/{BUILTIN_NAME}", headers=users["a"]["headers"])
        assert got.status_code == 200, f"built-in definition must be readable: {got.text}"
        hitl = got.json().get("hitl") or {}
        rules = hitl.get("rules") or {}
        missing = [t for t in DELETION_TOOLS if not _rule_interrupts(rules.get(t))]
        assert not missing, f"delete tools missing HITL interrupt rules: {missing}"


class TestDirectMcpDeleteExecutesWithoutHitl:
    async def test_direct_delete_tool_call_executes_immediately(self, mcp_base_url, client, users):
        """REQ-043: direct MCP tools/call of a delete tool is an explicit client
        command — no assistant HITL gate, normal validation/authorization apply."""
        doomed = await _create_doomed_agent(client, users)
        result = await call_mcp_tool(
            mcp_base_url, users["a"]["headers"], "composable_agents_delete", {"agent_name": doomed}
        )
        assert result["ok"], f"direct MCP delete must execute without assistant HITL: {result}"

        listing = await client.get("/api/v1/agents", headers=users["a"]["headers"])
        assert doomed not in {a["name"] for a in listing.json()}, "resource must actually be deleted"

    async def test_direct_delete_still_authorizes(self, mcp_base_url, client, users):
        doomed = await _create_doomed_agent(client, users)
        foreign = await call_mcp_tool(
            mcp_base_url, users["b"]["headers"], "composable_agents_delete", {"agent_name": doomed}
        )
        assert not foreign["ok"], "a foreign user must not delete another user's agent"


class TestAssistantDeleteProposalFlow:
    """The assistant (scripted LLM) proposes an agent delete → HITL gate."""

    async def test_delete_proposal_reaches_awaiting_hitl(self, mcp_base_url, client, users, stub_llm):
        """REQ-027: the one-shot result reports awaiting_hitl + thread_id and does
        NOT claim completion, and the resource is NOT yet deleted."""
        doomed = await _create_doomed_agent(client, users)
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(assistant_tool_call("composable_agents_delete", {"agent_name": doomed}))

        result = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_agents_run",
            {"agent_name": BUILTIN_NAME, "message": "clean up"},
        )
        assert result["ok"], f"proposing run failed: {result}"
        payload = result["result"]
        assert payload.get("thread_id"), f"awaiting_hitl result must include the thread id: {payload}"
        final = payload["message"]
        assert final.get("status") == "awaiting_hitl", f"expected awaiting_hitl status, got: {final}"
        blob = str(final).lower()
        assert "completed" not in blob, "an interrupted one-shot must not claim completion"

        listing = await client.get("/api/v1/agents", headers=users["a"]["headers"])
        assert doomed in {a["name"] for a in listing.json()}, "deletion must NOT happen before approval"

        return payload["thread_id"]

    async def test_reject_keeps_resource_intact_and_executes_nothing(self, mcp_base_url, client, users, stub_llm):
        doomed = await _create_doomed_agent(client, users)
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(assistant_tool_call("composable_agents_delete", {"agent_name": doomed}))

        propose = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_agents_run",
            {"agent_name": BUILTIN_NAME, "message": "clean up"},
        )
        thread_id = propose["result"]["thread_id"]

        reject = await client.post(
            f"/api/v1/chat/{thread_id}",
            headers=users["a"]["headers"],
            json={"decisions": [{"tool_call_id": "call_1", "action": "reject"}]},
        )
        assert reject.status_code == 200, f"reject decision failed: {reject.text}"

        listing = await client.get("/api/v1/agents", headers=users["a"]["headers"])
        assert doomed in {a["name"] for a in listing.json()}, "rejecting must leave the resource intact"

    async def test_approve_executes_exactly_one_delete(self, mcp_base_url, client, users, stub_llm):
        doomed = await _create_doomed_agent(client, users)
        await configure_llm_credentials(client, users["a"]["headers"])
        stub_llm.script(assistant_tool_call("composable_agents_delete", {"agent_name": doomed}))

        propose = await call_mcp_tool(
            mcp_base_url,
            users["a"]["headers"],
            "composable_agents_run",
            {"agent_name": BUILTIN_NAME, "message": "clean up"},
        )
        thread_id = propose["result"]["thread_id"]

        approve = await client.post(
            f"/api/v1/chat/{thread_id}",
            headers=users["a"]["headers"],
            json={"decisions": [{"tool_call_id": "call_1", "action": "approve"}]},
        )
        assert approve.status_code == 200, f"approve decision failed: {approve.text}"

        listing = await client.get("/api/v1/agents", headers=users["a"]["headers"])
        assert doomed not in {a["name"] for a in listing.json()}, "approve must execute the deletion"

        assert approve.json().get("status") != "awaiting_hitl", "resumed turn must not immediately interrupt again"
