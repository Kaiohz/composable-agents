"""Behavioral tests — LLM usage capture from the REAL chat path (dashboard §1, §2).

Feature under test (spec llm-usage-traces-dashboard.md, backend changes 1-2):

The real chat chain (POST /api/v1/chat/{thread_id} → SendMessageUseCase →
PersistentAgentRegistry → DeepAgentRunner.invoke → PostgresTraceEventRepository,
every layer real, ONLY the OpenAI-compatible LLM API stubbed via
``StubLlmServer``) must enrich the terminal ``AI_MESSAGE`` TraceEvent of each
turn with a ``metadata.llm_usage`` object:

    {
      "input_tokens":  int,   # from LangGraph usage_metadata.input_tokens
      "output_tokens": int,   # from LangGraph usage_metadata.output_tokens
      "total_tokens":  int,   # from LangGraph usage_metadata.total_tokens
      "model_name":    str,   # the resolved model that served the turn
      "provider":      str,   # the serving provider (openai-compatible chain)
      "duration_ms":   int,   # HUMAN_MESSAGE → AI_MESSAGE wall time, int >= 0
    }

Contract decisions fixed here for the implementation (Step 2):

* ``duration_ms`` is an int (ms resolution) >= 0, measured from the turn's
  persisted HUMAN_MESSAGE timestamp to its terminal AI_MESSAGE timestamp.
* When the LLM returns no ``usage_metadata`` (stub mode ``usage=None``):
  the turn still completes (HTTP 200), the AI_MESSAGE is persisted WITHOUT an
  ``llm_usage`` key and the aggregation endpoints later report that run with
  zero tokens, cost=0 and priced=false (asserted in the usage-routes suite).
* The existing TraceEvent shape / thread-history HTTP contract stays intact:
  the AI_MESSAGE payload (content/Message reconstruction) is unchanged —
  only ``metadata`` is enriched.
"""

import uuid

import pytest
from sqlalchemy import text

from src.infrastructure.database.rls_context import system_rls_context
from tests.behavioral.conftest import configure_llm_credentials, create_thread, invoke_agent

pytestmark = pytest.mark.asyncio(loop_scope="session")


def agent_yaml(name: str, model: str = "stub-model") -> str:
    return f'name: {name}\nmodel: {model}\nsystem_prompt: "Usage capture agent."\ntools: []\ndebug: false\n'


async def _create_agent(client, headers, name: str, model: str = "stub-model") -> str:
    resp = await client.post(
        "/api/v1/agents",
        headers=headers,
        data={"agent_name": name},
        files={"file": ("agent.yaml", agent_yaml(name, model).encode(), "text/yaml")},
    )
    assert resp.status_code == 201, resp.text
    return name


async def _ai_message_metadata(stack, thread_id: str):
    """The persisted terminal AI_MESSAGE metadata of a thread (side effect)."""
    async with system_rls_context(), stack["engine"].begin() as conn:
        rows = (
            await conn.execute(
                text("SELECT metadata FROM trace_events WHERE thread_id = :t AND type = 'ai_message'"),
                {"t": thread_id},
            )
        ).all()
    assert len(rows) == 1, f"expected exactly one persisted AI_MESSAGE event, got {len(rows)}"
    return rows[0][0]


class TestUsageCaptureContract:
    async def test_chat_invocation_persists_llm_usage_on_ai_message(self, stack, client, users, stub_llm):
        """A real invocation with usage_metadata (stub: 11/7/18) must persist
        the complete llm_usage object on the turn's AI_MESSAGE event."""
        stub_llm.reset(usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18})
        agent_name = await _create_agent(client, users["a"]["headers"], f"UsageCap{uuid.uuid4().hex[:6]}")
        thread = await create_thread(client, users["a"]["headers"], agent_name)
        assert thread.status_code == 201, thread.text
        thread_id = thread.json()["id"]

        await configure_llm_credentials(client, users["a"]["headers"], api_key="sk-cred-usage-capture")
        resp = await invoke_agent(client, users["a"]["headers"], thread_id)
        assert resp.status_code == 200, resp.text

        metadata = await _ai_message_metadata(stack, thread_id)
        assert metadata is not None, "AI_MESSAGE metadata must carry the llm_usage enrichment"
        assert "llm_usage" in metadata, f"llm_usage missing from AI_MESSAGE metadata: {metadata!r}"
        usage = metadata["llm_usage"]
        assert set(usage) == {
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "model_name",
            "provider",
            "duration_ms",
        }, f"llm_usage exposes unexpected fields: {set(usage)}"
        assert usage["input_tokens"] == 11, f"input_tokens must come from usage_metadata: {usage}"
        assert usage["output_tokens"] == 7, f"output_tokens must come from usage_metadata: {usage}"
        assert usage["total_tokens"] == 18, f"total_tokens must come from usage_metadata: {usage}"
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
        assert usage["model_name"] == "stub-model", f"model_name must be the resolved model: {usage}"
        assert usage["provider"] == "openai", f"provider must identify the serving provider: {usage}"
        assert isinstance(usage["duration_ms"], int) and usage["duration_ms"] >= 0, (
            f"duration_ms must be an int >= 0 (HUMAN_MESSAGE → AI_MESSAGE wall time): {usage}"
        )

    async def test_second_turn_is_captured_individually(self, stack, client, users, stub_llm):
        """Every turn produces its own AI_MESSAGE llm_usage capture — two
        turns ⇒ two persisted AI_MESSAGE events, each with llm_usage."""
        stub_llm.reset(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        agent_name = await _create_agent(client, users["a"]["headers"], f"Usage2Turns{uuid.uuid4().hex[:6]}")
        thread = await create_thread(client, users["a"]["headers"], agent_name)
        thread_id = thread.json()["id"]
        await configure_llm_credentials(client, users["a"]["headers"], api_key="sk-cred-usage-2turns")

        resp = await invoke_agent(client, users["a"]["headers"], thread_id)
        assert resp.status_code == 200, resp.text
        resp = await invoke_agent(client, users["a"]["headers"], thread_id, message="second turn")
        assert resp.status_code == 200, resp.text

        async with system_rls_context(), stack["engine"].begin() as conn:
            rows = (
                await conn.execute(
                    text("SELECT metadata->'llm_usage' FROM trace_events WHERE thread_id = :t AND type = 'ai_message'"),
                    {"t": thread_id},
                )
            ).all()
        assert len(rows) == 2, "each turn must capture its own llm_usage"
        for (usage,) in rows:
            assert usage is not None and usage["total_tokens"] == 15, usage

    async def test_chat_with_human_and_ai_events_only_enriches_metadata(self, stack, client, users, stub_llm):
        """The enrichment must not break the existing event stream shape: the
        turn still persists HUMAN_MESSAGE + AI_MESSAGE in sequence order with
        the AI payload content intact (only metadata is added)."""
        agent_name = await _create_agent(client, users["a"]["headers"], f"ShapeSafe{uuid.uuid4().hex[:6]}")
        thread = await create_thread(client, users["a"]["headers"], agent_name)
        thread_id = thread.json()["id"]
        await configure_llm_credentials(client, users["a"]["headers"], api_key="sk-cred-shape")
        resp = await invoke_agent(client, users["a"]["headers"], thread_id)
        assert resp.status_code == 200, resp.text

        async with system_rls_context(), stack["engine"].begin() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT type, sequence, content FROM trace_events "
                        "WHERE thread_id = :t AND type IN ('human_message', 'ai_message') ORDER BY timestamp"
                    ),
                    {"t": thread_id},
                )
            ).all()
        assert [(r[0], r[1]) for r in rows] == [("human_message", 0), ("ai_message", 1)], rows
        ai_content = rows[1][2]
        assert ai_content and '"stub-ok"' in ai_content, "AI_MESSAGE payload content must stay intact"


class TestMissingUsageMetadata:
    async def test_chat_without_usage_still_completes_without_llm_usage(self, stack, client, users, stub_llm):
        """Missing usage_metadata (stub: usage=None) must not break the chat
        path — the turn completes and the AI_MESSAGE carries NO llm_usage."""
        stub_llm.reset(usage=None)
        agent_name = await _create_agent(client, users["a"]["headers"], f"NoUsage{uuid.uuid4().hex[:6]}")
        thread = await create_thread(client, users["a"]["headers"], agent_name)
        thread_id = thread.json()["id"]
        await configure_llm_credentials(client, users["a"]["headers"], api_key="sk-cred-nousage")

        resp = await invoke_agent(client, users["a"]["headers"], thread_id)
        assert resp.status_code == 200, resp.text

        metadata = await _ai_message_metadata(stack, thread_id)
        assert metadata is None or "llm_usage" not in metadata, (
            f"llm_usage must stay absent when the LLM returns no usage_metadata: {metadata!r}"
        )


class TestHistoryEndpointStaysUnchanged:
    async def test_thread_history_contract_survives_llm_usage_enrichment(self, client, users, stub_llm):
        """The existing GET /api/v1/threads/{id}/history contract (dashboard
        sub-tab) keeps its exact shape while usage capture runs on the same
        invocation: thread summary source + turns with human/ai messages."""
        stub_llm.reset(usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7})
        agent_name = await _create_agent(client, users["a"]["headers"], f"HistShape{uuid.uuid4().hex[:6]}")
        thread = await create_thread(client, users["a"]["headers"], agent_name)
        thread_id = thread.json()["id"]
        await configure_llm_credentials(client, users["a"]["headers"], api_key="sk-cred-history")

        resp = await invoke_agent(client, users["a"]["headers"], thread_id)
        assert resp.status_code == 200, resp.text

        history = await client.get(f"/api/v1/threads/{thread_id}/history", headers=users["a"]["headers"])
        assert history.status_code == 200, history.text
        payload = history.json()
        assert payload["thread"]["id"] == thread_id
        assert payload["thread"]["source"] == "chat"
        assert isinstance(payload["turns"], list) and len(payload["turns"]) == 1
        turn = payload["turns"][0]
        assert set(turn) == {"turn_id", "human_message", "ai_message", "events"}
        assert turn["ai_message"]["content"] == "stub-ok"
        assert turn["human_message"]["content"] == "hello"
