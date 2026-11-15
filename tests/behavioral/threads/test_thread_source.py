"""Behavioral tests — immutable thread source chat|cron (spec §5.6 SRC-001/002).

Routes under test:
* POST /api/v1/threads          (public creation — source NOT client-settable)
* GET  /api/v1/threads, GET /api/v1/threads/{id}   (source included)
* GET  /api/v1/threads/{id}/history (thread payload includes source)
* POST /api/v1/chat/{thread_id} (chat path never mutates source)

Cron-created threads (source="cron") are covered by the crons suite — this
file pins the chat path and spoof prevention.
"""

import uuid

import pytest

from tests.behavioral.conftest import configure_llm_credentials, create_thread, invoke_agent

pytestmark = pytest.mark.asyncio(loop_scope="session")


def agent_yaml(name: str) -> str:
    return f'name: {name}\nmodel: stub-model\nsystem_prompt: "Source test agent."\ntools: []\ndebug: false\n'


async def _agent_with_thread_resp(client, users, *, spoofed_body: dict | None = None):
    name = f"agent-{uuid.uuid4().hex[:8]}"
    resp = await client.post(
        "/api/v1/agents",
        headers=users["a"]["headers"],
        data={"agent_name": name},
        files={"file": ("agent.yaml", agent_yaml(name).encode(), "text/yaml")},
    )
    assert resp.status_code == 201, resp.text
    thread_resp = await create_thread(client, users["a"]["headers"], name, **(spoofed_body or {}))
    return thread_resp


async def _assert_thread_source_is_chat(client, headers, thread_id: str) -> None:
    got = await client.get(f"/api/v1/threads/{thread_id}", headers=headers)
    assert got.status_code == 200
    assert got.json()["source"] == "chat"


class TestChatThreadsDefaultToSourceChat:
    async def test_new_thread_has_source_chat_in_get_and_list(self, client, users):
        thread = (await _agent_with_thread_resp(client, users)).json()

        got = await client.get(f"/api/v1/threads/{thread['id']}", headers=users["a"]["headers"])
        assert got.status_code == 200
        assert got.json()["source"] == "chat"

        listing = await client.get("/api/v1/threads", headers=users["a"]["headers"])
        assert listing.status_code == 200
        sources = {t["id"]: t["source"] for t in listing.json()}
        assert sources[thread["id"]] == "chat"

    async def test_history_payload_includes_source(self, client, users):
        thread = (await _agent_with_thread_resp(client, users)).json()
        history = await client.get(f"/api/v1/threads/{thread['id']}/history", headers=users["a"]["headers"])
        assert history.status_code == 200
        assert history.json()["thread"]["source"] == "chat"


class TestSpoofPrevention:
    async def test_client_cannot_set_source_cron_on_public_creation(self, client, users):
        """SRC-002 / §9: public POST /api/v1/threads accepts only existing
        public inputs; `source` is not client-controllable. Either it has no
        effect (thread created as chat) or the request is rejected."""
        resp = await _agent_with_thread_resp(client, users, spoofed_body={"source": "cron"})
        if resp.status_code == 201:
            await _assert_thread_source_is_chat(client, users["a"]["headers"], resp.json()["id"])
        else:
            assert resp.status_code in {400, 422}, resp.text

    async def test_client_cannot_set_arbitrary_source_value(self, client, users):
        resp = await _agent_with_thread_resp(client, users, spoofed_body={"source": "totally-real"})
        if resp.status_code == 201:
            await _assert_thread_source_is_chat(client, users["a"]["headers"], resp.json()["id"])
        else:
            assert resp.status_code in {400, 422}, resp.text


class TestSourceImmutability:
    async def test_chat_message_stream_never_changes_source(self, client, users, stub_llm):
        thread = (await _agent_with_thread_resp(client, users)).json()
        # Same per-user credential setup convention as the agents/crons suites.
        await configure_llm_credentials(client, users["a"]["headers"])
        resp = await invoke_agent(client, users["a"]["headers"], thread["id"], message="hello")
        assert resp.status_code == 200, resp.text

        history = await client.get(f"/api/v1/threads/{thread['id']}/history", headers=users["a"]["headers"])
        assert history.status_code == 200
        assert history.json()["thread"]["source"] == "chat"

    async def test_no_thread_update_api_for_source(self, client, users):
        """SRC-001: no thread update API is introduced for source — PATCH/PUT
        on a thread must not exist (immutable after insert)."""
        thread = (await _agent_with_thread_resp(client, users)).json()
        for method in ("patch", "put"):
            resp = await getattr(client, method)(
                f"/api/v1/threads/{thread['id']}", headers=users["a"]["headers"], json={"source": "cron"}
            )
            assert resp.status_code in {404, 405}, (
                f"Thread source must be immutable — {method.upper()} must not be a supported update path "
                f"(got {resp.status_code})"
            )


class TestCrossUser:
    async def test_threads_scoped_to_owner(self, client, users):
        thread = (await _agent_with_thread_resp(client, users)).json()
        foreign = await client.get(f"/api/v1/threads/{thread['id']}", headers=users["b"]["headers"])
        assert foreign.status_code == 404
