"""Behavioral tests — thread deletion store cascade (REQ-097..100, AC-015).

``DELETE /api/v1/threads/{thread_id}`` must cascade-delete the associated
LangGraph Store objects under:

* ``/chat_attachments/<thread_id>/`` (chat images, REQ-025);
* ``/documents/<thread_id>/``       (generated PDFs, REQ-056).

The cascade is silent and idempotent (REQ-098/REQ-099): empty prefix trees
never raise. A store failure must NOT block the thread row deletion
(REQ-100 / EDG-017): the thread is gone from the user's listing even when the
store access errors out, and the failure is logged (REQ-100, NFR-018).
"""

import uuid

import pytest
from sqlalchemy import text

from src.infrastructure.database.rls_context import system_rls_context

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _put_file(client, headers, path: str, content: str) -> None:
    resp = await client.put(f"/api/v1/store/files/{path.lstrip('/')}", headers=headers, json={"content": content})
    assert resp.status_code == 200, resp.text


async def _list_under(client, headers, prefix: str) -> list[str]:
    resp = await client.get("/api/v1/store/files", params={"prefix": prefix}, headers=headers)
    assert resp.status_code == 200, resp.text
    return [p for p in resp.json() if p.startswith(prefix)]


class TestThreadDeletionCascade:
    async def test_deleting_thread_removes_chat_attachments_and_documents(self, client, users):
        """AC-015: both prefixes are cleaned up — images and PDFs."""
        headers = users["a"]["headers"]
        agent_name = f"cascade-{uuid.uuid4().hex[:8]}"
        from tests.behavioral.conftest import create_thread

        resp = await client.post(
            "/api/v1/agents",
            headers=headers,
            data={"agent_name": agent_name},
            files={"file": ("agent.yaml", f"name: {agent_name}\nmodel: stub-model\ntools: []\n".encode(), "text/yaml")},
        )
        assert resp.status_code == 201, resp.text
        thread_id = (await create_thread(client, headers, agent_name)).json()["id"]

        await _put_file(client, headers, f"/chat_attachments/{thread_id}/att-1/capture.png", "aW1hZ2U=")
        await _put_file(client, headers, f"/documents/{thread_id}/uuid-1/rapport.pdf", "JVBERi0xLjc=")

        deleted = await client.delete(f"/api/v1/threads/{thread_id}", headers=headers)
        assert deleted.status_code in {200, 204}, deleted.text

        assert not await _list_under(client, headers, f"/chat_attachments/{thread_id}/")
        assert not await _list_under(client, headers, f"/documents/{thread_id}/")

    async def test_deleting_thread_preserves_other_threads_files(self, client, users):
        """The cascade is scoped to the deleted thread only."""
        from tests.behavioral.conftest import create_thread

        headers = users["a"]["headers"]
        name_a, name_b = (f"cascade-a-{uuid.uuid4().hex[:6]}", f"cascade-b-{uuid.uuid4().hex[:6]}")
        for name in (name_a, name_b):
            resp = await client.post(
                "/api/v1/agents",
                headers=headers,
                data={"agent_name": name},
                files={"file": ("agent.yaml", f"name: {name}\nmodel: stub-model\ntools: []\n".encode(), "text/yaml")},
            )
            assert resp.status_code == 201, resp.text
        thread_a = (await create_thread(client, headers, name_a)).json()["id"]
        thread_b = (await create_thread(client, headers, name_b)).json()["id"]

        await _put_file(client, headers, f"/chat_attachments/{thread_b}/att/keep.png", "a2VlcA==")

        deleted = await client.delete(f"/api/v1/threads/{thread_a}", headers=headers)
        assert deleted.status_code in {200, 204}, deleted.text
        remaining = await _list_under(client, headers, f"/chat_attachments/{thread_b}/")
        assert remaining, "files of OTHER threads must survive the cascade"

    async def test_cascade_is_idempotent_for_empty_prefixes(self, client, users):
        """EDG-018: deleting a thread with no store files must not raise."""
        from tests.behavioral.conftest import create_thread

        headers = users["a"]["headers"]
        agent_name = f"cascade-empty-{uuid.uuid4().hex[:8]}"
        resp = await client.post(
            "/api/v1/agents",
            headers=headers,
            data={"agent_name": agent_name},
            files={"file": ("agent.yaml", f"name: {agent_name}\nmodel: stub-model\ntools: []\n".encode(), "text/yaml")},
        )
        assert resp.status_code == 201, resp.text
        thread_id = (await create_thread(client, headers, agent_name)).json()["id"]

        deleted = await client.delete(f"/api/v1/threads/{thread_id}", headers=headers)
        assert deleted.status_code in {200, 204}, deleted.text

    async def test_store_failure_does_not_block_thread_deletion(self, client, users, stack, caplog):
        """REQ-100 / EDG-017: unreachable store still deletes the thread row.

        The LangGraph store table is made inaccessible (renamed away via the
        superuser connection — a REAL infrastructure failure, no internal
        mocking); the thread delete must still succeed and the failure must
        be logged at WARNING+ level.
        """
        from tests.behavioral.conftest import create_thread

        headers = users["a"]["headers"]
        agent_name = f"cascade-fail-{uuid.uuid4().hex[:8]}"
        resp = await client.post(
            "/api/v1/agents",
            headers=headers,
            data={"agent_name": agent_name},
            files={"file": ("agent.yaml", f"name: {agent_name}\nmodel: stub-model\ntools: []\n".encode(), "text/yaml")},
        )
        assert resp.status_code == 201, resp.text
        thread_id = (await create_thread(client, headers, agent_name)).json()["id"]

        # Make the store UNAVAILABLE for real: rename the underlying table.
        await _put_file(client, headers, f"/chat_attachments/{thread_id}/att/broken.png", "YnJva2Vu")
        async with system_rls_context(), stack["engine"].begin() as conn:
            await conn.execute(text("ALTER TABLE store RENAME TO store_cascade_test_bak"))

        # Restore the store table whatever happens to the test run (I11).
        try:
            with caplog.at_level("WARNING"):
                deleted = await client.delete(f"/api/v1/threads/{thread_id}", headers=headers)
        finally:
            async with system_rls_context(), stack["engine"].begin() as conn:
                await conn.execute(text("ALTER TABLE store_cascade_test_bak RENAME TO store"))

        assert deleted.status_code in {200, 204}, deleted.text
        listing = await client.get("/api/v1/threads", headers=headers)
        assert all(t["id"] != thread_id for t in listing.json()), "thread row must be deleted"
        assert any(
            record.levelno >= 30 and thread_id in record.getMessage()
            for record in caplog.records
        ), "the cascade failure must be explicitly logged (REQ-100)"
