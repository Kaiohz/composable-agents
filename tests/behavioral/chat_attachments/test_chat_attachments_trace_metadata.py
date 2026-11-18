"""Behavioral tests — attachments trace metadata + Message projection.

* REQ-020: ``HUMAN_MESSAGE`` events persist ``metadata.attachments`` with
  id / filename / content_type / size_bytes / kind / source.
* REQ-021 / REQ-016: NO trace contains ``content_base64``.
* REQ-022: the ``Message`` projection (history/messages endpoints) exposes
  the attachments metadata for the conversation history.
* REQ-023: the internal RustFS ref kept for cleanup is NOT exposed by the
  ``Message`` projection (document object paths invisible in history).
"""

import json

import pytest

from tests.behavioral.chat_attachments.conftest import (
    agent_with_thread,
    document_attachment,
    image_attachment,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

SPEC_ATTACHMENT_KEYS = {"id", "filename", "content_type", "size_bytes", "kind", "source"}


async def _invoke_mixed(client, headers, thread_id):
    return await client.post(
        f"/api/v1/chat/{thread_id}",
        headers=headers,
        json={
            "message": "Analyse cette image et ce document",
            "attachments": [image_attachment(), document_attachment()],
        },
    )


class TestHumanMessageTraceMetadata:
    """REQ-020 / REQ-021."""

    async def test_human_message_event_records_attachment_metadata(
        self, client, users, stub_llm, mcp_files
    ):
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        resp = await _invoke_mixed(client, users["a"]["headers"], thread_id)
        assert resp.status_code == 200, resp.text

        trace = await client.get(f"/api/v1/threads/{thread_id}/trace", headers=users["a"]["headers"])
        assert trace.status_code == 200, trace.text
        human_events = [e for e in trace.json()["events"] if e["type"] == "human_message"]
        assert human_events, "the turn must produce a HUMAN_MESSAGE event"
        metadata = (human_events[-1].get("metadata") or {}).get("attachments") or []
        assert len(metadata) == 2, f"both attachments must be traced: {metadata}"

        by_kind = {a["kind"]: a for a in metadata}
        image = by_kind.get("image")
        document = by_kind.get("document")
        assert image, f"an image attachment entry is expected: {metadata}"
        assert document, f"a document attachment entry is expected: {metadata}"
        for entry in (image, document):
            assert SPEC_ATTACHMENT_KEYS.issubset(entry.keys()), entry
            assert entry["id"], "each attachment carries a generated id"
            assert isinstance(entry["size_bytes"], int) and entry["size_bytes"] > 0

        assert image["filename"] == "capture.png"
        assert image["content_type"] == "image/png"
        assert image["source"] == "langgraph_store"
        assert document["filename"] == "rapport.pdf"
        assert document["content_type"] == "application/pdf"
        assert document["source"] == "rustfs_temp"

    async def test_no_base64_anywhere_in_trace_or_history(self, client, users, stub_llm, mcp_files):
        """REQ-021 / REQ-016: neither the trace nor the history leaks base64."""
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        image_b64 = image_attachment()["content_base64"]
        doc_raw = b"%PDF-1.7 secret document payload"
        doc_b64 = document_attachment(content_base64=__import__("base64").b64encode(doc_raw).decode())
        resp = await client.post(
            f"/api/v1/chat/{thread_id}",
            headers=users["a"]["headers"],
            json={
                "message": "Analyse",
                "attachments": [image_attachment(), doc_b64],
            },
        )
        assert resp.status_code == 200, resp.text

        trace = await client.get(f"/api/v1/threads/{thread_id}/trace", headers=users["a"]["headers"])
        assert trace.status_code == 200
        assert image_b64 not in trace.text, "the image base64 must never be persisted in traces"
        assert __import__("base64").b64encode(doc_raw).decode() not in trace.text

        messages = await client.get(f"/api/v1/threads/{thread_id}/messages", headers=users["a"]["headers"])
        assert messages.status_code == 200
        assert image_b64 not in messages.text

        history = await client.get(f"/api/v1/threads/{thread_id}/history", headers=users["a"]["headers"])
        assert history.status_code == 200
        assert image_b64 not in history.text


class TestMessageProjection:
    """REQ-022 / REQ-023."""

    async def test_message_projection_exposes_attachments(self, client, users, stub_llm, mcp_files):
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        resp = await _invoke_mixed(client, users["a"]["headers"], thread_id)
        assert resp.status_code == 200, resp.text

        messages = await client.get(f"/api/v1/threads/{thread_id}/messages", headers=users["a"]["headers"])
        assert messages.status_code == 200, messages.text
        human = [m for m in messages.json() if m["role"] == "human"]
        assert human, "a human message must be projected"
        attachments = human[-1].get("attachments")
        assert attachments is not None, "the Message projection must expose attachments (REQ-022)"
        assert len(attachments) == 2
        for entry in attachments:
            # REQ-023: projection exposes ONLY the spec metadata keys — no
            # internal refs (rustfs object path / store path / base64).
            extra_keys = set(entry.keys()) - SPEC_ATTACHMENT_KEYS
            assert not extra_keys, f"internal refs must not leak into the projection: {extra_keys}"

        # The projected metadata must not contain the temporary rustfs path.
        assert (
            "chat_tmp/" not in json.dumps(attachments)
        ), "the internal cleanup ref (chat_tmp/...) must not be exposed (REQ-023)"

        # History (turn-based projection) exposes the same metadata.
        history = await client.get(f"/api/v1/threads/{thread_id}/history", headers=users["a"]["headers"])
        assert history.status_code == 200
        turns = history.json()["turns"]
        assert turns, "history turns expected"
        human_message = turns[0]["human_message"]
        assert human_message is not None
        assert human_message.get("attachments"), "history human message carries attachments (REQ-070)"
