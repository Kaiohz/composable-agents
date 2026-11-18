"""Behavioral tests — image attachments → vision content blocks (REQ-012..014).

Images follow the VISION path (REQ-025..028): stored in the user's LangGraph
Store namespace (observable through the real ``/api/v1/store`` routes), sent
to the LLM as standard multimodal content blocks, and NEVER routed to the
Files/Classical MCP servers (REQ-028).

The whole chain is real: route → use case → attachments service → image store
→ runner → (stub LLM = the only external boundary, which RECORDS requests, so
we can assert the multimodal content blocks).
"""

import base64

import pytest

from tests.behavioral.chat_attachments.conftest import (
    agent_with_thread,
    image_attachment,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

PNG_BASE64 = image_attachment()["content_base64"]
DATA_URL_PREFIX = "data:image/png;base64,"


async def _invoke_with_image(client, headers, thread_id, attachments, message="Regarde cette image"):
    return await client.post(
        f"/api/v1/chat/{thread_id}",
        headers=headers,
        json={"message": message, "attachments": attachments},
    )


class TestVisionContentBlocks:
    """AC-001 / REQ-012 / REQ-013: image + text reaches the LLM."""

    async def test_image_and_text_reach_llm_as_multimodal_blocks(
        self, client, users, stub_llm, mcp_files
    ):
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        resp = await _invoke_with_image(
            client,
            users["a"]["headers"],
            thread_id,
            [image_attachment()],
            message="Regarde cette image",
        )
        assert resp.status_code == 200, resp.text

        assert stub_llm.app.requests, "the LLM must have been called"
        user_messages = [
            m for r in stub_llm.app.requests for m in r.body.get("messages", []) if m.get("role") == "user"
        ]
        assert user_messages, "a user message must be sent to the LLM"
        content = user_messages[-1].get("content")
        assert isinstance(content, list), f"multimodal content blocks expected, got: {content!r}"

        types = [b.get("type") for b in content]
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        text_blocks = [b for b in content if b.get("type") == "text"]
        assert image_blocks, f"an image block must be present: {types}"
        assert text_blocks, f"the user text must be preserved (REQ-013): {types}"
        # REQ-012/REQ-013: the user text comes FIRST, the image block carries
        # the base64 payload as a data URL.
        assert "Regarde cette image" in text_blocks[0].get("text", "")
        assert image_blocks[0]["image_url"]["url"].startswith(DATA_URL_PREFIX)
        assert image_blocks[0]["image_url"]["url"].endswith(PNG_BASE64)

    async def test_image_only_without_text_is_allowed(self, client, users, stub_llm):
        """EDG-001 / REQ-014: message empty but image present → multimodal."""
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        resp = await client.post(
            f"/api/v1/chat/{thread_id}",
            headers=users["a"]["headers"],
            json={"attachments": [image_attachment()]},
        )
        assert resp.status_code == 200, resp.text
        user_messages = [
            m for r in stub_llm.app.requests for m in r.body.get("messages", []) if m.get("role") == "user"
        ]
        assert user_messages
        content = user_messages[-1].get("content")
        image_blocks = [b for b in (content if isinstance(content, list) else []) if b.get("type") == "image_url"]
        assert image_blocks, "image-only turns must still deliver the image content block"

    async def test_images_never_sent_to_files_mcp(self, client, users, stub_llm, mcp_files):
        """REQ-028: the vision path never touches the Files MCP server."""
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        resp = await _invoke_with_image(client, users["a"]["headers"], thread_id, [image_attachment()])
        assert resp.status_code == 200, resp.text
        assert mcp_files.upload_calls == []
        assert mcp_files.delete_calls == []


class TestImageStorePersistence:
    """REQ-025..027: images are stored in the user's LangGraph Store."""

    async def test_image_stored_under_chat_attachments_prefix(self, client, users, stub_llm):
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        attachment = image_attachment(filename="capture.png")
        resp = await _invoke_with_image(client, users["a"]["headers"], thread_id, [attachment])
        assert resp.status_code == 200, resp.text

        listing = await client.get(
            "/api/v1/store/files", params={"prefix": f"/chat_attachments/{thread_id}/"}, headers=users["a"]["headers"]
        )
        assert listing.status_code == 200, listing.text
        paths = [p for p in listing.json() if p.startswith(f"/chat_attachments/{thread_id}/")]
        assert paths, f"the image must be stored under /chat_attachments/<thread_id>/: {listing.json()}"
        stored = paths[0]
        # REQ-025: <thread_id>/<attachment_id>/<filename> with a NON-empty
        # unique attachment id per attachment (EDG-006).
        parts = stored.strip("/").split("/")
        assert len(parts) >= 4, f"path structure /chat_attachments/<thread>/<attachment_id>/<filename>: {stored}"
        assert parts[-1] == "capture.png"

        # REQ-027: the image content is retrievable from the store API.
        fetched = await client.get(f"/api/v1/store/files/{stored.lstrip('/')}", headers=users["a"]["headers"])
        assert fetched.status_code == 200, fetched.text
        content = fetched.json()["content"]
        assert base64.b64decode(content)  # decodable payload

    async def test_two_identical_images_get_distinct_stored_paths(self, client, users, stub_llm):
        """EDG-006: two uploads never collide — one attachment_id per file."""
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        resp = await _invoke_with_image(
            client,
            users["a"]["headers"],
            thread_id,
            [image_attachment(filename="same.png"), image_attachment(filename="same.png")],
        )
        assert resp.status_code == 200, resp.text
        listing = await client.get(
            "/api/v1/store/files", params={"prefix": f"/chat_attachments/{thread_id}/"}, headers=users["a"]["headers"]
        )
        paths = [p for p in listing.json() if p.endswith("same.png")]
        assert len(paths) == 2
        assert len(set(paths)) == 2

    async def test_relative_filename_reduced_to_basename_in_trace(self, client, users, stub_llm):
        """EDG-004: 'some/dir/photo.png' is reduced to 'photo.png'."""
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        attachment = image_attachment(filename="some/dir/photo.png")
        resp = await _invoke_with_image(client, users["a"]["headers"], thread_id, [attachment])
        assert resp.status_code == 200, resp.text

        trace = await client.get(f"/api/v1/threads/{thread_id}/trace", headers=users["a"]["headers"])
        assert trace.status_code == 200
        human_events = [e for e in trace.json()["events"] if e["type"] == "human_message"]
        assert human_events
        attachments = (human_events[-1].get("metadata") or {}).get("attachments") or []
        assert attachments, "HUMAN_MESSAGE metadata must include attachments (REQ-020)"
        assert all("/" not in a["filename"] and "\\" not in a["filename"] for a in attachments)
        assert any(a["filename"] == "photo.png" for a in attachments)
