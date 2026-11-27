"""Behavioral tests — ChatRequest ``attachments`` validation (REQ-001..008).

All tests hit the REAL routes ``POST /api/v1/chat/{thread_id}`` (and, where
relevant, ``POST /api/v1/chat/{thread_id}/stream``) over the production app:
validation failures must return an explicit 4xx BEFORE any agent invocation.
"""

import pytest

from tests.behavioral.chat_attachments.conftest import (
    agent_with_thread,
    byte_attachment,
    files_mcp_yaml,
    image_attachment,
)
from tests.behavioral.conftest import invoke_agent

pytestmark = pytest.mark.asyncio(loop_scope="session")

VALIDATION_CODES = {400, 422}


async def _chat_with_attachments(client, headers, thread_id, attachments, message="Analyse ça"):
    return await client.post(
        f"/api/v1/chat/{thread_id}",
        headers=headers,
        json={"message": message, "attachments": attachments},
    )


class TestAttachmentsValidationRejects:
    """EC-001..005 + REQ-006: every invalid attachment payload is rejected."""

    @pytest.fixture(autouse=True)
    async def _thread(self, client, users):
        # A real thread is required only so a VALID payload could proceed;
        # invalid payloads must fail at request validation (thread-less 422).
        self.thread_id = await agent_with_thread(client, users["a"]["headers"])
        self.headers = users["a"]["headers"]

    async def test_invalid_base64_rejected_422(self, client, stub_llm):
        """EC-001: strictly undecodable base64 → explicit 422."""
        resp = await _chat_with_attachments(
            client,
            self.headers,
            self.thread_id,
            [image_attachment(content_base64="not-valid-base64!!!")],
        )
        assert resp.status_code == 422, resp.text
        assert not stub_llm.app.requests  # no LLM call on rejected payload

    async def test_more_than_ten_attachments_rejected(self, client):
        """EC-005: more than 10 attachments per request → rejected."""
        resp = await _chat_with_attachments(
            client, self.headers, self.thread_id, [image_attachment() for _ in range(11)]
        )
        assert resp.status_code in VALIDATION_CODES, resp.text

    async def test_total_decoded_size_over_50_mib_rejected(self, client):
        """REQ-006: total decoded size > 50 MiB → rejected (30 + 25 MiB)."""
        resp = await _chat_with_attachments(
            client,
            self.headers,
            self.thread_id,
            [
                byte_attachment("a.bin", "application/octet-stream", 30 * 1024 * 1024),
                byte_attachment("b.bin", "application/octet-stream", 25 * 1024 * 1024),
            ],
        )
        assert resp.status_code in VALIDATION_CODES, resp.text

    async def test_image_over_10_mib_rejected(self, client):
        """EC-003: a single image over 10 MiB → rejected."""
        resp = await _chat_with_attachments(
            client,
            self.headers,
            self.thread_id,
            [byte_attachment("big.png", "image/png", 10 * 1024 * 1024 + 1)],
        )
        assert resp.status_code in VALIDATION_CODES, resp.text

    async def test_document_over_50_mib_rejected(self, client):
        """EC-004: a single document over 50 MiB → rejected."""
        resp = await _chat_with_attachments(
            client,
            self.headers,
            self.thread_id,
            [byte_attachment("big.pdf", "application/pdf", 51 * 1024 * 1024)],
        )
        assert resp.status_code in VALIDATION_CODES, resp.text

    async def test_unsupported_image_mime_rejected(self, client, stub_llm):
        """EC-002 / REQ-007: image/tiff is not a supported vision MIME."""
        resp = await _chat_with_attachments(
            client,
            self.headers,
            self.thread_id,
            [image_attachment(filename="image.tiff", content_type="image/tiff")],
        )
        assert resp.status_code in VALIDATION_CODES, resp.text

    async def test_mime_not_coherent_with_extension_rejected(self, client):
        """REQ-006: MIME must be coherent with the filename extension."""
        resp = await _chat_with_attachments(
            client,
            self.headers,
            self.thread_id,
            [image_attachment(filename="report.pdf", content_type="image/png")],
        )
        assert resp.status_code in VALIDATION_CODES, resp.text

    async def test_empty_filename_rejected(self, client):
        """EDG-005: empty filename → rejected (REQ-006)."""
        resp = await _chat_with_attachments(
            client,
            self.headers,
            self.thread_id,
            [image_attachment(filename="")],
        )
        assert resp.status_code in VALIDATION_CODES, resp.text

    async def test_filename_with_traversal_rejected(self, client):
        """REQ-006 / NFR-009: no path traversal (`..`) in filename.

        A traversal-containing filename must be REJECTED, not silently
        reduced — the basename-reduction rule (EDG-004) applies to plain
        relative paths, never to traversal attempts.
        """
        for evil in ("../../etc/passwd.png", "/abs/path.png", "..\\evil.png"):
            resp = await _chat_with_attachments(
                client,
                self.headers,
                self.thread_id,
                [image_attachment(filename=evil)],
            )
            assert resp.status_code in VALIDATION_CODES, (evil, resp.text)

    async def test_missing_mimetype_rejected(self, client):
        """REQ-006: MIME must be present on every attachment."""
        attachment = image_attachment()
        del attachment["content_type"]
        resp = await _chat_with_attachments(client, self.headers, self.thread_id, [attachment])
        assert resp.status_code in VALIDATION_CODES, resp.text


class TestMessageAttachmentsRules:
    """REQ-004 / EDG-003 / REQ-005."""

    async def test_empty_message_without_attachments_still_rejected(self, client, users):
        """EDG-003: empty message string must remain rejected (unchanged)."""
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        resp = await client.post(f"/api/v1/chat/{thread_id}", headers=users["a"]["headers"], json={"message": ""})
        assert resp.status_code in VALIDATION_CODES, resp.text

    async def test_missing_message_without_attachments_rejected(self, client, users):
        """REQ-004: no message and no attachments → validation error."""
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        resp = await client.post(f"/api/v1/chat/{thread_id}", headers=users["a"]["headers"], json={})
        assert resp.status_code in VALIDATION_CODES, resp.text

    async def test_message_and_hitl_fields_with_attachments_rejected(self, client, users):
        """REQ-005: message + HITL fields remain mutually exclusive."""
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        resp = await client.post(
            f"/api/v1/chat/{thread_id}",
            headers=users["a"]["headers"],
            json={
                "message": "hi",
                "attachments": [image_attachment()],
                "decisions": [{"tool_call_id": "tc_1", "action": "approve"}],
            },
        )
        assert resp.status_code in VALIDATION_CODES, resp.text

    async def test_attachments_without_message_rejected_when_attachments_empty(self, client, users):
        """REQ-004: attachments field alone with an empty list fails (no
        message AND no attachment content → nothing to send). The FULL
        message-less-but-attached path is covered in the vision suite (EDG-001)."""
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        resp = await client.post(f"/api/v1/chat/{thread_id}", headers=users["a"]["headers"], json={"attachments": []})
        assert resp.status_code in VALIDATION_CODES, resp.text


class TestBackwardCompatibility:
    """REQ-003 / AC-011: payloads without attachments behave exactly as before."""

    async def test_plain_message_accepted_unchanged(self, client, users, stub_llm):
        thread_id = await agent_with_thread(client, users["a"]["headers"], files_mcp_yaml("compat-agent"))
        resp = await invoke_agent(client, users["a"]["headers"], thread_id, message="hello")
        assert resp.status_code == 200, resp.text
        assert resp.json()["role"] == "ai"

    async def test_stream_plain_message_accepted_unchanged(self, client, users, stub_llm):
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        resp = await client.post(
            f"/api/v1/chat/{thread_id}/stream",
            headers=users["a"]["headers"],
            json={"message": "hello"},
        )
        assert resp.status_code == 200, resp.text
        assert "[DONE]" in resp.text
