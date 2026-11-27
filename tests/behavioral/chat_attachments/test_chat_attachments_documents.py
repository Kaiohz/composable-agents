"""Behavioral tests — document attachments policy + temporary RustFS lifecycle.

REQ-009/010/011: documents are only allowed for agents whose MCP config
contains an HTTP server whose URL ends with ``/files/mcp``; ``/classical/mcp``
alone is NOT sufficient; the check happens backend-side BEFORE the agent is
invoked (REQ-011: the upload precedes the graph invocation).

REQ-015/016: the graph input references the temporary RustFS path and hints
``read_file``; the binary content NEVER enters the prompt.

REQ-038..047 / AC-004: the backend deletes the temporary object through the
Files MCP server after the terminal response, including on graph failure —
the MCP calls are observed on the REAL in-process Files-MCP stub server
(``mcp_files`` fixture — the ONLY mocked boundary).
"""

import uuid

import pytest

from tests.behavioral.chat_attachments.conftest import (
    agent_with_thread,
    document_attachment,
    files_mcp_yaml,
    image_attachment,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _invoke_document(client, headers, thread_id, attachment, message="Analyse ce document"):
    return await client.post(
        f"/api/v1/chat/{thread_id}",
        headers=headers,
        json={"message": message, "attachments": [attachment]},
    )


class TestDocumentPolicyFrontier:
    """REQ-009/010, EC-006, AC-005, EDG-008/009."""

    async def test_document_on_agent_without_files_mcp_rejected_before_graph(self, client, users, stub_llm):
        """EC-006 / AC-005: explicit 4xx, the graph is never invoked."""
        yaml_text = files_mcp_yaml("no-files-agent")  # no MCP servers at all
        thread_id = await agent_with_thread(client, users["a"]["headers"], yaml_text)
        resp = await _invoke_document(client, users["a"]["headers"], thread_id, document_attachment())
        assert 400 <= resp.status_code < 500, resp.text
        assert not stub_llm.app.requests, "the graph/LLM must NOT be invoked for a rejected document"

    async def test_document_on_classical_only_agent_rejected(self, client, users, stub_llm):
        """EDG-009: /classical/mcp alone is not sufficient for documents."""
        yaml_text = files_mcp_yaml("classical-only-agent", extra_servers=["http://127.0.0.1:59998/classical-mcp"])
        thread_id = await agent_with_thread(client, users["a"]["headers"], yaml_text)
        resp = await _invoke_document(client, users["a"]["headers"], thread_id, document_attachment())
        assert 400 <= resp.status_code < 500, resp.text
        assert not stub_llm.app.requests

    async def test_document_accepted_with_files_mcp_and_uses_the_files_server(self, client, users, stub_llm, mcp_files):
        """EDG-008: with several MCP servers, the backend picks the /files/mcp"""
        yaml_text = files_mcp_yaml(
            "multi-server-agent",
            mcp_url=mcp_files.url,
            extra_servers=["http://127.0.0.1:59998/classical-mcp"],
        )
        thread_id = await agent_with_thread(client, users["a"]["headers"], yaml_text)
        attachment = document_attachment()
        resp = await _invoke_document(client, users["a"]["headers"], thread_id, attachment)
        assert resp.status_code == 200, resp.text
        # The FILE server received the temporary upload, not the classical one.
        assert len(mcp_files.upload_calls) == 1, mcp_files.upload_calls

    async def test_images_still_allowed_without_files_mcp(self, client, users, stub_llm, mcp_files):
        """REQ-009 applies to DOCUMENTS only — images always allowed."""
        yaml_text = files_mcp_yaml("no-files-agent-vision")
        thread_id = await agent_with_thread(client, users["a"]["headers"], yaml_text)
        resp = await client.post(
            f"/api/v1/chat/{thread_id}",
            headers=users["a"]["headers"],
            json={"message": "analyze", "attachments": [image_attachment()]},
        )
        assert resp.status_code == 200, resp.text


class TestTemporaryDocumentUpload:
    """REQ-0011, REQ-031, REQ-038, REQ-015, REQ-016, NFR-007."""

    async def test_upload_sent_to_mcp_with_user_isolated_prefix(self, client, users, stub_llm, mcp_files):
        """REQ-038/NFR-007: prefix = chat_tmp/<user_id>/<thread_id>/..."""
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        attachment = document_attachment(filename="rapport.pdf")
        resp = await _invoke_document(client, users["a"]["headers"], thread_id, attachment)
        assert resp.status_code == 200, resp.text

        calls = mcp_files.upload_calls
        assert len(calls) == 1
        call = calls[0]
        assert call["file_name"] == "rapport.pdf"
        assert call["content_type"] == "application/pdf"
        assert call["prefix"].startswith(f"chat_tmp/user-a/{thread_id}/"), call["prefix"]
        # decoded bytes match what the client sent (strict base64 handling)
        assert call["decoded_bytes"].startswith(b"%PDF")

    async def test_prompt_references_temp_path_and_read_file_never_the_bytes(self, client, users, stub_llm, mcp_files):
        """REQ-015 / REQ-016 / REQ-017: graph input = reference text block."""
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        unique_marker = f"UNIQUE-DOCUMENT-{uuid.uuid4().hex[:8]}"
        attachment = document_attachment(
            filename="rapport.pdf", content_base64=base64_of(b"%PDF-1.7 " + unique_marker.encode())
        )
        base64_content = attachment["content_base64"]
        resp = await _invoke_document(client, users["a"]["headers"], thread_id, attachment)
        assert resp.status_code == 200, resp.text

        assert mcp_files.upload_calls, "the document must be uploaded before the graph runs (REQ-011)"
        call = mcp_files.upload_calls[0]
        object_name = f"{call['prefix'].rstrip('/')}/rapport.pdf"

        user_messages = [
            m for r in stub_llm.app.requests for m in r.body.get("messages", []) if m.get("role") == "user"
        ]
        # REQ-015: the prompt references the temporary path + read_file usage.
        prompt_texts: list[str] = []
        for m in user_messages:
            content = m.get("content")
            if isinstance(content, str):
                prompt_texts.append(content)
            elif isinstance(content, list):
                prompt_texts.extend(str(b.get("text", "")) for b in content if b.get("type") == "text")
        joined = "\n".join(prompt_texts)
        assert object_name in joined, "the prompt must reference the temp RustFS path (REQ-015)"
        assert "read_file" in joined, "the prompt must instruct the agent to use read_file (REQ-015)"
        # REQ-016: the binary/base64 content never enters the prompt / the LLM.
        assert unique_marker not in joined
        assert base64_content not in joined
        for request in stub_llm.app.requests:
            assert base64_content not in json_dumps(request.body), "base64 document content must never reach the LLM"

    async def test_document_and_image_coexistence(self, client, users, stub_llm, mcp_files):
        """REQ-001 sample payload: one image + one document in the same turn."""
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        resp = await client.post(
            f"/api/v1/chat/{thread_id}",
            headers=users["a"]["headers"],
            json={
                "message": "Analyse cette image et ce document",
                "attachments": [image_attachment(), document_attachment()],
            },
        )
        assert resp.status_code == 200, resp.text
        assert len(mcp_files.upload_calls) == 1, "only the document goes through MCP"

    async def test_document_only_without_text_is_allowed(self, client, users, stub_llm, mcp_files):
        """EDG-002 (review C1): message absent + document only → agent runs.

        The turn is a legal human turn regardless of the message (REQ-004) and
        the temporary document must be uploaded BEFORE the graph runs (REQ-011).
        """
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        resp = await client.post(
            f"/api/v1/chat/{thread_id}",
            headers=users["a"]["headers"],
            json={"attachments": [document_attachment()]},
        )
        assert resp.status_code == 200, resp.text
        assert mcp_files.upload_calls, "the temp document must be uploaded before the graph runs"


def base64_of(raw: bytes) -> str:
    import base64

    return base64.b64encode(raw).decode()


def json_dumps(payload) -> str:
    import json

    return json.dumps(payload, default=str)


class TestTemporaryDocumentCleanup:
    """REQ-041 / REQ-042 / AC-004 / NFR-017."""

    async def test_document_deleted_after_terminal_response(self, client, users, stub_llm, mcp_files):
        """AC-004 / REQ-041: invoke path deletes the temp object at the end."""
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        resp = await _invoke_document(client, users["a"]["headers"], thread_id, document_attachment())
        assert resp.status_code == 200, resp.text

        calls = mcp_files.upload_calls
        assert calls, "upload must have happened"
        expected_object = f"{calls[0]['prefix'].rstrip('/')}/rapport.pdf"
        assert expected_object in mcp_files.delete_calls, (
            f"after the terminal response the object must be deleted: {mcp_files.delete_calls}"
        )

    async def test_stream_path_deletes_document_after_done(self, client, users, stub_llm, mcp_files):
        """REQ-041 (streaming): cleanup after the terminal event + [DONE]."""
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        resp = await client.post(
            f"/api/v1/chat/{thread_id}/stream",
            headers=users["a"]["headers"],
            json={"message": "Analyse ce document", "attachments": [document_attachment(filename="doc-stream.pdf")]},
        )
        assert resp.status_code == 200, resp.text
        assert "[DONE]" in resp.text

        calls = mcp_files.upload_calls
        assert calls, "upload must have happened"
        expected_object = f"{calls[0]['prefix'].rstrip('/')}/doc-stream.pdf"
        assert expected_object in mcp_files.delete_calls

    async def test_cleanup_attempted_even_when_graph_fails(self, client, users, stub_llm, mcp_files):
        """REQ-042 / NFR-017: graph error does not orphan the temp document."""
        thread_id = await agent_with_thread(client, users["a"]["headers"])
        stub_llm.reset(mode="fail")
        resp = await _invoke_document(client, users["a"]["headers"], thread_id, document_attachment())
        # the agent failure surfaces as an error status/stream error...
        body = resp.text
        assert resp.status_code >= 400 or '"type": "error"' in body or "error" in body.lower()

        calls = mcp_files.upload_calls
        if calls:  # upload succeeded before the failure → cleanup must run
            expected_object = f"{calls[0]['prefix'].rstrip('/')}/rapport.pdf"
            assert expected_object in mcp_files.delete_calls, (
                f"cleanup must run in a finally block even on graph failure: {mcp_files.delete_calls}"
            )
