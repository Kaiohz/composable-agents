"""Builtin ``create_pdf`` tool tests (REQ-049..060) — Docker-independent.

The tool is a LangChain tool (REQ-049:
``src.infrastructure.deepagent.pdf_tools:create_pdf``) that renders Markdown
with mistletoe + fpdf2 into an A4 PDF and stores it in the USER's LangGraph
Store under a thread-scoped, UUID-scoped path (REQ-056:
``/documents/<thread_id>/<uuid>/<filename>.pdf``). It never returns binary
content (REQ-059) — the result is a structured JSON object (REQ-058)
``{"path", "filename", "mime_type"}`` whose ``path`` already carries the
thread/UUID scoping.

Contract under test (defined here for the implementation to satisfy):
  ``src.infrastructure.deepagent.pdf_tools.build_create_pdf_tool(
      file_store: StoreFileRepository, thread_id: str) -> BaseTool``
— the factory binds the store and the thread scoping.

Layer-level tests (tool logic has no HTTP entry point of its own) executed
WITHOUT the Docker-backed behavioral stack: the tool writes through a REAL
``LangGraphStoreFileRepository`` over an ``InMemoryStore`` with the same
``user_namespaced("filesystem")`` namespace convention the composition root
uses for its non-Docker fallback — no container required, assertions unchanged.
"""

import base64
import json
import re
import uuid

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _tool_factory():
    from src.infrastructure.deepagent.pdf_tools import build_create_pdf_tool

    return build_create_pdf_tool


def _store():
    """Docker-independent user-namespaced store for the tool invocations.

    Mirrors the composition root's own non-Docker fallback convention
    (``dependencies._require_store_file_repository``): a REAL
    :class:`LangGraphStoreFileRepository` over an ``InMemoryStore`` with the
    ``user_namespaced("filesystem")`` namespace provider, so the tool writes
    isolated under the ``current_user_id`` each test sets. A fresh store per
    request keeps every test independent without any container.
    """
    from langgraph.store.memory import InMemoryStore

    from src.infrastructure.deepagent.namespace import user_namespaced
    from src.infrastructure.store_file.adapter import LangGraphStoreFileRepository

    return LangGraphStoreFileRepository(
        store=InMemoryStore(),
        namespace_provider=lambda: user_namespaced("filesystem"),
    )


SAMPLE_MARKDOWN = (
    "# Rapport trimestriel\n\n"
    "Voici un **texte important** avec de *l'italique* et du `code`.\n\n"
    "## Points clés\n\n"
    "- Première puce\n- Deuxième puce\n\n"
    "1. Premier point\n2. Second point\n\n"
    "> Une citation\n\n"
    "| Colonne A | Colonne B |\n|---|---|\n| 1 | 2 |\n\n"
    "[Lien](https://example.com)\n\n---\n"
)


def _result_as_dict(result) -> dict:
    return result if isinstance(result, dict) else json.loads(str(result))


async def _invoke_and_error_seen(tool, args: dict) -> bool:
    """A tool rejection either raises or returns an error payload."""
    try:
        result = await tool.ainvoke(args)
    except Exception:  # noqa: BLE001 — tool-level rejections are allowed to raise
        return True
    return "error" in str(result).lower()


class TestCreatePdfHappyPath:
    async def test_valid_markdown_produces_pdf_stored_in_user_store(self, monkeypatch):
        """AC-006 / REQ-049/050/053/056/058: valid Markdown → PDF in store."""
        from src.infrastructure.database.rls_context import current_user_id

        file_store = _store()
        thread_id = f"pdf-thread-{uuid.uuid4().hex[:8]}"
        token = current_user_id.set("user-a")
        try:
            tool = _tool_factory()(file_store=file_store, thread_id=thread_id)
            result = await tool.ainvoke({"filename": "rapport", "markdown_content": SAMPLE_MARKDOWN})
            payload = _result_as_dict(result)

            assert payload["mime_type"] == "application/pdf"
            assert payload["filename"] == "rapport.pdf"  # REQ-051 auto-extension
            assert re.fullmatch(
                rf"/documents/{re.escape(thread_id)}/[0-9a-f-]{{36}}/rapport\.pdf", payload["path"]
            ), payload["path"]

            listed = await file_store.list_files(prefix=f"/documents/{thread_id}/")
            assert listed, "the PDF must be stored under /documents/<thread_id>/"
            content = await file_store.get_file(listed[0])
            pdf_bytes = base64.b64decode(content)
            assert pdf_bytes.startswith(b"%PDF"), pdf_bytes[:8]
            assert len(pdf_bytes) > 100, "a real PDF document body is expected"
        finally:
            current_user_id.reset(token)

    async def test_a4_media_box(self, monkeypatch):
        """REQ-052: the generated PDF is A4 (595 x 842 pt)."""
        from src.infrastructure.database.rls_context import current_user_id

        file_store = _store()
        thread_id = f"pdf-thread-{uuid.uuid4().hex[:8]}"
        token = current_user_id.set("user-a")
        try:
            tool = _tool_factory()(file_store=file_store, thread_id=thread_id)
            await tool.ainvoke({"filename": "a4.pdf", "markdown_content": "# A4\n\nContenu."})
            listed = await file_store.list_files(prefix=f"/documents/{thread_id}/")
            content = await file_store.get_file(listed[0])
            pdf_bytes = base64.b64decode(content)
            assert b"595" in pdf_bytes and b"841" in pdf_bytes, "A4 MediaBox expected (595x842 pt)"
        finally:
            current_user_id.reset(token)


class TestCreatePdfConstraints:
    async def test_duplicate_filenames_both_succeed_under_distinct_uuid_folders(self):
        """REQ-057: no conflict logic — distinct UUID folders per upload."""
        from src.infrastructure.database.rls_context import current_user_id

        file_store = _store()
        thread_id = f"pdf-thread-{uuid.uuid4().hex[:8]}"
        token = current_user_id.set("user-a")
        try:
            tool = _tool_factory()(file_store=file_store, thread_id=thread_id)
            first = await tool.ainvoke({"filename": "same-name", "markdown_content": "# V1\n"})
            second = await tool.ainvoke({"filename": "same-name", "markdown_content": "# V2\n"})

            paths = {_result_as_dict(first)["path"], _result_as_dict(second)["path"]}
            assert len(paths) == 2, f"duplicate names must not collide: {paths}"
            stored = set(await file_store.list_files(prefix=f"/documents/{thread_id}/"))
            for path in paths:
                assert path in stored, f"{path} must be stored: {stored}"
        finally:
            current_user_id.reset(token)

    async def test_markdown_over_max_chars_rejected(self, monkeypatch):
        """REQ-092/REQ-050: markdown over the env limit is rejected."""
        from src.infrastructure.database.rls_context import current_user_id

        monkeypatch.setenv("CREATE_PDF_MAX_MARKDOWN_CHARS", "100")
        file_store = _store()
        thread_id = f"pdf-thread-{uuid.uuid4().hex[:8]}"
        token = current_user_id.set("user-a")
        try:
            tool = _tool_factory()(file_store=file_store, thread_id=thread_id)
            rejected = await _invoke_and_error_seen(
                tool, {"filename": "big.pdf", "markdown_content": "x" * 101}
            )
            assert rejected, "oversized markdown must be rejected (error to the LLM, no PDF written)"
        finally:
            current_user_id.reset(token)

    @pytest.mark.parametrize(
        "bad_name",
        [
            "a" * 201,  # too long (REQ-050)
            "",  # empty (REQ-050)
            "../escape",  # traversal
            "some/dir/escape.pdf",  # separator
            "back\\slash.pdf",  # backslash exploit
            ".",  # reduces to the current dir
            "..",  # reduces to a traversal token
            ".pdf",  # empty base once the extension is stripped
            "report\nforged.pdf",  # control characters (log-forging vector)
            "   ",  # whitespace-only
        ],
    )
    async def test_filename_bounds_and_basename_safety_enforced(self, bad_name):
        """REQ-050: filename 1–200 chars, basename-safe, no control characters."""
        from src.infrastructure.database.rls_context import current_user_id

        file_store = _store()
        thread_id = f"pdf-thread-{uuid.uuid4().hex[:8]}"
        token = current_user_id.set("user-a")
        try:
            tool = _tool_factory()(file_store=file_store, thread_id=thread_id)
            rejected = await _invoke_and_error_seen(
                tool, {"filename": bad_name, "markdown_content": "# x\n"}
            )
            assert rejected, f"invalid filename {bad_name!r} must be rejected"
        finally:
            current_user_id.reset(token)

    async def test_result_contains_no_binary_content(self):
        """REQ-058/059: structured JSON result only — never the PDF bytes."""
        from src.infrastructure.database.rls_context import current_user_id

        file_store = _store()
        thread_id = f"pdf-thread-{uuid.uuid4().hex[:8]}"
        token = current_user_id.set("user-a")
        try:
            tool = _tool_factory()(file_store=file_store, thread_id=thread_id)
            result = await tool.ainvoke({"filename": "clean", "markdown_content": "# Clean\n"})
            payload = _result_as_dict(result)
            assert set(payload.keys()) <= {"path", "filename", "mime_type"}, payload
            for value in payload.values():
                assert not value.startswith("JVBERi"), "the result must not embed base64 PDF bytes"
        finally:
            current_user_id.reset(token)
