"""Builtin ``create_pdf`` LangChain tool (chat-pdf-attachments spec §5.5).

REQ-049: the tool stays YAML-referencable as
``src.infrastructure.deepagent.pdf_tools:create_pdf`` — the module-level
``create_pdf`` symbol is a ready-to-load :class:`StructuredTool` whose store is
resolved lazily from the composition root at call time and whose thread scope
comes from the injected LangGraph ``RunnableConfig`` (the deepagents ToolNode
always passes ``configurable.thread_id``).

The explicit factory :func:`build_create_pdf_tool` binds BOTH collaborator
values up-front (used by tests and any future wiring that wants a statically
scoped tool).

Storage (REQ-056/REQ-057): the generated PDF is written to the USER's
LangGraph Store under a thread-scoped, UUID-scoped path
``/documents/<thread_id>/<uuid>/<filename>.pdf`` — no filename-conflict logic
by construction. The tool result is a structured JSON dict
``{"path", "filename", "mime_type"}`` and NEVER carries binary content
(REQ-058/059).
"""

import asyncio
import base64
import json
import logging
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from mistletoe import Document
from pydantic import BaseModel, Field

from src.config import CREATE_PDF_MAX_MARKDOWN_CHARS_DEFAULT, resolve_env_int
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage
from src.infrastructure.deepagent.pdf_renderer import FpdfHtmlRenderer

logger = logging.getLogger(__name__)

#: YAML tool reference (REQ-049) — the exact attribute path agents declare.
CREATE_PDF_TOOL_REF = "src.infrastructure.deepagent.pdf_tools:create_pdf"

#: Mime type of every produced document (REQ-058).
PDF_MIME_TYPE = "application/pdf"

#: Store prefix of generated PDFs (REQ-056) — thread-scoped for the cascade.
DOCUMENTS_STORE_ROOT = "/documents"

#: Filename constraints (REQ-050).
FILENAME_MAX_LENGTH = 200


class CreatePdfArgs(BaseModel):
    """Tool argument schema (REQ-050)."""

    filename: str = Field(
        ..., description="Target document name, 1-200 chars. The .pdf extension is added automatically if missing."
    )
    markdown_content: str = Field(..., description="Markdown content to render into the PDF (1 to 1,000,000 chars).")


def _default_store_provider():
    """Return the wired user-namespaced store repository (or ``None``).

    The composition root (``src.dependencies.init_persistence``) registers the
    provider at boot; a YAML-referenced tool resolves it lazily on first call.
    """
    return None


_default_provider = _default_store_provider


def set_default_store_provider(provider: Any) -> None:
    """Register the store provider used by the module-level ``create_pdf``.

    Args:
        provider: A zero-arg callable returning a :class:`StoreFileRepository`
            or ``None`` when persistence is not initialized.
    """
    global _default_provider
    _default_provider = provider


def _resolve_default_store():
    """Resolve the default store through the registered provider."""
    store = _default_provider()
    if store is None:
        raise RuntimeError(
            "create_pdf is not wired to a store repository; the composition root must call set_default_store_provider at startup."
        )
    return store


def _has_unsafe_name_chars(filename: str) -> bool:
    """Return True when the name carries a separator, a traversal token or a control character."""
    return ".." in filename or "\\" in filename or "/" in filename or any(ord(char) < 32 for char in filename)


def _validate_filename(raw_filename: str) -> str:
    """Validate and normalize the requested document name (REQ-050/051)."""
    filename = (raw_filename or "").strip()
    if not filename:
        raise ValueError(ErrorMessage.PDF_TOOL_ERROR.format(reason="filename must be a non-empty string (1-200 chars)"))
    if len(filename) > FILENAME_MAX_LENGTH:
        raise ValueError(
            ErrorMessage.PDF_TOOL_ERROR.format(
                reason=f"filename must be at most {FILENAME_MAX_LENGTH} chars, got {len(filename)}"
            )
        )
    if _has_unsafe_name_chars(filename) or filename in (".", ".."):
        # Path separators, traversal tokens and control characters are refused
        # — the latter prevents LLM-supplied filenames from forging log lines
        # (the name is interpolated into CREATE_PDF_GENERATING).
        raise ValueError(
            ErrorMessage.PDF_TOOL_ERROR.format(reason="filename must be a plain basename without path separators")
        )
    if filename.lower().endswith(".pdf"):
        base = filename[: -len(".pdf")].strip()
        if not base:
            raise ValueError(
                ErrorMessage.PDF_TOOL_ERROR.format(reason="filename must be a plain basename without path separators")
            )
        return f"{base}.pdf"
    # EDG-010: accented names are preserved verbatim (store paths are UTF-8).
    return f"{filename}.pdf"


def _validate_markdown(markdown_content: str, max_chars: int) -> str:
    """Validate the markdown payload length (REQ-050 / REQ-092).

    Args:
        markdown_content: The LLM-provided markdown argument.
        max_chars: The env-configured limit.

    Returns:
        The payload unchanged when valid.

    Raises:
        ValueError: When the payload is empty or over the limit — the error
            surfaces to the LLM, no PDF is written.
    """
    if not markdown_content or not markdown_content.strip():
        raise ValueError(
            ErrorMessage.PDF_TOOL_ERROR.format(reason="markdown_content must be a non-empty string (1-1000000 chars)")
        )
    if len(markdown_content) > max_chars:
        raise ValueError(
            ErrorMessage.PDF_TOOL_ERROR.format(
                reason=f"markdown_content exceeds the maximum of {max_chars} chars (got {len(markdown_content)})"
            )
        )
    return markdown_content


def _render_pdf_bytes(markdown_content: str) -> bytes:
    """Render markdown into an A4 PDF (REQ-052/053/054).

    Args:
        markdown_content: The validated markdown payload.

    Returns:
        The PDF bytes (a real ``%PDF-`` document, A4 MediaBox 595x842).
    """
    from fpdf import FPDF

    document = Document(markdown_content)
    html = FpdfHtmlRenderer().render(document)
    pdf = FPDF(format="A4")
    pdf.add_page()
    try:
        pdf.write_html(html)
    except Exception:
        # Degradation guard (NFR-019 robustness): an unsupported construct must
        # never leave the user without a document — fall back to a plain-text
        # paragraph rendering of the SAME content.
        logger.warning(LogMessage.CREATE_PDF_ERROR, "write_html failed; falling back to plain text rendering")
        pdf = FPDF(format="A4")
        pdf.add_page()
        for line in markdown_content.splitlines() or [""]:
            pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(w=0, h=5, text=_latin_safe(line))
    return bytes(pdf.output())


def _latin_safe(text: str) -> str:
    """Latin-1 coercion fallback (see :mod:`pdf_renderer`)."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_create_pdf_tool(
    file_store,  # noqa: ANN001 — StoreFileRepository port contract
    thread_id: str | None = None,
    max_markdown_chars: int | None = None,
) -> BaseTool:
    """Build the thread-scoped ``create_pdf`` tool (test/infra contract).

    Args:
        file_store: The user-namespaced :class:`StoreFileRepository`.
        thread_id: The thread the PDFs belong to (cascade scoping, REQ-056).
            When ``None`` the thread id is read from the injected
            ``RunnableConfig`` at call time.
        max_markdown_chars: Optional static limit override; defaults to the
            ``CREATE_PDF_MAX_MARKDOWN_CHARS`` env value resolved per call
            (REQ-096).

    Returns:
        A :class:`StructuredTool` named ``create_pdf``.
    """

    async def create_pdf_impl(filename: str, markdown_content: str, config: RunnableConfig) -> str:
        """Render markdown → A4 PDF → user store; JSON result (REQ-058)."""
        configurable = dict(config or {}).get("configurable")
        if thread_id:
            effective_thread: str | None = thread_id
        elif isinstance(configurable, dict) and configurable.get("thread_id"):
            effective_thread = str(configurable.get("thread_id"))
        else:
            effective_thread = None
        if not effective_thread:
            raise ValueError(ErrorMessage.PDF_TOOL_ERROR.format(reason="no thread scope available for create_pdf"))
        store = file_store or _resolve_default_store()
        limit = max_markdown_chars or resolve_env_int(
            "CREATE_PDF_MAX_MARKDOWN_CHARS", CREATE_PDF_MAX_MARKDOWN_CHARS_DEFAULT
        )
        safe_name = _validate_filename(filename)
        _validate_markdown(markdown_content, limit)
        logger.info(LogMessage.CREATE_PDF_GENERATING, safe_name, effective_thread, len(markdown_content))

        # CPU-bound markdown→PDF layout: offload to a worker thread so large
        # documents never block the event loop while a turn streams.
        pdf_bytes = await asyncio.to_thread(_render_pdf_bytes, markdown_content)
        # REQ-056: a fresh UUID per generation excludes any collision by design
        # (EDG-006 analog); the visible filename stays the one chosen in args.
        unique = str(uuid.uuid4())
        store_path = f"{DOCUMENTS_STORE_ROOT}/{effective_thread}/{unique}/{safe_name}"
        # REQ-059: the store holds the document content; the result only
        # carries the path/filename/mime triple.
        await store.put_file(store_path, base64.b64encode(pdf_bytes).decode())
        payload = {"path": store_path, "filename": safe_name, "mime_type": PDF_MIME_TYPE}
        # The LLM-facing result is a compact JSON string (REQ-058 contract).
        return json.dumps(payload)

    return StructuredTool(
        name="create_pdf",
        description=(
            "Generate a PDF document (A4) from Markdown content and store it in the user's document space. "
            "Returns a JSON object with the stored path, the filename (with .pdf) and the mime type."
        ),
        args_schema=CreatePdfArgs,
        coroutine=create_pdf_impl,
    )


#: REQ-049: module-level YAML-referencable tool. The store provider is wired
#: by the composition root at boot; the thread scope comes from the injected
#: RunnableConfig.
create_pdf = build_create_pdf_tool(file_store=None, thread_id=None)
