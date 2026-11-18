"""Attachment domain entities (chat PDF attachments spec §5.1/§5.2).

Attachment metadata carries ONLY the spec keys — id, filename, content_type,
size_bytes, kind and source. The base64 content is never part of the metadata
(REQ-021): it stays in the transient request payload and the prepared turn
only holds it inside content blocks during the turn.
"""

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AttachmentKind(StrEnum):
    """Classification of an attachment (REQ-020 trace contract)."""

    IMAGE = "image"
    DOCUMENT = "document"


class AttachmentSource(StrEnum):
    """Where the attachment content lives for the duration of the turn."""

    LANGGRAPH_STORE = "langgraph_store"
    RUSTFS_TEMP = "rustfs_temp"


class AttachmentLimits(BaseModel, frozen=True):
    """Env-configured attachment limits (REQ-092), resolved from Settings.

    Attributes:
        max_count: Maximum number of attachments per request.
        max_total_bytes: Maximum total decoded size per request.
        max_image_bytes: Maximum size of a single image.
        max_document_bytes: Maximum size of a single document.
    """

    max_count: int = Field(gt=0)
    max_total_bytes: int = Field(gt=0)
    max_image_bytes: int = Field(gt=0)
    max_document_bytes: int = Field(gt=0)


#: Projection keys exposed by the trace contract (REQ-020) and by the
#: ``Message`` projection (REQ-022). Any other key is internal backend state
#: and MUST NOT leak (REQ-023). A TUPLE keeps the projected API key order
#: stable (independent of the interpreter's hash seed).
PROJECTION_KEYS = ("id", "filename", "content_type", "size_bytes", "kind", "source")


class ValidatedAttachment(BaseModel):
    """A strict-decoded attachment payload ready for preparation.

    ``content_base64`` keeps the ORIGINAL request payload (never re-encoded)
    so images and temporary uploads carry exactly the bytes the client sent.
    The field is excluded from ``repr``/``json`` to make accidental logging or
    trace persistence of the content impossible (NFR-004).
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    content_type: str
    kind: AttachmentKind
    size_bytes: int
    content_base64: str = Field(repr=False, exclude=True)


def project_attachment_entries(entries: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Reduce trace metadata attachment entries to the spec projection keys.

    Args:
        entries: Raw ``metadata.attachments`` entries from a HUMAN_MESSAGE event.

    Returns:
        A list of dicts containing ONLY :data:`PROJECTION_KEYS` (REQ-023);
        empty list when ``entries`` is empty or ``None``.
    """
    if not entries:
        return []
    return [{key: entry[key] for key in PROJECTION_KEYS if key in entry} for entry in entries]
