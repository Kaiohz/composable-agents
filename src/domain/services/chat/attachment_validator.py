"""Pure domain service: strict chat attachment validation (REQ-006..008).

The validator decodes and classifies attachment payloads BEFORE any I/O step
(store write / Files MCP upload / agent invocation). It is pure Python — no
framework or infrastructure import — so the business rules stay in the domain
layer.

Validation order (fail-fast, every failure maps to an explicit 422):

1. attachment count limit (EC-005);
2. per attachment: filename safety (EDG-004/EDG-005, NFR-009), strict base64
   decode (EC-001), MIME presence, kind classification (REQ-007), MIME↔
   extension coherence, document family rules (REQ-008), per-kind size limit
   (EC-003/EC-004);
3. total decoded size limit (REQ-006).

Images are never routed through the Files MCP family rules (REQ-008 only
applies to documents; the vision path accepts ONLY the four whitelisted image
types).
"""

import base64
import binascii
import posixpath
from uuid import uuid4

from src.domain.entities.attachment import (
    AttachmentKind,
    AttachmentLimits,
    ValidatedAttachment,
)
from src.domain.errors.attachment import AttachmentLimitError, AttachmentValidationError
from src.domain.errors.messages import ErrorMessage

#: The four vision MIME types (REQ-007).
IMAGE_MIME_TYPES: frozenset[str] = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/webp"}
)

#: Image filename extension → expected MIME (coherence rule, REQ-006).
IMAGE_EXTENSION_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

#: Document extensions accepted for chat documents — the SAME family as the
#: REST upload validation of the Files service (REQ-008).
DOCUMENT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".txt",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".md",
        ".csv",
        ".html",
        ".xml",
        ".json",
        ".rtf",
        ".odt",
        ".ods",
    }
)

#: Document MIME families accepted for chat documents (REQ-008).
DOCUMENT_MIME_PREFIXES: tuple[str, ...] = (
    "application/pdf",
    "text/",
    "application/vnd.openxmlformats-officedocument",
    "application/vnd.ms-",
    "application/json",
    "application/rtf",
    "application/vnd.oasis.opendocument",
)

#: Document extension → expected MIME prefix (coherence rule, REQ-006).
DOCUMENT_EXTENSION_MIME_PREFIX: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/",
    ".md": "text/",
    ".csv": "text/",
    ".html": "text/",
    ".xml": "text/",
    ".doc": "application/vnd.ms-",
    ".docx": "application/vnd.openxmlformats-officedocument",
    ".xls": "application/vnd.ms-",
    ".xlsx": "application/vnd.openxmlformats-officedocument",
    ".ppt": "application/vnd.ms-",
    ".pptx": "application/vnd.openxmlformats-officedocument",
    ".json": "application/json",
    ".rtf": "application/rtf",
    ".odt": "application/vnd.oasis.opendocument",
    ".ods": "application/vnd.oasis.opendocument",
}


def _has_traversal_sequence(filename: str) -> bool:
    """Return True when the filename carries a traversal/exploit/control sequence."""
    return (
        ".." in filename
        or "\\" in filename
        or filename.startswith("/")
        or any(ord(char) < 32 for char in filename)
    )


def _refuse_unsafe_filename(filename: str) -> None:
    """Refuse traversal attempts and control characters outright (NFR-009)."""
    if _has_traversal_sequence(filename):
        # Traversal attempts AND control characters are refused: the sibling
        # path policy (``is_safe_segment``) rejects control chars in object
        # path segments, so the same payload must never pass validation.
        raise AttachmentValidationError(ErrorMessage.ATTACHMENT_FILE_NAME_TRAVERSAL.format(filename=repr(filename)))


def _safe_basename(filename: str) -> str:
    """Reduce a filename to its basename, refusing unsafe shapes outright.

    Plain relative sub-paths (``some/dir/photo.png``) are REDUCED to the
    basename (EDG-004). Any traversal attempt — ``..`` segment, an initial
    slash (absolute path) or an exploited backslash — is REJECTED, never
    silently reduced.

    Returns:
        The basename (single path segment).

    Raises:
        AttachmentValidationError: On traversal attempts or empty names.
    """
    if not filename.strip():
        raise AttachmentValidationError(ErrorMessage.ATTACHMENT_EMPTY_FILE_NAME)
    _refuse_unsafe_filename(filename)
    basename = posixpath.basename(filename)
    if not basename or basename in (".", ".."):
        raise AttachmentValidationError(ErrorMessage.ATTACHMENT_FILE_NAME_TRAVERSAL.format(filename=repr(filename)))
    return basename


def _strict_decode(filename: str, content_base64: str) -> bytes:
    """Decode ``content_base64`` strictly (EC-001).

    Args:
        filename: Attachment filename (for the error message).
        content_base64: Payload candidate.

    Returns:
        The decoded bytes.

    Raises:
        AttachmentValidationError: When the payload is not strictly decodable.
    """
    if not isinstance(content_base64, str) or not content_base64:
        raise AttachmentValidationError(ErrorMessage.ATTACHMENT_INVALID_BASE64.format(filename=filename))
    try:
        return base64.b64decode(content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AttachmentValidationError(
            ErrorMessage.ATTACHMENT_INVALID_BASE64.format(filename=filename)
        ) from exc


def _classify_and_validate(
    filename: str,
    content_type: str,
    data: bytes,
    limits: AttachmentLimits,
) -> AttachmentKind:
    """Classify one attachment and apply its kind-specific rules (REQ-006..008).

    Returns:
        The attachment kind.

    Raises:
        AttachmentValidationError: On any unsupported/coherent/oversize case.
    """
    content_type = (content_type or "").strip()
    extension = posixpath.splitext(filename)[1].lower()
    size = len(data)

    if extension in IMAGE_EXTENSION_MIME:
        # Images MUST declare exactly the matching vision MIME (REQ-006/REQ-007).
        expected = IMAGE_EXTENSION_MIME[extension]
        if content_type not in IMAGE_MIME_TYPES or content_type != expected:
            raise AttachmentValidationError(
                ErrorMessage.ATTACHMENT_TYPE_NOT_COHERENT.format(
                    filename=filename, content_type=content_type, extension=extension
                )
            )
        if size > limits.max_image_bytes:
            raise AttachmentLimitError(
                ErrorMessage.ATTACHMENT_IMAGE_TOO_LARGE.format(filename=filename, max_bytes=limits.max_image_bytes)
            )
        return AttachmentKind.IMAGE

    if content_type.startswith("image/"):
        # A non-whitelisted image kind is refused outright (EC-002).
        raise AttachmentValidationError(
            ErrorMessage.ATTACHMENT_IMAGE_TYPE_NOT_SUPPORTED.format(
                filename=filename, content_type=content_type, allowed=", ".join(sorted(IMAGE_MIME_TYPES))
            )
        )

    # Document kind: same extension/MIME families as the REST upload (REQ-008).
    if extension not in DOCUMENT_EXTENSIONS or extension not in DOCUMENT_EXTENSION_MIME_PREFIX:
        raise AttachmentValidationError(
            ErrorMessage.ATTACHMENT_TYPE_NOT_COHERENT.format(
                filename=filename, content_type=content_type, extension=extension or "''"
            )
        )
    if not content_type.startswith(DOCUMENT_EXTENSION_MIME_PREFIX[extension]) or not any(
        content_type.startswith(prefix) for prefix in DOCUMENT_MIME_PREFIXES
    ):
        raise AttachmentValidationError(
            ErrorMessage.ATTACHMENT_TYPE_NOT_COHERENT.format(
                filename=filename, content_type=content_type, extension=extension
            )
        )
    if size > limits.max_document_bytes:
        raise AttachmentLimitError(
            ErrorMessage.ATTACHMENT_DOCUMENT_TOO_LARGE.format(filename=filename, max_bytes=limits.max_document_bytes)
        )
    return AttachmentKind.DOCUMENT


def decode_and_validate_attachments(
    attachments: list[dict] | None,
    limits: AttachmentLimits,
) -> list[ValidatedAttachment]:
    """Validate and decode a request's attachment payloads (REQ-006..008).

    Args:
        attachments: The raw request attachment dicts (``None`` means no
            attachments — an empty payload after this point is the caller's
            decision to make).
        limits: The env-configured attachment limits (REQ-092).

    Returns:
        Decoded, classified and validated attachments (each with a fresh
        unique id — EDG-006).

    Raises:
        AttachmentValidationError: When the count/size/type/coherence rules
            are violated (mapped to HTTP 422 by the domain error handler).
    """
    payload = attachments or []
    if len(payload) > limits.max_count:
        raise AttachmentLimitError(
            ErrorMessage.ATTACHMENT_COUNT_EXCEEDED.format(count=len(payload), max_count=limits.max_count)
        )

    validated: list[ValidatedAttachment] = []
    total = 0
    for raw in payload:
        raw = dict(raw) if isinstance(raw, dict) else {}
        raw_filename = raw.get("filename")
        # Non-str filenames are refused BEFORE coercion: a silent ``str()``
        # would turn ``None`` into the accepted basename "None".
        if not isinstance(raw_filename, str):
            raise AttachmentValidationError(ErrorMessage.ATTACHMENT_EMPTY_FILE_NAME)
        filename = _safe_basename(raw_filename)
        content_type = raw.get("content_type")
        if not isinstance(content_type, str) or not content_type.strip():
            raise AttachmentValidationError(ErrorMessage.ATTACHMENT_MIME_REQUIRED.format(filename=filename))
        # Non-str/empty payloads are refused BEFORE any attempt to decode: a
        # silent ``str()`` coercion could turn ``None`` into a decodable "None".
        content_base64 = raw.get("content_base64")
        if not isinstance(content_base64, str) or not content_base64:
            raise AttachmentValidationError(ErrorMessage.ATTACHMENT_INVALID_BASE64.format(filename=filename))
        data = _strict_decode(filename, content_base64)
        kind = _classify_and_validate(filename, content_type, data, limits)
        total += len(data)
        validated.append(
            ValidatedAttachment(
                id=str(uuid4()),
                filename=filename,
                content_type=content_type.strip(),
                kind=kind,
                size_bytes=len(data),
                content_base64=raw["content_base64"],
            )
        )

    if total > limits.max_total_bytes:
        raise AttachmentLimitError(
            ErrorMessage.ATTACHMENT_TOTAL_TOO_LARGE.format(total_bytes=total, max_bytes=limits.max_total_bytes)
        )
    return validated
