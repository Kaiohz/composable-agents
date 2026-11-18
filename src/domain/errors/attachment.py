"""Attachment domain errors (chat PDF attachments spec §5.1/§5.4).

Every attachment/policy/temporary-tool failure maps to an explicit 4xx/5xx
status so malformed payloads never reach the agent graph (REQ-010, EC-001..007).
"""

from src.domain.errors.base import DomainError
from src.domain.errors.codes import ErrorCode
from src.domain.errors.mcp import McpError


class AttachmentError(DomainError):
    """Base error for chat attachment handling."""

    status_code = ErrorCode.UNPROCESSABLE_ENTITY


class AttachmentValidationError(AttachmentError):
    """A payload violates the strict attachment contract (REQ-006..008)."""

    status_code = ErrorCode.UNPROCESSABLE_ENTITY


class DocumentsNotAllowedError(AttachmentError):
    """A document was sent to an agent WITHOUT the Files MCP server (REQ-010)."""

    status_code = ErrorCode.UNPROCESSABLE_ENTITY


class FilesMcpTemporaryError(McpError):
    """The Files MCP temporary tool call failed (upload/delete)."""

    status_code = ErrorCode.BAD_GATEWAY


class AttachmentLimitError(AttachmentValidationError):
    """A configured attachment limit was exceeded (count/size/total)."""


class PdfToolError(AttachmentError):
    """The builtin create_pdf tool rejected its arguments."""

    status_code = ErrorCode.BAD_REQUEST
