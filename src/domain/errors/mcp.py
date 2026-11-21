"""MCP (Model Context Protocol) domain errors."""

from src.domain.errors.base import DomainError
from src.domain.errors.codes import ErrorCode


class McpError(DomainError):
    """Base error for MCP operations."""

    status_code = ErrorCode.BAD_GATEWAY


class McpConnectionError(McpError):
    """Error connecting to an MCP server."""

    status_code = ErrorCode.BAD_GATEWAY


class McpToolLoadError(McpError):
    """Error loading MCP tools."""

    status_code = ErrorCode.BAD_GATEWAY


class McpRegistryError(McpError):
    """Base error for MCP-registry (mcp-raganything) boundary failures."""

    status_code = ErrorCode.BAD_GATEWAY


class McpRegistryUnavailableError(McpRegistryError):
    """The trusted MCP registry service is unavailable (safe retryable).

    Spec: an unavailable registry must produce a safe retryable error and
    never a false success (Error case: registry outage).
    """

    status_code = ErrorCode.SERVICE_UNAVAILABLE


class McpRegistryRequestError(McpRegistryError):
    """The MCP registry answered with a non-2xx response.

    The detail never echoes the registry response body (it may contain
    secrets); only the status code is surfaced safely.
    """

    status_code = ErrorCode.BAD_GATEWAY


class McpRegistryNotFoundError(McpRegistryError):
    """Registered MCP server not found (non-enumerating not-found)."""

    status_code = ErrorCode.NOT_FOUND


class McpRegistryConflictError(McpRegistryError):
    """The registry refused a creation that would overwrite an existing server.

    The import contract turns registration conflicts into explicit, safe
    errors (REQ-TX-002): the existing resource is never overwritten, renamed
    or silently reused. The detail names the candidate server only — the
    registry response body (which may carry secrets) is never reflected.
    """

    status_code = ErrorCode.CONFLICT
