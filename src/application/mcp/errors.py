"""MCP tool error mapping (REQ-009).

Every MCP tool wraps its body with :func:`safe_mcp_call`: domain failures are
translated into safe, structured MCP errors with a stable code prefix and an
actionable message. Raw stack traces, credentials, connection strings,
encrypted values and downstream response bodies containing secrets are NEVER
returned — the mapper only forwards the centralized detail text of the raised
domain error, and collapses everything unexpected into a generic failure.

The SDK's FastMCP converts ANY exception raised by a tool into an
``isError`` tool result (verified against the installed ``mcp`` SDK), so the
mapper re-raises a :class:`McpToolError` carrying the mapped detail; no
third-party ``fastmcp`` dependency exists here.
"""

import functools
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import pydantic

from src.domain.errors.agent import (
    AgentConfigAlreadyExistsError,
    AgentError,
    AgentInvocationRejectedError,
    AgentNotFoundError,
    BuiltinAgentError,
)
from src.domain.errors.base import DomainError
from src.domain.errors.config import ConfigError, ConfigValidationError
from src.domain.errors.cron import CronNotFoundError
from src.domain.errors.mcp import (
    McpRegistryNotFoundError,
    McpRegistryRequestError,
    McpRegistryUnavailableError,
)
from src.domain.errors.messages import ErrorMessage
from src.domain.errors.model import ModelNotFoundError
from src.domain.errors.prompt import PromptNotFoundError, PromptVersionNotFoundError
from src.domain.errors.storage import StorageError
from src.domain.errors.store_file import StoreFileNotFoundError
from src.domain.errors.thread import ThreadNotFoundError
from src.domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)

#: Agent-facing MCP server instructions (visible in MCP initialize results).
MCP_INSTRUCTIONS = (
    "Composable control plane: typed management tools for agents, MCP servers, skills, memories, "
    "prompts, models, crons, threads and traces, plus one-shot agent execution "
    "(composable_agents_run). All operations are scoped to the authenticated user."
)

T = TypeVar("T")


class McpToolError(RuntimeError):
    """A mapped, SAFE tool failure surfaced as an MCP ``isError`` result.

    The plain ``RuntimeError`` base keeps the SDK's generic conversion
    (exception → isError tool result); the message carries the stable code
    prefix and the centralized detail only.
    """


#: Stable, structured error-code prefixes (REQ-009) — safe to expose; they
#: never embed request arguments or secrets.
_CODE_VALIDATION = "MCP_VALIDATION_ERROR"
_CODE_CONFLICT = "MCP_CONFLICT_ERROR"
_CODE_NOT_FOUND = "MCP_NOT_FOUND_ERROR"
_CODE_READ_ONLY = "MCP_READ_ONLY_ERROR"
_CODE_REGISTRY = "MCP_REGISTRY_ERROR"
_CODE_STORAGE = "MCP_STORAGE_ERROR"
_CODE_INVOCATION = "MCP_INVOCATION_ERROR"
_CODE_UNEXPECTED = "MCP_INTERNAL_ERROR"


@dataclass(frozen=True)
class _ErrorFamily:
    """One domain-error family → stable MCP code mapping rule."""

    exception_type: type[BaseException]
    code: str
    #: ``None`` renders ``exc.detail``; a static string replaces it (used when
    #: the exception's own detail is considered too raw for the wire).
    detail_override: str | None = None


#: Ordered domain-error family → stable MCP code. MOST SPECIFIC FIRST — the
#: first ``isinstance`` hit wins, so subclasses must precede their bases
#: (e.g. every not-found subtype before its ``AgentError`` parent).
_ERROR_FAMILIES: tuple[_ErrorFamily, ...] = (
    _ErrorFamily(AgentInvocationRejectedError, _CODE_INVOCATION),
    _ErrorFamily(AgentConfigAlreadyExistsError, _CODE_CONFLICT),
    _ErrorFamily(BuiltinAgentError, _CODE_READ_ONLY),
    _ErrorFamily(AgentNotFoundError, _CODE_NOT_FOUND),
    _ErrorFamily(StoreFileNotFoundError, _CODE_NOT_FOUND),
    _ErrorFamily(ThreadNotFoundError, _CODE_NOT_FOUND),
    _ErrorFamily(PromptNotFoundError, _CODE_NOT_FOUND),
    _ErrorFamily(PromptVersionNotFoundError, _CODE_NOT_FOUND),
    _ErrorFamily(ModelNotFoundError, _CODE_NOT_FOUND),
    _ErrorFamily(CronNotFoundError, _CODE_NOT_FOUND),
    _ErrorFamily(McpRegistryUnavailableError, _CODE_REGISTRY, ErrorMessage.MCP_REGISTRY_UNAVAILABLE),
    _ErrorFamily(McpRegistryNotFoundError, _CODE_NOT_FOUND),
    _ErrorFamily(McpRegistryRequestError, _CODE_REGISTRY),
    _ErrorFamily(StorageError, _CODE_STORAGE, "A storage operation failed."),
    _ErrorFamily(AgentError, _CODE_NOT_FOUND),
    _ErrorFamily(ConfigValidationError, _CODE_VALIDATION),
    _ErrorFamily(ConfigError, _CODE_VALIDATION),
    # Catch-all for the remaining domain families (e.g. in-use conflicts,
    # cron validation, HITL actions): their centralized detail is actionable
    # and non-secret, rendered under the validation code like the REST layer.
    _ErrorFamily(DomainError, _CODE_VALIDATION),
)


def _render_pydantic_issues(exc: Exception) -> str:
    """Render pydantic issues: field location + schema message ONLY.

    NEITHER the rejected input nor its value is echoed — a malformed tool
    argument may carry credential material (REQ-009), matching the REST
    request-validation handler's redaction of ``input``/``ctx``.
    """
    errors: list[dict] = [error for error in (getattr(exc, "errors", list)() or []) if isinstance(error, dict)]
    issues = [
        f"{'.'.join(str(part) for part in error.get('loc', ()))}: {error.get('msg', 'invalid value')}"
        for error in errors
    ]
    return "; ".join(issues[:5])


def error_detail(exc: Exception) -> str:
    """Map one exception to its SAFE, structured MCP error text (REQ-009).

    The mapping is centralized: each domain family contributes a stable code
    prefix plus its centralized (non-secret) detail. Unexpected exceptions
    collapse into a generic message.
    """
    if isinstance(exc, pydantic.ValidationError):
        return f"[{_CODE_VALIDATION}] Invalid tool arguments ({_render_pydantic_issues(exc)})"
    for family in _ERROR_FAMILIES:
        if isinstance(exc, family.exception_type):
            # ``detail_override`` pins a SAFE generic wording; the family's own
            # ``exc.detail`` comes from the centralized ErrorMessage catalogue.
            detail = family.detail_override if family.detail_override is not None else _detail_of(exc)
            return f"[{family.code}] {detail}"
    return f"[{_CODE_UNEXPECTED}] Tool failed unexpectedly; check the server logs."


def _detail_of(exc: Exception) -> str:
    """Return the exception's centralized detail text (DomainError.detail)."""
    return str(getattr(exc, "detail", exc))


def safe_mcp_call(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Decorate an MCP tool coroutine with the safe error mapping (REQ-009).

    Any exception escaping the wrapped body is mapped to its safe,
    code-prefixed form and re-raised as :class:`McpToolError`; the SDK then
    renders it as an ``isError`` tool result. ``functools.wraps`` preserves
    the original signature so the FastMCP schema derives from the declared
    tool parameters (verified against the installed SDK).
    """
    from src.infrastructure.database.rls_context import current_user_id

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs) -> T:
        user = current_user_id.get() or ""
        started = time.monotonic()
        try:
            result = await fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — mapped centrally below
            detail = error_detail(exc)
            # Server-side diagnostics carry the exception TYPE + traceback;
            # the wire payload carries ONLY the mapped safe detail (REQ-009).
            logger.warning(LogMessage.MCP_TOOL_FAILED, user, fn.__name__, detail)
            logger.debug(LogMessage.MCP_TOOL_FAILURE_TYPE, type(exc).__name__)
            raise McpToolError(detail) from None
        logger.info(LogMessage.MCP_TOOL_EXECUTED, user, fn.__name__, time.monotonic() - started, True)
        return result

    return wrapper
