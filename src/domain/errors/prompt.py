"""Prompt management domain errors (local registry)."""

from src.domain.errors.base import DomainError
from src.domain.errors.codes import ErrorCode


class PromptError(DomainError):
    """Base error for prompt registry operations."""

    status_code = ErrorCode.INTERNAL_SERVER_ERROR


class PromptNotFoundError(PromptError):
    """Prompt identifier not found (or owned by another user — non-enumerating)."""

    status_code = ErrorCode.NOT_FOUND


class PromptVersionNotFoundError(PromptError):
    """The referenced prompt version does not exist (no Latest fallback, §9)."""

    status_code = ErrorCode.NOT_FOUND


class PromptInUseError(PromptError):
    """The prompt is referenced by at least one agent and cannot be deleted (PR-005)."""

    status_code = ErrorCode.CONFLICT
