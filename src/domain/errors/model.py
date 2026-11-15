"""Model catalog domain errors (spec §5.2)."""

from src.domain.errors.base import DomainError
from src.domain.errors.codes import ErrorCode


class ModelError(DomainError):
    """Base error for model catalog operations."""

    status_code = ErrorCode.INTERNAL_SERVER_ERROR


class ModelNotFoundError(ModelError):
    """Model id not found (or owned by another user — non-enumerating, §9)."""

    status_code = ErrorCode.NOT_FOUND


class ModelInUseError(ModelError):
    """The model is referenced by at least one agent and cannot be deleted (MDL-004)."""

    status_code = ErrorCode.CONFLICT
