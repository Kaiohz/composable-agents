"""Cron subsystem domain errors (spec §5.3)."""

from src.domain.errors.base import DomainError
from src.domain.errors.codes import ErrorCode


class CronError(DomainError):
    """Base error for cron operations."""

    status_code = ErrorCode.INTERNAL_SERVER_ERROR


class CronNotFoundError(CronError):
    """Cron id not found (or owned by another user — non-enumerating)."""

    status_code = ErrorCode.NOT_FOUND


class CronValidationError(CronError):
    """Cron definition failed validation: bad expression, invalid timezone,
    missing/inaccessible agent, or unattended user-placeholder requirement."""
    status_code = ErrorCode.UNPROCESSABLE_ENTITY


class CronRunConflictError(CronError):
    """A run is already active for this cron — manual overlap is rejected (CRN-005)."""

    status_code = ErrorCode.CONFLICT
