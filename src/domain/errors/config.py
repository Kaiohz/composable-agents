"""Configuration-related domain errors."""

from src.domain.errors.base import DomainError
from src.domain.errors.codes import ErrorCode


class ConfigError(DomainError):
    """Configuration error (invalid input, malformed file, bad request body)."""

    status_code = ErrorCode.BAD_REQUEST


class ConfigNotFoundError(ConfigError):
    """Configuration file or resource not found."""

    status_code = ErrorCode.NOT_FOUND


class ConfigValidationError(ConfigError):
    """Schema validation error carrying structured error details.

    Structured details are sanitized on construction: pydantic error entries
    echo the REJECTED INPUT (which may carry credential material), so every
    payload is projected onto ``{loc, msg, type}`` — the same rule as the
    request-validation handler. Nothing the YAML declared inline leaks into
    a 422 body or a log.
    """

    status_code = ErrorCode.UNPROCESSABLE_ENTITY

    def __init__(self, errors: list[dict]) -> None:
        self.errors = [_sanitized_error(e) for e in errors]
        messages = [f"  - {e.get('loc', '?')}: {e.get('msg', '?')}" for e in self.errors]
        super().__init__("Validation errors:\n" + "\n".join(messages))


def _sanitized_error(error: dict) -> dict:
    """Project one error entry onto its safe fields (never echo the input)."""
    return {
        "loc": list(error.get("loc", [])) if isinstance(error.get("loc"), (list, tuple)) else error.get("loc", []),
        "msg": str(error.get("msg", "")),
        "type": str(error.get("type", "value_error")),
    }
