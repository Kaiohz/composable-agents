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
    """Schema validation error carrying structured error details."""

    status_code = ErrorCode.UNPROCESSABLE_ENTITY

    def __init__(self, errors: list[dict]) -> None:
        self.errors = [_jsonable(e) for e in errors]
        messages = [f"  - {e.get('loc', '?')}: {e.get('msg', '?')}" for e in errors]
        super().__init__("Validation errors:\n" + "\n".join(messages))


def _jsonable(value):
    """Deeply coerce pydantic error payloads into JSON-serializable values.

    Pydantic wraps ``ctx`` exceptions (e.g. ``ValueError`` instances raised by
    model validators) into the error dict; without coercion the response layer
    crashes serializing them.
    """
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
