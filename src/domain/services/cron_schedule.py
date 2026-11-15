"""Cron schedule evaluation service (spec §5.3 CRN-001/CRN-005).

Pure domain logic over standard five-field cron expressions (no seconds) and
IANA timezones. The original expression and timezone are retained by callers;
occurrence timestamps are evaluated here and handled in UTC.
"""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from croniter import croniter

from src.domain.errors.cron import CronValidationError
from src.domain.errors.messages import ErrorMessage

# Five fields exactly (minute hour day-of-month month day-of-week). Expressions
# with seconds (six+ fields) are rejected outright (§9).
_CRON_FIELD_COUNT = 5


def validate_expression(expression: str) -> None:
    """Validate a five-field cron expression (no seconds).

    Args:
        expression: The raw expression string.

    Raises:
        CronValidationError: If the expression does not have exactly five
            valid, parseable fields.
    """
    if not isinstance(expression, str):
        # Defensive guard (QA BUG-007 root cause): a non-string (e.g. an
        # explicit ``null`` that slipped past every earlier boundary) must be a
        # mapped validation error, never an unhandled ``AttributeError`` from
        # ``expression.split(...)``. The malformed value is not interpolated
        # into the message (no Python repr leaks to the client).
        raise CronValidationError(
            ErrorMessage.CRON_INVALID_EXPRESSION.format(
                expression="<none>",
                reason="expression must be a five-field cron string",
            )
        )
    fields = expression.split()
    if len(fields) != _CRON_FIELD_COUNT:
        raise CronValidationError(
            ErrorMessage.CRON_INVALID_EXPRESSION.format(
                expression=expression,
                reason=f"expected exactly {_CRON_FIELD_COUNT} fields, got {len(fields)} (seconds are not supported)",
            )
        )
    try:
        croniter(expression, datetime.now(UTC))
    except (ValueError, KeyError) as exc:
        raise CronValidationError(
            ErrorMessage.CRON_INVALID_EXPRESSION.format(expression=expression, reason=str(exc))
        ) from exc


def validate_timezone(timezone_name: str) -> ZoneInfo:
    """Validate an IANA timezone identifier and return its ZoneInfo.

    Args:
        timezone_name: e.g. ``UTC`` or ``Europe/Paris``.

    Returns:
        The resolved :class:`zoneinfo.ZoneInfo`.

    Raises:
        CronValidationError: If the timezone is not a valid IANA identifier.
    """
    try:
        return ZoneInfo(timezone_name)
    except Exception as exc:
        raise CronValidationError(ErrorMessage.CRON_INVALID_TIMEZONE.format(timezone=timezone_name)) from exc


def next_occurrence(expression: str, timezone_name: str, after: datetime) -> datetime:
    """Return the first occurrence strictly after ``after`` (UTC).

    Args:
        expression: Five-field cron expression.
        timezone_name: IANA timezone the expression is evaluated in.
        after: Reference instant (aware UTC).

    Returns:
        The next occurrence as an aware UTC datetime.
    """
    zone = validate_timezone(timezone_name)
    iterator = croniter(expression, after.astimezone(zone))
    return iterator.get_next(datetime).astimezone(UTC)


def latest_due_occurrence(
    expression: str,
    timezone_name: str,
    frontier: datetime,
    now: datetime,
) -> datetime | None:
    """Return the due occurrence to execute for a cron whose frontier elapsed.

    CRN-005 coalescing: when a downtime spans several occurrences, at most one
    execution runs — for the latest boundary that is due at ``now``. The
    returned value is never older than ``frontier``, so a frontier that is not
    itself an aligned boundary still claims its own instant.

    Args:
        expression: Five-field cron expression.
        timezone_name: IANA timezone of the expression.
        frontier: The stored ``next_run_at`` (aware UTC) that is due.
        now: The evaluation instant (aware UTC).

    Returns:
        The occurrence timestamp to claim (aware UTC), or ``None`` when the
        frontier is not yet due.
    """
    if frontier > now:
        return None
    zone = validate_timezone(timezone_name)
    iterator = croniter(expression, now.astimezone(zone))
    latest = iterator.get_prev(datetime).astimezone(UTC)
    if latest < frontier:
        return frontier
    return latest
