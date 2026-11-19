"""Shared route-boundary validation (REQ-007) used by REST routes AND MCP tools.

The validation rules described here used to exist only inside REST route
modules. MCP tools must apply EXACTLY the same rules — agent-name validation,
YAML size/UTF-8 checks, store-path normalization and UTC normalization for
trace filters — so the rules live once, in this module, and both boundaries
call them. This prevents REST/MCP behavioral drift.

Pure functions; they raise the centralized domain errors that REST handlers
and the MCP error mapper already understand.
"""

import re
from datetime import UTC, datetime

from src.domain.errors.config import ConfigError
from src.domain.errors.messages import ErrorMessage
from src.domain.errors.store_file import StoreFileNotFoundError

#: REST agent-name pattern — shared verbatim with the MCP agent tools.
AGENT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,98}[a-zA-Z0-9]$")
#: 1 MB YAML upload limit — shared verbatim with the MCP agent tools.
MAX_YAML_SIZE = 1024 * 1024

_SKILLS_PREFIX = "/skills/"
_MEMORIES_PREFIX = "/memories/"


def _reject_unsafe_component(component: str, invalid_message: str) -> str:
    """Validate one plain path component for the global store prefixes (REQ-013).

    A component must be a plain directory/file name: no empty value, no
    path separators, no ``..`` traversal and no current/parent references.
    Returns the stripped component.

    Raises:
        StoreFileNotFoundError: On any traversal/escape attempt.
    """
    normalized = (component or "").strip().strip("/")
    if not normalized or "/" in normalized or "\\" in normalized or ".." in normalized or normalized in (".", ".."):
        raise StoreFileNotFoundError(invalid_message)
    return normalized


def validate_agent_name(name: str) -> None:
    """Validate an agent name (pattern shared by REST and MCP, REQ-007).

    Raises:
        ConfigError: When the name does not match the pattern.
    """
    if not AGENT_NAME_PATTERN.match(name):
        raise ConfigError(ErrorMessage.INVALID_AGENT_NAME.format(name=name))


def read_yaml_bytes(data: bytes) -> str:
    """Validate raw upload bytes and decode them as UTF-8 YAML text (REQ-007/012).

    MCP clients send the YAML text directly; REST uploads carry the same bytes
    through multipart. Both boundaries enforce the identical size limit and
    UTF-8 requirement.

    Raises:
        ConfigError: When the payload exceeds the size limit or is not UTF-8.
    """
    if len(data) > MAX_YAML_SIZE:
        raise ConfigError(ErrorMessage.FILE_TOO_LARGE.format(max_size=MAX_YAML_SIZE))
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ConfigError(ErrorMessage.FILE_NOT_UTF8) from e


def normalize_store_path(path: str) -> str:
    """Normalize a store path: leading slash + traversal rejection (REQ-013).

    Raises:
        StoreFileNotFoundError: When the path tries to escape the store.
    """
    normalized = path if path.startswith("/") else f"/{path}"
    if ".." in normalized:
        raise StoreFileNotFoundError(f"Invalid path: {path}")
    return normalized


def skill_store_path(skill_name: str, filename: str = "SKILL.md") -> str:
    """Resolve the global ``/skills/<name>/<filename>`` path (REQ-013).

    The skill tools may only write inside the global ``/skills/`` prefix: the
    name must be a plain directory name (no separators, no ``..``). The
    filename is a fixed constant (``SKILL.md``), never caller-controlled.

    Raises:
        StoreFileNotFoundError: On any traversal/escape attempt.
    """
    name = _reject_unsafe_component(skill_name, ErrorMessage.MCP_SKILL_NAME_INVALID.format(name=skill_name))
    return normalize_store_path(f"{_SKILLS_PREFIX}{name}/{filename}")


def skills_prefix() -> str:
    """Return the global skills store prefix."""
    return _SKILLS_PREFIX


def skill_directory_path(skill_name: str) -> str:
    """Resolve the global ``/skills/<name>/`` directory prefix (REQ-013)."""
    name = _reject_unsafe_component(skill_name, ErrorMessage.MCP_SKILL_NAME_INVALID.format(name=skill_name))
    return normalize_store_path(f"{_SKILLS_PREFIX}{name}/")


def memory_store_path(filename: str) -> str:
    """Resolve the global ``/memories/<filename>`` path (REQ-013).

    Raises:
        StoreFileNotFoundError: On any traversal/escape attempt.
    """
    name = _reject_unsafe_component(filename, ErrorMessage.MCP_MEMORY_NAME_INVALID.format(name=filename))
    return normalize_store_path(f"{_MEMORIES_PREFIX}{name}")


def memories_prefix() -> str:
    """Return the global memories store prefix."""
    return _MEMORIES_PREFIX


def utc_bound(value):
    """Normalize a trace-filter bound to UTC (shared with REST, REQ-007).

    A naive datetime bound against a ``timestamptz`` column would be
    interpreted in the database session timezone; ISO bounds without an offset
    therefore mean UTC.

    Args:
        value: The parsed bound (or ``None``).

    Returns:
        The UTC-normalized bound (or ``None`` when unset).
    """
    if value is None:
        return None
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


async def list_skill_usage_agent_names(list_files, skill_name: str) -> list[str]:
    """List the agents whose namespace carries ``/skills/{skill_name}/``.

    Shared helper used by the REST ``GET /api/v1/store/skills/{name}/usage``
    route and the MCP ``composable_skills_usage`` tool so the scan semantics
    never drift.

    Args:
        list_files: An awaitable callable ``(prefix: str) -> list[str]``.
        skill_name: The skill directory name (plain, validated by callers).

    Returns:
        The sorted list of agent names that own a copy of the skill.
    """
    all_files = await list_files("/agents/")
    agents: set[str] = set()
    for path in all_files:
        if f"/skills/{skill_name}/" in path:
            parts = path.split("/")
            if len(parts) >= 3 and parts[1] == "agents":
                agents.add(parts[2])
    return sorted(agents)
