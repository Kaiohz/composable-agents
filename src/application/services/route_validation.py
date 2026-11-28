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

from src.domain.constants import MAX_YAML_SIZE
from src.domain.errors.config import ConfigError
from src.domain.errors.messages import ErrorMessage
from src.domain.errors.store_file import StoreFileNotFoundError
from src.domain.services.agent_scope import MEMORIES_PREFIX, agent_memory_prefix, agent_scope_slug, is_valid_scope_slug

#: REST agent-name pattern — shared verbatim with the MCP agent tools.
AGENT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,98}[a-zA-Z0-9]$")
# ``MAX_YAML_SIZE`` (1 MB YAML upload limit) is imported from
# :mod:`src.domain.constants` — the single owner shared with the read-side
# agent-topology resolver (review NV-1).

_SKILLS_PREFIX = "/skills/"


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


def validate_agent_slug(slug: str) -> str:
    """Validate one agent-slug segment (REQ-002/REQ-015 charset rule).

    The single controlled nesting segment of an agent-scoped memory path:
    lowercase alphanumerics and dashes, 1..100 characters (the pattern is
    owned by :mod:`src.domain.services.agent_scope` — one rule, one owner).

    Args:
        slug: The candidate agent segment.

    Returns:
        The validated (unchanged) slug.

    Raises:
        StoreFileNotFoundError: When the segment violates the charset rule.
    """
    if not is_valid_scope_slug(slug):
        raise StoreFileNotFoundError(ErrorMessage.MCP_MEMORY_AGENT_SEGMENT_INVALID.format(slug=slug))
    return slug


def memory_store_path(filename: str, agent_scope: str | None = None) -> str:
    """Resolve a ``/memories/<name>`` store path (REQ-013 + REQ-015 + REQ-020).

    Two accepted shapes:

    * ``agent_scope`` bound (REQ-020 closure, agent-facing platform tools):
      ``filename`` must be a PLAIN component and the file resolves inside the
      bound agent's space — ``/memories/{slug}/{filename}`` where ``slug`` is
      the bound agent's canonical scope slug (C-1: platform-controlled, so
      no agent/scope can be smuggled through the filename and the segment is
      always slug-valid even for non-slug agent names).
    * ``agent_scope`` unset (external MCP sessions): ``filename`` MUST be the
      nested agent shape ``{agent_slug}/{name}`` with EXACTLY one agent
      segment validated against :func:`validate_agent_slug` (REQ-015 —
      provided segments are validated STRICTLY, never mapped). The FLAT
      ``{name}`` personal shape is REFUSED — memories are scoped user × agent
      (user directive): the personal namespace is a system-internal fallback
      and is not reachable from callers. Deeper nesting stays rejected.

    Raises:
        StoreFileNotFoundError: On any traversal/escape, invalid segment or a
            flat (non-agent-scoped) filename.
    """
    if agent_scope is not None:
        name = _reject_unsafe_component(filename, ErrorMessage.MCP_MEMORY_NAME_INVALID.format(name=filename))
        return normalize_store_path(f"{MEMORIES_PREFIX}{agent_scope_slug(agent_scope)}/{name}")

    normalized = (filename or "").strip().strip("/")
    if "/" in normalized:
        agent_segment, _, nested_name = normalized.partition("/")
        validate_agent_slug(agent_segment)
        name = _reject_unsafe_component(nested_name, ErrorMessage.MCP_MEMORY_NAME_INVALID.format(name=filename))
        return normalize_store_path(f"{MEMORIES_PREFIX}{agent_segment}/{name}")
    # User × agent directive: a FLAT (no '/') filename never resolves into the
    # personal namespace — memories are scoped user × agent; scope-less
    # sessions must carry the '{agent_slug}/{name}' shape.
    raise StoreFileNotFoundError(
        ErrorMessage.MCP_MEMORY_AGENT_REQUIRED.format(name=filename)
    )


def memories_prefix(agent_scope: str | None = None) -> str:
    """Return the memories store prefix, scoped to an agent when given.

    Args:
        agent_scope: When set, the caller's agent space prefix
            ``/memories/{slug}/`` (the canonical scope slug of the bound
            agent name, REQ-020 + C-1); ``None`` returns the flat global
            personal prefix ``/memories/`` (REQ-019).
    """
    return agent_memory_prefix(agent_scope)


def validate_memories_store_path(path: str) -> str:
    """Validate one ``/memories/...`` store path for the REST store-file route.

    Both boundaries of the same store MUST enforce REQ-015 identically: the
    MCP memories tools validate the agent segment and the depth through
    :func:`memory_store_path`, while the REST store-file route historically
    accepted ANY ``/memories/...`` shape (QA BUG-005). The path must be
    exactly the flat personal shape ``/memories/{name}`` or the
    single-segment agent shape ``/memories/{agent_slug}/{name}`` with an
    ``agent_slug`` matching :func:`validate_agent_slug` (REQ-002 charset);
    deeper nesting stays rejected. Paths OUTSIDE the ``/memories/`` prefix
    (skills, agent namespaces) keep the general ``normalize_store_path``
    contract unchanged.

    Args:
        path: The store path as captured by the route (leading slash added
            and ``..`` traversal rejected first, unchanged behavior).

    Returns:
        The normalized, validated store path to use.

    Raises:
        StoreFileNotFoundError: On any traversal/escape, invalid agent
            segment or deeper nesting — the same centralized error the
            route already raises for traversal attempts (mapped to 404).
    """
    normalized = normalize_store_path(path)
    if not normalized.startswith(MEMORIES_PREFIX):
        return normalized
    return memory_store_path(normalized[len(MEMORIES_PREFIX) :])


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
