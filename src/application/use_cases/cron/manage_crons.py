"""Cron definition CRUD use cases with schedule validation (spec §5.3 CRN-001/002).

Create and update validate the expression (exactly five fields, no seconds),
the timezone (IANA identifier), the agent (must exist and belong to the cron
owner) and the unattended-run requirement (the resolved agent configuration
must not need ``${USER_JWT}``/``${USER_API_KEY}``). The next occurrence is
recalculated after create, schedule/timezone changes and enable/disable
operations; disabled crons carry a ``None`` frontier.
"""

import logging
from datetime import UTC, datetime

from src.application.services.cron_unattended import find_user_placeholder_sources
from src.domain.entities.cron import Cron
from src.domain.errors.cron import CronValidationError
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_config_repository import AgentConfigRepository
from src.domain.ports.agent_config_store import AgentConfigStore
from src.domain.ports.cron_repository import CronRepository
from src.domain.ports.mcp_registry import McpRegistryClient
from src.domain.services.cron_schedule import (
    next_occurrence,
    validate_expression,
    validate_timezone,
)

logger = logging.getLogger(__name__)

def compute_next_run(cron_expression: str, timezone: str, enabled: bool, now: datetime) -> datetime | None:
    """Frontier for a definition state: ``None`` when disabled, else the next occurrence."""
    if not enabled:
        return None
    return next_occurrence(cron_expression, timezone, now)


async def assert_agent_unattended_safe(
    agent_name: str,
    agent_config_repository: AgentConfigRepository,
    config_store: AgentConfigStore,
    mcp_registry: McpRegistryClient | None = None,
) -> str:
    """Validate that the agent exists, is owned and runs unattended (CRN-002).

    The scan covers BOTH the stored YAML (legacy inline values) and the
    agent's ``mcp_refs`` selection resolved through the trusted registry
    execution-config path (resource-only agents keep their MCP selection in
    the registry).

    Args:
        agent_name: Exact agent name reference.
        agent_config_repository: Metadata repository (ownership via context).
        config_store: YAML store — scanned for user-credential placeholders.
        mcp_registry: Trusted registry client (resolves ``mcp_refs``).

    Returns:
        The raw agent YAML text (reused by callers when handy).

    Raises:
        CronValidationError: Missing/malformed agent reference or an agent
            requiring ``${USER_JWT}``/``${USER_API_KEY}`` for unattended runs.
        AgentNotFoundError: Missing agent (or foreign — non-enumerating).
    """
    if not isinstance(agent_name, str) or not agent_name.strip():
        # Defensive guard (QA BUG-010 root cause): a client-supplied ``null``
        # sentinel must be a clean validation error at the boundary — never a
        # 404 whose message interpolates a Python ``None`` repr.
        raise CronValidationError(ErrorMessage.CRON_AGENT_NAME_REQUIRED)
    await agent_config_repository.get(agent_name)
    yaml_text = await config_store.get(agent_name)
    sources = await find_user_placeholder_sources(yaml_text, mcp_registry, agent_name)
    if sources:
        raise CronValidationError(
            ErrorMessage.CRON_AGENT_UNATTENDED_PLACEHOLDER.format(
                agent_name=f"{agent_name} ({', '.join(sources)})"
            )
        )
    return yaml_text


class CreateCronUseCase:
    """Create and validate a cron owned by the current user."""

    def __init__(
        self,
        cron_repository: CronRepository,
        agent_config_repository: AgentConfigRepository,
        config_store: AgentConfigStore,
        mcp_registry: McpRegistryClient | None = None,
    ) -> None:
        self._cron_repository = cron_repository
        self._agent_config_repository = agent_config_repository
        self._config_store = config_store
        self._mcp_registry = mcp_registry

    async def execute(
        self,
        name: str,
        agent_name: str,
        message: str,
        cron_expression: str,
        timezone: str,
        enabled: bool,
    ) -> Cron:
        """Validate the definition and persist it with its computed frontier.

        Raises:
            CronValidationError: Bad expression/timezone or unattended
                placeholder requirement.
            AgentNotFoundError: Missing or foreign agent.
        """
        validate_expression(cron_expression)
        validate_timezone(timezone)
        await assert_agent_unattended_safe(
            agent_name, self._agent_config_repository, self._config_store, self._mcp_registry
        )

        now = datetime.now(UTC)
        cron = Cron(
            name=name,
            agent_name=agent_name,
            message=message,
            cron_expression=cron_expression,
            timezone=timezone,
            enabled=enabled,
        )
        created = await self._cron_repository.create(cron, compute_next_run(cron_expression, timezone, enabled, now))
        logger.info(LogMessage.CRON_CREATED, created.id, created.next_run_at)
        return created


class ListCronsUseCase:
    """List owned crons with scheduling state and last-run summary."""

    def __init__(self, cron_repository: CronRepository) -> None:
        self._cron_repository = cron_repository

    async def execute(self) -> list[Cron]:
        return await self._cron_repository.list_all()


class GetCronUseCase:
    """Fetch an owned cron definition (non-enumerating 404 otherwise)."""

    def __init__(self, cron_repository: CronRepository) -> None:
        self._cron_repository = cron_repository

    async def execute(self, cron_id: str) -> Cron:
        return await self._cron_repository.get(cron_id)


class UpdateCronUseCase:
    """Edit definition fields (including ``enabled``) and recalculate the frontier."""

    def __init__(
        self,
        cron_repository: CronRepository,
        agent_config_repository: AgentConfigRepository,
        config_store: AgentConfigStore,
        mcp_registry: McpRegistryClient | None = None,
    ) -> None:
        self._cron_repository = cron_repository
        self._agent_config_repository = agent_config_repository
        self._config_store = config_store
        self._mcp_registry = mcp_registry

    async def execute(self, cron_id: str, changes: dict) -> Cron:
        """Apply provided fields; revalidate what changed.

        The frontier is recalculated when ``enabled``, ``cron_expression`` or
        ``timezone`` changes (or the agent reference changes), preventing
        stale, unclaimed occurrences from a previous schedule from executing
        (CRN-006). Only explicitly-provided fields are applied.

        Raises:
            CronNotFoundError: Missing or foreign cron.
            CronValidationError: Invalid new expression/timezone/agent.
        """
        current = await self._cron_repository.get(cron_id)
        merged = {
            "name": changes.get("name", current.name),
            "agent_name": changes.get("agent_name", current.agent_name),
            "message": changes.get("message", current.message),
            "cron_expression": changes.get("cron_expression", current.cron_expression),
            "timezone": changes.get("timezone", current.timezone),
            "enabled": changes.get("enabled", current.enabled),
        }
        schedule_changed = (
            "cron_expression" in changes
            or "timezone" in changes
            or "enabled" in changes
            or "agent_name" in changes
        )
        if "cron_expression" in changes:
            validate_expression(merged["cron_expression"])
        if "timezone" in changes:
            validate_timezone(merged["timezone"])
        if "agent_name" in changes:
            await assert_agent_unattended_safe(
                merged["agent_name"], self._agent_config_repository, self._config_store, self._mcp_registry
            )

        if not schedule_changed:
            next_run_at = current.next_run_at
        else:
            next_run_at = compute_next_run(
                merged["cron_expression"], merged["timezone"], merged["enabled"], datetime.now(UTC)
            )
        updated = current.model_copy(update=merged)
        saved = await self._cron_repository.update(updated, next_run_at)
        logger.info(LogMessage.CRON_UPDATED, saved.id, saved.next_run_at)
        return saved


class DeleteCronUseCase:
    """Delete a cron; future scheduling stops (runs are removed by cascade)."""

    def __init__(self, cron_repository: CronRepository) -> None:
        self._cron_repository = cron_repository

    async def execute(self, cron_id: str) -> None:
        await self._cron_repository.delete(cron_id)
        logger.info(LogMessage.CRON_DELETED, cron_id)
