"""Shared helpers for subagent ``agent_ref`` validation and dependent invalidation."""

import logging

import yaml  # type: ignore[import-untyped]

from src.domain.entities.agent_config import AgentConfig, SubAgentConfig
from src.domain.errors.config import ConfigError
from src.domain.ports.agent_config_repository import AgentConfigRepository
from src.domain.ports.agent_config_store import AgentConfigStore
from src.domain.ports.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)


async def validate_subagent_refs(
    config: AgentConfig,
    repository: AgentConfigRepository,
) -> None:
    """Validate subagent ``agent_ref`` references against the metadata repository.

    The metadata repository is the single source of truth for "does this agent
    exist?" — injecting the port (rather than a bound ``exists`` callable) keeps
    the use case decoupled from any one adapter's notion of existence and avoids
    drift if the store/repository split is refactored later.

    Each ref is validated by an extracted helper, keeping this orchestrator at a
    minimal cognitive complexity.

    Raises:
        ConfigError: On self-reference or when a referenced agent does not exist.
    """
    for sa in config.subagents:
        await _assert_subagent_ref_exists(sa, config.name, repository)


async def _assert_subagent_ref_exists(
    sa: SubAgentConfig,
    container_name: str,
    repository: AgentConfigRepository,
) -> None:
    """Validate one subagent's ``agent_ref`` against the metadata repository."""
    if sa.agent_ref is None:
        return
    if sa.agent_ref == container_name:
        raise ConfigError(
            f"Subagent '{sa.name}' references its own agent '{container_name}' (self-reference is not allowed)."
        )
    if not await repository.exists(sa.agent_ref):
        raise ConfigError(f"Subagent '{sa.name}' references unknown agent '{sa.agent_ref}'.")


async def invalidate_dependent_agents(
    config_store: AgentConfigStore,
    agent_registry: AgentRegistry,
    agent_name: str,
) -> None:
    """Invalidate cached runners of agents that reference ``agent_name`` via subagent ``agent_ref``.

    Scans every stored YAML, parses it with ``yaml.safe_load`` (lightweight — full
    AgentConfig validation is not needed here) and looks for subagents whose
    ``agent_ref`` equals ``agent_name``. The ``agent_name`` itself is not invalidated
    a second time (the caller is expected to have already invalidated it).

    Corrupted or unparseable YAMLs are logged and skipped so a single bad config
    cannot prevent invalidation of the rest. After the loop, a summary error lists
    every agent that could not be inspected so the failure surface is
    observable — not silently swallowed.

    Args:
        config_store: Store exposing all persisted YAML configs.
        agent_registry: Registry whose cached runners must be invalidated.
        agent_name: Name of the agent whose dependents must be invalidated.
    """
    try:
        all_names = await config_store.list_all()
    except Exception:
        logger.exception(
            "Failed to list stored agent configs during dependent invalidation for '%s'", agent_name
        )
        return

    dependent_agents: dict[str, str] = {}
    failed_agents: list[str] = []

    for other_name in all_names:
        if other_name == agent_name:
            continue
        try:
            subagent_ref = await _find_agent_ref_in(other_name, agent_name, config_store)
            if subagent_ref is not None:
                dependent_agents[other_name] = subagent_ref
        except Exception:
            logger.exception(
                "Failed to parse YAML for agent '%s' during dependent invalidation of '%s'",
                other_name,
                agent_name,
            )
            failed_agents.append(other_name)

    await _invalidate_dependents(dependent_agents, agent_registry, agent_name, failed_agents)


async def _find_agent_ref_in(
    other_name: str,
    agent_name: str,
    config_store: AgentConfigStore,
) -> str | None:
    """Return this config's ``agent_ref`` equal to ``agent_name``, if any.

    Raises:
        Exception: On store read / YAML parse failures (surface as failed agent).
    """
    yaml_content = await config_store.get(other_name)
    data = yaml.safe_load(yaml_content) or {}
    if not isinstance(data, dict):
        return None
    subagents = data.get("subagents")
    if not isinstance(subagents, list):
        return None
    for sa in subagents:
        if isinstance(sa, dict) and sa.get("agent_ref") == agent_name:
            return str(sa["agent_ref"])
    return None


async def _invalidate_dependents(
    dependent_agents: dict[str, str],
    agent_registry: AgentRegistry,
    agent_name: str,
    failed_agents: list[str],
) -> None:
    """Invalidate the found dependents and summarise the parse failures."""
    for other_name in dependent_agents:
        await agent_registry.invalidate(other_name)

    if failed_agents:
        logger.error(
            "Dependent invalidation for '%s' skipped %d agent(s) with parse errors: %s",
            agent_name,
            len(failed_agents),
            ", ".join(failed_agents),
        )
