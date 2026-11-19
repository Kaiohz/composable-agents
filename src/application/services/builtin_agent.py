"""Built-in Composable AI Assistant overlay (REQ-031..REQ-044).

The assistant is a VIRTUAL, centrally managed, read-only agent definition:

* it is never seeded into the globally keyed ``agent_configs`` table nor the
  shared object store (no per-user rows, no ``{name}.yaml`` object);
* it is overlaid into each authenticated user's agent catalog (list/get,
  registry discovery, thread creation, one-shot execution, runner building);
* its name is reserved: creating a custom agent with it conflicts, and
  updating/deleting the assistant itself raises the read-only
  :class:`~src.domain.errors.agent.BuiltinAgentError` (REQ-034);
* ``is_builtin`` is computed here (REQ-033) — never persisted;
* its definition carries the general-purpose system prompt (REQ-035) and the
  HITL interrupt rules for EVERY destructive delete tool (REQ-042).
"""

import logging
from datetime import UTC, datetime

from src.domain.entities.agent_config import AgentConfig, HITLConfig
from src.domain.entities.agent_config_metadata import AgentConfigMetadata
from src.domain.errors.agent import AgentConfigAlreadyExistsError, BuiltinAgentError
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)

#: The virtual built-in assistant name (spec: fixed, reserved platform-wide).
BUILTIN_AGENT_NAME = "composable-ai-assistant"

#: Serving model of the built-in definition (metadata only; execution always
#: uses the running user's configured LLM credentials, REQ-041).
BUILTIN_AGENT_MODEL = AgentConfig.model_fields["model"].default

#: General-purpose system prompt (REQ-035): manage Composable resources, use
#: the available MCP capabilities and run other user agents, while respecting
#: authorization, avoiding settings/credentials and requesting
#: destructive-delete approval.
BUILTIN_SYSTEM_PROMPT = """You are the Composable AI Assistant, the platform's built-in general-purpose agent.
You can manage the current user's Composable resources through the platform tools available to you:
create and update agents, MCP servers, skills, memories, prompts (and prompt versions), models, crons,
and inspect threads, traces and usage.
You can also use the available MCP capabilities (built-in and user-registered MCP servers) and run
other agents owned by the current user as one-shot tasks.
Rules you must always follow:
- Act strictly on behalf of the authenticated Composable user you are serving; never attempt to access,
  mention or request another user's resources, settings, credentials or API keys.
- Destructive deletion tools trigger a human approval flow: propose a deletion only when the user asks
  for it, then let the approval gate decide.
- When a capability is unavailable or fails, say so plainly instead of guessing.
"""

#: The destructive delete tools of REQ-011 — the built-in definition
#: configures an interrupt rule for EVERY one of them (REQ-042).
BUILTIN_HITL_DELETE_TOOLS: tuple[str, ...] = (
    "composable_agents_delete",
    "composable_mcp_servers_delete",
    "composable_skills_delete",
    "composable_memories_delete",
    "composable_prompts_delete",
    "composable_models_delete",
    "composable_crons_delete",
    "composable_threads_delete",
)


def is_builtin_agent_name(name: str) -> bool:
    """Return whether ``name`` addresses the virtual built-in assistant."""
    return name == BUILTIN_AGENT_NAME


def _builtin_description() -> str:
    """Return the display description of the built-in assistant."""
    return "Built-in, centrally managed Composable AI Assistant."


def builtin_agent_metadata() -> AgentConfigMetadata:
    """Build the overlay metadata projection of the built-in assistant.

    The timestamps are "now" placeholders: the assistant has no persisted row;
    clients display the live definition, never stored metadata.
    """
    now = datetime.now(UTC)
    return AgentConfigMetadata(
        name=BUILTIN_AGENT_NAME,
        model=BUILTIN_AGENT_MODEL,
        minio_path="",
        description=_builtin_description(),
        created_at=now,
        updated_at=now,
        is_builtin=True,
    )


def builtin_agent_config() -> AgentConfig:
    """Build the virtual read-only definition of the built-in assistant (REQ-032).

    The definition carries the system prompt (REQ-035) and an interrupt rule
    for every destructive delete tool (REQ-042). Its ``tools`` list is empty:
    the dynamic toolset (platform tools, built-in MCP mounts, user-registered
    MCP servers, ``create_pdf``) is resolved per build in the running user's
    context (REQ-036/038) — see the agent registry's built-in resolution.
    """
    return AgentConfig(
        name=BUILTIN_AGENT_NAME,
        model=BUILTIN_AGENT_MODEL,
        system_prompt=BUILTIN_SYSTEM_PROMPT,
        tools=[],
        description=_builtin_description(),
        hitl=HITLConfig(rules=dict.fromkeys(BUILTIN_HITL_DELETE_TOOLS, True)),
        is_builtin=True,
    )


def ensure_custom_agent_name(name: str) -> None:
    """Reserve the built-in name against custom agent creation (REQ-034).

    Args:
        name: The requested agent name.

    Raises:
        AgentConfigAlreadyExistsError: With an actionable message when the
            name is reserved (HTTP 409).
    """
    if is_builtin_agent_name(name):
        logger.warning(LogMessage.LOG_AGENT_CONFIG_ALREADY_EXISTS, name)
        raise AgentConfigAlreadyExistsError(ErrorMessage.AGENT_NAME_RESERVED.format(name=name))


def ensure_not_builtin(name: str) -> None:
    """Refuse built-in agent mutations (update/delete) in any entry point.

    Args:
        name: The addressed agent name.

    Raises:
        BuiltinAgentError: Read-only built-in-agent error (HTTP 403) shared by
            REST and MCP (REQ-034).
    """
    if is_builtin_agent_name(name):
        raise BuiltinAgentError(ErrorMessage.BUILTIN_AGENT_READ_ONLY.format(name=name))
