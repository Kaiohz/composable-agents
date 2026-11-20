"""RunAgentOneShotUseCase — trusted one-shot agent execution (REQ-022..REQ-030).

Creates a NEW thread with the trusted ``source=mcp`` marker, executes exactly
ONE human turn through the existing :class:`SendMessageUseCase` (the standard
trace persistence, HITL and runner behavior are reused unchanged), and returns
the final Message together with the persisted turn trace.

Entry points:

* the MCP ``composable_agents_run`` tool (external coding agents);
* the agent-facing ``composable_agents_run`` platform tool (agent-to-agent
  one-shot execution, REQ-028) — those calls carry an INTERNAL INVOCATION
  CHAIN used to reject self-invocation, cycles and depth beyond four nested
  calls (REQ-029) BEFORE another agent runner is invoked.

The invocation chain flows through a contextvar that the use case sets for the
duration of the turn: nested ``composable_agents_run`` platform tools read it
to extend the chain. A rejected nested run surfaces a stable
``"Agent invocation rejected:"`` marker in the calling agent's ToolMessage; the
caller's one-shot scan re-raises it so the rejection propagates through the
whole chain instead of the outer call silently succeeding.
"""

import logging
import time
from contextvars import ContextVar

from src.application.responses.agent_run import DegradedCapability, RunAgentOneShotResult
from src.application.use_cases.create_thread import CreateThreadUseCase
from src.application.use_cases.send_message import SendMessageUseCase
from src.domain.entities.thread import ThreadSource
from src.domain.entities.trace_event import TraceEventType
from src.domain.errors.agent import AgentInvocationRejectedError
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_registry import AgentRegistry
from src.domain.ports.trace_event_repository import TraceEventRepository

logger = logging.getLogger(__name__)

#: Name of the one-shot platform tool — the chain marker is scanned on its
#: TOOL_RESULT trace events.
RUN_TOOL_NAME = "composable_agents_run"

#: Stable rejection marker prefix (centralized wording, lowercase for scans).
REJECTION_MARKER = "agent invocation rejected"

#: Maximum chain size: the entry execution + at most FOUR nested
#: ``composable_agents_run`` calls (REQ-029). A fifth nested call is rejected.
MAX_INVOCATION_CHAIN_SIZE = 5

#: Runtime invocation chain of the one-shot executions in the current task.
#: Set by :meth:`RunAgentOneShotUseCase.execute`; read by the platform tool.
current_invocation_chain: ContextVar[tuple[str, ...]] = ContextVar("agent_one_shot_invocation_chain", default=())


def get_current_invocation_chain() -> tuple[str, ...]:
    """Return the invocation chain of the executing one-shot (may be empty)."""
    return current_invocation_chain.get()


class RunAgentOneShotUseCase:
    """Execute one agent as a single one-shot task in a trusted ``mcp`` thread."""

    def __init__(
        self,
        create_thread: CreateThreadUseCase,
        send_message: SendMessageUseCase,
        registry: AgentRegistry,
        trace_repo: TraceEventRepository,
    ) -> None:
        self._create_thread = create_thread
        self._send_message = send_message
        self._registry = registry
        self._trace_repo = trace_repo

    async def execute(
        self,
        agent_name: str,
        message: str,
        *,
        invocation_chain: tuple[str, ...] | list[str] | None = None,
    ) -> RunAgentOneShotResult:
        """Run one visible agent for exactly one turn in a new ``mcp`` thread.

        Args:
            agent_name: The agent to execute (any agent visible to the
                authenticated user, including the built-in assistant).
            message: The non-empty human turn message.
            invocation_chain: The INTERNAL invocation chain (agent names of
                the already-executing one-shot runs) — provided only by the
                agent-to-agent platform tool. ``None`` (MCP entry point)
                starts a fresh chain with this agent alone.

        Returns:
            The one-shot result: thread id, agent name, source, final Message
            projection, the bounded turn trace and degraded capability info.

        Raises:
            AgentInvocationRejectedError: Self-invocation, a cycle, or a call
                that would exceed the maximum nesting depth (REQ-029) — always
                BEFORE another agent runner is invoked.
            AgentNotFoundError: The agent does not exist or is foreign.
            AgentInvocationRejectedError: Empty message (REQ-022 non-empty
                contract, safe structural error).
            AgentError / LlmNotConfiguredError: Runner build/execution
                failures surface as-is (the user-facing LLM-not-configured
                error travels unchanged).
        """
        started = time.monotonic()
        # REQ-022: a non-empty message is mandatory for every one-shot run.
        if not isinstance(message, str) or not message.strip():
            raise AgentInvocationRejectedError(ErrorMessage.MCP_TOOL_MESSAGE_EMPTY)

        chain = [str(name) for name in (invocation_chain or [])]
        if agent_name in chain:
            reason = (
                f"agent '{agent_name}' cannot invoke itself"
                if chain and chain[-1] == agent_name
                else f"cyclic invocation of agent '{agent_name}' detected in the invocation chain"
            )
            raise _rejection(agent_name, chain, reason)
        full_chain = [*chain, agent_name]
        if len(full_chain) > MAX_INVOCATION_CHAIN_SIZE:
            reason = (
                f"invocation chain depth {len(full_chain)} exceeds the maximum of "
                f"{MAX_INVOCATION_CHAIN_SIZE - 1} nested {RUN_TOOL_NAME} calls"
            )
            raise _rejection(agent_name, chain, reason)

        # RLS-scoped visibility is owned by the trusted CreateThreadUseCase:
        # it checks the agent BEFORE creating the thread row (no orphan empty
        # threads for unknown/foreign agents) and raises the same AgentNotFoundError,
        # so this use case never re-queries the registry for the same call.
        logger.info(LogMessage.AGENT_ONE_SHOT_STARTED, agent_name, tuple(full_chain))
        token = current_invocation_chain.set(tuple(full_chain))
        try:
            thread = await self._create_thread.execute(agent_name, source=ThreadSource.MCP)
            final_message = await self._send_message.execute(thread.id, message=message)
            events = await self._trace_repo.list_by_thread(thread.id)
            self._raise_on_rejected_nested_run(events)
        finally:
            current_invocation_chain.reset(token)

        degraded = [DegradedCapability(**entry) for entry in self._registry.degraded_capabilities(agent_name)]
        logger.info(
            LogMessage.AGENT_ONE_SHOT_COMPLETED,
            thread.id,
            agent_name,
            final_message.status,
            time.monotonic() - started,
        )
        return RunAgentOneShotResult(
            thread_id=thread.id,
            agent_name=agent_name,
            source=ThreadSource.MCP,
            message=final_message,
            trace=events,
            degraded=degraded,
        )

    def _raise_on_rejected_nested_run(self, events) -> None:
        """Re-raise a nested one-shot rejection so it propagates up the chain.

        A nested ``composable_agents_run`` inside the agent's turn is executed
        by the graph as a tool call; its rejection surfaces as an error
        ToolMessage carrying the stable marker. Without this scan the nested
        failure would be swallowed (the graph continues) and the outer MCP
        call would report a false success.
        """
        for event in events:
            if event.type != TraceEventType.TOOL_RESULT or event.name != RUN_TOOL_NAME:
                continue
            if REJECTION_MARKER in (event.content or "").lower():
                raise AgentInvocationRejectedError(
                    ErrorMessage.AGENT_INVOCATION_REJECTED.format(
                        reason="a nested one-shot invocation executed by this turn was rejected"
                    )
                )


def _rejection(agent_name: str, chain: list[str], reason: str) -> AgentInvocationRejectedError:
    """Build the rejection error and log it safely (no arguments, no secrets)."""
    logger.warning(LogMessage.AGENT_ONE_SHOT_REJECTED, agent_name, tuple(chain), reason)
    return AgentInvocationRejectedError(ErrorMessage.AGENT_INVOCATION_REJECTED.format(reason=reason))
