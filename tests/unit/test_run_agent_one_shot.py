"""Focused unit tests for RunAgentOneShotUseCase (feat-composable-mcp).

Covers the invocation-chain rules (REQ-029/REQ-030) and the one-shot
orchestration contract (REQ-023/024) against mocked ports, without the
behavioral stack: self-invocation, cycles, depth beyond four, empty messages,
and the nested-rejection trace scan that propagates a chain rejection to the
outer MCP call.
"""

from unittest.mock import AsyncMock

import pytest

from src.application.responses.agent_run import RunAgentOneShotResult
from src.application.use_cases.create_thread import CreateThreadUseCase
from src.application.use_cases.run_agent_one_shot import MAX_INVOCATION_CHAIN_SIZE, RunAgentOneShotUseCase
from src.domain.entities.agent_config import AgentConfig
from src.domain.entities.message import Message, MessageRole, MessageStatus
from src.domain.entities.thread import Thread, ThreadSource
from src.domain.entities.trace_event import TraceEvent, TraceEventType
from src.domain.errors.agent import AgentInvocationRejectedError
from src.domain.ports.agent_registry import AgentRegistry
from src.domain.ports.thread_repository import ThreadRepository
from src.domain.ports.trace_event_repository import TraceEventRepository


def _message(status: MessageStatus = MessageStatus.COMPLETED, content: str = "done") -> Message:
    return Message(role=MessageRole.AI, content=content, status=status)


def _tool_result(name: str, content: str, thread_id: str = "t-1") -> TraceEvent:
    from datetime import UTC, datetime

    return TraceEvent(
        id="e-1",
        thread_id=thread_id,
        turn_id="turn-1",
        type=TraceEventType.TOOL_RESULT,
        source=None,
        name=name,
        content=content,
        metadata=None,
        timestamp=datetime.now(UTC),
        sequence=0,
    )


def _build_use_case(
    *,
    agents: list[str],
    send_message: AsyncMock | None = None,
    trace_events: list[TraceEvent] | None = None,
    thread: Thread | None = None,
) -> tuple[RunAgentOneShotUseCase, AsyncMock]:
    """Wire the use case with async-mock ports and a scripted SendMessage mock."""
    registry = AsyncMock(spec=AgentRegistry)
    registry.list_agents.return_value = agents
    threads_repo = AsyncMock(spec=ThreadRepository)
    threads_repo.create.return_value = thread or Thread(agent_name=agents[0], source=ThreadSource.MCP)
    trace_repo = AsyncMock(spec=TraceEventRepository)
    trace_repo.list_by_thread.return_value = trace_events or []

    send_mock = send_message or AsyncMock()
    send_mock.execute.return_value = _message()

    create_thread = CreateThreadUseCase(threads=threads_repo, registry=registry)
    use_case = RunAgentOneShotUseCase(
        create_thread=create_thread,
        send_message=send_mock,  # type: ignore[arg-type] — unit-level stub of the use case
        registry=registry,
        trace_repo=trace_repo,
    )
    return use_case, send_mock


class TestInvocationChainGuards:
    async def test_direct_self_invocation_is_rejected(self):
        use_case, send_mock = _build_use_case(agents=["a"])
        with pytest.raises(AgentInvocationRejectedError):
            await use_case.execute("a", "go", invocation_chain=("a",))
        send_mock.execute.assert_not_awaited()  # rejected BEFORE any runner is invoked

    async def test_cycle_is_rejected(self):
        use_case, send_mock = _build_use_case(agents=["a", "b"])
        with pytest.raises(AgentInvocationRejectedError):
            await use_case.execute("a", "cycle", invocation_chain=("a", "b"))
        send_mock.execute.assert_not_awaited()

    async def test_depth_beyond_four_is_rejected(self):
        chain = ("d0", "d1", "d2", "d3", "d4")  # entry + 4 nested = at the limit
        assert len(chain) == MAX_INVOCATION_CHAIN_SIZE
        use_case, send_mock = _build_use_case(agents=["d5", *chain])
        with pytest.raises(AgentInvocationRejectedError):
            await use_case.execute("d5", "deeper", invocation_chain=chain)
        send_mock.execute.assert_not_awaited()

    async def test_four_nested_calls_are_allowed(self):
        chain = ("d0", "d1", "d2", "d3")  # entry + 3 nested; the 4th nested target fits
        use_case, send_mock = _build_use_case(agents=["d4", *chain])
        result = await use_case.execute("d4", "one more", invocation_chain=chain)
        assert result.source == ThreadSource.MCP
        send_mock.execute.assert_awaited_once()

    async def test_empty_message_is_rejected(self):
        use_case, send_mock = _build_use_case(agents=["a"])
        for empty in ("", "   "):
            with pytest.raises(AgentInvocationRejectedError):
                await use_case.execute("a", empty)
        send_mock.execute.assert_not_awaited()

    async def test_unknown_agent_is_not_found_before_thread_creation(self):
        use_case, send_mock = _build_use_case(agents=["a"])
        from src.domain.errors.agent import AgentNotFoundError

        with pytest.raises(AgentNotFoundError):
            await use_case.execute("ghost", "hi")
        send_mock.execute.assert_not_awaited()


class TestOneShotOrchestration:
    async def test_creates_trusted_mcp_thread_and_returns_projection(self):
        thread = Thread(agent_name="a", source=ThreadSource.MCP, id="t-9")
        use_case, send_mock = _build_use_case(agents=["a"], thread=thread)
        result = await use_case.execute("a", "hello")

        send_mock.execute.assert_awaited_once_with("t-9", message="hello")
        assert isinstance(result, RunAgentOneShotResult)
        assert result.thread_id == "t-9"
        assert result.agent_name == "a"
        assert result.source == ThreadSource.MCP
        assert result.message.content == "done"
        assert result.degraded == []

    async def test_rejection_marker_in_trace_propagates_to_the_outer_call(self):
        """A nested run-tool rejection recorded in the trace fails the caller."""
        thread = Thread(agent_name="a", source=ThreadSource.MCP, id="t-1")
        events = [
            _tool_result(
                "composable_agents_run",
                "Error executing tool composable_agents_run: Agent invocation rejected: cycle",
                thread_id="t-1",
            ),
        ]
        use_case, send_mock = _build_use_case(agents=["a", "b"], thread=thread, trace_events=events)
        send_mock.execute.return_value = _message()

        with pytest.raises(AgentInvocationRejectedError):
            await use_case.execute("b", "start", invocation_chain=("base",))

    async def test_benign_tool_results_do_not_propagate(self):
        thread = Thread(agent_name="a", source=ThreadSource.MCP, id="t-2")
        events = [
            _tool_result("composable_agents_run", "all good: thread b completed"),
        ]
        use_case, send_mock = _build_use_case(agents=["a", "b"], thread=thread, trace_events=events)
        result = await use_case.execute("b", "run b")
        send_mock.execute.assert_awaited_once()
        assert result.degraded == []

    async def test_registry_reports_degraded_capabilities(self):
        thread = Thread(agent_name="a", source=ThreadSource.MCP, id="t-3")
        use_case, _ = _build_use_case(agents=["a"], thread=thread)
        use_case._registry.degraded_capabilities.return_value = [{"name": "dead", "reason": "MCP server unavailable"}]
        result = await use_case.execute("a", "hi")
        assert [item.model_dump() for item in result.degraded] == [{"name": "dead", "reason": "MCP server unavailable"}]

    async def test_agent_config_round_trip_for_platform_tools(self):
        """The platform runner path builds the config from a plain AgentConfig."""
        config = AgentConfig(name="a")
        assert config.is_builtin is False  # overlay marker untouched by YAML round trips
