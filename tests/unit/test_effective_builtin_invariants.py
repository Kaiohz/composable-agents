"""Projection invariants of the effective built-in assistant definition.

``_apply_invariants`` is the READ projection: dynamic server-side capability
fields are forced out of the user-facing shape EXCEPT the capability FLAGS the
UI needs (files_mcp for document attachments — the dynamic toolset itself
stays invisible via ``mcp_servers: []``).
"""

from unittest.mock import MagicMock

from src.application.services.effective_builtin_agent import EffectiveBuiltinAgentService
from src.domain.entities.agent_config import AgentConfig


def _config() -> AgentConfig:
    return AgentConfig(
        name="composable-ai-assistant",
        system_prompt="sys",
        mcp_servers=[],
    )


class TestApplyInvariants:
    def test_capabilities_declare_files_mcp(self):
        projected = EffectiveBuiltinAgentService._apply_invariants(_config())

        assert projected.capabilities == {"files_mcp": True}

    def test_dynamic_toolset_stays_forced_empty(self):
        projected = EffectiveBuiltinAgentService._apply_invariants(_config())

        assert projected.mcp_servers == []
        assert projected.tools == []
        assert projected.subagents == []
        assert projected.is_builtin is True


class TestStreamReasoningEffortPassthrough:
    """StreamMessageUseCase threads ChatRequest.reasoning_effort to the
    registry's get_runner."""

    async def test_reasoning_effort_reaches_get_runner(self):
        import uuid
        from datetime import UTC, datetime
        from unittest.mock import AsyncMock

        from src.application.use_cases.stream_message import StreamMessageUseCase
        from src.domain.entities.thread import Thread
        from src.domain.entities.trace_event import TraceEvent, TraceEventType

        async def single_event():
            now = datetime.now(UTC)
            yield TraceEvent(
                id=str(uuid.uuid4()),
                thread_id="t1",
                turn_id="tr1",
                type=TraceEventType.AI_MESSAGE,
                content="done",
                timestamp=now,
                sequence=1,
            )

        runner = MagicMock()
        runner.stream = MagicMock(return_value=single_event())
        registry = MagicMock()
        registry.get_runner = AsyncMock(return_value=runner)
        threads = AsyncMock()
        threads.get = AsyncMock(return_value=Thread(id="t1", agent_name="composable-ai-assistant", user_id="u1"))
        trace_repo = MagicMock()
        trace_repo.add = AsyncMock(return_value=None)

        use_case = StreamMessageUseCase(registry=registry, threads=threads, trace_repo=trace_repo)
        events = [event async for event in use_case.execute("t1", "hello", reasoning_effort="high")]

        assert len(events) == 1
        registry.get_runner.assert_awaited_once_with("composable-ai-assistant", reasoning_effort="high")
