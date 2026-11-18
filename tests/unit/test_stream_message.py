"""Tests for StreamMessageUseCase (Ticket 3 rewrite).

The use case now depends on TraceEventRepository + the new runner API
``stream(thread_id, message, turn_id) -> AsyncIterator[TraceEvent]``.
Internal repositories are used for real; only the LLM runner is mocked.
"""

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.application.use_cases.stream_message import StreamMessageUseCase
from src.domain.entities.message import Message, MessageRole, MessageStatus
from src.domain.entities.prepared_chat_turn import PreparedChatTurn, TemporaryDocumentRef
from src.domain.entities.trace_event import TraceEvent, TraceEventType
from src.domain.errors.agent import AgentError
from src.domain.errors.thread import ThreadNotFoundError
from src.domain.ports.agent_registry import AgentRegistry
from src.domain.ports.agent_runner import AgentRunner


class _FakeRegistry(AgentRegistry):
    """Real in-test registry that returns a single runner."""

    def __init__(self, runner: AgentRunner) -> None:
        self._runner = runner

    async def get_runner(self, agent_name: str) -> AgentRunner:  # noqa: ARG002
        return self._runner

    async def list_agents(self) -> list[str]:
        return ["test-agent"]

    async def invalidate(self, agent_name: str) -> None:  # noqa: ARG002
        pass

    async def close(self) -> None:
        pass


def _event(thread_id: str, turn_id: str, type_: TraceEventType, content: str, seq: int) -> TraceEvent:
    return TraceEvent(
        id=str(uuid4()),
        thread_id=thread_id,
        turn_id=turn_id,
        type=type_,
        content=content,
        timestamp=datetime.now(UTC),
        sequence=seq,
    )


class TestStreamMessageUseCase:
    @pytest.fixture
    def registry(self, mock_agent_runner):
        return _FakeRegistry(mock_agent_runner)

    @pytest.fixture
    def use_case(self, registry, thread_repo, trace_repo):
        return StreamMessageUseCase(registry, thread_repo, trace_repo)

    async def test_execute_yields_events_and_persists(self, use_case, mock_agent_runner, thread_repo, trace_repo):
        # Arrange
        thread = await thread_repo.create("test-agent")
        final_msg = Message(role=MessageRole.AI, content="Hi!", status=MessageStatus.COMPLETED)
        events = [
            _event(thread.id, "turn-1", TraceEventType.HUMAN_MESSAGE, "Hello", seq=0),
            _event(thread.id, "turn-1", TraceEventType.THINKING, "hmm", seq=1),
            _event(thread.id, "turn-1", TraceEventType.CONTENT, "Hi", seq=2),
            _event(thread.id, "turn-1", TraceEventType.AI_MESSAGE, final_msg.model_dump_json(), seq=3),
        ]

        async def _stream(_tid: str, _message: str, _turn_id: str) -> AsyncIterator[TraceEvent]:
            for ev in events:
                yield ev

        mock_agent_runner.stream = _stream

        # Act
        yielded: list[TraceEvent] = [ev async for ev in use_case.execute(thread.id, "Hello")]

        # Assert — all events yielded
        assert len(yielded) == 4
        assert [e.type for e in yielded] == [
            TraceEventType.HUMAN_MESSAGE,
            TraceEventType.THINKING,
            TraceEventType.CONTENT,
            TraceEventType.AI_MESSAGE,
        ]
        # And all events persisted via trace_repo.add
        persisted = await trace_repo.list_by_thread(thread.id)
        assert len(persisted) == 4

    async def test_execute_persists_human_message(self, use_case, mock_agent_runner, thread_repo, trace_repo):
        # Arrange
        thread = await thread_repo.create("test-agent")

        async def _stream(_tid: str, _message: str, _turn_id: str) -> AsyncIterator[TraceEvent]:
            yield _event(thread.id, _turn_id, TraceEventType.HUMAN_MESSAGE, "Hello", seq=0)
            yield _event(
                thread.id,
                _turn_id,
                TraceEventType.AI_MESSAGE,
                Message(role=MessageRole.AI, content="Hi!", status=MessageStatus.COMPLETED).model_dump_json(),
                seq=1,
            )

        mock_agent_runner.stream = _stream

        # Act
        _ = [ev async for ev in use_case.execute(thread.id, "Hello")]

        # Assert — HUMAN_MESSAGE is persisted
        persisted = await trace_repo.list_by_thread(thread.id)
        human_events = [e for e in persisted if e.type == TraceEventType.HUMAN_MESSAGE]
        assert len(human_events) == 1
        assert human_events[0].content == "Hello"

    async def test_execute_persists_ai_message(self, use_case, mock_agent_runner, thread_repo, trace_repo):
        # Arrange
        thread = await thread_repo.create("test-agent")
        final_msg = Message(role=MessageRole.AI, content="Final answer.", status=MessageStatus.COMPLETED)

        async def _stream(_tid: str, _message: str, _turn_id: str) -> AsyncIterator[TraceEvent]:
            yield _event(thread.id, _turn_id, TraceEventType.HUMAN_MESSAGE, "Hello", seq=0)
            yield _event(thread.id, _turn_id, TraceEventType.AI_MESSAGE, final_msg.model_dump_json(), seq=1)

        mock_agent_runner.stream = _stream

        # Act
        _ = [ev async for ev in use_case.execute(thread.id, "Hello")]

        # Assert — AI_MESSAGE is persisted and its content is valid JSON (the Message payload)
        persisted = await trace_repo.list_by_thread(thread.id)
        ai_events = [e for e in persisted if e.type == TraceEventType.AI_MESSAGE]
        assert len(ai_events) == 1
        payload = json.loads(ai_events[0].content)
        assert payload["content"] == "Final answer."

    async def test_execute_propagates_runner_error(self, use_case, mock_agent_runner, thread_repo):
        # Arrange
        thread = await thread_repo.create("test-agent")

        async def _stream(_tid: str, _message: str, _turn_id: str) -> AsyncIterator[TraceEvent]:
            yield _event(thread.id, _turn_id, TraceEventType.HUMAN_MESSAGE, "Hello", seq=0)
            raise AgentError("graph crashed")

        mock_agent_runner.stream = _stream

        # Act / Assert — runner error propagates out of the async generator
        with pytest.raises(AgentError, match="graph crashed"):
            _ = [ev async for ev in use_case.execute(thread.id, "Hello")]


# ---------------------------------------------------------------------------
# C1 stream twin — no-orphan cleanup on ANY failure after preparation
# ---------------------------------------------------------------------------


class _RecordingCleaner:
    """Collects the temporary-document deletions (port boundary double)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def delete_temporary_documents(self, agent_name: str, refs) -> None:  # noqa: ANN001
        self.calls.append((agent_name, tuple(ref.object_path for ref in refs)))


class _FailingRunnerRegistry(AgentRegistry):
    """Registry whose runner build fails (get_runner raises)."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def get_runner(self, agent_name: str) -> AgentRunner:  # noqa: ARG002
        raise self._error

    async def list_agents(self) -> list[str]:
        return []

    async def invalidate(self, agent_name: str) -> None:  # noqa: ARG002
        pass

    async def close(self) -> None:
        pass


class _VanishingThreadRepo:
    """Delegates to a real repo but reports the thread as not found."""

    def __init__(self, real) -> None:  # noqa: ANN001
        self._real = real

    async def get(self, thread_id: str):
        raise ThreadNotFoundError(f"Thread not found: {thread_id}")

    def __getattr__(self, name: str):  # noqa: ANN204
        return getattr(self._real, name)


def _prepared_document_turn(agent_name: str) -> PreparedChatTurn:
    """A prepared turn carrying one temporary document (upload-time scheme)."""
    return PreparedChatTurn(
        text=None,
        content_blocks=[
            {
                "type": "text",
                "text": "A document named 'rapport.pdf' is available at chat_tmp/u/t/att/rapport.pdf. "
                "Use the read_file tool from the Files MCP server.",
            }
        ],
        attachments_metadata=[
            {
                "id": "att-1",
                "filename": "rapport.pdf",
                "content_type": "application/pdf",
                "size_bytes": 22,
                "kind": "document",
                "source": "rustfs_temp",
            }
        ],
        temporary_documents=[
            TemporaryDocumentRef(
                attachment_id="att-1",
                filename="rapport.pdf",
                object_path="chat_tmp/u/t/att/rapport.pdf",
            )
        ],
        agent_name=agent_name,
    )


class TestStreamingTemporaryDocumentCleanup:
    """The streaming path guarantees the same cleanup as the invoke path."""

    async def test_prepared_documents_are_cleaned_when_the_runner_build_fails(self, thread_repo, trace_repo):
        """Runner BUILD failure → the prepared temp document is NOT stranded."""
        cleaner = _RecordingCleaner()
        use_case = StreamMessageUseCase(
            _FailingRunnerRegistry(AgentError("runner build failed")),
            thread_repo,
            trace_repo,
            attachments=object(),
            cleaner=cleaner,
        )
        thread = await thread_repo.create("test-agent")

        with pytest.raises(AgentError):
            async for _ in use_case.execute(thread.id, "Analyse", _prepared_document_turn("test-agent")):
                pass

        assert cleaner.calls == [("test-agent", ("chat_tmp/u/t/att/rapport.pdf",))]

    async def test_prepared_documents_are_cleaned_when_the_thread_lookup_fails(self, thread_repo, trace_repo):
        """Even a vanished thread cleans up via prepared.agent_name (no mask)."""
        cleaner = _RecordingCleaner()
        use_case = StreamMessageUseCase(
            _FakeRegistry(__import__("unittest").mock.AsyncMock()),
            _VanishingThreadRepo(thread_repo),
            trace_repo,
            attachments=object(),
            cleaner=cleaner,
        )

        with pytest.raises(ThreadNotFoundError):
            async for _ in use_case.execute("missing-thread", "Analyse", _prepared_document_turn("test-agent")):
                pass

        assert cleaner.calls == [("test-agent", ("chat_tmp/u/t/att/rapport.pdf",))]

    async def test_stream_failure_with_a_prepared_turn_cleans_up(self, mock_agent_runner, thread_repo, trace_repo):
        """A mid-stream runner failure still cleans the prepared document."""
        cleaner = _RecordingCleaner()
        registry = _FakeRegistry(mock_agent_runner)
        use_case = StreamMessageUseCase(registry, thread_repo, trace_repo, attachments=object(), cleaner=cleaner)
        thread = await thread_repo.create("test-agent")

        async def _stream(_tid: str, _turn_input: str, _turn_id: str) -> AsyncIterator[TraceEvent]:
            yield _event(thread.id, _turn_id, TraceEventType.HUMAN_MESSAGE, "Analyse", seq=0)
            raise AgentError("graph crashed")

        mock_agent_runner.stream = _stream

        with pytest.raises(AgentError):
            async for _ in use_case.execute(thread.id, "Analyse", _prepared_document_turn("test-agent")):
                pass

        assert cleaner.calls == [("test-agent", ("chat_tmp/u/t/att/rapport.pdf",))]

    async def test_awaiting_hitl_stream_keeps_the_documents(self, mock_agent_runner, thread_repo, trace_repo):
        """REQ-043: a stream ending in AWAITING_HITL keeps the prepared docs."""
        cleaner = _RecordingCleaner()
        use_case = StreamMessageUseCase(
            _FakeRegistry(mock_agent_runner), thread_repo, trace_repo, attachments=object(), cleaner=cleaner
        )
        thread = await thread_repo.create("test-agent")
        final_message = Message(role=MessageRole.AI, content="?", status=MessageStatus.AWAITING_HITL)

        async def _stream(_tid: str, _turn_input: str, _turn_id: str) -> AsyncIterator[TraceEvent]:
            yield _event(thread.id, _turn_id, TraceEventType.HUMAN_MESSAGE, "Analyse", seq=0)
            yield _event(thread.id, _turn_id, TraceEventType.AI_MESSAGE, final_message.model_dump_json(), seq=1)

        mock_agent_runner.stream = _stream
        prepared = _prepared_document_turn("test-agent")

        events = [ev async for ev in use_case.execute(thread.id, "Analyse", prepared)]

        assert events[-1].type == TraceEventType.AI_MESSAGE
        assert cleaner.calls == [], "the document must survive an AWAITING_HITL stream"
