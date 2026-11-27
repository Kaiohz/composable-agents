"""Tests for SendMessageUseCase (HITL refactor — TDD red phase).

The use case now depends on TraceEventRepository + the new runner API
``invoke(thread_id, message, turn_id) -> (Message, list[TraceEvent])`` and the
unified HITL resume method
``resume_hitl(thread_id, decisions, turn_id) -> (Message, list[TraceEvent])``.

Internal repositories (PostgresThreadRepository, PostgresTraceEventRepository)
are used for real; only the LLM runner (AgentRunner) is mocked at the port
boundary.
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.application.use_cases.send_message import SendMessageUseCase
from src.domain.entities.hitl_decision import HitlDecision
from src.domain.entities.message import Message, MessageRole, MessageStatus
from src.domain.entities.prepared_chat_turn import PreparedChatTurn, TemporaryDocumentRef
from src.domain.entities.trace_event import TraceEvent, TraceEventType
from src.domain.errors.agent import AgentError
from src.domain.errors.hitl import InvalidHitlActionError
from src.domain.errors.thread import ThreadNotFoundError
from src.domain.ports.agent_registry import AgentRegistry
from src.domain.ports.agent_runner import AgentRunner


class _FakeRegistry(AgentRegistry):
    """Real in-test registry that returns a single runner."""

    def __init__(self, runner: AgentRunner) -> None:
        self._runner = runner

    async def get_runner(self, agent_name: str, reasoning_effort: str | None = None) -> AgentRunner:  # noqa: ARG002
        return self._runner

    async def list_agents(self) -> list[str]:
        return ["test-agent"]

    async def invalidate(self, agent_name: str) -> None:  # noqa: ARG002
        pass

    async def close(self) -> None:
        pass


def _human_event(thread_id: str, turn_id: str, content: str, seq: int = 0) -> TraceEvent:
    return TraceEvent(
        id=str(uuid4()),
        thread_id=thread_id,
        turn_id=turn_id,
        type=TraceEventType.HUMAN_MESSAGE,
        content=content,
        timestamp=datetime.now(UTC),
        sequence=seq,
    )


def _ai_event(thread_id: str, turn_id: str, message: Message, seq: int = 1) -> TraceEvent:
    return TraceEvent(
        id=str(uuid4()),
        thread_id=thread_id,
        turn_id=turn_id,
        type=TraceEventType.AI_MESSAGE,
        content=message.model_dump_json(),
        timestamp=datetime.now(UTC),
        sequence=seq,
    )


def _hitl_decision_event(thread_id: str, turn_id: str, name: str, content: str, seq: int) -> TraceEvent:
    """Build a HITL_DECISION trace event emitted by the runner on resume."""
    return TraceEvent(
        id=str(uuid4()),
        thread_id=thread_id,
        turn_id=turn_id,
        type=TraceEventType.HITL_DECISION,
        name=name,
        content=content,
        timestamp=datetime.now(UTC),
        sequence=seq,
    )


class TestSendMessageUseCase:
    @pytest.fixture
    def registry(self, mock_agent_runner):
        return _FakeRegistry(mock_agent_runner)

    @pytest.fixture
    def use_case(self, registry, thread_repo, trace_repo):
        return SendMessageUseCase(registry, thread_repo, trace_repo)

    async def test_unsupported_hitl_action_raises(self, use_case, thread_repo):
        # Arrange
        thread = await thread_repo.create("test-agent")

        # Act / Assert
        with pytest.raises(InvalidHitlActionError, match="Unsupported HITL action"):
            await use_case.execute(thread.id, action="unknown_action", tool_call_id="tc-1")

    async def test_sends_message_and_persists_trace(self, use_case, mock_agent_runner, thread_repo, trace_repo):
        # Arrange
        thread = await thread_repo.create("test-agent")
        final_message = Message(role=MessageRole.AI, content="Hello human!", status=MessageStatus.COMPLETED)
        # Runner.invoke returns (Message, list[TraceEvent]).
        # Provide a minimal trace: HUMAN_MESSAGE + AI_MESSAGE.
        turn_id = "turn-returned-by-runner"  # the use case generates its own; runner just echoes
        trace = [
            _human_event(thread.id, turn_id, "Hello agent!", seq=0),
            _ai_event(thread.id, turn_id, final_message, seq=1),
        ]
        mock_agent_runner.invoke.return_value = (final_message, trace)

        # Act
        result = await use_case.execute(thread.id, message="Hello agent!")

        # Assert — returns the final AI Message
        assert result.role == MessageRole.AI
        assert result.content == "Hello human!"
        assert result.status == MessageStatus.COMPLETED
        # And persists the trace events in a batch
        mock_agent_runner.invoke.assert_awaited_once()
        # Verify trace events are now in the repository
        events = await trace_repo.list_by_thread(thread.id)
        assert len(events) == 2
        assert events[0].type == TraceEventType.HUMAN_MESSAGE
        assert events[1].type == TraceEventType.AI_MESSAGE

    async def test_each_call_generates_new_turn_id(self, use_case, mock_agent_runner, thread_repo, trace_repo):
        # Arrange — two consecutive calls must produce two distinct turns
        thread = await thread_repo.create("test-agent")
        msg1 = Message(role=MessageRole.AI, content="first", status=MessageStatus.COMPLETED)
        msg2 = Message(role=MessageRole.AI, content="second", status=MessageStatus.COMPLETED)

        # Capture the turn_id passed to the runner across two calls
        captured_turn_ids: list[str] = []

        async def _invoke(_tid: str, _message: str, turn_id: str) -> tuple[Message, list[TraceEvent]]:
            captured_turn_ids.append(turn_id)
            trace = [_human_event(_tid, turn_id, _message, seq=0), _ai_event(_tid, turn_id, msg1, seq=1)]
            return (msg1, trace)

        mock_agent_runner.invoke.side_effect = _invoke
        mock_agent_runner.invoke.return_value = None  # clear the default to use side_effect

        await use_case.execute(thread.id, message="q1")

        # Reset return for second call with msg2
        async def _invoke2(_tid: str, _message: str, turn_id: str) -> tuple[Message, list[TraceEvent]]:
            captured_turn_ids.append(turn_id)
            trace = [_human_event(_tid, turn_id, _message, seq=0), _ai_event(_tid, turn_id, msg2, seq=1)]
            return (msg2, trace)

        mock_agent_runner.invoke.side_effect = _invoke2
        await use_case.execute(thread.id, message="q2")

        # Assert — two distinct turn_ids generated by the use case
        assert len(captured_turn_ids) == 2
        assert captured_turn_ids[0] != captured_turn_ids[1]
        # And 4 trace events persisted (2 per turn)
        events = await trace_repo.list_by_thread(thread.id)
        assert len(events) == 4

    # ------------------------------------------------------------------ #
    # NEW HITL resume contract (decisions-based)
    # ------------------------------------------------------------------ #

    async def test_resume_hitl_persists_trace(self, use_case, mock_agent_runner, thread_repo, trace_repo):
        # Arrange
        thread = await thread_repo.create("test-agent")
        final_message = Message(role=MessageRole.AI, content="Resumed after approval.", status=MessageStatus.COMPLETED)
        decisions = [HitlDecision(tool_call_id="tc-1", action="approve")]

        captured_turn_ids: list[str] = []

        async def _resume(_tid: str, _decisions, turn_id: str):
            captured_turn_ids.append(turn_id)
            hitl_decision_event = _hitl_decision_event(_tid, turn_id, "approve", "tc-1", seq=0)
            ai_event = _ai_event(_tid, turn_id, final_message, seq=1)
            return final_message, [hitl_decision_event, ai_event]

        mock_agent_runner.resume_hitl.side_effect = _resume

        # Act
        result = await use_case.execute(thread.id, decisions=decisions)

        # Assert — resume_hitl awaited once with (thread.id, decisions, turn_id)
        mock_agent_runner.resume_hitl.assert_awaited_once()
        assert mock_agent_runner.resume_hitl.await_args.args[0] == thread.id
        assert mock_agent_runner.resume_hitl.await_args.args[1] == decisions
        # The use case generated a turn_id (non-empty, captured)
        assert len(captured_turn_ids) == 1
        assert captured_turn_ids[0] and isinstance(captured_turn_ids[0], str)
        # And the final message is returned
        assert result == final_message
        # And trace events are now persisted (HITL_DECISION then AI_MESSAGE)
        events = await trace_repo.list_by_thread(thread.id)
        assert len(events) == 2
        assert events[0].type == TraceEventType.HITL_DECISION
        assert events[1].type == TraceEventType.AI_MESSAGE

    async def test_resume_hitl_generates_turn_id(self, use_case, mock_agent_runner, thread_repo, trace_repo):
        # Arrange — two consecutive resume calls must produce two distinct turns
        thread = await thread_repo.create("test-agent")
        msg1 = Message(role=MessageRole.AI, content="first", status=MessageStatus.COMPLETED)
        msg2 = Message(role=MessageRole.AI, content="second", status=MessageStatus.COMPLETED)

        captured_turn_ids: list[str] = []

        async def _resume1(_tid: str, _decisions, turn_id: str):
            captured_turn_ids.append(turn_id)
            return msg1, [
                _hitl_decision_event(_tid, turn_id, "approve", "tc-1", seq=0),
                _ai_event(_tid, turn_id, msg1, seq=1),
            ]

        mock_agent_runner.resume_hitl.side_effect = _resume1
        await use_case.execute(thread.id, decisions=[HitlDecision(tool_call_id="tc-1", action="approve")])

        async def _resume2(_tid: str, _decisions, turn_id: str):
            captured_turn_ids.append(turn_id)
            return msg2, [
                _hitl_decision_event(_tid, turn_id, "approve", "tc-2", seq=0),
                _ai_event(_tid, turn_id, msg2, seq=1),
            ]

        mock_agent_runner.resume_hitl.side_effect = _resume2
        await use_case.execute(thread.id, decisions=[HitlDecision(tool_call_id="tc-2", action="approve")])

        # Assert — two distinct turn_ids generated by the use case
        assert len(captured_turn_ids) == 2
        assert captured_turn_ids[0] != captured_turn_ids[1]
        # And 4 trace events persisted (2 per turn)
        events = await trace_repo.list_by_thread(thread.id)
        assert len(events) == 4

    async def test_legacy_single_approve_converted_to_decisions(
        self, use_case, mock_agent_runner, thread_repo, trace_repo
    ):
        # Arrange
        thread = await thread_repo.create("test-agent")
        final_message = Message(role=MessageRole.AI, content="Action approved.", status=MessageStatus.COMPLETED)
        mock_agent_runner.resume_hitl.return_value = (
            final_message,
            [
                _hitl_decision_event(thread.id, "turn-x", "approve", "tc-1", seq=0),
                _ai_event(thread.id, "turn-x", final_message, seq=1),
            ],
        )

        # Act — legacy single-decision shape
        result = await use_case.execute(thread.id, action="approve", tool_call_id="tc-1")

        # Assert — resume_hitl awaited with a 1-element decisions list
        mock_agent_runner.resume_hitl.assert_awaited_once()
        args = mock_agent_runner.resume_hitl.await_args.args
        assert args[0] == thread.id
        decisions = args[1]
        assert isinstance(decisions, list)
        assert len(decisions) == 1
        assert isinstance(decisions[0], HitlDecision)
        assert decisions[0].tool_call_id == "tc-1"
        assert decisions[0].action == "approve"
        # And the final message is returned
        assert result == final_message
        # And trace events are now persisted (HITL_DECISION then AI_MESSAGE)
        events = await trace_repo.list_by_thread(thread.id)
        assert len(events) == 2
        assert events[0].type == TraceEventType.HITL_DECISION
        assert events[1].type == TraceEventType.AI_MESSAGE

    async def test_legacy_single_reject_converted_to_decisions(
        self, use_case, mock_agent_runner, thread_repo, trace_repo
    ):
        # Arrange
        thread = await thread_repo.create("test-agent")
        final_message = Message(
            role=MessageRole.AI, content="Action rejected: Too risky", status=MessageStatus.COMPLETED
        )
        mock_agent_runner.resume_hitl.return_value = (
            final_message,
            [
                _hitl_decision_event(thread.id, "turn-y", "reject", "Too risky", seq=0),
                _ai_event(thread.id, "turn-y", final_message, seq=1),
            ],
        )

        # Act — legacy single-decision shape
        result = await use_case.execute(thread.id, action="reject", tool_call_id="tc-1", reason="Too risky")

        # Assert — resume_hitl awaited with a 1-element decisions list
        mock_agent_runner.resume_hitl.assert_awaited_once()
        args = mock_agent_runner.resume_hitl.await_args.args
        assert args[0] == thread.id
        decisions = args[1]
        assert isinstance(decisions, list)
        assert len(decisions) == 1
        assert isinstance(decisions[0], HitlDecision)
        assert decisions[0].tool_call_id == "tc-1"
        assert decisions[0].action == "reject"
        assert decisions[0].reason == "Too risky"
        # And the final message is returned
        assert result == final_message
        # And trace events are now persisted
        events = await trace_repo.list_by_thread(thread.id)
        assert len(events) == 2
        assert events[0].type == TraceEventType.HITL_DECISION
        assert events[1].type == TraceEventType.AI_MESSAGE

    async def test_legacy_single_edit_converted_to_decisions(
        self, use_case, mock_agent_runner, thread_repo, trace_repo
    ):
        # Arrange
        thread = await thread_repo.create("test-agent")
        final_message = Message(
            role=MessageRole.AI, content="Action edited and approved.", status=MessageStatus.COMPLETED
        )
        mock_agent_runner.resume_hitl.return_value = (
            final_message,
            [
                _hitl_decision_event(thread.id, "turn-z", "edit", "edited", seq=0),
                _ai_event(thread.id, "turn-z", final_message, seq=1),
            ],
        )

        # Act — legacy single-decision shape
        result = await use_case.execute(thread.id, action="edit", tool_call_id="tc-1", edits={"param": "value"})

        # Assert — resume_hitl awaited with a 1-element decisions list
        mock_agent_runner.resume_hitl.assert_awaited_once()
        args = mock_agent_runner.resume_hitl.await_args.args
        assert args[0] == thread.id
        decisions = args[1]
        assert isinstance(decisions, list)
        assert len(decisions) == 1
        assert isinstance(decisions[0], HitlDecision)
        assert decisions[0].tool_call_id == "tc-1"
        assert decisions[0].action == "edit"
        assert decisions[0].edits == {"param": "value"}
        # And the final message is returned
        assert result == final_message
        # And trace events are now persisted
        events = await trace_repo.list_by_thread(thread.id)
        assert len(events) == 2
        assert events[0].type == TraceEventType.HITL_DECISION
        assert events[1].type == TraceEventType.AI_MESSAGE

    async def test_resume_hitl_runner_error_propagates(self, use_case, mock_agent_runner, thread_repo):
        # Arrange
        thread = await thread_repo.create("test-agent")
        mock_agent_runner.resume_hitl.side_effect = AgentError("Backend failed")

        # Act / Assert
        with pytest.raises(AgentError, match="Backend failed"):
            await use_case.execute(thread.id, decisions=[HitlDecision(tool_call_id="tc-1", action="approve")])

    async def test_runner_error_propagates(self, use_case, mock_agent_runner, thread_repo):
        # Arrange
        thread = await thread_repo.create("test-agent")
        mock_agent_runner.invoke.side_effect = AgentError("Backend failed")
        mock_agent_runner.invoke.return_value = None

        # Act / Assert
        with pytest.raises(AgentError, match="Backend failed"):
            await use_case.execute(thread.id, message="Hello")


# --------------------------------------------------------------------------- #
# C1 — message-less attachment turns (REQ-004 / EDG-001 / EDG-002) + no-orphan
# --------------------------------------------------------------------------- #


class _RecordingCleaner:
    """Collects the temporary-document deletions (port boundary double)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def delete_temporary_documents(self, agent_name: str, refs) -> None:  # noqa: ANN001
        self.calls.append((agent_name, tuple(ref.object_path for ref in refs)))


@pytest.fixture
def cleaner() -> _RecordingCleaner:
    """The cleanup port double, shared by the C1 and C5 test classes."""
    return _RecordingCleaner()


class _FailingRunnerRegistry(AgentRegistry):
    """Registry whose runner build fails (get_runner raises)."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def get_runner(self, agent_name: str, reasoning_effort: str | None = None) -> AgentRunner:  # noqa: ARG002
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
    """A message-less prepared turn carrying one temporary document."""
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


class TestMessagelessAttachmentTurns:
    """C1: HITL is derived from the HITL FIELDS, never from ``message is None``."""

    @pytest.fixture
    def prepared(self) -> PreparedChatTurn:
        return _prepared_document_turn("test-agent")

    def _use_case(self, mock_agent_runner, thread_repo, trace_repo, cleaner):
        return SendMessageUseCase(
            _FakeRegistry(mock_agent_runner), thread_repo, trace_repo, attachments=object(), cleaner=cleaner
        )

    async def test_messageless_attachment_turn_reaches_the_runner(
        self, mock_agent_runner, thread_repo, trace_repo, prepared, cleaner
    ):
        """EDG-001: ``message`` absent + attachments → human turn, NOT HTTP 400."""
        thread = await thread_repo.create("test-agent")
        final_message = Message(role=MessageRole.AI, content="stub-ok", status=MessageStatus.COMPLETED)
        mock_agent_runner.invoke.return_value = (
            final_message,
            [_human_event(thread.id, "turn-x", None, seq=0), _ai_event(thread.id, "turn-x", final_message, seq=1)],
        )
        use_case = self._use_case(mock_agent_runner, thread_repo, trace_repo, cleaner)

        result = await use_case.execute(thread.id, message="", prepared=prepared)

        mock_agent_runner.invoke.assert_awaited_once()
        turn_input = mock_agent_runner.invoke.await_args.args[1]
        # The runner receives the multimodal content blocks with NO user text.
        assert turn_input.text is None
        assert turn_input.content_blocks == prepared.content_blocks
        assert result == final_message

    async def test_messageless_turn_via_none_message_reaches_the_runner_too(
        self, mock_agent_runner, thread_repo, trace_repo, prepared, cleaner
    ):
        """EDG-002: document-only turns work even when ``message`` stays None."""
        thread = await thread_repo.create("test-agent")
        final_message = Message(role=MessageRole.AI, content="stub-ok", status=MessageStatus.COMPLETED)
        mock_agent_runner.invoke.return_value = (
            final_message,
            [_human_event(thread.id, "turn-x", None, seq=0), _ai_event(thread.id, "turn-x", final_message, seq=1)],
        )
        use_case = self._use_case(mock_agent_runner, thread_repo, trace_repo, cleaner)

        result = await use_case.execute(thread.id, message=None, prepared=prepared)

        assert result == final_message
        mock_agent_runner.invoke.assert_awaited_once()

    async def test_no_message_and_no_attachments_still_rejects(
        self, mock_agent_runner, thread_repo, trace_repo, cleaner
    ):
        """The legacy contract stands: nothing to send → InvalidHitlActionError."""
        thread = await thread_repo.create("test-agent")
        use_case = self._use_case(mock_agent_runner, thread_repo, trace_repo, cleaner)

        with pytest.raises(InvalidHitlActionError):
            await use_case.execute(thread.id, message=None, prepared=None)

        mock_agent_runner.invoke.assert_not_awaited()
        mock_agent_runner.resume_hitl.assert_not_awaited()

    async def test_prepared_documents_are_cleaned_when_the_runner_build_fails(
        self, thread_repo, trace_repo, prepared, cleaner
    ):
        """C1 no-orphan: a get_runner failure cleans up the uploaded document."""
        thread = await thread_repo.create("test-agent")
        registry = _FailingRunnerRegistry(AgentError("runner build failed"))
        use_case = SendMessageUseCase(registry, thread_repo, trace_repo, attachments=object(), cleaner=cleaner)

        with pytest.raises(AgentError):
            await use_case.execute(thread.id, message="Analyse", prepared=prepared)

        assert cleaner.calls == [("test-agent", ("chat_tmp/u/t/att/rapport.pdf",))]

    async def test_prepared_documents_are_cleaned_when_the_thread_lookup_fails(
        self, thread_repo, trace_repo, prepared, cleaner
    ):
        """C1 no-orphan: even a vanished thread cleans up via prepared.agent_name."""
        use_case = SendMessageUseCase(
            _FakeRegistry(__import__("unittest").mock.AsyncMock()),
            _VanishingThreadRepo(thread_repo),
            trace_repo,
            attachments=object(),
            cleaner=cleaner,
        )

        with pytest.raises(ThreadNotFoundError):
            await use_case.execute("missing-thread", message="Analyse", prepared=prepared)

        assert cleaner.calls == [("test-agent", ("chat_tmp/u/t/att/rapport.pdf",))]

    async def test_prepared_documents_are_cleaned_when_the_graph_fails(
        self, mock_agent_runner, thread_repo, trace_repo, prepared, cleaner
    ):
        """REQ-042: a graph error triggers the cleanup in the outer finally."""
        thread = await thread_repo.create("test-agent")
        mock_agent_runner.invoke.side_effect = AgentError("graph exploded")
        use_case = self._use_case(mock_agent_runner, thread_repo, trace_repo, cleaner)

        with pytest.raises(AgentError):
            await use_case.execute(thread.id, message="Analyse", prepared=prepared)

        assert cleaner.calls == [("test-agent", ("chat_tmp/u/t/att/rapport.pdf",))]

    async def test_awaiting_hitl_human_turn_keeps_the_documents(
        self, mock_agent_runner, thread_repo, trace_repo, prepared, cleaner
    ):
        """REQ-043: a human-attachment turn ending in AWAITING_HITL keeps its docs."""
        thread = await thread_repo.create("test-agent")
        final_message = Message(role=MessageRole.AI, content="?", status=MessageStatus.AWAITING_HITL)
        mock_agent_runner.invoke.return_value = (
            final_message,
            [_human_event(thread.id, "turn-x", None, seq=0), _ai_event(thread.id, "turn-x", final_message, seq=1)],
        )
        use_case = self._use_case(mock_agent_runner, thread_repo, trace_repo, cleaner)

        result = await use_case.execute(thread.id, message="", prepared=prepared)

        assert result.status == MessageStatus.AWAITING_HITL
        assert cleaner.calls == []

    async def test_awaiting_hitl_resume_keeps_the_original_documents(
        self, mock_agent_runner, thread_repo, trace_repo, cleaner
    ):
        """REQ-043: a resumed flow ending in AWAITING_HITL keeps the docs."""
        thread = await thread_repo.create("test-agent")
        final_message = Message(role=MessageRole.AI, content="?", status=MessageStatus.AWAITING_HITL)
        mock_agent_runner.resume_hitl.return_value = (
            final_message,
            [_hitl_decision_event(thread.id, "turn-x", "approve", "tc-1", seq=0)],
        )
        use_case = self._use_case(mock_agent_runner, thread_repo, trace_repo, cleaner)

        await use_case.execute(thread.id, decisions=[HitlDecision(tool_call_id="tc-1", action="approve")])

        assert cleaner.calls == []

    async def test_failed_resume_does_not_delete_the_pending_documents(
        self, mock_agent_runner, thread_repo, trace_repo, cleaner
    ):
        """REQ-043: a FAILED resume leaves the turn pending — documents survive."""
        thread = await thread_repo.create("test-agent")
        mock_agent_runner.resume_hitl.side_effect = AgentError("resume failed")
        use_case = self._use_case(mock_agent_runner, thread_repo, trace_repo, cleaner)

        with pytest.raises(AgentError):
            await use_case.execute(thread.id, decisions=[HitlDecision(tool_call_id="tc-1", action="approve")])

        assert cleaner.calls == []


class TestHitlResumeCleanup:
    """REQ-044 / AC-013 / NFR-017 (reprise HITL): the resumed flow's terminal
    response deletes the ORIGINAL turn's temporary documents — and the
    re-derived object path matches the path derived at upload time."""

    async def test_hitl_resume_deletes_the_original_turn_documents(
        self, mock_agent_runner, thread_repo, trace_repo, cleaner
    ):
        from src.infrastructure.database.rls_context import current_user_id

        token = current_user_id.set("user-a")
        try:
            thread = await thread_repo.create("test-agent")
            # The ORIGINAL human turn carried a temporary document; REQ-043
            # kept it alive while the turn was awaiting HITL.
            await trace_repo.add(
                thread.id,
                TraceEvent(
                    id=str(uuid4()),
                    thread_id=thread.id,
                    turn_id=str(uuid4()),
                    type=TraceEventType.HUMAN_MESSAGE,
                    content="Analyse",
                    metadata={
                        "attachments": [
                            {
                                "id": "att-1",
                                "filename": "rapport.pdf",
                                "content_type": "application/pdf",
                                "size_bytes": 22,
                                "kind": "document",
                                "source": "rustfs_temp",
                            }
                        ]
                    },
                    timestamp=datetime.now(UTC),
                    sequence=0,
                ),
            )
            final_message = Message(role=MessageRole.AI, content="done", status=MessageStatus.COMPLETED)
            mock_agent_runner.resume_hitl.return_value = (final_message, [])

            use_case = SendMessageUseCase(
                _FakeRegistry(mock_agent_runner), thread_repo, trace_repo, attachments=object(), cleaner=cleaner
            )
            await use_case.execute(thread.id, decisions=[HitlDecision(tool_call_id="tc-1", action="approve")])
        finally:
            current_user_id.reset(token)

        assert cleaner.calls, "the original turn's document must be deleted after the HITL terminal response"
        agent_name, paths = cleaner.calls[0]
        # The derivation (owner + thread + attachment id + filename) matches
        # the upload-time scheme chat_tmp/<user>/<thread>/<id>/<filename>.
        assert agent_name == "test-agent"
        assert paths == (f"chat_tmp/user-a/{thread.id}/att-1/rapport.pdf",)

    def _doc_metadata(self) -> dict:
        """The attachment metadata of the ORIGINAL turn's temporary document."""
        return {
            "attachments": [
                {
                    "id": "att-1",
                    "filename": "rapport.pdf",
                    "content_type": "application/pdf",
                    "size_bytes": 22,
                    "kind": "document",
                    "source": "rustfs_temp",
                }
            ]
        }

    async def test_non_human_events_are_skipped(self, mock_agent_runner, thread_repo, trace_repo, cleaner):
        """Only HUMAN_MESSAGE metadata drives the derivation (other events skipped)."""
        from src.infrastructure.database.rls_context import current_user_id

        token = current_user_id.set("user-a")
        try:
            thread = await thread_repo.create("test-agent")
            await trace_repo.add(
                thread.id,
                TraceEvent(
                    id=str(uuid4()),
                    thread_id=thread.id,
                    turn_id=str(uuid4()),
                    type=TraceEventType.AI_MESSAGE,
                    content="intermediate",
                    metadata={"attachments": [{"id": "x", "kind": "document"}]},
                    timestamp=datetime.now(UTC),
                    sequence=0,
                ),
            )
            await trace_repo.add(
                thread.id,
                TraceEvent(
                    id=str(uuid4()),
                    thread_id=thread.id,
                    turn_id=str(uuid4()),
                    type=TraceEventType.HUMAN_MESSAGE,
                    content="Analyse",
                    metadata=self._doc_metadata(),
                    timestamp=datetime.now(UTC),
                    sequence=1,
                ),
            )
            use_case = SendMessageUseCase(
                _FakeRegistry(mock_agent_runner), thread_repo, trace_repo, attachments=object(), cleaner=cleaner
            )
            await use_case._cleanup_after_hitl_terminal(
                thread.id, "test-agent", Message(role=MessageRole.AI, status=MessageStatus.COMPLETED)
            )
        finally:
            current_user_id.reset(token)
        assert cleaner.calls and cleaner.calls[0][1] == (f"chat_tmp/user-a/{thread.id}/att-1/rapport.pdf",)

    async def test_no_owner_event_is_skipped_with_a_loud_warning(
        self, mock_agent_runner, thread_repo, trace_repo, cleaner
    ):
        """I1: an owner-less event skips the derivation WITHOUT a silent fallback.

        The thread (and therefore its trace events) carries NO user id and no
        auth context exists — a fabricated path would be invalid.
        """
        thread = await thread_repo.create("test-agent")
        await trace_repo.add(
            thread.id,
            TraceEvent(
                id=str(uuid4()),
                thread_id=thread.id,
                turn_id=str(uuid4()),
                type=TraceEventType.HUMAN_MESSAGE,
                content="Analyse",
                metadata=self._doc_metadata(),
                timestamp=datetime.now(UTC),
                sequence=0,
            ),
        )
        use_case = SendMessageUseCase(
            _FakeRegistry(mock_agent_runner), thread_repo, trace_repo, attachments=object(), cleaner=cleaner
        )
        await use_case._cleanup_after_hitl_terminal(
            thread.id, "test-agent", Message(role=MessageRole.AI, status=MessageStatus.COMPLETED)
        )
        assert cleaner.calls == [], "an owner-less event must not be deleted under a fabricated path"

    async def test_non_document_entries_are_skipped(self, mock_agent_runner, thread_repo, trace_repo, cleaner):
        """Only ``kind: document`` entries are derived — images live in the store."""
        thread = await thread_repo.create("test-agent")
        await trace_repo.add(
            thread.id,
            TraceEvent(
                id=str(uuid4()),
                thread_id=thread.id,
                turn_id=str(uuid4()),
                type=TraceEventType.HUMAN_MESSAGE,
                content="Analyse",
                metadata={
                    "attachments": [
                        {
                            "id": "att-9",
                            "filename": "capture.png",
                            "content_type": "image/png",
                            "size_bytes": 9,
                            "kind": "image",
                            "source": "langgraph_store",
                        }
                    ]
                },
                timestamp=datetime.now(UTC),
                sequence=0,
            ),
        )
        use_case = SendMessageUseCase(
            _FakeRegistry(mock_agent_runner), thread_repo, trace_repo, attachments=object(), cleaner=cleaner
        )
        await use_case._cleanup_after_hitl_terminal(
            thread.id, "test-agent", Message(role=MessageRole.AI, status=MessageStatus.COMPLETED)
        )
        assert cleaner.calls == [], "image attachments must never be deleted as temp docs"

    async def test_malformed_document_entries_are_logged_and_skipped(
        self, mock_agent_runner, thread_repo, trace_repo, cleaner
    ):
        """A malformed entry is a WARNING, never a crash and never a deletion."""
        thread = await thread_repo.create("test-agent")
        await trace_repo.add(
            thread.id,
            TraceEvent(
                id=str(uuid4()),
                thread_id=thread.id,
                turn_id=str(uuid4()),
                type=TraceEventType.HUMAN_MESSAGE,
                content="Analyse",
                metadata={"attachments": [{"kind": "document"}]},  # no id/filename → KeyError
                timestamp=datetime.now(UTC),
                sequence=0,
            ),
        )
        use_case = SendMessageUseCase(
            _FakeRegistry(mock_agent_runner), thread_repo, trace_repo, attachments=object(), cleaner=cleaner
        )
        await use_case._cleanup_after_hitl_terminal(
            thread.id, "test-agent", Message(role=MessageRole.AI, status=MessageStatus.COMPLETED)
        )
        assert cleaner.calls == []

    async def test_trace_scan_failure_never_crashes_the_cleanup(
        self, mock_agent_runner, thread_repo, trace_repo, cleaner
    ):
        """The trace scan failure is logged; the cleanup stays best-effort."""
        thread = await thread_repo.create("test-agent")

        class _BrokenTraceRepo:
            async def list_by_thread(self, _thread_id: str):
                raise RuntimeError("trace repository down")

            def __getattr__(self, name: str):  # noqa: ANN204
                return getattr(trace_repo, name)

        use_case = SendMessageUseCase(
            _FakeRegistry(mock_agent_runner), thread_repo, _BrokenTraceRepo(), attachments=object(), cleaner=cleaner
        )
        await use_case._cleanup_after_hitl_terminal(
            thread.id, "test-agent", Message(role=MessageRole.AI, status=MessageStatus.COMPLETED)
        )
        assert cleaner.calls == []
