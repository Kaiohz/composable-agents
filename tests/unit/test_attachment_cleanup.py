"""Unit tests for the shared post-turn temporary-document cleanup helper
(REQ-041..047 / NFR-017, Docker-free).

Covers every branch of :func:`run_temporary_document_cleanup`: the no-op
guards, the AWAITING_HITL skip (REQ-043), the success path, the plain failure
path (REQ-046: never raises, never masks the response) and the cancellation
path (EC-013: retry exactly once, unshielded).
"""

import asyncio

import pytest

from src.application.use_cases._attachment_cleanup import (
    resolve_cleanup_agent,
    run_temporary_document_cleanup,
)
from src.domain.entities.prepared_chat_turn import PreparedChatTurn, TemporaryDocumentRef
from src.domain.entities.thread import Thread


def _prepared() -> PreparedChatTurn:
    """A prepared turn carrying one temporary document (upload-time scheme)."""
    return PreparedChatTurn(
        temporary_documents=[
            TemporaryDocumentRef(
                attachment_id="att-1",
                filename="rapport.pdf",
                object_path="chat_tmp/u/t/att/rapport.pdf",
            )
        ],
        agent_name="test-agent",
    )


class _Cleaner:
    """Scripted cleanup port double."""

    def __init__(self, outcomes) -> None:  # noqa: ANN001
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def delete_temporary_documents(self, agent_name: str, refs) -> None:  # noqa: ANN001
        self.calls.append((agent_name, tuple(ref.object_path for ref in refs)))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return None


async def test_no_cleaner_is_a_no_op():
    assert await run_temporary_document_cleanup(None, "agent", "t", _prepared(), awaiting_hitl=False) is False


async def test_no_prepared_turn_is_a_no_op():
    cleaner = _Cleaner([None])
    assert await run_temporary_document_cleanup(cleaner, "agent", "t", None, awaiting_hitl=False) is False
    assert cleaner.calls == []


async def test_no_temporary_documents_is_a_no_op():
    cleaner = _Cleaner([None])
    assert await run_temporary_document_cleanup(cleaner, "agent", "t", PreparedChatTurn(), awaiting_hitl=False) is False
    assert cleaner.calls == []


async def test_awaiting_hitl_turn_is_skipped():
    cleaner = _Cleaner([None])
    assert await run_temporary_document_cleanup(cleaner, "agent", "t", _prepared(), awaiting_hitl=True) is False
    assert cleaner.calls == [], "REQ-043: an awaiting turn keeps its documents"


async def test_success_deletes_once_with_the_agent_resolved_from_the_prepared_turn():
    cleaner = _Cleaner([None])
    assert await run_temporary_document_cleanup(cleaner, "test-agent", "t", _prepared(), awaiting_hitl=False) is True
    assert cleaner.calls == [("test-agent", ("chat_tmp/u/t/att/rapport.pdf",))]


async def test_plain_failure_never_raises():
    cleaner = _Cleaner([RuntimeError("MCP down")])
    assert await run_temporary_document_cleanup(cleaner, "agent", "t", _prepared(), awaiting_hitl=False) is False


async def test_cancellation_retries_once_and_succeeds():
    """EC-013 + S7497: cancelled → ONE unshielded retry runs, then the raise."""
    cleaner = _Cleaner([asyncio.CancelledError(), None])
    with pytest.raises(asyncio.CancelledError):
        await run_temporary_document_cleanup(cleaner, "agent", "t", _prepared(), awaiting_hitl=False)
    assert len(cleaner.calls) == 2, "exactly one unshielded retry after the cancellation (EC-013)"


async def test_cancellation_retry_failing_still_reraises_the_cancellation():
    """EC-013 + S7497: the retry failure stays logged-only; cancellation raises."""
    cleaner = _Cleaner([asyncio.CancelledError(), RuntimeError("retry failed too")])
    with pytest.raises(asyncio.CancelledError):
        await run_temporary_document_cleanup(cleaner, "agent", "t", _prepared(), awaiting_hitl=False)
    assert len(cleaner.calls) == 2, "the retry still ran exactly once"


async def test_resolve_cleanup_agent_prefers_the_prepared_turn():
    thread = Thread(id="t", agent_name="thread-agent")
    assert resolve_cleanup_agent(_prepared(), thread) == "test-agent"


async def test_resolve_cleanup_agent_falls_back_to_the_thread():
    thread = Thread(id="t", agent_name="thread-agent")
    assert resolve_cleanup_agent(None, thread) == "thread-agent"
    assert resolve_cleanup_agent(PreparedChatTurn(agent_name=""), thread) == "thread-agent"


async def test_resolve_cleanup_agent_handles_both_missing():
    assert resolve_cleanup_agent(None, None) == ""
