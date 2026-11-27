"""StreamMessageUseCase — stream all TraceEvents of a turn and persist each one.

The use case depends on TraceEventRepository + the runner API
``stream(thread_id, message, turn_id) -> AsyncIterator[TraceEvent]`` and
persists each event before yielding it. When a prepared attachment turn is
supplied, the runner receives the multimodal content blocks (REQ-018) and the
temporary documents are cleaned up in the generator's ``finally`` — the whole
execution (thread lookup, runner build, streaming loop) sits inside the
``try``, so ANY failure after preparation cleans the already-uploaded
documents through ``prepared.agent_name`` exactly like the invoke path
(REQ-041/042/045, C1 no-orphan). It covers normal completion AND cancellation
(EC-013), skipped when the last AI message ended in ``AWAITING_HITL``
(REQ-043), and never masks a response already produced (REQ-046).
"""

import logging
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from src.application.use_cases._attachment_cleanup import (
    resolve_cleanup_agent,
    run_temporary_document_cleanup,
    turn_ended_awaiting_hitl,
)
from src.domain.entities.human_turn import HumanTurnContent
from src.domain.entities.message import Message
from src.domain.entities.prepared_chat_turn import PreparedChatTurn
from src.domain.entities.trace_event import TraceEvent, TraceEventType
from src.domain.errors.attachment import AttachmentValidationError
from src.domain.errors.messages import ErrorMessage
from src.domain.errors.storage import StorageError
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_registry import AgentRegistry
from src.domain.ports.temporary_document_cleaner import TemporaryDocumentCleaner
from src.domain.ports.thread_repository import ThreadRepository
from src.domain.ports.trace_event_repository import TraceEventRepository

if TYPE_CHECKING:
    from src.application.services.chat_attachments_service import ChatAttachmentsService

logger = logging.getLogger(__name__)


class StreamMessageUseCase:
    """Stream all TraceEvents of a turn and persist each one.

    Each event emitted by the runner is persisted via ``trace_repo.add`` before
    being yielded, so the trace is durably stored even if the client disconnects
    mid-stream.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        threads: ThreadRepository,
        trace_repo: TraceEventRepository,
        attachments: "ChatAttachmentsService | None" = None,
        cleaner: TemporaryDocumentCleaner | None = None,
    ) -> None:
        self._registry = registry
        self._threads = threads
        self._trace_repo = trace_repo
        self._attachments = attachments
        self._cleaner = cleaner

    async def execute(
        self,
        thread_id: str,
        message: str,
        prepared: PreparedChatTurn | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncGenerator[TraceEvent, None]:
        """Execute the use case.

        Args:
            thread_id: Conversation thread identifier.
            message: Human message text (legacy string turn).
            prepared: Prepared attachments for this turn (``None`` keeps the
                exact legacy string flow — REQ-003/REQ-017). When set, the
                runner receives the multimodal content blocks (REQ-018) and the
                HUMAN_MESSAGE event carries the attachment metadata (REQ-020).
            reasoning_effort: Optional thinking level for THIS turn
                (``low``/``medium``/``high``; ``None``/absent = thinking off)
                — threaded to the runner's LLM call payload.

        Yields:
            Each :class:`TraceEvent` emitted by the runner (HUMAN_MESSAGE,
            intermediates, then AI_MESSAGE), in turn order.

        Raises:
            ThreadNotFoundError: If the thread does not exist.
            AgentError: On runner failure.
            AttachmentValidationError: When attachments ride along but the
                attachment service is not wired.
            StorageError: If persisting an event fails.
        """
        if self._attachments is None and prepared is not None:
            raise AttachmentValidationError(ErrorMessage.ATTACHMENT_UNAVAILABLE)
        turn_id = str(uuid.uuid4())
        chunk_count = 0
        last_ai_event = None
        thread = None
        try:
            thread = await self._threads.get(thread_id)
            runner = await self._registry.get_runner(
                thread.agent_name, reasoning_effort=reasoning_effort
            )
            logger.info(LogMessage.CHAT_STREAM_STARTED, thread_id, thread.agent_name)
            turn_input = message if prepared is None else HumanTurnContent.from_prepared(prepared)
            async for event in runner.stream(thread_id, turn_input, turn_id):
                try:
                    await self._trace_repo.add(thread_id, event)
                except Exception as exc:
                    logger.exception(LogMessage.CHAT_STREAM_PERSIST_FAILED, thread_id, thread.agent_name)
                    raise StorageError(ErrorMessage.STORAGE_FAILED_PERSIST_STREAM.format(error=exc)) from exc
                if event.type == TraceEventType.AI_MESSAGE:
                    last_ai_event = event
                chunk_count += 1
                yield event
        except Exception:
            logger.exception(
                LogMessage.CHAT_STREAM_ERROR_UC, thread_id, resolve_cleanup_agent(prepared, thread), chunk_count
            )
            raise
        finally:
            # C1 no-orphan guarantee, stream twin (REQ-041/042/043/045): the
            # thread lookup and runner build happen INSIDE the try, so a turn
            # that already carries a prepared document is cleaned up on ANY
            # failure — including before the runner exists — using
            # ``prepared.agent_name`` exactly like the invoke path. The
            # generator ``finally`` also fires on client cancellation (EC-013).
            # Skipped only when the turn ended in AWAITING_HITL; never masks a
            # response already produced (REQ-046).
            awaiting = _stream_awaiting_hitl(last_ai_event)
            agent = resolve_cleanup_agent(prepared, thread)
            await run_temporary_document_cleanup(self._cleaner, agent, thread_id, prepared, awaiting_hitl=awaiting)


def _stream_awaiting_hitl(last_ai_event) -> bool:  # noqa: ANN401 (TraceEvent | None)
    """Return True when the stream's last AI_MESSAGE awaits HITL (REQ-043)."""
    if last_ai_event is None or last_ai_event.type != TraceEventType.AI_MESSAGE:
        return False
    try:
        message = Message.model_validate_json(str(last_ai_event.content or "{}"))
    except Exception:  # noqa: BLE001 — malformed payloads are treated as terminal
        return False
    return turn_ended_awaiting_hitl(message.status)
