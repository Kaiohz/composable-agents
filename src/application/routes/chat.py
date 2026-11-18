"""Chat HTTP routes: POST /chat/{id} and POST /chat/{id}/stream.

Emits TraceEvent records (not the legacy StreamEvent). On stream error, the
generator emits a plain JSON error payload ``{"type": "error", "data": "..."}``
that is NOT a TraceEvent (TraceEventType has no ERROR variant) — this matches
the previous SSE contract for errors; the frontend already handles it.
"""

import asyncio
import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from src.application.requests.chat import ChatRequest
from src.application.services.chat_attachments_service import ChatAttachmentsService
from src.application.use_cases.get_thread import GetThreadUseCase
from src.application.use_cases.send_message import SendMessageUseCase
from src.application.use_cases.stream_message import StreamMessageUseCase
from src.dependencies import (
    get_chat_attachments_service,
    get_get_thread_use_case,
    get_send_message_use_case,
    get_stream_message_use_case,
)
from src.domain.entities.message import Message
from src.domain.errors.attachment import AttachmentValidationError
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


async def _prepare_attachments(
    attachments_service: ChatAttachmentsService | None,
    thread_id: str,
    body: ChatRequest,
):
    """Prepare the request attachments BEFORE any agent invocation.

    Args:
        attachments_service: The shared preparation service (``None`` when the
            persistence layer is down — attachments are then refused).
        thread_id: Conversation thread identifier.
        body: The parsed chat request.

    Returns:
        A :class:`PreparedChatTurn` when attachments were prepared, ``None``
        for the legacy attachments-free flow (REQ-003/REQ-017).

    Raises:
        AttachmentValidationError: When attachments are requested but the
            preparation service is unavailable.
    """
    if body.attachments is None:
        return None
    if attachments_service is None:
        raise AttachmentValidationError(ErrorMessage.ATTACHMENT_UNAVAILABLE)
    return await attachments_service.prepare_for_request(thread_id, body.message, body.attachments)


@router.post("/{thread_id}")
async def send_message(
    thread_id: str,
    body: ChatRequest,
    use_case: Annotated[SendMessageUseCase, Depends(get_send_message_use_case)],
    attachments: Annotated[ChatAttachmentsService, Depends(get_chat_attachments_service)],
) -> Message:
    """Send a human message or an HITL decision and return the final AI Message.

    When the request carries attachments they are validated, stored (images)
    or temporarily uploaded (documents) BEFORE the agent is invoked (REQ-011);
    any violation returns an explicit 4xx without invoking the graph.

    Args:
        thread_id: Conversation thread identifier.
        body: Chat request (message XOR HITL fields; optional attachments).
        use_case: SendMessageUseCase wired at startup.
        attachments: The shared attachment preparation service.

    Returns:
        The final AI Message.
    """
    logger.info(LogMessage.CHAT_RECEIVE, thread_id, "HITL" if body.message is None else body.message[:80])
    prepared = await _prepare_attachments(attachments, thread_id, body)
    # C1 route symmetry: BOTH chat entry points pass the same normalized
    # message value ("" when absent) — the use case derives the HITL nature
    # from the HITL fields only, so a message-less attachment turn reaches
    # the runner on both routes (REQ-004, EDG-001/EDG-002).
    result = await use_case.execute(
        thread_id,
        message=body.message or "",
        action=body.action,
        tool_call_id=body.tool_call_id,
        reason=body.reason,
        edits=body.edits,
        decisions=body.decisions,
        prepared=prepared,
    )
    logger.info(LogMessage.CHAT_RESPONSE, thread_id, result.status, len(result.content or ""))
    return result


@router.post("/{thread_id}/stream")
async def stream_message(
    thread_id: str,
    body: ChatRequest,
    use_case: Annotated[StreamMessageUseCase, Depends(get_stream_message_use_case)],
    get_thread: Annotated[GetThreadUseCase, Depends(get_get_thread_use_case)],
    attachments: Annotated[ChatAttachmentsService, Depends(get_chat_attachments_service)],
) -> EventSourceResponse:
    """Stream all TraceEvents of a turn as SSE, then a ``[DONE]`` terminator.

    Each ``data:`` line carries a JSON-serialized TraceEvent. Attachments are
    prepared BEFORE the stream starts so any validation/upload failure is a
    plain HTTP error (not an SSE error event). On stream error, a plain JSON
    error payload ``{"type": "error", "data": "..."}`` is emitted (NOT a
    TraceEvent), matching the SSE contract so the frontend keeps working.

    Args:
        thread_id: Conversation thread identifier.
        body: Chat request (must contain ``message``).
        use_case: StreamMessageUseCase wired at startup.
        get_thread: GetThreadUseCase used to validate the thread exists.
        attachments: The shared attachment preparation service.

    Returns:
        An ``EventSourceResponse`` streaming TraceEvents.
    """
    logger.info(LogMessage.CHAT_STREAM_RECEIVE, thread_id, (body.message or "")[:80])
    await get_thread.execute(thread_id)
    prepared = await _prepare_attachments(attachments, thread_id, body)

    async def event_generator():
        event_count = 0
        try:
            async for event in use_case.execute(thread_id, body.message or "", prepared):
                event_count += 1
                yield {"data": event.model_dump_json()}
            yield {"data": "[DONE]"}
            logger.info(LogMessage.CHAT_STREAM_COMPLETE, thread_id, event_count)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(LogMessage.CHAT_STREAM_ERROR, thread_id, event_count)
            yield {"data": json.dumps({"type": "error", "data": str(exc)})}

    return EventSourceResponse(event_generator(), sep="\r\n", ping=15)
