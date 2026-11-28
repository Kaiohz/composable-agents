"""SendMessageUseCase — send a message or HITL decisions to the agent.

The use case depends on TraceEventRepository + the runner API
``invoke(thread_id, message, turn_id) -> (Message, list[TraceEvent])`` and the
unified HITL resume method
``resume_hitl(thread_id, decisions, turn_id) -> (Message, list[TraceEvent])``.
The full trace is persisted in a single batch via ``trace_repo.add_batch`` for
both the human-message and the HITL resume paths.

The HITL nature is derived from the HITL FIELDS only (``decisions`` /
``tool_call_id``): a message-less turn with prepared attachments is a legal
human turn (REQ-004, EDG-001/EDG-002) and must reach the runner. The whole
execution is wrapped in a ``finally`` so the temporary documents of an
already-prepared turn are cleaned up on ANY failure (runner build, guards,
graph error) — never orphaned, never masking the response/error (REQ-041/042/
045/046). Attachment preparation rides along through the optional ``prepared``
parameter (see :class:`ChatAttachmentsService`); the cleanup is delegated to
the :class:`TemporaryDocumentCleaner` port so the MCP plumbing stays behind
the boundary.
"""

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from src.application.use_cases._attachment_cleanup import (
    resolve_cleanup_agent,
    run_temporary_document_cleanup,
)
from src.domain.entities.hitl_decision import HitlDecision
from src.domain.entities.human_turn import HumanTurnContent
from src.domain.entities.message import Message, MessageStatus
from src.domain.entities.prepared_chat_turn import PreparedChatTurn
from src.domain.errors.attachment import AttachmentValidationError
from src.domain.errors.hitl import InvalidHitlActionError
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_registry import AgentRegistry
from src.domain.ports.temporary_document_cleaner import TemporaryDocumentCleaner
from src.domain.ports.thread_repository import ThreadRepository
from src.domain.ports.trace_event_repository import TraceEventRepository

if TYPE_CHECKING:
    from src.application.services.chat_attachments_service import ChatAttachmentsService

logger = logging.getLogger(__name__)


def _as_prepared(refs: list) -> PreparedChatTurn | None:  # noqa: ANN401 (TemporaryDocumentRef list)
    """Wrap temp refs in a minimal PreparedChatTurn for the cleanup helper."""
    if not refs:
        return None

    return PreparedChatTurn(temporary_documents=refs)  # type: ignore[arg-type]


def _is_hitl_request(decisions: list[HitlDecision] | None, tool_call_id: str | None) -> bool:
    """Return True when the request carries HITL fields (C1: never message-based)."""
    return decisions is not None or tool_call_id is not None


def _validate_turn_request(
    is_hitl: bool,
    message: str | None,
    action: str | None,
    decisions: list[HitlDecision] | None,
    prepared: PreparedChatTurn | None,
) -> None:
    """Enforce the message/HITL/attachments contract (REQ-004/REQ-005).

    Raises:
        InvalidHitlActionError: On an unsupported legacy HITL action, or when
            the request carries neither a message, HITL fields nor attachments.
    """
    if is_hitl and decisions is None and action not in {"approve", "reject", "edit"}:
        raise InvalidHitlActionError(ErrorMessage.INVALID_HITL_ACTION.format(action=action))
    if not is_hitl and not message and prepared is None:
        # Legacy contract preserved: a turn must carry a message, HITL fields
        # or prepared attachments.
        raise InvalidHitlActionError(ErrorMessage.INVALID_HITL_ACTION.format(action=action))


class SendMessageUseCase:
    """Send a human message or HITL decisions to the agent and return the response.

    For a human message: generates a fresh ``turn_id``, invokes the runner,
    persists the full trace in a batch, and returns the final AI Message.
    When prepared attachments ride along (``prepared``), the runner receives a
    content-block turn (REQ-018) and the temporary documents are deleted after
    the terminal response — skipped when the turn ends in ``AWAITING_HITL``
    (REQ-043), attempted in a ``finally`` even on graph error (REQ-042).

    For HITL decisions (approve/reject/edit, single or multiple): generates a
    fresh ``turn_id``, calls ``runner.resume_hitl`` with the decisions list,
    persists the returned trace in a batch, and returns the final AI Message.
    When the resumed flow reaches a terminal response, the temporary documents
    of the ORIGINAL turn (traced on the thread) are cleaned up (REQ-044).
    The legacy single-decision shape (``action`` + ``tool_call_id`` +
    optional ``reason``/``edits``) is converted into a one-element decisions
    list for backward compatibility.
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
        *,
        message: str | None = None,
        action: str | None = None,
        tool_call_id: str | None = None,
        reason: str | None = None,
        edits: dict[str, Any] | None = None,
        decisions: list[HitlDecision] | None = None,
        prepared: PreparedChatTurn | None = None,
        reasoning_effort: str | None = None,
    ) -> Message:
        """Dispatch the turn and guarantee the document cleanup (C1 no-orphan).

        Raises:
            InvalidHitlActionError: If ``action`` is not a supported HITL action.
            AttachmentValidationError: When attachments ride along but the
                attachment service is not wired.
            AgentError: On runner failure.
            ThreadNotFoundError: If the thread does not exist.
        """
        is_hitl = _is_hitl_request(decisions, tool_call_id)
        _validate_turn_request(is_hitl, message, action, decisions, prepared)

        final_message: Message | None = None
        thread = None
        try:
            thread = await self._threads.get(thread_id)
            runner = await self._registry.get_runner(
                thread.agent_name, reasoning_effort=reasoning_effort
            )

            if not is_hitl:
                final_message = await self._run_human_turn(thread_id, thread, runner, message, prepared)
            else:
                final_message = await self._run_hitl_turn(
                    thread_id, thread, runner, action, tool_call_id, reason, edits, decisions
                )
            return final_message
        finally:
            # C1 no-orphan guarantee (REQ-041/042/045): every failure path and
            # the graph path itself fall through here — documents are never
            # stranded, never masking the response (REQ-046).
            await self._cleanup_turn_documents(thread_id, thread, prepared, final_message, is_hitl=is_hitl)

    async def _run_human_turn(
        self, thread_id: str, thread, runner, message: str | None, prepared: PreparedChatTurn | None
    ) -> Message:
        """Run a human turn (message and/or attachments) and return the AI Message.

        ``message`` may legitimately be empty when the attachments carry the
        turn (REQ-004 / EDG-001/EDG-002).
        """
        logger.info(LogMessage.CHAT_SENDING_HUMAN, thread_id, thread.agent_name)
        turn_id = str(uuid.uuid4())
        start = time.monotonic()
        if self._attachments is None and prepared is not None:
            raise AttachmentValidationError(ErrorMessage.ATTACHMENT_UNAVAILABLE)
        turn_input = HumanTurnContent.from_prepared(prepared) if prepared is not None else str(message)
        final_message, trace = await runner.invoke(thread_id, turn_input, turn_id)
        await self._trace_repo.add_batch(thread_id, trace)
        elapsed = time.monotonic() - start
        logger.info(
            LogMessage.CHAT_INVOKE_COMPLETE,
            thread_id,
            thread.agent_name,
            elapsed,
            final_message.status,
            len(final_message.content or ""),
        )
        return final_message

    async def _run_hitl_turn(
        self,
        thread_id: str,
        thread,
        runner,
        action: str | None,
        tool_call_id: str | None,
        reason: str | None,
        edits: dict[str, Any] | None,
        decisions: list[HitlDecision] | None,
    ) -> Message:
        """Resume a HITL turn with the decisions and return the AI Message."""
        if decisions is None:
            decisions = [HitlDecision(tool_call_id=tool_call_id, action=action, reason=reason, edits=edits)]  # type: ignore[arg-type]

        logger.info(LogMessage.CHAT_HITL_RECEIVED, thread_id, thread.agent_name, "decisions", len(decisions))
        turn_id = str(uuid.uuid4())
        start = time.monotonic()
        final_message, trace = await runner.resume_hitl(thread_id, decisions, turn_id)
        await self._trace_repo.add_batch(thread_id, trace)
        elapsed = time.monotonic() - start
        logger.info(LogMessage.CHAT_HITL_COMPLETE, thread_id, thread.agent_name, elapsed, final_message.status)
        return final_message

    async def _cleanup_turn_documents(
        self,
        thread_id: str,
        thread,
        prepared: PreparedChatTurn | None,
        final_message: Message | None,
        *,
        is_hitl: bool,
    ) -> None:
        """Clean the turn's temporary documents — best-effort, never raises (REQ-046)."""
        if is_hitl:
            if thread is not None and final_message is not None:
                await self._cleanup_after_hitl_terminal(thread_id, thread.agent_name, final_message)
            # A FAILED resume leaves the turn pending (REQ-043): documents stay.
            return
        agent = resolve_cleanup_agent(prepared, thread)
        awaiting = final_message is not None and final_message.status == MessageStatus.AWAITING_HITL
        await run_temporary_document_cleanup(self._cleaner, agent, thread_id, prepared, awaiting_hitl=awaiting)

    async def _cleanup_after_hitl_terminal(
        self, thread_id: str, agent_name: str, final_message: Message | None
    ) -> None:
        """Delete the ORIGINAL turn's temp docs after a TERMINAL HITL resume (REQ-044)."""
        if self._cleaner is None or final_message is None or final_message.status == MessageStatus.AWAITING_HITL:
            # REQ-043: an awaiting turn keeps its documents alive.
            return
        try:
            events = await self._trace_repo.list_by_thread(thread_id)
        except Exception:
            logger.exception(LogMessage.CHAT_TEMP_CLEANUP_FAILED, "(trace scan)", thread_id, "list_by_thread failed")
            return
        refs = self._collect_pending_temporary_refs(events, thread_id)
        await run_temporary_document_cleanup(
            self._cleaner, agent_name, thread_id, _as_prepared(refs), awaiting_hitl=False
        )

    def _collect_pending_temporary_refs(self, events, thread_id: str):  # noqa: ANN001, ANN202
        """Re-derive the ORIGINAL turn's document refs from the trace events.

        chat_tmp Option 2: the relative object path carries NO user segment
        (the identity is the server-side agent scope), so the derivation
        needs ONLY the metadata id + filename — the trace owner is no longer
        a path segment. Only ``kind: document`` entries are derived — images
        live in the LangGraph Store (REQ-025) and are never deleted as temp
        docs.
        """
        from src.domain.entities.attachment import AttachmentKind, project_attachment_entries
        from src.domain.entities.prepared_chat_turn import TemporaryDocumentRef, temporary_object_path
        from src.domain.entities.trace_event import TraceEventType

        refs: dict[str, TemporaryDocumentRef] = {}
        for event in events:
            if event.type != TraceEventType.HUMAN_MESSAGE:
                continue
            for entry in project_attachment_entries((event.metadata or {}).get("attachments")):
                if entry.get("kind") != AttachmentKind.DOCUMENT.value:
                    continue
                try:
                    refs[entry["id"]] = TemporaryDocumentRef(
                        attachment_id=entry["id"],
                        filename=entry["filename"],
                        object_path=temporary_object_path(thread_id, entry["id"], entry["filename"]),
                    )
                except (KeyError, ValueError) as exc:
                    logger.warning(LogMessage.CHAT_TEMP_CLEANUP_FAILED, "(derivation)", thread_id, exc)
        return list(refs.values())
