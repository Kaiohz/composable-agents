"""Shared post-turn temporary document cleanup helper for the chat use cases
(REQ-041..047).

Used by :class:`SendMessageUseCase` and :class:`StreamMessageUseCase` so both
entry points guarantee the same cleanup semantics:

* the cleanup agent resolves from the prepared turn first, then the thread
  (:func:`resolve_cleanup_agent`) so a failure BEFORE the thread lookup or the
  runner build still cleans the already-uploaded documents (C1 no-orphan);
* skipped when the turn ended in ``AWAITING_HITL`` (the document must stay
  available until the HITL flow reaches a terminal response — REQ-043);
* attempted after the terminal response, in a ``finally`` (REQ-041/REQ-042);
* on cancellation, retried once unshielded so a client disconnect still
  cleans up (EC-013);
* every failure stays inside the helper — an agent response already produced
  must never be masked (REQ-046).
"""

import asyncio
import contextlib
import logging

from src.domain.entities.message import MessageStatus
from src.domain.entities.prepared_chat_turn import PreparedChatTurn
from src.domain.logging.messages import LogMessage
from src.domain.ports.temporary_document_cleaner import TemporaryDocumentCleaner

logger = logging.getLogger(__name__)


async def run_temporary_document_cleanup(
    cleaner: TemporaryDocumentCleaner | None,
    agent_name: str,
    thread_id: str,
    prepared: PreparedChatTurn | None,
    *,
    awaiting_hitl: bool,
) -> bool:
    """Best-effort cleanup of the turn's temporary documents.

    Args:
        cleaner: The cleanup port (``None`` means not wired — no-op).
        agent_name: The turn's agent (resolves the Files MCP server).
        thread_id: Conversation thread identifier (log context).
        prepared: The prepared turn (cleanup only when it carries temp docs).
        awaiting_hitl: True when the turn ended in ``AWAITING_HITL`` — cleanup
            is then SKIPPED (REQ-043).

    Returns:
        True when the cleanup actually ran (used by tests/diagnostics).

    Raises:
        asyncio.CancelledError: When the surrounding scope was cancelled —
            the single unshielded retry completes first (EC-013: the temp
            document is still deleted once), then the cancellation is RAISED,
            never swallowed (S7497).
    """
    if cleaner is None or prepared is None or not prepared.temporary_documents:
        return False
    if awaiting_hitl:
        logger.info(LogMessage.CHAT_TEMP_CLEANUP_SKIPPED_HITL, thread_id)
        return False
    try:
        await asyncio.shield(cleaner.delete_temporary_documents(agent_name, prepared.temporary_documents))
        return True
    except asyncio.CancelledError as cancelled:
        # The response was cancelled (client disconnect / stream abort): the
        # temp document still gets ONE unshielded retry (EC-013) — a cancelled
        # task may await again (the cancellation is delivered once) — and then
        # the cancellation propagates (S7497: never swallow a cancellation).
        logger.warning(LogMessage.CHAT_TEMP_CLEANUP_CANCELLED_RETRY, thread_id)
        with contextlib.suppress(Exception):  # noqa: SIM105 — REQ-046: the retry failure stays logged-only
            await cleaner.delete_temporary_documents(agent_name, prepared.temporary_documents)
        raise cancelled
    except Exception:  # noqa: BLE001 — REQ-046: cleanup failures never propagate
        return False


def turn_ended_awaiting_hitl(status: MessageStatus | None) -> bool:
    """Return True when a final Message status means the turn awaits HITL."""
    return status == MessageStatus.AWAITING_HITL


def resolve_cleanup_agent(prepared: PreparedChatTurn | None, thread) -> str:  # noqa: ANN001
    """Resolve the cleanup agent: the prepared turn first, then the thread.

    The invoke and stream entry points share this resolution so a failure
    BEFORE the thread lookup / runner build (both routes) still cleans the
    already-uploaded documents using ``prepared.agent_name`` (C1 no-orphan).
    """
    if prepared is not None and prepared.agent_name:
        return prepared.agent_name
    return thread.agent_name if thread is not None else ""
