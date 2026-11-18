import logging

from src.domain.logging.messages import LogMessage
from src.domain.ports.store_file_repository import StoreFileRepository
from src.domain.ports.thread_repository import ThreadRepository

logger = logging.getLogger(__name__)

#: Store prefixes wiped with the thread (chat-pdf-attachments spec §5.2.1).
_CASCADE_PREFIXES = ("/chat_attachments/", "/documents/")


class DeleteThreadUseCase:
    """Delete a thread and cascade-wipe its LangGraph Store files (REQ-097..100).

    Both cascades are SILENT (no UI confirmation, REQ-098) and IDEMPOTENT
    (REQ-099): empty prefix trees never raise. A store failure is logged
    explicitly (WARNING+ with the thread id) and NEVER blocks the thread row
    deletion (REQ-100 / EDG-017) — the thread disappears from the user's
    listing even if orphaned store objects remain.
    """

    def __init__(self, threads: ThreadRepository, store_files: StoreFileRepository | None = None) -> None:
        self._threads = threads
        self._store_files = store_files

    async def execute(self, thread_id: str) -> None:
        """Delete the thread row, then best-effort wipe its store prefixes.

        Args:
            thread_id: Conversation thread identifier.

        Raises:
            ThreadNotFoundError: If the thread does not exist (nothing is
                wiped — the cascade never touches other threads' files).
            StorageError: If the DB deletion fails (store untouched).
        """
        await self._threads.delete(thread_id)
        await self._cascade_delete_store_files(thread_id)

    async def _cascade_delete_store_files(self, thread_id: str) -> None:
        """Wipe ``/chat_attachments/<thread_id>/`` and ``/documents/<thread_id>/``.

        Idempotent by construction (REQ-099): missing files/prefixes are no-ops.
        Each prefix is attempted independently so one prefix's failure cannot
        mask the other; every failure is logged WARNING+ with the thread id
        (REQ-100 / NFR-018) and retried once before being dropped with an
        orphan-notice.

        Args:
            thread_id: The deleted thread identifier.
        """
        if self._store_files is None:
            return
        store_files = self._store_files
        prefixes = tuple(f"{prefix}{thread_id}/" for prefix in _CASCADE_PREFIXES)
        logger.info(LogMessage.THREAD_STORE_CASCADE_STARTED, thread_id, len(prefixes))
        deleted_total = 0
        for prefix in prefixes:
            paths = await self._list_or_log(store_files, thread_id, prefix)
            if not paths:
                continue
            for path in paths:
                deleted = await self._delete_or_log(store_files, thread_id, path)
                if deleted:
                    deleted_total += 1
        logger.info(LogMessage.THREAD_STORE_CASCADE_COMPLETED, thread_id, deleted_total)

    async def _list_or_log(self, store_files: StoreFileRepository, thread_id: str, prefix: str) -> list[str]:
        """List a cascade prefix, logging (and retrying once) on store failure."""
        try:
            return await store_files.list_files(prefix)
        except Exception:  # noqa: BLE001 — REQ-100: a store failure must NOT block the deletion
            logger.warning(LogMessage.THREAD_STORE_CASCADE_FAILED, thread_id, prefix, "list_files failed")
            logger.info(LogMessage.THREAD_STORE_CASCADE_RETRY, thread_id, prefix)
            try:
                return await store_files.list_files(prefix)
            except Exception:  # noqa: BLE001 — same REQ-100 contract
                logger.exception(LogMessage.THREAD_STORE_CASCADE_FAILED, thread_id, prefix, "list_files retry failed")
                return []

    async def _delete_or_log(self, store_files: StoreFileRepository, thread_id: str, path: str) -> bool:
        """Delete one store file idempotently, logging failures (REQ-100)."""
        try:
            await store_files.delete_file(path)
            return True
        except Exception:  # noqa: BLE001 — REQ-100: never block the thread deletion
            logger.warning(LogMessage.THREAD_STORE_CASCADE_FAILED, thread_id, path, "delete_file failed")
            logger.info(LogMessage.THREAD_STORE_CASCADE_RETRY, thread_id, path)
            try:
                await store_files.delete_file(path)
                return True
            except Exception:  # noqa: BLE001 — same REQ-100 contract
                logger.exception(LogMessage.THREAD_STORE_CASCADE_FAILED, thread_id, path, "delete_file retry failed")
                return False
