"""ChatAttachmentsService — shared attachment preparation (spec §9.1 step 2).

Three-stage flow shared by :class:`SendMessageUseCase` and
:class:`StreamMessageUseCase` (spec §9.1 composable-agents):

1. validation/decoding — :mod:`src.domain.services.chat.attachment_validator`;
2. preparation — image persistence in the user's LangGraph Store
   (REQ-025..028) + document temporary upload through the agent's Files MCP
   server (REQ-009..011, EDG-008);
3. handover — a :class:`PreparedChatTurn` with standard content blocks
   (REQ-012), projection-safe trace metadata (REQ-020/021) and the temp refs
   the use cases must clean up in their ``finally`` blocks (REQ-042).

The service NEVER logs nor persists attachment base64 content (NFR-004) and
always builds the temporary prefix from the ``current_user_id`` contextvar
(NFR-007).
"""

import logging
from collections.abc import Awaitable, Callable

from src.application.requests.chat import ChatAttachmentIn
from src.domain.entities.agent_config import AgentConfig
from src.domain.entities.attachment import (
    AttachmentKind,
    AttachmentLimits,
    AttachmentSource,
    ValidatedAttachment,
)
from src.domain.entities.mcp_server_config import McpServerConfig
from src.domain.entities.prepared_chat_turn import (
    TEMP_ROOT_PREFIX,
    PreparedChatTurn,
    TemporaryDocumentRef,
    temporary_object_path,
    temporary_prefix,
)
from src.domain.errors.attachment import (
    AttachmentValidationError,
    DocumentsNotAllowedError,
    FilesMcpTemporaryError,
)
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_config_loader import AgentConfigLoader
from src.domain.ports.agent_config_store import AgentConfigStore
from src.domain.ports.files_mcp_client import FilesMcpClient
from src.domain.ports.store_file_repository import StoreFileRepository
from src.domain.ports.thread_repository import ThreadRepository
from src.domain.services.chat.attachment_validator import decode_and_validate_attachments
from src.domain.services.chat.mcp_files_policy import find_files_mcp_server
from src.infrastructure.database.rls_context import current_user_id

logger = logging.getLogger(__name__)

#: Root of the user's persistent attachments tree inside the LangGraph Store
#: (REQ-025). The tree is threaded on the user namespace (NFR-007).
CHAT_ATTACHMENTS_STORE_ROOT = "chat_attachments"

#: Prompt reference text appended for each document (REQ-015) — carries ONLY
#: the object path and the read_file instruction, never any binary content
#: (REQ-016).
DOCUMENT_REFERENCE_TEMPLATE = (
    "A document named '{filename}' (MIME: {content_type}) has been shared with you for this turn. "
    "Its temporary file is available at the path: {object_path}. "
    "Use the read_file tool from the Files MCP server to open and analyze this document."
)


class ChatAttachmentsService:
    """Prepares chat attachments for the runner and cleans them up after.

    The service implements the orchestration ports consumed by the chat use
    cases; the Files MCP plumbing stays behind :class:`FilesMcpClient` and the
    cleanup uses the :class:`TemporaryDocumentCleaner` adapter.
    """

    def __init__(
        self,
        threads: ThreadRepository,
        config_store: AgentConfigStore,
        config_loader: AgentConfigLoader,
        store_files: StoreFileRepository,
        files_client: FilesMcpClient,
        limits: AttachmentLimits,
        builtin_config_provider: Callable[[], Awaitable[AgentConfig]] | None = None,
    ) -> None:
        """Wire the ports (no per-request state).

        Args:
            threads: Thread repository (thread → agent resolution).
            config_store: Agent YAML store (same chain as the agents routes).
            config_loader: Agent YAML loader.
            store_files: User-namespaced LangGraph Store file repository.
            files_client: Direct Files MCP temporary tool client.
            limits: Env-configured attachment limits (REQ-092).
        """
        self._threads = threads
        self._config_store = config_store
        self._config_loader = config_loader
        self._store_files = store_files
        self._files_client = files_client
        self._limits = limits
        self._builtin_config_provider = builtin_config_provider

    async def prepare_for_request(
        self,
        thread_id: str,
        message: str | None,
        attachments: list[ChatAttachmentIn] | None,
    ) -> PreparedChatTurn | None:
        """Validate, store/upload and block-ify the request attachments.

        Args:
            thread_id: Conversation thread identifier.
            message: The user text (``None`` and no attachments is impossible
                — :class:`ChatRequest` rejects it).
            attachments: The request attachment DTOs (``None``/empty means a
                legacy attachments-free turn → returns ``None`` so the use
                case keeps the exact legacy runner input, REQ-003/REQ-017).

        Returns:
            A :class:`PreparedChatTurn` when at least one attachment was
            accepted, otherwise ``None``.

        Raises:
            AttachmentValidationError: On any contract violation (422).
            DocumentsNotAllowedError: When a document targets an agent without
                the Files MCP server (REQ-010).
            FilesMcpTemporaryError: When the temporary upload fails (EC-007).
        """
        if not attachments:
            return None
        validated = decode_and_validate_attachments([att.model_dump() for att in attachments], self._limits)
        thread = await self._threads.get(thread_id)
        config = await self._load_agent_config(thread.agent_name)

        metadata: list[dict] = []
        blocks: list[dict] = []
        temp_docs: list[TemporaryDocumentRef] = []
        if message:
            # REQ-013: the user text is preserved as the FIRST text block.
            blocks.append({"type": "text", "text": message})

        images = [att for att in validated if att.kind == AttachmentKind.IMAGE]
        documents = [att for att in validated if att.kind == AttachmentKind.DOCUMENT]
        server = None
        if documents:
            server = find_files_mcp_server(config)
            if server is None:
                # REQ-010: an explicit 4xx BEFORE any agent invocation, the
                # backend being the authoritative policy point (REQ-009).
                logger.warning(LogMessage.CHAT_ATTACHMENTS_DOCUMENT_POLICY_FAILED, thread_id, thread.agent_name)
                raise DocumentsNotAllowedError(ErrorMessage.ATTACHMENT_DOCUMENT_POLICY)

        for att in images:
            store_path = await self._store_image(thread_id, att)
            metadata.append(
                {
                    "id": att.id,
                    "filename": att.filename,
                    "content_type": att.content_type,
                    "size_bytes": att.size_bytes,
                    "kind": att.kind.value,
                    "source": AttachmentSource.LANGGRAPH_STORE.value,
                }
            )
            # REQ-012: standard multimodal image block.
            blocks.append({"type": "image", "base64": att.content_base64, "mime_type": att.content_type})
            logger.info(LogMessage.CHAT_ATTACHMENTS_IMAGE_STORED, att.id, store_path, thread_id)

        for att in documents:
            # `server` is guaranteed non-None for documents by the policy check
            # above (REQ-010 raises otherwise); the assert pins it for mypy.
            assert server is not None
            ref = await self._upload_document(server, thread_id, att)
            temp_docs.append(ref)
            metadata.append(
                {
                    "id": att.id,
                    "filename": att.filename,
                    "content_type": att.content_type,
                    "size_bytes": att.size_bytes,
                    "kind": att.kind.value,
                    "source": AttachmentSource.RUSTFS_TEMP.value,
                }
            )
            # REQ-015: reference block with the temp path + read_file hint;
            # the binary content never enters the prompt (REQ-016).
            blocks.append(
                {
                    "type": "text",
                    "text": DOCUMENT_REFERENCE_TEMPLATE.format(
                        filename=att.filename,
                        content_type=att.content_type,
                        object_path=ref.object_path,
                    ),
                }
            )
            logger.info(LogMessage.CHAT_ATTACHMENTS_DOCUMENT_UPLOADED, att.id, ref.object_path, thread_id)

        # C1 no-orphan: the turn's agent is carried on the prepared turn so a
        # post-preparation failure (runner build, guards) still cleans up.
        return PreparedChatTurn(
            text=message,
            content_blocks=blocks,
            attachments_metadata=metadata,
            temporary_documents=temp_docs,
            agent_name=thread.agent_name,
        )

    async def _store_image(self, thread_id: str, att: ValidatedAttachment) -> str:
        """Persist an image under the user's chat-attachments tree (REQ-025).

        The store content is the ORIGINAL base64 payload, retrievable by the
        existing store API (REQ-027). The store never logs the content.
        """
        store_path = f"/{CHAT_ATTACHMENTS_STORE_ROOT}/{thread_id}/{att.id}/{att.filename}"
        await self._store_files.put_file(store_path, att.content_base64)
        return store_path

    async def _upload_document(
        self,
        server: McpServerConfig,
        thread_id: str,
        att: ValidatedAttachment,
    ) -> TemporaryDocumentRef:
        """Upload a document through the Files MCP server (REQ-011/EDG-008).

        The prefix is built by the backend from the authenticated user id
        (REQ-038, NFR-007) and the attachment id is unique per attachment
        (EDG-006/EDG-007).
        """
        user_id = current_user_id.get()
        if not user_id:
            # I9: an unauthenticated context must fail with a domain error,
            # never a raw ValueError → 500 (routes are auth-guarded; this is a
            # defensive boundary).
            logger.warning(LogMessage.CHAT_ATTACHMENTS_DOCUMENT_POLICY_FAILED, thread_id, "(no authenticated user)")
            raise AttachmentValidationError(ErrorMessage.ATTACHMENT_UNAVAILABLE)
        prefix = temporary_prefix(user_id, thread_id, att.id)
        logger.info(LogMessage.CHAT_ATTACHMENTS_DOCUMENT_UPLOADING, att.filename, server.url or "", thread_id)
        try:
            upload = await self._files_client.temporary_upload(
                server,
                file_name=att.filename,
                content_base64=att.content_base64,
                content_type=att.content_type,
                prefix=prefix,
            )
        except FilesMcpTemporaryError:
            raise
        except Exception as exc:
            logger.error(LogMessage.CHAT_ATTACHMENTS_MCP_UPLOAD_FAILED, att.filename, thread_id)
            raise FilesMcpTemporaryError(
                ErrorMessage.ATTACHMENT_MCP_UPLOAD_FAILED.format(filename=att.filename, error=exc)
            ) from exc
        # I7 (NFR-008/NFR-009): the object path embedded in the prompt and used
        # as the cleanup target MUST stay inside the user's own chat_tmp/
        # namespace — derive locally and verify the server's answer against it.
        user_root = f"{TEMP_ROOT_PREFIX}/{user_id}/"
        object_path = upload.object_name or temporary_object_path(user_id, thread_id, att.id, att.filename)
        if not object_path.startswith(user_root):
            logger.error(LogMessage.CHAT_ATTACHMENTS_MCP_UPLOAD_FAILED, att.filename, thread_id)
            raise FilesMcpTemporaryError(
                ErrorMessage.ATTACHMENT_MCP_UPLOAD_FAILED.format(
                    filename=att.filename, error="object outside the user temporary namespace"
                )
            )
        return TemporaryDocumentRef(attachment_id=att.id, filename=att.filename, object_path=object_path)

    async def _load_agent_config(self, agent_name: str) -> AgentConfig:
        """Load the agent YAML through the same chain as the agents routes.

        The virtual built-in Composable AI Assistant has NO stored YAML
        (REQ-032): it resolves to its centrally managed read-only definition
        so attaching a file to a conversation with it behaves like any other
        agent instead of a confusing not-found.
        """
        from src.application.services.builtin_agent import builtin_agent_config, is_builtin_agent_name

        if is_builtin_agent_name(agent_name):
            if self._builtin_config_provider is not None:
                return await self._builtin_config_provider()
            return builtin_agent_config()
        yaml_content = await self._config_store.get(agent_name)
        return self._config_loader.load_from_string(yaml_content)
