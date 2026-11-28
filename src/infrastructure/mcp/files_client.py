"""Infrastructure adapters: direct Files MCP temporary tool client + cleaner.

Two adapters share the MCP streamable-HTTP plumbing:

* :class:`McpStreamableFilesClient` implements :class:`FilesMcpClient` — one
  per-call mcp ``ClientSession`` against the agent's ``/files/mcp`` server,
  with the McpServerConfig headers resolved through the SAME helper as the
  agent tool loader (``${USER_JWT}`` / ``${USER_API_KEY}`` → RLS contextvars,
  NFR-011) and the global MCP tool timeout applied (NFR-005).
* :class:`McpFilesTemporaryCleaner` implements :class:`TemporaryDocumentCleaner`
  — post-turn cleanup that resolves the target server from the agent config,
  deletes every ``chat_tmp/...`` ref, retries once on failure, logs ERROR on
  any failure and NEVER raises (REQ-045/046/047).

Nothing here ever deletes outside ``chat_tmp/<user_id>/`` — the object paths
are built exclusively by the attachments preparation service
(:func:`domain.temporary_object_path`). The temporary tools are never loaded
for the LLM: :mod:`src.infrastructure.mcp.adapter` filters them out (REQ-036).
"""

import asyncio
import json
import logging
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.domain.entities.mcp_server_config import McpServerConfig
from src.domain.errors.attachment import FilesMcpTemporaryError
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_config_loader import AgentConfigLoader
from src.domain.ports.agent_config_store import AgentConfigStore
from src.domain.ports.files_mcp_client import FilesMcpClient, FilesMcpUpload
from src.domain.ports.temporary_document_cleaner import TemporaryDocumentCleaner
from src.infrastructure.env_utils import resolve_headers_drop_empty

logger = logging.getLogger(__name__)


def _resolve_call_timeout(timeout_s: float) -> Any:
    """Return the mcp ``read_timeout_seconds`` value for one tool call."""
    from datetime import timedelta

    return timedelta(seconds=timeout_s)


class McpStreamableFilesClient(FilesMcpClient):
    """Call the Files MCP temporary tools over streamable HTTP (REQ-029..033)."""

    def __init__(self, tool_timeout: float = 60.0) -> None:
        """Initialize the client.

        Args:
            tool_timeout: Per-call MCP timeout in seconds (same value as the
                agent tool loader — NFR-005).
        """
        self._tool_timeout = tool_timeout

    def _resolved_headers(self, server: McpServerConfig) -> dict[str, str]:
        """Resolve the server's credential headers (loader-equivalent).

        SAME precedence as the agent tool loader (NFR-011): env/user
        placeholders of the configured headers resolve first, then the
        credential of the ORIGINATING request is injected for
        ``auth_mode: "user_token"`` servers (feat-mcp-oauth2 REQ-004) and
        wins over a colliding header name. Without the injected credential
        the dual-auth guard of the Files MCP server denies every
        initialize (401) — temporary uploads/cleanup degrade.
        """
        from src.domain.services.mcp_auth_mode import user_credential_headers
        from src.infrastructure.database.rls_context import current_auth_context

        injected = user_credential_headers(server.auth_mode, current_auth_context.get())
        return resolve_headers_drop_empty({**dict(server.headers), **injected})

    async def temporary_upload(
        self,
        server: McpServerConfig,
        *,
        file_name: str,
        content_base64: str,
        content_type: str,
        prefix: str,
    ) -> FilesMcpUpload:
        """Upload a temporary document (see port contract, REQ-030..033)."""
        payload = await self._call_tool(
            server,
            "temporary_upload_file",
            {
                "file_name": file_name,
                "content_base64": content_base64,
                "content_type": content_type,
                "prefix": prefix,
            },
        )
        try:
            return FilesMcpUpload(
                object_name=str(payload["object_name"]),
                size=int(payload["size"]),
                content_type=str(payload.get("content_type") or content_type),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.error(LogMessage.CHAT_ATTACHMENTS_MCP_UPLOAD_FAILED, file_name, server.name)
            raise FilesMcpTemporaryError(
                ErrorMessage.ATTACHMENT_MCP_UPLOAD_FAILED.format(filename=file_name, error=exc)
            ) from exc

    async def temporary_delete(self, server: McpServerConfig, *, object_path: str) -> None:
        """Delete a temporary object (see port contract, REQ-034/035)."""
        await self._call_tool(server, "temporary_delete_file", {"object_path": object_path})

    async def _call_tool(self, server: McpServerConfig, tool_name: str, arguments: dict[str, str]) -> dict[str, Any]:
        """Open a session, invoke the tool and return its structured payload.

        The WHOLE call (connection + handshake + tool execution) is bounded by
        the MCP tool timeout (NFR-005).

        Args:
            server: The Files MCP server config.
            tool_name: ``temporary_upload_file`` or ``temporary_delete_file``.
            arguments: Tool arguments.

        Returns:
            The tool's structured JSON payload.

        Raises:
            FilesMcpTemporaryError: On connection/timeout/tool errors.
        """
        headers = self._resolved_headers(server)
        try:
            return await asyncio.wait_for(
                self._invoke(server.url or "", headers, tool_name, arguments),
                timeout=self._tool_timeout,
            )
        except TimeoutError as exc:
            logger.error(LogMessage.CHAT_ATTACHMENTS_MCP_TOOL_FAILED, tool_name, server.name)
            raise FilesMcpTemporaryError(
                ErrorMessage.MCP_TOOL_CALL_TIMEOUT.format(name=tool_name, timeout=self._tool_timeout)
            ) from exc
        except FilesMcpTemporaryError:
            raise
        except Exception as exc:
            logger.error(LogMessage.CHAT_ATTACHMENTS_MCP_TOOL_FAILED, tool_name, server.name)
            # Log the FULL exception (incl. anyio ExceptionGroup sub-exceptions)
            # — without exc_info the outer "unhandled errors in a TaskGroup
            # (1 sub-exception)" message hides the real transport/tool error.
            logger.error("Files MCP %r: exception detail: %r", tool_name, exc, exc_info=exc)
            raise FilesMcpTemporaryError(
                ErrorMessage.ATTACHMENT_MCP_UPLOAD_FAILED.format(filename=tool_name, error=exc)
            ) from exc

    async def _invoke(
        self, url: str, headers: dict[str, str], tool_name: str, arguments: dict[str, str]
    ) -> dict[str, Any]:
        """Run the MCP session and return the parsed structured result."""
        async with (
            streamablehttp_client(url, headers=headers or None) as (read, write, _get_session_id),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(
                tool_name,
                arguments=dict(arguments),
                read_timeout_seconds=_resolve_call_timeout(self._tool_timeout),
            )
            if result.isError:
                raise FilesMcpTemporaryError(
                    ErrorMessage.ATTACHMENT_MCP_UPLOAD_FAILED.format(filename=tool_name, error=result.content)
                )
            return _parse_structured_payload(result)


def _parse_structured_payload(result: Any) -> dict[str, Any]:
    """Extract the tool's structured payload from a ``CallToolResult``.

    Prefers ``structuredContent`` (FastMCP structured output), falls back to
    the first JSON ``TextContent`` block.

    Args:
        result: The mcp ``CallToolResult``.

    Returns:
        The structured dict payload.

    Raises:
        FilesMcpTemporaryError: When no structured payload can be parsed.
    """
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                return decoded
    raise FilesMcpTemporaryError(ErrorMessage.ATTACHMENT_MCP_DELETE_FAILED.format(error="no structured payload"))


class McpFilesTemporaryCleaner(TemporaryDocumentCleaner):
    """Best-effort cleanup port adapter (REQ-041..047) — never raises."""

    def __init__(
        self,
        files_client: FilesMcpClient,
        config_store: AgentConfigStore,
        config_loader: AgentConfigLoader,
        mcp_refs_resolver=None,
    ) -> None:
        """Initialize the cleaner.

        Args:
            files_client: The direct Files MCP client used for deletions.
            config_store: Agent YAML store (resolves the Files server).
            config_loader: Agent YAML loader (same chain as the agents routes).
            mcp_refs_resolver: Optional resolver of ``mcp_refs`` registry
                identities into trusted execution configs — resource-only
                agents keep their MCP selection in the registry (REQ-MCP-003).
        """
        self._files_client = files_client
        self._config_store = config_store
        self._config_loader = config_loader
        self._mcp_refs_resolver = mcp_refs_resolver

    async def delete_temporary_documents(self, agent_name: str, refs: list) -> None:  # noqa: ANN401 (TemporaryDocumentRef)
        """Delete all refs via the agent's Files MCP server (idempotent, REQ-045).

        REQ-046: each failure is logged ERROR and retried exactly once; the
        method never raises so an agent response already produced can never be
        masked by a cleanup failure. REQ-047: deletions target only the refs
        supplied by the backend — all under ``chat_tmp/<user_id>/...``.
        """
        if not refs:
            return
        thread_id = refs[0].object_path.split("/")[2] if len(refs[0].object_path.split("/")) > 2 else ""
        logger.info(LogMessage.CHAT_TEMP_CLEANUP_STARTED, len(refs), thread_id)
        server = await self._resolve_files_server(agent_name)
        if server is None:
            logger.error(LogMessage.CHAT_ATTACHMENTS_DOCUMENT_POLICY_FAILED, thread_id, agent_name)
            return
        deleted = 0
        for ref in refs:
            if await self._delete_once(server, ref, thread_id):
                deleted += 1
        if deleted:
            logger.info(LogMessage.CHAT_TEMP_CLEANUP_SUCCEEDED, thread_id, deleted)

    async def _delete_once(self, server: McpServerConfig, ref, thread_id: str) -> bool:  # noqa: ANN401
        """Attempt one deletion with exactly one retry (REQ-046).

        Returns:
            True when a delete call succeeded.
        """
        for attempt in (1, 2):
            try:
                await self._files_client.temporary_delete(server, object_path=ref.object_path)
                return True
            except Exception:  # noqa: BLE001 — cleanup NEVER masks the turn (REQ-046)
                logger.exception(
                    LogMessage.CHAT_TEMP_CLEANUP_FAILED,
                    ref.object_path,
                    thread_id,
                    "temporary_delete call failed",
                )
                if attempt == 1:
                    logger.warning(LogMessage.CHAT_TEMP_CLEANUP_RETRY, ref.object_path, thread_id)
        return False

    async def _resolve_files_server(self, agent_name: str) -> McpServerConfig | None:
        """Resolve the agent's Files MCP server through the store + loader chain.

        Uses the same domain policy helper as the preparation flow (REQ-009);
        a resource-only agent's ``mcp_refs`` selection is merged from the
        registry first (REQ-MCP-003). Any resolution failure logs and returns
        ``None`` (never raises).
        """
        from src.domain.services.chat.mcp_files_policy import find_files_mcp_server

        try:
            yaml_content = await self._config_store.get(agent_name)
            config = self._config_loader.load_from_string(yaml_content)
            if config.mcp_refs and self._mcp_refs_resolver is not None:
                resolved = await self._mcp_refs_resolver(config.mcp_refs)
                if resolved:
                    config = config.model_copy(update={"mcp_servers": [*config.mcp_servers, *resolved]})
        except Exception:  # noqa: BLE001 — cleanup NEVER masks the turn (REQ-046)
            logger.exception(LogMessage.CHAT_TEMP_CLEANUP_FAILED, "(agent config)", agent_name, "resolution failed")
            return None
        return find_files_mcp_server(config)
