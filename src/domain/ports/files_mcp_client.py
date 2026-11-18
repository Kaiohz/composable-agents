"""Outbound port: direct Files MCP temporary tool client (REQ-029..035).

The chat attachments flow talks to the ``/files/mcp`` FastMCP server of the
RAG service through direct MCP tool invocations of the two technical tools
``temporary_upload_file`` / ``temporary_delete_file`` (REQ-029). These tools
are NEVER exposed to the LLM (REQ-036) — only the backend calls them.

Implementations open a per-call streamable-HTTP MCP client session with the
user-credential headers resolved exactly like the agent tool loader
(``${USER_JWT}`` / ``${USER_API_KEY}`` reuse, NFR-011) and enforce the
existing MCP tool timeout (NFR-005).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.domain.entities.mcp_server_config import McpServerConfig


@dataclass(frozen=True)
class FilesMcpUpload:
    """Result of a ``temporary_upload_file`` MCP call (REQ-033 contract).

    Attributes:
        object_name: The ``chat_tmp/...`` object path stored in RustFS.
        size: Decoded payload size in bytes.
        content_type: Declared MIME type of the uploaded object.
    """

    object_name: str
    size: int
    content_type: str


class FilesMcpClient(ABC):
    """Outbound port driving the Files MCP temporary tools directly."""

    @abstractmethod
    async def temporary_upload(
        self,
        server: McpServerConfig,
        *,
        file_name: str,
        content_base64: str,
        content_type: str,
        prefix: str,
    ) -> FilesMcpUpload:
        """Upload a temporary document via the MCP ``temporary_upload_file`` tool.

        Args:
            server: The agent's Files MCP server config (URL + credential
                placeholder headers).
            file_name: Attachment basename.
            content_base64: Strict-validated base64 payload.
            content_type: Attachment MIME type.
            prefix: User-isolated temporary prefix
                (``chat_tmp/<user_id>/<thread_id>/<attachment_id>/``).

        Returns:
            The :class:`FilesMcpUpload` contract returned by the server.

        Raises:
            FilesMcpTemporaryError: When the MCP call fails or times out.
        """
        ...

    @abstractmethod
    async def temporary_delete(self, server: McpServerConfig, *, object_path: str) -> None:
        """Delete a temporary object via the MCP ``temporary_delete_file`` tool.

        Idempotent (REQ-045): an already-deleted object is not an error.

        Args:
            server: The agent's Files MCP server config.
            object_path: The ``chat_tmp/<user_id>/...`` path to remove.

        Raises:
            FilesMcpTemporaryError: When the MCP call fails or times out.
        """
        ...
