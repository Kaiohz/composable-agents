"""Pure domain service: Files MCP authorization policy (REQ-009/REQ-011).

The backend determines document-attachment authorization from the agent's
``McpServerConfig`` list — the frontend is NEVER the only enforcement point.
A server qualifies only when it is an HTTP transport whose URL ends with
``/files/mcp``; ``/classical/mcp`` alone MUST NOT suffice (EDG-009), and when
an agent declares several servers the one ending with ``/files/mcp`` is
selected (EDG-008).
"""

from src.domain.entities.agent_config import AgentConfig
from src.domain.entities.mcp_server_config import McpServerConfig, McpTransportType

#: URL suffix a Files server MUST end with (REQ-009).
FILES_MCP_URL_SUFFIX = "/files/mcp"


def find_files_mcp_server(config: AgentConfig) -> McpServerConfig | None:
    """Return the agent's Files MCP HTTP server, or ``None`` when absent.

    Args:
        config: The agent configuration.

    Returns:
        The first HTTP McpServerConfig whose URL ends with ``/files/mcp``
        (EDG-008 selection rule), otherwise ``None``.
    """
    for server in config.mcp_servers:
        if server.transport != McpTransportType.HTTP or not server.url:
            continue
        if server.url.rstrip("/").endswith(FILES_MCP_URL_SUFFIX):
            return server
    return None


def allows_document_attachments(config: AgentConfig) -> bool:
    """Return True when an agent may receive document attachments (REQ-009)."""
    return find_files_mcp_server(config) is not None
