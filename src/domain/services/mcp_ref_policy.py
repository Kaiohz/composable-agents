"""Owner-scoped MCP reference policy (req agent-resource-import).

A ``mcp_refs`` selection names servers REGISTERED BY THE CURRENT USER via the
trusted ``mcp-raganything`` registry: a selection referencing an unknown
(or foreign, non-enumerating) server must be refused on every write surface
(built-in assistant customization, agent YAML import/update) BEFORE anything
is persisted, and resolved through the trusted execution-config path at
runner build time.
"""

from src.domain.errors.config import ConfigError
from src.domain.errors.messages import ErrorMessage
from src.domain.ports.mcp_registry import McpRegistryClient


async def ensure_mcp_refs_registered(
    registry_client: McpRegistryClient,
    refs: list[str],
) -> None:
    """Verify every selected identity is a server registered by the current user.

    One masked listing call validates the WHOLE selection (NFR: avoid
    unnecessary remote calls). The registry client is owner-scoped: a foreign
    server is indistinguishable from an unregistered one, so the refusal is
    non-enumerating.

    Args:
        registry_client: The trusted MCP-registry client (originating
            credential forwarded by the adapter).
        refs: The selected registry identities (non-empty list expected).

    Raises:
        ConfigError: When at least one reference is not a registered server
            of the current user.
        McpRegistryError: When the registry is unreachable or unusable — a
            controlled failure, never persisted nor degraded silently here
            (REQ-TX-003).
    """
    if not refs:
        return
    servers = await registry_client.list_servers()
    known = {server.name for server in servers}
    unknown = sorted({ref for ref in refs if ref not in known})
    if unknown:
        raise ConfigError(ErrorMessage.AGENT_IMPORT_MCP_REF_NOT_REGISTERED.format(names=", ".join(unknown)))
