"""Shared ``mcp_refs`` execution-config resolution (REQ-MCP-003).

One implementation guarantees the NFR invariant everywhere a selection is
resolved (agent registry runner builds, chat-attachment Files-MCP policy):
each unique registry identity of a selection resolves through the trusted
execution-config path EXACTLY ONCE per construction, and the decrypted
material travels only to the runner's tool loader.
"""

from collections.abc import Awaitable, Callable
from logging import getLogger

from src.domain.entities.mcp_server_config import McpServerConfig
from src.domain.logging.messages import LogMessage
from src.domain.ports.mcp_registry import McpRegistryClient

#: Resolver of one agent's ``mcp_refs`` selection into trusted execution configs.
McpRefsResolver = Callable[[list[str]], Awaitable[list[McpServerConfig]]]

logger = getLogger(__name__)


def build_mcp_refs_resolver(registry_client: McpRegistryClient | None) -> McpRefsResolver | None:
    """Build the registry-backed ``mcp_refs`` resolver, or ``None``.

    Args:
        registry_client: The trusted MCP-registry client (``None`` means no
            registry is wired — the resolver is absent and callers raise their
            own controlled error when a selection is declared).

    Returns:
        An async resolver deduplicating the selection and resolving each
        unique identity once. ``None`` when the client is absent.
    """
    if registry_client is None:
        return None

    async def resolve(refs: list[str]) -> list[McpServerConfig]:
        """Resolve once in declaration order after deduplicating identities."""
        if not refs:
            return []
        configs = [await registry_client.get_execution_config(ref) for ref in dict.fromkeys(refs)]
        logger.info(LogMessage.AGENT_MCP_REFS_RESOLVED, len(configs), len(refs))
        return configs

    return resolve
