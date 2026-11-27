import asyncio
import logging
from collections import deque
from typing import Any, cast

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection

from src.domain.entities.mcp_server_config import McpServerConfig, McpTransportType
from src.domain.errors.mcp import McpConnectionError, McpToolLoadError
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage
from src.domain.ports.mcp_tool_loader import McpToolLoader
from src.domain.services.agent_scope import agent_scope_slug
from src.domain.services.mcp_auth_mode import user_credential_headers
from src.infrastructure.database.rls_context import current_auth_context
from src.infrastructure.env_utils import resolve_all_vars, resolve_headers_drop_empty

logger = logging.getLogger(__name__)

#: Technical MCP tools reserved for the BACKEND (chat-pdf-attachments REQ-036).
#: They drive the temporary document upload/cleanup directly (spec §5.3) and
#: are NEVER exposed to the LLM; the normal ``list_files`` / ``list_folders``
#: / ``read_file`` tools keep flowing through (REQ-037).
BACKEND_ONLY_TOOL_NAMES = frozenset({"temporary_upload_file", "temporary_delete_file"})


def _merge_headers(base: dict[str, str], injected: dict[str, str]) -> dict[str, str]:
    """Merge header mappings case-insensitively; the injected mapping wins.

    HTTP header names are case-insensitive, so a configured ``authorization``
    and an injected ``Authorization`` are the SAME header: the merged mapping
    keeps the INJECTED spelling/value (feat-mcp-oauth2 credential-precedence
    contract, explicitly normalized instead of relying on httpx's accidental
    last-write-wins ordering).

    Args:
        base: The configured (explicit) headers.
        injected: The credential headers taking precedence.

    Returns:
        A fresh mapping; the last-written spelling per lowercase key wins.
    """
    merged: dict[str, tuple[str, str]] = {key.lower(): (key, value) for key, value in base.items()}
    for key, value in injected.items():
        merged[key.lower()] = (key, value)
    return dict(merged.values())


#: Automatically injected agent-scope header (REQ-016): every outbound HTTP
#: MCP server connection of a scoped agent carries ``X-Composable-Agent:
#: {scope_slug}`` (the canonical slug of the agent/subagent name — review
#: C-1) so the receiving mcp-raganything server resolves the caller's space
#: without ever seeing an illegal value. Platform-injected at tool-load
#: time — NEVER an LLM-supplied parameter.
AGENT_SCOPE_HEADER = "X-Composable-Agent"


class LangchainMcpToolLoader(McpToolLoader):
    """Adapter MCP utilisant langchain-mcp-adapters pour charger des outils."""

    def __init__(self, tool_timeout: float = 60.0) -> None:
        # Bounded reference list for the lifecycle sweep (close()). Since
        # langchain-mcp-adapters >= 0.3 the per-call MultiServerMCPClient is
        # STATELESS (sessions are created per call and close() is a no-op), so
        # the accumulation is diagnostic only; the maxlen keeps a worker that
        # rebuilds runners every turn (built-in assistant) at a flat memory.
        self._clients: deque[MultiServerMCPClient] = deque(maxlen=64)
        # Per-call timeout for each MCP tool invocation. A hung tool is converted
        # to a ToolException -> ToolMessage(error) so the agent can recover
        # (retry/skip) instead of stalling or being killed.
        self._tool_timeout = tool_timeout

    def _http_connection_spec(self, config: McpServerConfig, agent_scope: str | None = None) -> dict[str, Any]:
        """Build the ``streamable_http`` connection spec for one HTTP server.

        Credential precedence: REQ-005 (stored auth_token as
        ``Authorization: Bearer <value>`` — langchain-mcp-adapters 0.3.0
        StreamableHttpConnection NEVER accepts the ``auth_token`` key,
        TypeError) then REQ-004 (the connected user's credential, injected
        by the domain rule). An empty resolution drops the entry again, and
        the user credential (last write) wins the precedence chain.

        REQ-004: ``auth_mode: "user_token"`` injects the connected user's
        credential (the domain rule decides which header carries it);
        colliding header names are replaced case-insensitively by the
        injected credential. One resolution pass resolves env/user
        placeholders and drops empty/credential-empty entries (never send a
        malformed ``Authorization: Bearer `` or empty ``X-API-Key``).

        REQ-016/REQ-017 + review C-1: a scoped build additionally injects
        ``X-Composable-Agent`` (the injected header WINS over a
        case-insensitive YAML-configured collision, ``_merge_headers``
        semantics) and resolves the ``${AGENT_NAME}`` placeholder of the
        YAML-configured headers. BOTH values go through the canonical
        name→slug mapping (:func:`agent_scope_slug` — the ONE owner shared
        with the spaces API and the memories closure), so the receiver's
        REQ-002 charset rule can never be violated and no illegal header
        byte (CR/LF, control chars) can ever be emitted: a slug-valid name
        passes through unchanged, any other name maps deterministically.
        The injection is platform-side, at tool-load time — never an LLM
        choice. ``agent_scope=None`` (built-in assistant, personal scope)
        injects nothing.
        """
        injected: dict[str, str] = {}
        if config.auth_token:
            token = resolve_all_vars(config.auth_token)
            if token:
                injected["Authorization"] = f"Bearer {token}"
        injected.update(user_credential_headers(config.auth_mode, current_auth_context.get()))
        # REQ-016 + C-1: the agent-scope header rides the injected (winning)
        # layer, and its value is the canonical slug — never the raw name.
        if agent_scope:
            scope = agent_scope_slug(agent_scope)
            injected[AGENT_SCOPE_HEADER] = scope
            # REQ-017 + C-1: the YAML-configured headers resolve ${AGENT_NAME}
            # against the SAME canonical slug (before os.environ).
            extra_vars = {"AGENT_NAME": scope}
        else:
            extra_vars = None
        return {
            "transport": "streamable_http",
            "url": config.url,
            "headers": resolve_headers_drop_empty(
                _merge_headers(dict(config.headers), injected), extra_vars=extra_vars
            ),
        }

    async def load_tools(self, configs: list[McpServerConfig], agent_scope: str | None = None) -> list[Any]:
        """Charge les outils depuis les serveurs MCP configures.

        Args:
            configs: Liste des configurations de serveurs MCP.
            agent_scope: Nom de l'agent courant dont les connexions HTTP
                sortantes relevent (mappé sur le slug canonique avant
                injection — ``None`` pour le scope personal, aucun en-tete
                ``X-Composable-Agent``).

        Returns:
            Liste des outils charges depuis les serveurs MCP.

        Raises:
            McpConnectionError: Si la connexion a un serveur MCP echoue.
            McpToolLoadError: Si le chargement des outils echoue.
        """
        server_configs: dict[str, dict[str, Any]] = {}

        for config in configs:
            if config.transport == McpTransportType.STDIO:
                server_configs[config.name] = {
                    "transport": "stdio",
                    "command": config.command,
                    "args": config.args,
                    "env": resolve_headers_drop_empty(config.env),
                }
            elif config.transport == McpTransportType.HTTP:
                server_configs[config.name] = self._http_connection_spec(config, agent_scope)

        try:
            # The client expects a mapping of validated connection TypedDicts.
            # Our runtime mapping is heterogeneous (stdio/url/headers/env, all
            # resolved and validated above), which the TypedDict union cannot
            # express statically, so the mapping is cast once here.
            client = MultiServerMCPClient(cast("dict[str, Connection]", server_configs))
            self._clients.append(client)
            tools = await client.get_tools()
            loaded = self._patch_sync_support(tools)
            # REQ-036: the backend-only temporary tools are filtered here, at
            # the loader level, so they never become LLM-facing tools.
            return [tool for tool in loaded if tool.name not in BACKEND_ONLY_TOOL_NAMES]
        except ConnectionError as e:
            logger.exception(LogMessage.MCP_CONNECT_FAILED)
            raise McpConnectionError(ErrorMessage.MCP_CONNECTION_ERROR.format(error=e)) from e
        except Exception as e:
            logger.exception(LogMessage.MCP_TOOLS_LOAD_FAILED)
            raise McpToolLoadError(ErrorMessage.MCP_TOOL_LOAD_ERROR.format(error=e)) from e

    async def close(self) -> None:
        """Ferme toutes les connexions MCP ouvertes.

        MultiServerMCPClient v0.1.0+ is stateless (no close method).
        Guard handles future library versions that may add cleanup.
        """
        for client in self._clients:
            close_fn = getattr(client, "close", None)
            if close_fn and callable(close_fn):
                result = close_fn()
                if hasattr(result, "__await__"):
                    await result
        self._clients.clear()

    def _patch_sync_support(self, tools: list[BaseTool]) -> list[BaseTool]:
        """Patch MCP tools: per-call timeout + sync invocation support.

        langchain-mcp-adapters creates async-only StructuredTools (coroutine only,
        no func). deepagents invokes subagent tools synchronously, which crashes
        with NotImplementedError. We recreate each tool with:
          - a coroutine wrapped in asyncio.wait_for(tool_timeout): a hung tool
            raises ToolException -> ToolMessage(error) so the agent RECOVERS
            (the LLM can retry/skip) instead of stalling or being killed.
          - a sync func that runs the same timed coroutine for deepagents.
        """
        patched: list[BaseTool] = []
        for tool in tools:
            if isinstance(tool, StructuredTool) and tool.coroutine is not None and tool.func is None:
                original_coro = tool.coroutine
                timeout = self._tool_timeout
                tool_name = tool.name

                async def timed_coro(*args, _oc=original_coro, _to=timeout, _name=tool_name, **kwargs):
                    try:
                        return await asyncio.wait_for(_oc(*args, **kwargs), timeout=_to)
                    except TimeoutError as e:
                        logger.warning(LogMessage.MCP_TOOL_TIMEOUT, _name, _to)
                        raise ToolException(ErrorMessage.MCP_TOOL_CALL_TIMEOUT.format(name=_name, timeout=_to)) from e

                def sync_wrapper(*args, _tc=timed_coro, **kwargs):
                    return asyncio.get_event_loop().run_until_complete(_tc(*args, **kwargs))

                patched.append(
                    StructuredTool(
                        name=tool.name,
                        description=tool.description,
                        args_schema=tool.args_schema,
                        func=sync_wrapper,
                        coroutine=timed_coro,
                        response_format=tool.response_format,
                        metadata=tool.metadata,
                    )
                )
            else:
                patched.append(tool)
        return patched
