"""Dynamic toolset resolution for the built-in Composable AI Assistant.

Per-build (per-turn) resolution in the RUNNING user's context (REQ-036/038):

* the built-in MCP mounts exposed by ``mcp-raganything`` (Files, Classical
  RAG, Web, Memories) when their internal base URL is configured;
* ONLY the MCP servers the current user EXPLICITLY SELECTED in the built-in
  customization (``mcp_refs``) — resolved through the trusted
  execution-config path (REQ-037); the entire user registry is NEVER mounted
  automatically (REQ-BUILTIN-003);
* the builtin ``create_pdf`` LangChain tool.

CONCURRENCY (spec NFR): optional MCP tool sources are loaded CONCURRENTLY
(``asyncio.gather`` with ``return_exceptions=True``, fanned out through a
``asyncio.Semaphore``) — each source stays bounded by the existing
``MCP_TOOL_TIMEOUT`` carried by the tool loader, so a slow source can never
stall the gather beyond the configured timeout.

Failure isolation (REQ-040): a failing optional source — connection error,
timeout, malformed registry payload, ANY exception type — is logged safely
and surfaced as degraded capability information; the remaining tools
(platform, healthy servers) stay available and the build never raises.

Deterministic namespacing (REQ-039): the PLATFORM mounts are collected in a
fixed order and the selected user mounts follow the customization's selection
order; the tool names are assigned AFTER the gather, sequentially, in that
same order — so the resolution is deterministic across turns. The platform
tool names are RESERVED first (callers pass ``reserved_names``), so a
registered MCP server can never silently shadow a platform ``composable_*``
tool; colliding candidates become ``<server>__<tool>``. A collision therefore
NEVER silently replaces one tool with another's.
"""

import asyncio
import logging
from typing import Any

from src.config import Settings
from src.domain.entities.mcp_server_config import McpServerConfig, McpTransportType
from src.domain.logging.messages import LogMessage
from src.domain.ports.mcp_registry import McpRegistryClient
from src.domain.ports.mcp_tool_loader import McpToolLoader

logger = logging.getLogger(__name__)

#: Built-in MCP mount paths owned by mcp-raganything (spec Technical Notes) —
#: resolved against the configured internal base URL, never the frontend's
#: config.json.
BUILTIN_MOUNTS: tuple[tuple[str, str], ...] = (
    ("files", "/files/mcp"),
    ("classical", "/classical/mcp"),
    ("web", "/web/mcp"),
    ("memories", "/memories/mcp"),
)

#: Maximum concurrent optional MCP tool sources resolved per build. Each
#: source is already bounded by the loader's ``MCP_TOOL_TIMEOUT``; the
#: semaphore caps the fan-out so a user with many registered servers cannot
#: open an unbounded number of simultaneous connections.
MAX_CONCURRENT_MCP_SOURCES = 8

#: Log/reason rendering bound for a degraded capability entry.
_REASON_MAX_CHARS = 300


def _namespaced_tool_name(server_name: str, tool_name: str, taken: set[str]) -> str:
    """Return the agent-facing tool name for one server tool (REQ-039).

    Deterministic: the caller iterates the sources in their stable collected
    order, so the FIRST source to claim a name keeps it unchanged; later
    claimants get ``<server>__<tool>`` (numeric suffix on double collisions).
    """
    if tool_name not in taken:
        taken.add(tool_name)
        return tool_name
    candidate = f"{server_name}__{tool_name}"
    if candidate in taken:
        suffix = 2
        while f"{candidate}__{suffix}" in taken:
            suffix += 1
        candidate = f"{candidate}__{suffix}"
    logger.info(LogMessage.BUILTIN_TOOLNAME_COLLISION, tool_name, server_name, candidate)
    taken.add(candidate)
    return candidate


def _rename(tool: Any, name: str) -> Any:
    """Return a copy of ``tool`` exposing ``name`` as its agent-facing name."""
    try:
        return tool.model_copy(update={"name": name})
    except Exception:  # pragma: no cover — defensive: never lose the tool
        return tool


def _safe_reason(exc: BaseException) -> str:
    """Render a bounded, non-secret failure reason for a degraded source."""
    detail = str(getattr(exc, "detail", "") or exc).strip().replace("\n", " ")
    return detail[:_REASON_MAX_CHARS] if detail else "MCP server unavailable"


def _uid() -> str:
    """Return the current RLS user id for logs (or an empty marker)."""
    from src.infrastructure.database.rls_context import current_user_id

    return current_user_id.get() or ""


class BuiltinToolsetResolver:
    """Resolve the built-in assistant's dynamic capabilities per build."""

    def __init__(
        self,
        mcp_tool_loader: McpToolLoader,
        registry_client: McpRegistryClient | None,
        builtin_mcp_base_url_provider,
        agent_name: str = "composable-ai-assistant",
    ) -> None:
        """Wire the resolver.

        Args:
            mcp_tool_loader: The outbound MCP tool loader port.
            registry_client: The trusted registry client (``None`` disables
                registered-server resolution rather than failing the build).
            builtin_mcp_base_url_provider: Zero-arg callable returning the
                configured ``mcp-raganything`` internal base URL (may be
                empty — the built-in mounts are then simply absent).
            agent_name: Log context name.
        """
        self._mcp_tool_loader = mcp_tool_loader
        self._registry_client = registry_client
        self._builtin_base_url_provider = builtin_mcp_base_url_provider
        self._agent_name = agent_name

    async def resolve_dynamic_tools(
        self, reserved_names: frozenset[str] = frozenset(), selected_refs: list[str] | None = None
    ) -> tuple[list, list[dict]]:
        """Resolve MCP mounts, the SELECTED registered servers and ``create_pdf``.

        Args:
            reserved_names: Agent-facing tool names already claimed by the
                platform toolset — a registered MCP server advertising one of
                these names is deterministically namespaced instead of
                silently shadowing the platform tool (REQ-039).
            selected_refs: The built-in customization's explicit MCP selection
                (registry identities). ``None``/empty mounts NO user-registered
                server: only the explicitly selected servers are mounted, in
                addition to the platform tooling (REQ-BUILTIN-003).

        Returns:
            ``(tools, degraded)`` — the resolved LangChain tools plus the
            JSON-safe degradation entries (REQ-040) for the failing optional
            sources. NEVER raises: a failing registry or source degrades.
        """
        sources, collection_degraded = await self._collect_sources(selected_refs)
        tools, load_degraded = await self._load_sources(sources, reserved_names)
        tools.append(self._pdf_tool())
        degraded = [*collection_degraded, *load_degraded]
        logger.info(LogMessage.BUILTIN_AGENT_TOOLS_RESOLVED, _uid(), len(tools), len(degraded))
        return tools, degraded

    async def _collect_sources(self, selected_refs: list[str] | None) -> tuple[list[McpServerConfig], list[dict]]:
        """Collect the optional MCP sources in their stable, sorted order.

        The platform mounts always apply; user-registered servers are mounted
        ONLY when explicitly selected (``mcp_refs``), resolved through the
        trusted execution-config path with per-construction deduplication.
        """
        sources: list[McpServerConfig] = []
        degraded: list[dict] = []
        base_url = (self._builtin_base_url_provider() or "").strip().rstrip("/")
        if base_url:
            for mount_name, mount_path in BUILTIN_MOUNTS:
                sources.append(
                    McpServerConfig(name=mount_name, transport=McpTransportType.HTTP, url=f"{base_url}{mount_path}")
                )
        if selected_refs:
            sources.extend(await self._collect_selected_sources(list(selected_refs), degraded))
            logger.info(LogMessage.BUILTIN_MCP_SELECTION_APPLIED, len(selected_refs), _uid())
        return sources, degraded

    async def _collect_selected_sources(self, selected_refs: list[str], degraded: list[dict]) -> list[McpServerConfig]:
        """Resolve ONLY the explicitly selected registered servers (REQ-BUILTIN-003).

        Each unique reference resolves its decrypted execution config through
        the trusted path (REQ-037), exactly once per construction (NFR). A
        failing registry or an unknown server degrades to capability
        information without ever failing the built-in build (REQ-040).
        """
        if self._registry_client is None:
            return []
        configured: list[McpServerConfig] = []
        seen: set[str] = set()
        for ref in selected_refs:
            if ref in seen:
                continue
            seen.add(ref)
            try:
                configured.append(await self._registry_client.get_execution_config(ref))
            # A single missing server degrades; it must never brick the build.
            except Exception as exc:  # noqa: BLE001
                self._record_degradation(degraded, ref, exc)
        return configured

    @staticmethod
    def _record_degradation(degraded: list[dict], name: str, exc: BaseException) -> None:
        """Log + record one failing optional source (never raise, REQ-040)."""
        reason = _safe_reason(exc)
        logger.warning(LogMessage.BUILTIN_AGENT_OPTIONAL_TOOL_FAILED, name, reason)
        degraded.append({"name": name, "reason": reason})

    async def _load_sources(
        self, sources: list[McpServerConfig], reserved_names: frozenset[str]
    ) -> tuple[list, list[dict]]:
        """Load every source's tools CONCURRENTLY, then assign names in order.

        Concurrency (NFR): one bounded ``gather`` over the semaphore; per-source
        failure isolation via ``return_exceptions=True``. Names are assigned
        AFTER the gather, sequentially, in the deterministic source order, so
        the collision resolution is stable across turns (REQ-039) and the
        platform's reserved names are never shadowed.
        """
        taken: set[str] = set(reserved_names)
        degraded: list[dict] = []
        if not sources:
            return [], degraded

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_MCP_SOURCES)
        results = await asyncio.gather(
            *(self._load_one_bounded(semaphore, config) for config in sources),
            return_exceptions=True,
        )
        tools = self._merge_source_results(sources, results, taken, degraded)
        return tools, degraded

    @staticmethod
    def _merge_source_results(
        sources: list[McpServerConfig],
        results: list,
        taken: set[str],
        degraded: list[dict],
    ) -> list:
        """Zip source configs with their gathered results, in stable order.

        A failed source (any exception type) is recorded as degraded
        capability information; the healthy sources' tools are appended with
        deterministic namespacing against the shared ``taken`` set.
        """
        tools: list = []
        for config, result in zip(sources, results, strict=True):
            if isinstance(result, BaseException):
                reason = _safe_reason(result)
                logger.warning(LogMessage.BUILTIN_AGENT_OPTIONAL_TOOL_FAILED, config.name, reason)
                degraded.append({"name": config.name, "reason": reason})
                continue
            for tool in result:
                facing_name = _namespaced_tool_name(config.name, str(tool.name), taken)
                tools.append(_rename(tool, facing_name))
        return tools

    async def _load_one_bounded(self, semaphore: asyncio.Semaphore, config: McpServerConfig) -> list:
        """Load one source's tools under the shared fan-out semaphore."""
        async with semaphore:
            return await self._mcp_tool_loader.load_tools([config])

    def _pdf_tool(self) -> Any:
        """Return the built-in ``create_pdf`` tool (module-level, REQ-036)."""
        from src.infrastructure.deepagent.pdf_tools import create_pdf

        return create_pdf


def builtin_mcp_base_url(settings: Settings):
    """Build the zero-arg provider returning the configured base URL."""

    def _provider() -> str:
        return settings.builtin_mcp_base_url

    return _provider
