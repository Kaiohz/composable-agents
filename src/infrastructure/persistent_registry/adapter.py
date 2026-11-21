"""User-scoped agent registry with dependency-aware runner freshness.

Two cache-correctness mechanisms (spec §7 CACHE-001/CACHE-002):

* CACHE-001 — cache identity is ``(user_id, agent_name)``. Cached runners
  embed user-specific state (per-user :class:`ChatOpenAI` credentials,
  resolved prompt content, model metadata), so entries are never shared
  across users: the user id comes from the request/scheduler RLS contextvar.
* CACHE-002 — every cache hit is validated against a cross-worker dependency
  fingerprint (committed database state via the ``RunnerFreshness`` port).
  Agent YAML edits, new prompt content versions on unpinned references and
  model metadata edits change the fingerprint, so EVERY worker rebuilds on
  the next invocation — not only the process that performed the mutation.
  Process-local eviction alone would not meet the acceptance criteria.

In-flight runs keep their snapshot: the fingerprint is read once per
``get_runner`` call and the returned runner is used for the whole invocation.
"""

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import UUID

from src.application.services.builtin_agent import (
    BUILTIN_AGENT_NAME,
    builtin_agent_config,
    is_builtin_agent_name,
)
from src.domain.entities.agent_config import AgentConfig, PromptRef
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_config_loader import AgentConfigLoader
from src.domain.ports.agent_config_repository import AgentConfigRepository
from src.domain.ports.agent_config_store import AgentConfigStore
from src.domain.ports.agent_registry import AgentRegistry
from src.domain.ports.agent_runner import AgentRunner
from src.domain.ports.mcp_registry import McpRegistryClient
from src.domain.ports.mcp_tool_loader import McpToolLoader
from src.domain.ports.model_repository import ModelRepository
from src.domain.ports.prompt_repository import PromptRepository
from src.domain.ports.runner_freshness import RunnerFreshness
from src.domain.ports.tracing_provider import TracingProvider
from src.domain.services.mcp_execution_resolution import build_mcp_refs_resolver
from src.infrastructure.database.rls_context import current_user_id
from src.infrastructure.deepagent.adapter import DeepAgentRunner
from src.infrastructure.deepagent.builtin_resolution import BuiltinToolsetResolver
from src.infrastructure.deepagent.factory import create_agent_from_config, resolve_model_identity
from src.infrastructure.deepagent.platform_tools import PLATFORM_TOOLS_SELECTOR, AgentPlatformToolsetBuilder
from src.infrastructure.deepagent.toolset_guard import BuiltInToolsetGuardMiddleware

logger = logging.getLogger(__name__)

# Serving-provider label recorded on the LLM-usage capture when the user has
# no per-user LLM settings (env-fallback serving chain): the agents are always
# served by the OpenAI-compatible ChatOpenAI chain.
_DEFAULT_PROVIDER = "openai"

#: LRU cap on the per-user degraded-capability records (REQ-040 advisory data
#: with tiny values; the cap keeps worker memory flat at any user scale).
_DEGRADED_CACHE_MAX_USERS = 256


@dataclass
class LlmRuntimeDependencies:
    """Optional per-user LLM resolution collaborators."""

    credentials_resolver: Callable[[str], Awaitable[tuple[str, str] | None]] | None = None
    provider_resolver: Callable[[str], Awaitable[str | None]] | None = None


@dataclass
class BuiltinRuntimeDependencies:
    """Optional built-in and registry runtime collaborators.

    Grouping them keeps the registry constructor below Sonar's parameter
    threshold while making the built-in resolution and trusted MCP references
    explicit.
    """

    resolver: BuiltinToolsetResolver | None = None
    config_provider: Callable[[], Awaitable[AgentConfig]] | None = None
    mcp_registry_client: McpRegistryClient | None = None


@dataclass
class _CacheEntry:
    """A cached runner plus the dependency fingerprint it was built against."""

    runner: AgentRunner
    fingerprint: str


class PersistentAgentRegistry(AgentRegistry):
    """Registry that loads agent configs from MinIO/PostgreSQL and caches runners per user."""

    def __init__(
        self,
        config_loader: AgentConfigLoader,
        config_store: AgentConfigStore,
        config_repository: AgentConfigRepository,
        mcp_tool_loader: McpToolLoader,
        tracing_provider: TracingProvider | None = None,
        stream_idle_timeout: float = 120.0,
        invoke_timeout: float = 120.0,
        llm_runtime: LlmRuntimeDependencies | None = None,
        prompt_repository: PromptRepository | None = None,
        model_repository: ModelRepository | None = None,
        runner_freshness: RunnerFreshness | None = None,
        builtin_runtime: BuiltinRuntimeDependencies | None = None,
        platform_toolset: AgentPlatformToolsetBuilder | None = None,
    ) -> None:
        self._config_loader = config_loader
        self._config_store = config_store
        self._config_repository = config_repository
        self._mcp_tool_loader = mcp_tool_loader
        self._tracing_provider = tracing_provider
        self._stream_idle_timeout = stream_idle_timeout
        self._invoke_timeout = invoke_timeout
        self._llm_credentials_resolver = llm_runtime.credentials_resolver if llm_runtime else None
        self._llm_provider_resolver = llm_runtime.provider_resolver if llm_runtime else None
        self._prompt_repository = prompt_repository
        self._model_repository = model_repository
        self._runner_freshness = runner_freshness
        self._builtin_resolver = builtin_runtime.resolver if builtin_runtime else None
        self._builtin_config_provider = builtin_runtime.config_provider if builtin_runtime else None
        self._platform_toolset = platform_toolset or AgentPlatformToolsetBuilder()
        # REQ-MCP-003: resource-only agents persist ``mcp_refs`` identities; the
        # runner resolves them against the user's registry at build time.
        self._mcp_registry_client = builtin_runtime.mcp_registry_client if builtin_runtime else None
        # CACHE-001: keyed by (user_id, agent_name).
        self._runners: dict[tuple[str, str], _CacheEntry] = {}
        self._lock = asyncio.Lock()
        # REQ-040 degradation info of the LATEST builtin build (read right
        # after the build through :meth:`degraded_capabilities`), LRU-bounded
        # per user so the worker memory stays flat over time.
        self._degraded: OrderedDict[str, list[dict]] = OrderedDict()

    async def get_runner(self, agent_name: str) -> AgentRunner:
        """Return a runner for the CURRENT user's agent, rebuilding when stale.

        The built-in Composable AI Assistant follows a DELIBERATE REBUILD
        strategy (REQ-032/REQ-038): its dynamic toolset (platform tools,
        built-in MCP mounts, the running user's registered MCP servers,
        ``create_pdf``) is resolved fresh on every construction in the
        current user's context, and the resulting runner is never cached —
        a registry mutation is reflected on the very next turn without any
        restart and without cross-worker cache concerns.

        Args:
            agent_name: Name of the agent (or the built-in assistant name).

        Returns:
            AgentRunner ready for invocation, reflecting the latest committed
            dependency state (agent YAML revision, unpinned prompt Latest
            version, model metadata) or the freshly resolved built-in
            toolset.

        Raises:
            AgentNotFoundError: If no config exists for this user/agent.
            PromptNotFoundError / PromptVersionNotFoundError: If a referenced
                prompt (or pinned version) is missing or inaccessible.
            ModelNotFoundError: If a referenced model is missing.
            LlmNotConfiguredError: If the running user has no LLM credentials.
        """
        if is_builtin_agent_name(agent_name):
            return await self._build_builtin_runner()

        uid = current_user_id.get() or ""
        key = (uid, agent_name)

        entry = self._runners.get(key)
        if entry is not None:
            if self._runner_freshness is None:
                logger.info(LogMessage.AGENT_CACHE_HIT, agent_name)
                return entry.runner
            current_fingerprint = await self._runner_freshness.fingerprint(agent_name)
            if current_fingerprint == entry.fingerprint:
                logger.info(LogMessage.AGENT_CACHE_HIT, agent_name)
                return entry.runner
            # CACHE-002: a dependency changed (or vanished) since the cached
            # runner was built — evict and rebuild below.
            logger.info(LogMessage.RUNNER_FRESHNESS_REBUILD, agent_name, uid)
            async with self._lock:
                self._runners.pop(key, None)

        async with self._lock:
            entry = self._runners.get(key)
            if entry is not None:
                return entry.runner

            logger.info(LogMessage.AGENT_BUILDING, agent_name)
            yaml_content = await self._config_store.get(agent_name)
            config = self._config_loader.load_from_string(yaml_content)

            async def config_resolver(name: str) -> AgentConfig:
                referenced_yaml = await self._config_store.get(name)
                return self._config_loader.load_from_string(referenced_yaml)

            # REQ-028: the declarative platform-tools selector ("mcp") swaps
            # in the LangChain platform toolset for custom agents.
            extra_tools, config = self._apply_platform_selector(config)

            runner = await self._construct_runner(
                config,
                config_resolver=config_resolver,
                extra_tools=extra_tools,
            )
            fingerprint = (
                await self._runner_freshness.fingerprint(agent_name) if self._runner_freshness is not None else ""
            )
            self._runners[key] = _CacheEntry(runner=runner, fingerprint=fingerprint)
            logger.info(LogMessage.AGENT_CACHED, agent_name)
            return runner

    def _apply_platform_selector(self, config: AgentConfig) -> tuple[list, AgentConfig]:
        """Swap the ``tools: ["mcp"]`` selector for the platform toolset.

        The selector is NOT a module path; it must never reach the YAML tool
        resolver. The platform tools (including the one-shot execution tool,
        REQ-029-guarded) are injected as pre-resolved objects.

        Returns:
            ``(extra_tools, adjusted_config)``.
        """
        selector_hits = [t for t in config.tools if t == PLATFORM_TOOLS_SELECTOR]
        if not selector_hits:
            return [], config
        remaining = [t for t in config.tools if t != PLATFORM_TOOLS_SELECTOR]
        adjusted = config.model_copy(update={"tools": remaining})
        tools = self._platform_toolset.build(include_run=True)
        logger.info(LogMessage.AGENT_MCP_TOOLS_LOADED, len(tools), config.name)
        return tools, adjusted

    async def _build_builtin_runner(self) -> DeepAgentRunner:
        """Resolve the built-in assistant dynamically and build a fresh runner.

        The dynamic toolset is resolved in the CURRENT user's context: the
        platform tools INCLUDING one-shot execution (REQ-011/036 — the
        assistant may run OTHER user agents; self-invocation is refused by
        the invocation chain and by the run tool itself, REQ-030), built-in
        MCP mounts, ONLY the MCP servers explicitly selected in the user's
        customization via ``mcp_refs`` (REQ-BUILTIN-003; per-server failure
        isolation, REQ-040; deterministic collision namespacing, REQ-039),
        the builtin ``create_pdf`` tool, and the standard non-sensitive
        runtime tools supplied by the deep-agent framework.
        """
        uid = current_user_id.get() or ""
        logger.info(LogMessage.BUILTIN_AGENT_RESOLVING, uid)
        if self._builtin_config_provider is not None:
            config = await self._builtin_config_provider()
        else:
            config = builtin_agent_config()

        # Platform tools are built FIRST so their names are reserved BEFORE
        # the dynamic MCP resolution: a registered server advertising a
        # composable_* tool name is namespaced instead of shadowing it (I3).
        platform_tools = self._platform_toolset.build(include_run=True)
        reserved = frozenset(tool.name for tool in platform_tools)

        dynamic_tools: list = []
        degraded: list[dict] = []
        if self._builtin_resolver is not None:
            dynamic_tools, degraded = await self._builtin_resolver.resolve_dynamic_tools(
                reserved_names=reserved,
                selected_refs=config.mcp_refs,
            )
        self._record_degraded_capabilities(uid, degraded)

        # The SELECTED servers are already mounted (and namespaced) by the
        # dynamic resolver above: the graph must NOT resolve ``mcp_refs`` a
        # second time (NFR: one resolution per MCP per construction), must not
        # bind a second raw-named copy of the same tools, and must keep the
        # per-source degradation isolation of REQ-040 (a resolution error in
        # the generic path would hard-fail the build instead of degrading).
        graph_config = config.model_copy(update={"mcp_refs": []})
        runner = await self._construct_runner(
            graph_config,
            config_resolver=None,
            extra_tools=[*platform_tools, *dynamic_tools],
            middleware=[BuiltInToolsetGuardMiddleware()],
        )
        logger.info(
            LogMessage.BUILTIN_AGENT_TOOLS_RESOLVED, uid, len(platform_tools) + len(dynamic_tools), len(degraded)
        )
        return runner

    def _record_degraded_capabilities(self, uid: str, degraded: list[dict]) -> None:
        """Record one build's degradation with an LRU bound on the user set.

        REQ-040: the one-shot use case reads this immediately after its build
        in the same task. The dict is bounded (LRU eviction) so an
        ever-growing user population cannot grow the worker's memory.
        """
        self._degraded[uid] = degraded
        self._degraded.move_to_end(uid)
        while len(self._degraded) > _DEGRADED_CACHE_MAX_USERS:
            self._degraded.popitem(last=False)

    async def _construct_runner(
        self,
        config: AgentConfig,
        *,
        config_resolver=None,
        extra_tools: list | None = None,
        middleware: list | None = None,
    ) -> DeepAgentRunner:
        """Build one DeepAgentRunner from a resolved configuration (shared path).

        Centralizes graph creation, the model/credential resolution and the
        LLM-usage identity so the cached custom-agent path and the fresh
        built-in path stay behaviorally identical. ``agent_name``-scoped
        logging happens at the call sites.
        """
        graph, response_format_model = await create_agent_from_config(
            config,
            self._mcp_tool_loader,
            config_resolver=config_resolver,
            llm_credentials_resolver=self._llm_credentials_resolver,
            prompt_resolver=self._build_prompt_resolver(),
            model_resolver=self._build_model_resolver(),
            extra_tools=extra_tools,
            middleware=middleware or (),
            mcp_refs_resolver=self._build_mcp_refs_resolver(),
        )
        # LLM-usage capture identity (dashboard spec §1): the resolved
        # serving model name (AGT-003 precedence) and the user's configured
        # provider label (fallback: the OpenAI-compatible serving chain).
        model_name, _model_base_url = await resolve_model_identity(config, self._build_model_resolver())
        provider = _DEFAULT_PROVIDER
        uid = current_user_id.get() or ""
        if self._llm_provider_resolver is not None:
            provider = await self._llm_provider_resolver(uid) or _DEFAULT_PROVIDER
        return DeepAgentRunner(
            graph,
            tracing_provider=self._tracing_provider,
            response_format_model=response_format_model,
            stream_idle_timeout=self._stream_idle_timeout,
            invoke_timeout=self._invoke_timeout,
            model_name=model_name,
            provider=provider,
        )

    def _build_prompt_resolver(self):
        """Build the registry-backed ``prompt_ref`` resolver (AGT-002), or None."""
        repo = self._prompt_repository
        if repo is None:
            return None

        async def resolve(ref: PromptRef) -> str:
            version = await repo.get_version(str(ref.id), ref.version)
            return version.content

        return resolve

    def _build_mcp_refs_resolver(self):
        """The shared ``mcp_refs`` resolver (REQ-MCP-003), or None.

        The closure is built ONCE (one implementation domain-wide — NFR: a
        drift would silently change the "resolved exactly once" guarantee) and
        resolves each unique registry identity through the trusted
        execution-config path exactly once per construction.
        """
        return build_mcp_refs_resolver(self._mcp_registry_client)

    def _build_model_resolver(self):
        """Build the catalog-backed ``model_ref`` resolver (AGT-003), or None."""
        repo = self._model_repository
        if repo is None:
            return None

        async def resolve(model_id: UUID):
            return await repo.get(str(model_id))

        return resolve

    async def list_agents(self) -> list[str]:
        """List agent names visible to the current user, built-in assistant first.

        REQ-031: the virtual built-in assistant is part of every
        authenticated user's catalog (registry discovery, thread creation,
        one-shot execution) even with zero custom agents. It is NEVER a
        persisted row and never duplicated by one.
        """
        metadata_list = await self._config_repository.list_all()
        names = [m.name for m in metadata_list if not is_builtin_agent_name(m.name)]
        return [BUILTIN_AGENT_NAME, *names]

    def degraded_capabilities(self, agent_name: str) -> list[dict]:
        """Return the degradation info of the LATEST build (REQ-040).

        The agent registry records the optional-capability failures of its
        most recent built-in assistant build; the one-shot use case forwards
        them into the run result so an outage is surfaced as degraded
        capability information, never a silent capability loss.
        """
        uid = current_user_id.get() or ""
        if is_builtin_agent_name(agent_name):
            return list(self._degraded.get(uid, []))
        return []

    async def invalidate(self, agent_name: str) -> None:
        """Evict cached runners for the agent under the current user's scope.

        With no user context set (system jobs) every user's entry for the
        agent is evicted. Cross-worker correctness does not rely on this:
        the CACHE-002 fingerprint check rebuilds stale entries everywhere.
        """
        uid = current_user_id.get()
        async with self._lock:
            if uid is not None:
                self._runners.pop((uid, agent_name), None)
            else:
                for key in [k for k in self._runners if k[1] == agent_name]:
                    self._runners.pop(key, None)
            logger.info(LogMessage.AGENT_CACHE_INVALIDATED, agent_name)

    async def close(self) -> None:
        """Clear all cached runners."""
        logger.info(LogMessage.REGISTRY_CLOSING, len(self._runners))
        self._runners.clear()
