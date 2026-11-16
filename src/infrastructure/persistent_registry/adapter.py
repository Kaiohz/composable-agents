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
from dataclasses import dataclass
from uuid import UUID

from src.domain.entities.agent_config import AgentConfig, PromptRef
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_config_loader import AgentConfigLoader
from src.domain.ports.agent_config_repository import AgentConfigRepository
from src.domain.ports.agent_config_store import AgentConfigStore
from src.domain.ports.agent_registry import AgentRegistry
from src.domain.ports.agent_runner import AgentRunner
from src.domain.ports.mcp_tool_loader import McpToolLoader
from src.domain.ports.model_repository import ModelRepository
from src.domain.ports.prompt_repository import PromptRepository
from src.domain.ports.runner_freshness import RunnerFreshness
from src.domain.ports.tracing_provider import TracingProvider
from src.infrastructure.database.rls_context import current_user_id
from src.infrastructure.deepagent.adapter import DeepAgentRunner
from src.infrastructure.deepagent.factory import create_agent_from_config, resolve_model_identity

logger = logging.getLogger(__name__)

# Serving-provider label recorded on the LLM-usage capture when the user has
# no per-user LLM settings (env-fallback serving chain): the agents are always
# served by the OpenAI-compatible ChatOpenAI chain.
_DEFAULT_PROVIDER = "openai"


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
        llm_credentials_resolver=None,
        llm_provider_resolver=None,
        prompt_repository: PromptRepository | None = None,
        model_repository: ModelRepository | None = None,
        runner_freshness: RunnerFreshness | None = None,
    ) -> None:
        self._config_loader = config_loader
        self._config_store = config_store
        self._config_repository = config_repository
        self._mcp_tool_loader = mcp_tool_loader
        self._tracing_provider = tracing_provider
        self._stream_idle_timeout = stream_idle_timeout
        self._invoke_timeout = invoke_timeout
        self._llm_credentials_resolver = llm_credentials_resolver
        self._llm_provider_resolver = llm_provider_resolver
        self._prompt_repository = prompt_repository
        self._model_repository = model_repository
        self._runner_freshness = runner_freshness
        # CACHE-001: keyed by (user_id, agent_name).
        self._runners: dict[tuple[str, str], _CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get_runner(self, agent_name: str) -> AgentRunner:
        """Return a runner for the CURRENT user's agent, rebuilding when stale.

        Args:
            agent_name: Name of the agent.

        Returns:
            AgentRunner ready for invocation, reflecting the latest committed
            dependency state (agent YAML revision, unpinned prompt Latest
            version, model metadata).

        Raises:
            AgentNotFoundError: If no config exists for this user/agent.
            PromptNotFoundError / PromptVersionNotFoundError: If a referenced
                prompt (or pinned version) is missing or inaccessible.
            ModelNotFoundError: If a referenced model is missing.
        """
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

            graph, response_format_model = await create_agent_from_config(
                config,
                self._mcp_tool_loader,
                config_resolver=config_resolver,
                llm_credentials_resolver=self._llm_credentials_resolver,
                prompt_resolver=self._build_prompt_resolver(),
                model_resolver=self._build_model_resolver(),
            )
            # LLM-usage capture identity (dashboard spec §1): the resolved
            # serving model name (AGT-003 precedence) and the user's configured
            # provider label (fallback: the OpenAI-compatible serving chain).
            model_name, _model_base_url = await resolve_model_identity(config, self._build_model_resolver())
            provider = _DEFAULT_PROVIDER
            if self._llm_provider_resolver is not None:
                provider = await self._llm_provider_resolver(uid) or _DEFAULT_PROVIDER
            runner = DeepAgentRunner(
                graph,
                tracing_provider=self._tracing_provider,
                response_format_model=response_format_model,
                stream_idle_timeout=self._stream_idle_timeout,
                invoke_timeout=self._invoke_timeout,
                model_name=model_name,
                provider=provider,
            )
            fingerprint = (
                await self._runner_freshness.fingerprint(agent_name) if self._runner_freshness is not None else ""
            )
            self._runners[key] = _CacheEntry(runner=runner, fingerprint=fingerprint)
            logger.info(LogMessage.AGENT_CACHED, agent_name)
            return runner

    def _build_prompt_resolver(self):
        """Build the registry-backed ``prompt_ref`` resolver (AGT-002), or None."""
        repo = self._prompt_repository
        if repo is None:
            return None

        async def resolve(ref: PromptRef) -> str:
            version = await repo.get_version(str(ref.id), ref.version)
            return version.content

        return resolve

    def _build_model_resolver(self):
        """Build the catalog-backed ``model_ref`` resolver (AGT-003), or None."""
        repo = self._model_repository
        if repo is None:
            return None

        async def resolve(model_id: UUID):
            return await repo.get(str(model_id))

        return resolve

    async def list_agents(self) -> list[str]:
        """List agent names from the metadata repository (current user)."""
        metadata_list = await self._config_repository.list_all()
        return [m.name for m in metadata_list]

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
