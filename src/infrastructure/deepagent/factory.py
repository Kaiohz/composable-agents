import asyncio
import importlib
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import StoreBackend
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import AsyncPostgresStore
from pydantic import BaseModel, SecretStr

from src.config import Settings
from src.domain.entities.agent_config import AgentConfig, SubAgentConfig
from src.domain.entities.mcp_server_config import McpServerConfig
from src.domain.entities.model_metadata import ModelMetadata
from src.domain.errors.config import ConfigError
from src.domain.errors.mcp import McpError
from src.domain.logging.messages import LogMessage
from src.domain.ports.mcp_tool_loader import McpToolLoader
from src.domain.services.mcp_execution_resolution import McpRefsResolver
from src.infrastructure.deepagent.namespace import user_namespaced
from src.infrastructure.deepagent.schema_utils import schema_to_pydantic_model

logger = logging.getLogger(__name__)

# Dependency resolution seams injected by the composition root (spec §5.5):
# * prompt_resolver: PromptRef → verbatim system prompt content (Latest or
#   pinned version) from the local prompt registry.
# * model_resolver: model UUID → catalog ModelMetadata (metadata only).
# * mcp_refs_resolver: registry identities → trusted execution configs
#   (resource-only agents, REQ-MCP-003).
PromptResolver = Callable[[Any], Awaitable[str]]
ModelResolver = Callable[[Any], Awaitable[ModelMetadata]]


async def resolve_model_identity(
    config: AgentConfig,
    model_resolver: ModelResolver | None,
) -> tuple[str, str | None]:
    """Resolve the effective serving model identity (AGT-003 precedence).

    Returns ``(model_name, base_url)``: the ``model_ref`` catalog's
    ``provider_model`` identifier and ``base_url`` when the reference is set,
    else the legacy ``model`` string with ``None`` base URL. Shared by
    ``_resolve_model`` and the agent registry (which passes the served model
    name to the runner for the LLM-usage trace enrichment, dashboard spec §1).

    Args:
        config: The agent configuration.
        model_resolver: Optional catalog model metadata resolver.

    Returns:
        Tuple of ``(model_name, base_url)``.

    Raises:
        ConfigError: When ``model_ref`` is set but no resolver is wired.
    """
    if config.model_ref is None:
        return config.model, None
    if model_resolver is None:
        raise ConfigError(f"Agent '{config.name}' sets model_ref but no model metadata resolver is wired.")
    metadata = await model_resolver(str(config.model_ref))
    return metadata.provider_model, metadata.base_url


async def _resolve_model(
    config: AgentConfig,
    llm_credentials_resolver: Callable[[str], Awaitable[tuple[str, str] | None]] | None,
    model_resolver: ModelResolver | None = None,
):
    """Resolve the ``model`` argument passed to ``create_deep_agent``.

    Resolution order (spec §5.5 AGT-003):

    * ``model_ref`` (catalog UUID) takes PRECEDENCE — the agent runs with the
      referenced model's ``provider_model`` identifier and, when present, its
      ``base_url``. An invalid/missing reference raises the matching error and
      NEVER falls back to the legacy ``model`` string; the legacy string stays
      in the YAML for schema compatibility only.
    * Without ``model_ref`` the legacy ``model`` string applies unchanged
      (legacy custom strings and singleton LLM settings keep their behavior).

    Runtime credentials are resolved exactly as before (unchanged by this
    feature): when ``llm_credentials_resolver`` is provided AND
    ``current_user_id`` is set, the user's ``(base_url, api_key)`` build a
    per-user :class:`ChatOpenAI`; an unconfigured user raises
    :class:`LlmNotConfiguredError`; when the resolver is ``None`` OR the user
    context is unset (tests), the env-based string model fallback applies.

    Args:
        config: The agent configuration.
        llm_credentials_resolver: Optional per-user credential resolver.
        model_resolver: Optional catalog model metadata resolver.

    Returns:
        Either a model string (env fallback) or a :class:`ChatOpenAI` instance
        (per-user credentials).

    Raises:
        LlmNotConfiguredError: When the user is authenticated but has no
            configured LLM provider.
    """
    from src.domain.errors.llm import LlmNotConfiguredError
    from src.domain.errors.messages import ErrorMessage
    from src.domain.errors.model import ModelNotFoundError
    from src.infrastructure.database.rls_context import current_user_id

    model_name = config.model
    model_base_url: str | None = None

    if config.model_ref is not None:
        try:
            model_name, model_base_url = await resolve_model_identity(config, model_resolver)
        except ModelNotFoundError:
            logger.error(LogMessage.AGENT_CREATE_ERROR, config.name)
            raise

    user_id = current_user_id.get()
    if llm_credentials_resolver is None or user_id is None:
        # Env-based fallback (existing behaviour) — model name only.
        return model_name

    credentials = await llm_credentials_resolver(user_id)
    if credentials is None:
        raise LlmNotConfiguredError(ErrorMessage.LLM_NOT_CONFIGURED.format(user_id=user_id))

    credentials_base_url, api_key = credentials
    return ChatOpenAI(
        model=model_name,
        base_url=model_base_url or credentials_base_url,
        api_key=SecretStr(api_key),
        # LLM-usage capture (dashboard spec §1-§2): the LangGraph agent node
        # invokes the model with ``ainvoke`` — a NON-streaming provider call —
        # so the completion carries ``usage_metadata``, which the capture
        # reads. ``streaming`` is pinned to the langchain-openai DEFAULT
        # (False) only to keep that guarantee explicit: setting it to True
        # would switch the provider call to a streamed one, where usage is
        # only returned when the provider honours stream_options.include_usage
        # (the SSE event stream itself is unaffected either way — TraceEvents
        # still stream in turn order, and per-token realtime updates are an
        # explicit dashboard non-goal).
        streaming=False,
    )


def _to_pg_conn_string(database_url: str) -> str:
    """Convert an asyncpg-normalized URL to a plain PostgreSQL connection string.

    The ``Settings.database_url`` is normalized to ``postgresql+asyncpg://`` for
    use with SQLAlchemy/asyncpg. langgraph-postgres uses ``psycopg`` and expects
    a plain ``postgresql://`` URL. This strips the ``+asyncpg`` driver suffix.

    Args:
        database_url: The normalized database URL (``postgresql+asyncpg://...``).

    Returns:
        A plain ``postgresql://`` connection string.
    """
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


_pg_store: AsyncPostgresStore | None = None
_pg_checkpointer: AsyncPostgresSaver | None = None
_pg_store_cm: Any = None
_pg_checkpointer_cm: Any = None
# Event loop the singletons were created on. asyncpg/psycopg primitives are
# loop-bound: a singleton acquired on another loop must not be reused (the
# behavioral suite runs a session loop; unit tests run per-function loops).
_pg_loop: asyncio.AbstractEventLoop | None = None
_pg_checkpointer_loop: asyncio.AbstractEventLoop | None = None
_memory_store: InMemoryStore | None = None


def _get_memory_store() -> InMemoryStore:
    """Get or create a singleton ``InMemoryStore`` shared across all agents.

    This ensures that files written via the Store File API (which also uses this
    singleton when ``store_backend`` is ``"memory"``) are visible to agents that
    use the in-memory store.
    """
    global _memory_store
    if _memory_store is None:
        _memory_store = InMemoryStore()
    return _memory_store


_store_lock = asyncio.Lock()


async def _get_shared_store():
    """Get the global shared store instance.

    If the Postgres store was initialized at startup (by ``dependencies.py``),
    use that. Otherwise fall back to the shared ``InMemoryStore`` singleton.
    This ensures the Store File API and all agents share the same store,
    regardless of per-agent ``store_backend`` config.

    The singleton is LOOP-SCOPED: asyncpg connections cannot cross event
    loops, so a store created on a different loop is transparently re-created.

    An ``asyncio.Lock`` protects the lazy initialization to prevent
    concurrent agent creations from opening multiple connection pools.
    """
    loop = asyncio.get_running_loop()
    if _pg_store is not None and _pg_loop is loop:
        return _pg_store
    async with _store_lock:
        if _pg_store is not None and _pg_loop is loop:
            return _pg_store
        try:
            return await _create_postgres_store()
        except Exception:
            return _get_memory_store()


async def _create_postgres_store(settings: Settings | None = None) -> AsyncPostgresStore:
    """Get or create a singleton ``AsyncPostgresStore`` backed by Postgres.

    The store is created once and reused across all agent creations to avoid
    leaking connection pools. Schema migrations are applied on first creation.
    The async context manager reference is kept alive to prevent premature
    connection closure by the garbage collector.

    Args:
        settings: Optional settings override. Defaults to a fresh ``Settings``.

    Returns:
        A configured ``AsyncPostgresStore`` with schema migrations applied.
    """
    global _pg_store, _pg_store_cm, _pg_loop
    if _pg_store is not None and _pg_loop is asyncio.get_running_loop():
        return _pg_store
    s = settings or Settings()  # type: ignore[call-arg]  # env-resolved required fields (pydantic-settings)
    conn_string = _to_pg_conn_string(s.database_url)
    _pg_store_cm = AsyncPostgresStore.from_conn_string(conn_string)
    _pg_store = await _pg_store_cm.__aenter__()
    await _pg_store.setup()
    _pg_loop = asyncio.get_running_loop()
    return _pg_store


async def _create_postgres_checkpointer(settings: Settings | None = None) -> AsyncPostgresSaver:
    """Get or create a singleton ``AsyncPostgresSaver`` checkpointer backed by Postgres.

    The checkpointer is created once and reused across all agent creations to
    avoid leaking connection pools. Schema migrations are applied on first
    creation. The async context manager reference is kept alive to prevent
    premature connection closure by the garbage collector.

    Args:
        settings: Optional settings override. Defaults to a fresh ``Settings``.

    Returns:
        A configured ``AsyncPostgresSaver`` with schema migrations applied.
    """
    global _pg_checkpointer, _pg_checkpointer_cm, _pg_checkpointer_loop
    loop = asyncio.get_running_loop()
    if _pg_checkpointer is not None and _pg_checkpointer_loop is loop:
        return _pg_checkpointer
    s = settings or Settings()  # type: ignore[call-arg]  # env-resolved required fields (pydantic-settings)
    conn_string = _to_pg_conn_string(s.database_url)
    _pg_checkpointer_cm = AsyncPostgresSaver.from_conn_string(conn_string)
    _pg_checkpointer = await _pg_checkpointer_cm.__aenter__()
    await _pg_checkpointer.setup()
    _pg_checkpointer_loop = loop
    return _pg_checkpointer


def _resolve_response_format(value: dict[str, Any] | None) -> tuple[type[BaseModel] | None, dict[str, Any] | None]:
    """Resolve a ``response_format`` config value into ``(model, schema_dict)``.

    Args:
        value: The agent/subagent ``response_format`` config (a JSON Schema dict
            or ``None``).

    Returns:
        Tuple of ``(pydantic_model_or_none, schema_dict_or_none)``. The model is
        used for post-extraction validation/stripping; the dict is passed to
        deepagents' native ``response_format`` parameter.
    """
    if value is None:
        return None, None
    return schema_to_pydantic_model(value), value


def _resolve_tools(config: AgentConfig) -> list | None:
    """Charge les tools depuis leurs chemins Python (module:attribute).

    The builtin ``create_pdf`` reference stays YAML-referencable (REQ-049):
    the module-level ``create_pdf`` symbol is a ready-to-load StructuredTool
    whose user-namespaced store provider is wired by the composition root at
    boot (``src.dependencies.init_persistence``) and whose thread scope comes
    from the injected LangGraph ``RunnableConfig`` — no per-tool factory
    special-casing required (OCP: new builtin tools follow the same contract).
    """
    if not config.tools:
        return None
    tools = []
    for tool_path in config.tools:
        module_path, _, attr_name = tool_path.rpartition(":")
        if not module_path or not attr_name:
            logger.error(LogMessage.TOOL_FORMAT_INVALID, tool_path)
            raise ValueError(
                f"Invalid tool format '{tool_path}'. "
                f"Expected: 'module.path:tool_name' (e.g., 'mypackage.tools:my_tool')"
            )
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            logger.exception(LogMessage.TOOL_MODULE_NOT_FOUND, tool_path)
            raise ValueError(f"Module not found for tool '{tool_path}': {e}") from e

        if not hasattr(module, attr_name):
            available = [a for a in dir(module) if not a.startswith("_")]
            logger.error(LogMessage.TOOL_ATTRIBUTE_NOT_FOUND, attr_name, module_path)
            raise ValueError(f"Attribute '{attr_name}' not found in '{module_path}'. Available: {available}")
        tools.append(getattr(module, attr_name))
    return tools


def _resolve_backend(store):
    """Create the backend for the agent.

    The only supported backend is ``StoreBackend`` which reads/writes files
    from the shared LangGraph store (Postgres). This ensures skills and
    memories created via the Store File API are visible to all agents.

    The namespace is resolved per-request via ``user_namespaced("filesystem")``
    so each authenticated user's skills/memories are isolated. When no user
    context is set (tests), the namespace falls back to ``("filesystem",)``.

    Args:
        store: The shared store instance (Postgres or InMemoryStore).

    Returns:
        A ``StoreBackend`` instance.
    """
    return StoreBackend(store=store, namespace=lambda _r: user_namespaced("filesystem"))


async def _load_mcp_tools_resilient(
    mcp_tool_loader: McpToolLoader,
    servers: list[McpServerConfig],
    agent_name: str,
) -> list:
    """Load MCP tools SERVER BY SERVER, degrading gracefully per server.

    REQ-036 companion (chat-pdf-attachments): agents may declare several MCP
    servers (EDG-008) — one unreachable server must not brick the whole agent
    build (a multi-server agent must still boot with the live server's tools).
    A failing server is logged (server name + error) and simply skipped.

    I6 refinement: when EVERY declared server fails, the previous hard-fail
    semantics are preserved — the first failure is re-raised instead of
    booting a silently tool-less agent.

    Args:
        mcp_tool_loader: The tool loader (each call receives ONE config).
        servers: The agent's MCP server configs.
        agent_name: The agent name (log context).

    Returns:
        The tools of every reachable server (empty only when NO server is
        declared).

    Raises:
        McpError: When every declared server failed (no live tool source).
    """
    tools: list = []
    failures: list[tuple[McpServerConfig, McpError]] = []
    for server in servers:
        try:
            tools.extend(await mcp_tool_loader.load_tools([server]))
        except McpError as exc:
            failures.append((server, exc))
            logger.warning(LogMessage.AGENT_MCP_TOOLS_SKIPPED_SERVER, agent_name, server.name, exc)
    if servers and not tools and failures:
        # Every declared server is unreachable: fail loudly (legacy behavior)
        # rather than boot an agent that silently lost all its MCP tools.
        raise failures[0][1]
    return tools


def _resolve_interrupt_on(config: AgentConfig) -> dict[str, Any] | None:
    """Convertit la config HITL en dict pour create_deep_agent."""
    if not config.hitl.rules:
        return None
    result: dict[str, Any] = {}
    for tool_name, rule in config.hitl.rules.items():
        if isinstance(rule, bool):
            result[tool_name] = rule
        else:
            result[tool_name] = {"allowed_decisions": rule.allowed_decisions}
    return result


def _resolve_subagent_instructions(sa: SubAgentConfig) -> str | None:
    """Resolve subagent instructions from the YAML only.

    Spec §8.8: the implicit Phoenix agent-name prompt lookup is removed —
    subagents use their configured ``instructions`` (or the referenced
    agent's ``system_prompt`` via the caller's fallback logic). Nothing here
    touches Phoenix.

    Args:
        sa: The subagent configuration.

    Returns:
        The resolved instructions or ``None`` when not configured.
    """
    return sa.instructions


async def _resolve_subagents(
    config: AgentConfig,
    mcp_tool_loader: McpToolLoader | None = None,
    config_resolver: Callable[[str], Awaitable[AgentConfig]] | None = None,
    mcp_refs_resolver: McpRefsResolver | None = None,
) -> list | None:
    """Convertit les configs de sous-agents.

    Args:
        config: Configuration de l'agent principal.
        mcp_tool_loader: Loader MCP optionnel pour les sous-agents.
        config_resolver: Résolveur optionnel permettant de charger la
            ``AgentConfig`` référencée par ``SubAgentConfig.agent_ref``.
        mcp_refs_resolver: Résolveur optionnel des sélections ``mcp_refs``
            (REQ-MCP-003) pour les sous-agents référencés.

    Returns:
        Liste de dicts de sous-agents ou None.
    """
    if not config.subagents:
        return None
    subagents = []
    for sa in config.subagents:
        if sa.agent_ref is not None:
            spec = await _resolve_referenced_subagent(sa, mcp_tool_loader, config_resolver, mcp_refs_resolver)
            subagents.append(spec)
            continue

        local_tools = _resolve_tools_list(sa.tools) if sa.tools else None
        mcp_tools: list = []
        if sa.mcp_servers and mcp_tool_loader:
            mcp_tools = await _load_mcp_tools_resilient(mcp_tool_loader, sa.mcp_servers, sa.name)
        all_tools = (local_tools or []) + mcp_tools if (local_tools or mcp_tools) else None

        instructions = _resolve_subagent_instructions(sa)

        subagents.append(
            {
                "name": sa.name,
                "description": sa.description,
                "system_prompt": instructions,
                "model": sa.model,
                "tools": all_tools,
                "response_format": sa.response_format,
            }
        )
    return subagents


async def _resolve_referenced_subagent(
    sa: SubAgentConfig,
    mcp_tool_loader: McpToolLoader | None,
    config_resolver: Callable[[str], Awaitable[AgentConfig]] | None,
    mcp_refs_resolver: McpRefsResolver | None = None,
) -> dict:
    """Build a subagent spec from a SubAgentConfig that references another agent.

    Args:
        sa: The subagent configuration carrying ``agent_ref``.
        mcp_tool_loader: Optional MCP tool loader.
        config_resolver: Async callable returning the referenced AgentConfig.
        mcp_refs_resolver: Optional resolver for the referenced agent's
            ``mcp_refs`` selection (REQ-MCP-003).

    Returns:
        Dict spec ready for ``create_deep_agent``.

    Raises:
        ConfigError: If ``agent_ref`` is set but no ``config_resolver`` is provided.
    """
    if config_resolver is None:
        raise ConfigError(
            f"Subagent '{sa.name}' references agent '{sa.agent_ref}' but no config_resolver was provided."
        )

    assert sa.agent_ref is not None
    ref = await config_resolver(sa.agent_ref)

    if ref.subagents:
        logger.warning(
            "Referenced agent '%s' has its own subagents; ignoring them (one level only).",
            sa.agent_ref,
        )

    instructions = _resolve_subagent_instructions(sa)
    system_prompt = instructions if instructions is not None else ref.system_prompt

    local_tools = _resolve_tools_list(sa.tools) if sa.tools else None
    ref_tools = _resolve_tools_list(ref.tools) if ref.tools else None
    mcp_tools = await _resolve_reference_mcp_tools(sa, ref, mcp_tool_loader, mcp_refs_resolver)
    all_tools = (local_tools or []) + (ref_tools or []) + mcp_tools if (local_tools or ref_tools or mcp_tools) else None

    return {
        "name": sa.name,
        "description": sa.description,
        "system_prompt": system_prompt,
        "model": sa.model if sa.model else ref.model,
        "tools": all_tools,
        "response_format": sa.response_format if sa.response_format else ref.response_format,
    }


async def _resolve_reference_mcp_tools(
    sa: SubAgentConfig,
    ref: AgentConfig,
    mcp_tool_loader: McpToolLoader | None,
    mcp_refs_resolver: McpRefsResolver | None,
) -> list:
    """Combine the MCP tool sources of a referenced subagent (legacy + refs).

    The legacy inline ``mcp_servers`` of the entry and of the referenced config
    load exactly as before; the referenced config's ``mcp_refs`` selection is
    resolved through the trusted registry path (REQ-MCP-003).
    """
    tools: list = []
    servers = list(sa.mcp_servers) + list(ref.mcp_servers)
    if servers and mcp_tool_loader:
        tools = tools + await _load_mcp_tools_resilient(mcp_tool_loader, servers, sa.name)
    if ref.mcp_refs and mcp_refs_resolver is not None and mcp_tool_loader:
        ref_configs = await mcp_refs_resolver(ref.mcp_refs)
        if ref_configs:
            tools = tools + await _load_mcp_tools_resilient(mcp_tool_loader, ref_configs, sa.name)
    return tools


def _resolve_tools_list(tool_paths: list[str]) -> list | None:
    """Helper pour resoudre une liste de tools."""
    if not tool_paths:
        return None
    tools = []
    for tool_path in tool_paths:
        module_path, _, attr_name = tool_path.rpartition(":")
        if module_path and attr_name:
            module = importlib.import_module(module_path)
            tools.append(getattr(module, attr_name))
    return tools or None


def _apply_optional_kwargs(kwargs: dict, config: AgentConfig, store) -> None:
    """Populate optional kwargs from config if their values are set.

    Args:
        kwargs: The kwargs dict passed to ``create_deep_agent``.
        config: Configuration de l'agent.
        store: The store instance (in-memory or Postgres) to wire into ``StoreBackend``.
    """
    backend = _resolve_backend(store)
    if backend:
        kwargs["backend"] = backend
    interrupt_on = _resolve_interrupt_on(config)
    if interrupt_on:
        kwargs["interrupt_on"] = interrupt_on


async def _prepare_agent_namespace(
    store,
    agent_name: str,
    skills: list[str],
    memory: list[str],
) -> tuple[str, list[str]]:
    """Copy selected skills and memories to the agent's namespace in the store.

    The agent namespace follows the pattern:
    - Skills: /agents/{agent_name}/skills/{skill_name}/SKILL.md
    - Memories: /agents/{agent_name}/memories/{filename}

    This ensures SkillsMiddleware only discovers skills explicitly selected
    for this agent, not all skills in the global /skills/ directory.

    Args:
        store: The shared LangGraph BaseStore instance.
        agent_name: Name of the agent.
        skills: List of skill directory paths (e.g. ["/skills/mcp/", "/skills/rag/"]).
        memory: List of memory file paths (e.g. ["/memories/AGENTS.md"]).

    Returns:
        Tuple of (skills_source_path, memory_paths) for create_deep_agent.
    """
    ns = user_namespaced("filesystem")
    agent_skills_dir = f"/agents/{agent_name}/skills/"
    agent_memories_dir = f"/agents/{agent_name}/memories/"

    selected_skill_names = {s.rstrip("/").split("/")[-1] for s in skills}
    selected_memory_files = {m.split("/")[-1] for m in memory}

    await _cleanup_stale_agent_files(
        store, ns, agent_skills_dir, agent_memories_dir, selected_skill_names, selected_memory_files
    )
    await _copy_skills_to_agent_ns(store, ns, agent_skills_dir, skills)
    new_memory_paths = await _copy_memories_to_agent_ns(store, ns, agent_memories_dir, memory)

    return agent_skills_dir, new_memory_paths


async def _cleanup_stale_agent_files(
    store,
    ns: tuple[str, ...],
    agent_skills_dir: str,
    agent_memories_dir: str,
    selected_skill_names: set[str],
    selected_memory_files: set[str],
) -> None:
    """Delete files in agent namespace that are no longer selected.

    Args:
        store: The shared LangGraph BaseStore instance.
        ns: The resolved user namespace.
        agent_skills_dir: Agent skills directory prefix.
        agent_memories_dir: Agent memories directory prefix.
        selected_skill_names: Currently selected skill names.
        selected_memory_files: Currently selected memory filenames.
    """
    existing_items = await store.asearch(ns, limit=1000)
    for item in existing_items:
        await _maybe_delete_stale_item(
            store,
            ns,
            item,
            agent_skills_dir,
            agent_memories_dir,
            selected_skill_names,
            selected_memory_files,
        )


async def _maybe_delete_stale_item(
    store,
    ns: tuple[str, ...],
    item,
    agent_skills_dir: str,
    agent_memories_dir: str,
    selected_skill_names: set[str],
    selected_memory_files: set[str],
) -> None:
    """Delete a single store item if it is a stale skill or memory.

    Args:
        store: The shared LangGraph BaseStore instance.
        ns: The resolved user namespace.
        item: A store item to evaluate.
        agent_skills_dir: Agent skills directory prefix.
        agent_memories_dir: Agent memories directory prefix.
        selected_skill_names: Currently selected skill names.
        selected_memory_files: Currently selected memory filenames.
    """
    if item.key.startswith(agent_skills_dir):
        remainder = item.key[len(agent_skills_dir) :]
        skill_name = remainder.split("/")[0] if "/" in remainder else remainder
        if skill_name not in selected_skill_names:
            await _safe_delete(store, ns, item.key, "stale agent skill")
    elif item.key.startswith(agent_memories_dir):
        filename = item.key[len(agent_memories_dir) :]
        if filename not in selected_memory_files:
            await _safe_delete(store, ns, item.key, "stale agent memory")


async def _safe_delete(store, ns: tuple[str, ...], key: str, label: str) -> None:
    """Best-effort delete of a store key, logging failures instead of raising.

    Args:
        store: The shared LangGraph BaseStore instance.
        ns: The resolved user namespace.
        key: The store key to delete.
        label: Human-readable label for the log message.
    """
    try:
        await store.adelete(ns, key)
    except Exception:
        logger.exception("Failed to delete %s: %s", label, key)


def _store_lookup_key(path: str) -> str:
    """The canonical Store lookup key of an authored file reference (C5).

    The Store File API persists files under CANONICAL keys (the REST routes
    normalize every path with a leading slash), while an agent YAML may
    reference a skill or a memory with or without the leading slash. The
    runtime lookup contract is: authored reference → canonical store key,
    in ONE place, so an imported resource is looked up exactly where it was
    written — regardless of the slash convention of the persisted reference.
    """
    return path if path.startswith("/") else f"/{path}"


async def _copy_skills_to_agent_ns(
    store,
    ns: tuple[str, ...],
    agent_skills_dir: str,
    skills: list[str],
) -> None:
    """Copy selected skills' ``SKILL.md`` into the agent namespace.

    Args:
        store: The shared LangGraph BaseStore instance.
        ns: The resolved user namespace.
        agent_skills_dir: Agent skills directory prefix.
        skills: List of skill directory paths (slash convention-agnostic).
    """
    for skill_dir in skills:
        skill_name = skill_dir.rstrip("/").split("/")[-1]
        src_path = _store_lookup_key(f"{skill_dir.rstrip('/')}/SKILL.md")
        dst_path = f"{agent_skills_dir}{skill_name}/SKILL.md"
        item = await store.aget(ns, src_path)
        if item is not None:
            await store.aput(ns, dst_path, item.value)


async def _copy_memories_to_agent_ns(
    store,
    ns: tuple[str, ...],
    agent_memories_dir: str,
    memory: list[str],
) -> list[str]:
    """Copy selected memory files into the agent namespace.

    Args:
        store: The shared LangGraph BaseStore instance.
        ns: The resolved user namespace.
        agent_memories_dir: Agent memories directory prefix.
        memory: List of memory file paths (slash convention-agnostic).

    Returns:
        List of destination paths written into the agent namespace.
    """
    new_memory_paths: list[str] = []
    for mem_path in memory:
        filename = mem_path.split("/")[-1]
        dst_path = f"{agent_memories_dir}{filename}"
        item = await store.aget(ns, _store_lookup_key(mem_path))
        if item is not None:
            await store.aput(ns, dst_path, item.value)
        new_memory_paths.append(dst_path)
    return new_memory_paths


async def create_agent_from_config(
    config: AgentConfig,
    mcp_tool_loader: McpToolLoader | None = None,
    config_resolver: Callable[[str], Awaitable[AgentConfig]] | None = None,
    llm_credentials_resolver: Callable[[str], Awaitable[tuple[str, str] | None]] | None = None,
    prompt_resolver: PromptResolver | None = None,
    model_resolver: ModelResolver | None = None,
    extra_tools: list | None = None,
    middleware: Sequence[Any] = (),
    mcp_refs_resolver: McpRefsResolver | None = None,
):
    """Create a compiled Deep Agent from configuration.

    Args:
        config: Agent configuration.
        mcp_tool_loader: Optional MCP tool loader for loading remote tools.
        config_resolver: Optional async callable used to resolve subagent
            ``agent_ref`` references into full ``AgentConfig`` objects.
        llm_credentials_resolver: Optional async callable returning
            ``(base_url, api_key_plaintext)`` for the current user. When
            provided AND ``current_user_id`` is set, a per-user
            :class:`ChatOpenAI` instance is built with these credentials and
            passed to ``create_deep_agent`` instead of the string model. When
            the resolver returns ``None`` for a set user,
            :class:`LlmNotConfiguredError` is raised. When the resolver is
            ``None`` OR ``current_user_id`` is unset (no auth context, tests),
            the env-based string model fallback is used (current behaviour).
        prompt_resolver: Optional resolver for ``prompt_ref`` (spec §5.5
            AGT-002): returns the verbatim system prompt content of the
            referenced registry prompt — Latest version when
            ``prompt_ref.version`` is omitted, exactly that immutable version
            when pinned. Missing/inaccessible prompts or versions raise
            explicit errors; there is NO Phoenix path and NO agent-name
            lookup (§8.8).
        model_resolver: Optional resolver for ``model_ref`` (AGT-003):
            returns the catalog model metadata used for invocation.

    Returns:
        Tuple of (compiled agent graph, response_format_model or None).
    """
    logger.info(LogMessage.AGENT_CREATING, config.name, config.model)
    checkpointer: MemorySaver | AsyncPostgresSaver
    if config.backend.checkpoint_backend == "postgres":
        try:
            checkpointer = await _create_postgres_checkpointer()
        except Exception as e:
            logger.warning(LogMessage.CHECKPOINTER_POSTGRES_UNAVAILABLE, e)
            checkpointer = MemorySaver()
    else:
        checkpointer = MemorySaver()

    # The store is a global singleton shared between the Store File API and all
    # agents. If Postgres was initialized at startup (by dependencies.py), use
    # that. Otherwise use the shared InMemoryStore. This ensures skills and
    # memories created via the API are visible to agents regardless of the
    # per-agent store_backend setting.
    store = await _get_shared_store()

    all_tools = await _resolve_agent_tools(config, mcp_tool_loader, extra_tools, mcp_refs_resolver)
    system_prompt = await _resolve_system_prompt(config, prompt_resolver)
    skills_source, memory_paths = await _resolve_skills_and_memory(store, config)

    kwargs = {
        "name": config.name,
        "model": await _resolve_model(config, llm_credentials_resolver, model_resolver),
        "system_prompt": system_prompt,
        "tools": all_tools,
        "checkpointer": checkpointer,
        "store": store,
    }

    if middleware:
        # Agent-facing middleware hooks (e.g. the built-in assistant's toolset
        # guard) — passthrough to deepagents' create_deep_agent.
        kwargs["middleware"] = list(middleware)

    if skills_source:
        kwargs["skills"] = [skills_source]
    if memory_paths:
        kwargs["memory"] = memory_paths

    _apply_optional_kwargs(kwargs, config, store)

    if config.response_format:
        response_format_model, response_format_dict = _resolve_response_format(config.response_format)
        kwargs["response_format"] = response_format_dict
    else:
        response_format_model = None

    subagents = await _resolve_subagents(config, mcp_tool_loader, config_resolver, mcp_refs_resolver)
    if subagents:
        kwargs["subagents"] = subagents
        logger.info(LogMessage.AGENT_SUBAGENTS, config.name, len(subagents))
    try:
        graph = create_deep_agent(**kwargs)
    except Exception:
        logger.exception(LogMessage.AGENT_CREATE_ERROR, config.name)
        raise
    logger.info(LogMessage.AGENT_CREATED, config.name)
    return graph, response_format_model


async def _resolve_system_prompt(
    config: AgentConfig,
    prompt_resolver: PromptResolver | None,
) -> str | None:
    """Resolve the agent's system prompt (spec §5.5 AGT-002).

    * ``prompt_ref`` set → resolve through the LOCAL registry (Latest when
      ``version`` is omitted, exactly that immutable version when pinned).
      Missing/inaccessible prompts or versions raise explicit errors — no
      Phoenix lookup and no agent-name fallback (§8.8).
    * Otherwise the legacy inline ``system_prompt`` applies as-is.
    * When nothing is configured the existing non-Phoenix default behaviour
      applies (``None``).

    Args:
        config: The agent configuration.
        prompt_resolver: Optional registry resolver wired by the composition
            root.

    Returns:
        The resolved system prompt content, or ``None``.
    """
    if config.prompt_ref is None:
        return config.system_prompt
    if prompt_resolver is None:
        raise ConfigError(f"Agent '{config.name}' sets prompt_ref but no prompt resolver is wired.")
    return await prompt_resolver(config.prompt_ref)


async def _resolve_agent_tools(
    config: AgentConfig,
    mcp_tool_loader: McpToolLoader | None,
    extra_tools: list | None = None,
    mcp_refs_resolver: McpRefsResolver | None = None,
) -> list | None:
    """Resolve an agent's combined local + MCP + injected tool list.

    The MCP section covers BOTH sources: the legacy inline ``mcp_servers``
    (agents created before the resource-only rule, REQ-LEGACY-001) and the
    ``mcp_refs`` registry selections resolved through the trusted
    execution-config path (REQ-MCP-003). A ``mcp_refs`` selection without a
    wired resolver is an explicit error — never a silent downgrade.

    Args:
        config: The agent configuration.
        mcp_tool_loader: Optional MCP tool loader for remote tools.
        extra_tools: Optional pre-resolved tool objects injected by the agent
            registry — the built-in assistant's dynamic toolset (REQ-036) and
            custom agents' platform toolset (REQ-028) arrive through this
            parameter instead of module paths.
        mcp_refs_resolver: Optional resolver of ``mcp_refs`` registry
            identities into trusted execution configs.

    Returns:
        Combined tools list or ``None`` when the agent has no tools.

    Raises:
        ConfigError: When ``mcp_refs`` is declared but no resolver is wired.
        McpRegistryError: When the registry selection cannot be resolved.
    """
    local_tools = _resolve_tools(config)
    mcp_tools: list = []
    if config.mcp_refs:
        if mcp_refs_resolver is None:
            raise ConfigError(f"Agent '{config.name}' sets mcp_refs but no MCP registry resolver is wired.")
        logger.info(LogMessage.AGENT_MCP_REFS_RESOLVING, config.name, len(config.mcp_refs))
        resolved_configs = await mcp_refs_resolver(config.mcp_refs)
        if resolved_configs and mcp_tool_loader:
            mcp_tools.extend(await _load_mcp_tools_resilient(mcp_tool_loader, resolved_configs, config.name))
    if config.mcp_servers and mcp_tool_loader:
        logger.info(LogMessage.AGENT_MCP_TOOLS_LOADING, config.name, len(config.mcp_servers))
        mcp_tools.extend(await _load_mcp_tools_resilient(mcp_tool_loader, config.mcp_servers, config.name))
        logger.info(LogMessage.AGENT_MCP_TOOLS_LOADED, len(mcp_tools), config.name)
    combined = (local_tools or []) + mcp_tools + (list(extra_tools) if extra_tools else [])
    all_tools = combined if combined else None
    logger.info(LogMessage.AGENT_TOOLS_TOTAL, config.name, len(all_tools) if all_tools else 0)
    return all_tools


async def _resolve_skills_and_memory(
    store,
    config: AgentConfig,
) -> tuple[str | None, list[str] | None]:
    """Prepare the agent namespace and resolve skills/memory paths.

    Args:
        store: The shared LangGraph BaseStore instance.
        config: The agent configuration.

    Returns:
        Tuple of ``(skills_source, memory_paths)``; both are ``None`` when
        the agent declares neither skills nor memory.
    """
    if config.skills:
        return await _prepare_agent_namespace(store, config.name, config.skills, config.memory)
    if config.memory:
        _, memory_paths = await _prepare_agent_namespace(store, config.name, [], config.memory)
        return None, memory_paths
    return None, None
