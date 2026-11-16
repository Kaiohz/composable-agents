import logging
import ssl
from dataclasses import dataclass

# NOTE: imported from the typed submodule — the package's own ``__init__.py``
# carries a file-level ``# type: ignore[...]`` comment, so mypy cannot see the
# ``Minio`` re-export through the package root (py.typed is only complete on
# the internal modules).
from miniopy_async.api import Minio
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import AsyncAdaptedQueuePool

from src.application.use_cases.api_key.create_api_key import CreateApiKeyUseCase
from src.application.use_cases.api_key.list_api_keys import ListApiKeysUseCase
from src.application.use_cases.api_key.revoke_api_key import RevokeApiKeyUseCase
from src.application.use_cases.create_agent_config import CreateAgentConfigUseCase
from src.application.use_cases.create_thread import CreateThreadUseCase
from src.application.use_cases.cron.manage_crons import (
    CreateCronUseCase,
    DeleteCronUseCase,
    GetCronUseCase,
    ListCronsUseCase,
    UpdateCronUseCase,
)
from src.application.use_cases.cron.run_crons import (
    ExecuteCronRunUseCase,
    ListCronRunsUseCase,
    RunCronNowUseCase,
    SchedulerTickUseCase,
)
from src.application.use_cases.delete_agent_config import DeleteAgentConfigUseCase
from src.application.use_cases.delete_thread import DeleteThreadUseCase
from src.application.use_cases.get_agent_config import GetAgentConfigUseCase
from src.application.use_cases.get_thread import GetThreadUseCase
from src.application.use_cases.get_thread_history import GetThreadHistoryUseCase
from src.application.use_cases.list_agent_configs import ListAgentConfigsUseCase
from src.application.use_cases.list_threads import ListThreadsUseCase
from src.application.use_cases.list_traces import ListTracesUseCase
from src.application.use_cases.load_agent_config import LoadAgentConfigUseCase
from src.application.use_cases.manage_store_file import (
    DeleteStoreFileUseCase,
    GetStoreFileUseCase,
    ListStoreFilePreviewsUseCase,
    ListStoreFilesUseCase,
    PutStoreFileUseCase,
)
from src.application.use_cases.model.manage_models import (
    CreateModelUseCase,
    DeleteModelUseCase,
    GetModelUseCase,
    ListModelsUseCase,
    UpdateModelUseCase,
)
from src.application.use_cases.prompt.create_prompt import CreatePromptUseCase
from src.application.use_cases.prompt.manage_prompt_versions import (
    AddPromptVersionUseCase,
    GetPromptVersionUseCase,
    ListPromptVersionsUseCase,
)
from src.application.use_cases.prompt.manage_prompts import (
    DeletePromptUseCase,
    GetPromptUseCase,
    ListPromptsUseCase,
    UpdatePromptUseCase,
)
from src.application.use_cases.send_message import SendMessageUseCase
from src.application.use_cases.stream_message import StreamMessageUseCase
from src.application.use_cases.trace_usage import GetTopUsageRunsUseCase, GetTraceUsageUseCase
from src.application.use_cases.update_agent_config import UpdateAgentConfigUseCase
from src.application.use_cases.user.get_current_user import GetCurrentUserUseCase
from src.application.use_cases.user_llm_settings.delete_user_llm_settings import DeleteUserLlmSettingsUseCase
from src.application.use_cases.user_llm_settings.get_user_llm_settings import GetUserLlmSettingsUseCase
from src.application.use_cases.user_llm_settings.upsert_user_llm_settings import UpsertUserLlmSettingsUseCase
from src.config import Settings
from src.domain.entities.auth.auth_context import AuthContext
from src.domain.errors.messages import ErrorMessage
from src.domain.errors.security import AuthenticationError
from src.domain.errors.storage import StorageError
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_registry import AgentRegistry
from src.domain.ports.auth.api_key_repository import ApiKeyRepository
from src.domain.ports.cron_repository import CronRepository, CronRunRepository
from src.domain.ports.llm_price_repository import LlmPriceRepository
from src.domain.ports.model_repository import ModelRepository
from src.domain.ports.prompt_repository import PromptRepository
from src.domain.ports.store_file_repository import StoreFileRepository
from src.domain.ports.thread_repository import ThreadRepository
from src.domain.ports.trace_event_repository import TraceEventRepository
from src.domain.ports.trace_summary_repository import TraceSummaryRepository
from src.domain.ports.trace_usage_repository import TraceUsageRepository
from src.domain.ports.tracing_provider import TracingProvider
from src.domain.ports.user_llm_settings_repository import UserLlmSettingsRepository
from src.infrastructure.auth.jwt_adapter import JwtAdapter
from src.infrastructure.crypto.fernet_crypto import FernetCrypto
from src.infrastructure.database.rls_context import current_auth_context, current_user_id
from src.infrastructure.mcp.adapter import LangchainMcpToolLoader
from src.infrastructure.persistent_registry.adapter import PersistentAgentRegistry
from src.infrastructure.postgres_api_key.adapter import PostgresApiKeyRepository
from src.infrastructure.postgres_cron.adapter import PostgresCronRepository, PostgresCronRunRepository
from src.infrastructure.postgres_llm_prices.adapter import PostgresLlmPriceRepository
from src.infrastructure.postgres_model.adapter import PostgresModelRepository
from src.infrastructure.postgres_prompt.adapter import PostgresPromptRepository
from src.infrastructure.postgres_repository.adapter import PostgresAgentConfigRepository
from src.infrastructure.postgres_thread.adapter import PostgresThreadRepository
from src.infrastructure.postgres_trace.adapter import PostgresTraceEventRepository
from src.infrastructure.postgres_trace.summary_repository import PostgresTraceSummaryRepository
from src.infrastructure.postgres_trace.usage_repository import PostgresTraceUsageRepository
from src.infrastructure.postgres_user_llm.adapter import PostgresUserLlmSettingsRepository
from src.infrastructure.runner_freshness.adapter import PostgresRunnerFreshness
from src.infrastructure.rustfs_store.adapter import RustfsAgentConfigStore
from src.infrastructure.store_file.adapter import LangGraphStoreFileRepository
from src.infrastructure.tracing.noop_adapter import NoopTracingProvider
from src.infrastructure.yaml_config.adapter import YamlAgentConfigLoader
from src.security import ComposableAgentsSecurity

logger = logging.getLogger(__name__)

# ============= CONFIG =============

# Runtime-equivalent of ``Settings()``: pydantic-settings resolves required
# fields (e.g. ``database_url``) from the environment at call time; mypy
# cannot model that, so the constructor call needs a scoped suppression.
settings = Settings()  # type: ignore[call-arg]

# ============= TRACING =============


def _create_tracing_provider(settings: Settings) -> TracingProvider:
    """Create the appropriate TracingProvider based on settings.

    Phoenix remains CONFIGURED SOLELY as an optional tracing provider
    (spec §8.9); no prompt path uses it anymore.

    Args:
        settings: Application settings containing tracing configuration.

    Returns:
        A TracingProvider instance matching the configured provider.
    """
    tracing = settings.tracing

    if tracing.enabled and tracing.provider == "langfuse":
        from src.infrastructure.tracing.langfuse_adapter import LangfuseTracingProvider

        logger.info(LogMessage.TRACING_LANGFUSE_INIT, tracing.langfuse_host)
        return LangfuseTracingProvider(
            public_key=tracing.langfuse_public_key or "",
            secret_key=tracing.langfuse_secret_key or "",
            host=tracing.langfuse_host,
        )

    if tracing.enabled and tracing.provider == "phoenix":
        from src.infrastructure.tracing.phoenix_adapter import PhoenixTracingProvider

        logger.info(LogMessage.TRACING_PHOENIX_INIT, tracing.phoenix_collector_endpoint)
        return PhoenixTracingProvider(
            endpoint=tracing.phoenix_collector_endpoint,
            api_key=tracing.phoenix_api_key,
            project_name=tracing.project_name,
        )

    logger.info(LogMessage.TRACING_DISABLED)
    return NoopTracingProvider()


# ============= ADAPTERS =============

agent_config_loader = YamlAgentConfigLoader()
mcp_tool_loader = LangchainMcpToolLoader(tool_timeout=settings.mcp_tool_timeout)
tracing_provider = _create_tracing_provider(settings)

# Security instance shared across all routes (HTTP + WebSocket).
security = ComposableAgentsSecurity(master_key=settings.api_key)


def get_security() -> ComposableAgentsSecurity:
    """Provide the singleton ``ComposableAgentsSecurity`` instance.

    Used as a FastAPI dependency by the WebSocket endpoint, which cannot
    inherit router-level ``dependencies=[...]``.
    """
    return security


# ============= PERSISTENCE (initialized at startup) =============


@dataclass
class CompositionRoot:
    """Container for application-wide persistence state, replacing module-level globals."""

    async_engine: AsyncEngine | None = None
    rustfs_store: RustfsAgentConfigStore | None = None
    pg_repository: PostgresAgentConfigRepository | None = None
    agent_registry: AgentRegistry | None = None
    thread_repository: ThreadRepository | None = None
    trace_event_repository: TraceEventRepository | None = None
    trace_summary_repository: TraceSummaryRepository | None = None
    store_file_repository: StoreFileRepository | None = None
    api_key_repository: ApiKeyRepository | None = None
    user_llm_settings_repository: UserLlmSettingsRepository | None = None
    prompt_repository: PromptRepository | None = None
    model_repository: ModelRepository | None = None
    cron_repository: CronRepository | None = None
    cron_run_repository: CronRunRepository | None = None
    trace_usage_repository: TraceUsageRepository | None = None
    llm_price_repository: LlmPriceRepository | None = None
    runner_freshness: PostgresRunnerFreshness | None = None
    fernet_crypto: FernetCrypto | None = None
    jwt_adapter: JwtAdapter | None = None


_root = CompositionRoot()


async def init_persistence() -> None:
    """Initialize persistent infrastructure: SQLAlchemy engine, MinIO store, PostgreSQL repositories.

    Must be called during application startup.
    """
    logger.info(LogMessage.PERSISTENCE_INITIALIZING)

    connect_args: dict = {}
    if settings.postgres_statement_cache_size is not None:
        connect_args["statement_cache_size"] = settings.postgres_statement_cache_size

    # Build asyncpg SSL context from the sslmode extracted by Settings.
    #
    # SSL decision (Sonar S5527/S4830, configuration-driven, safe by default):
    # every encryption mode builds a single validated context — certificate AND
    # hostname verification on. Local/self-signed deployments may explicitly
    # opt out via ``DATABASE_SSL_INSECURE=true``: the flag relaxes exactly one
    # isolated context (each branch constructs its own connection object so a
    # relaxed flow can never pollute the validated ones) and logs a loud
    # warning. The bypass is configuration-driven and never the default.
    # - ``disable`` / ``allow``: no SSL is built (unchanged).
    ssl_mode = settings.ssl_mode
    if ssl_mode in ("verify-ca", "verify-full"):
        connect_args["ssl"] = ssl.create_default_context()
    elif ssl_mode in ("require", "prefer"):
        if settings.database_ssl_insecure:
            logger.warning(
                LogMessage.DATABASE_SSL_INSECURE_ENABLED,
                ssl_mode,
            )
            # Explicit, configuration-driven opt-out for local/self-signed
            # stacks (see config.py): the context is created then deliberately
            # relaxed. Sonar S5527/S4830 correctly flag relaxed verification on
            # any context they can trace; the marker on each line documents the
            # single deliberate opt-out (env-gated, warned, never the
            # production default). Justification: local/self-signed TLS test
            # stacks need unverified TLS while production defaults validate.
            insecure_ctx = ssl.create_default_context()  # NOSONAR
            insecure_ctx.check_hostname = False  # NOSONAR
            insecure_ctx.verify_mode = ssl.CERT_NONE  # NOSONAR
            connect_args["ssl"] = insecure_ctx
        else:
            connect_args["ssl"] = ssl.create_default_context()

    _root.async_engine = create_async_engine(
        settings.database_url,
        poolclass=AsyncAdaptedQueuePool,
        pool_size=20,
        max_overflow=20,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
    logger.info(LogMessage.SQLALCHEMY_ENGINE_CREATED)

    # Register the RLS ``before_cursor_execute`` listener on the engine. The
    # listener is a no-op on SQLite (tests) and emits ``SET LOCAL`` GUCs on
    # PostgreSQL so Row-Level Security policies can filter rows per user.
    from src.infrastructure.database.rls_listener import register_rls_listener

    register_rls_listener(_root.async_engine)

    _root.pg_repository = PostgresAgentConfigRepository(engine=_root.async_engine)
    _root.thread_repository = PostgresThreadRepository(engine=_root.async_engine)
    _root.trace_event_repository = PostgresTraceEventRepository(engine=_root.async_engine)
    _root.trace_summary_repository = PostgresTraceSummaryRepository(engine=_root.async_engine)
    _root.api_key_repository = PostgresApiKeyRepository(engine=_root.async_engine)
    _root.prompt_repository = PostgresPromptRepository(engine=_root.async_engine)
    _root.model_repository = PostgresModelRepository(engine=_root.async_engine)
    _root.cron_repository = PostgresCronRepository(engine=_root.async_engine)
    _root.cron_run_repository = PostgresCronRunRepository(engine=_root.async_engine)
    _root.trace_usage_repository = PostgresTraceUsageRepository(engine=_root.async_engine)
    _root.llm_price_repository = PostgresLlmPriceRepository(engine=_root.async_engine)
    _root.runner_freshness = PostgresRunnerFreshness(engine=_root.async_engine)
    logger.info(LogMessage.POSTGRES_REPOS_INITIALIZED)

    # Seed/refresh the known LLM price catalog (dashboard spec §3). Runs on the
    # system write path (bypass-aware RLS policy) and is deliberately NON-fatal:
    # a seeding failure must never block startup — unpriced models degrade to
    # cost=0 / priced=false in the usage endpoints instead.
    if _root.llm_price_repository is not None:
        try:
            seeded = await _root.llm_price_repository.seed_known_defaults()
            logger.info(LogMessage.LLM_PRICES_SEEDED, seeded)
        except Exception:
            # NON-fatal, but NEVER detail-free: logger.exception carries the
            # traceback, and a raised StorageError already embeds the driver
            # cause — the operator keeps the diagnosis an operator needs.
            logger.exception(LogMessage.LLM_PRICES_SEED_FAILED)

    # FernetCrypto for per-user LLM API key encryption. If the configured key
    # is empty (dev / test), generate a throwaway in-memory key so init does
    # not crash — production must set SECRET_ENCRYPTION_KEY.
    from cryptography.fernet import Fernet

    fernet_key = settings.secret_encryption_key
    if not fernet_key:
        logger.warning(LogMessage.LLM_CRYPTO_KEY_EMPTY)
        fernet_key = Fernet.generate_key().decode()
    _root.fernet_crypto = FernetCrypto(key=fernet_key)
    _root.user_llm_settings_repository = PostgresUserLlmSettingsRepository(
        engine=_root.async_engine, crypto=_root.fernet_crypto
    )
    logger.info(LogMessage.LLM_SETTINGS_REPO_INITIALIZED)

    # Wire the dual-auth AuthService (JWT + per-user API key) into the
    # singleton security so verify_credentials can resolve an AuthContext and
    # set the RLS contextvars on each authenticated request.
    from src.domain.services.auth.auth_service import AuthService

    _root.jwt_adapter = JwtAdapter(
        jwks_url=f"{settings.logto_url}/oidc/jwks" if settings.logto_url else "",
        audience=settings.jwt_audience,
    )
    auth_service = AuthService(jwt_port=_root.jwt_adapter, api_key_repo=_root.api_key_repository)
    security.set_auth_service(auth_service)

    minio_client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    _root.rustfs_store = RustfsAgentConfigStore(client=minio_client, bucket=settings.minio_bucket)
    await _root.rustfs_store.ensure_bucket()
    logger.info(LogMessage.RUSTFS_STORE_INITIALIZED, settings.minio_bucket)

    _root.agent_registry = PersistentAgentRegistry(
        config_loader=agent_config_loader,
        config_store=_root.rustfs_store,
        config_repository=_root.pg_repository,
        mcp_tool_loader=mcp_tool_loader,
        tracing_provider=tracing_provider,
        stream_idle_timeout=settings.agent_stream_idle_timeout,
        invoke_timeout=settings.agent_invoke_timeout,
        llm_credentials_resolver=get_llm_credentials_resolver(),
        llm_provider_resolver=get_llm_provider_resolver(),
        prompt_repository=_root.prompt_repository,
        model_repository=_root.model_repository,
        runner_freshness=_root.runner_freshness,
    )

    # Store file repository — reuse the singleton LangGraph BaseStore from the
    # deepagent factory (AsyncPostgresStore) so the file API shares the same
    # connection pool as the agents. Falls back to the shared InMemoryStore
    # singleton on init failure.
    #
    # The namespace is resolved per-request via a ``namespace_provider``
    # callable: ``(user_id, "filesystem")`` when ``current_user_id`` is set
    # (authenticated request), ``("filesystem",)`` when it is ``None`` (tests,
    # background jobs). This isolates skills/memories per user.
    try:
        from src.infrastructure.deepagent.factory import _create_postgres_store
        from src.infrastructure.deepagent.namespace import user_namespaced

        store = await _create_postgres_store(settings)
        _root.store_file_repository = LangGraphStoreFileRepository(
            store=store, namespace_provider=lambda: user_namespaced("filesystem")
        )
        logger.info(LogMessage.PERSISTENCE_STORE_FILE_INITIALIZED)

        # NOTE: Row-Level Security is NOT applied on the LangGraph ``store``
        # table. ``AsyncPostgresStore`` uses its own asyncpg connection pool
        # (not the SQLAlchemy engine), so the RLS listener that sets the
        # ``app.user_id`` GUC never runs on store connections. With FORCE RLS
        # the policy would filter out every row (GUC is NULL) and reject
        # inserts (WITH CHECK fails). Per-user isolation for skills/memories
        # is enforced at the application layer via the namespace prefix
        # (``user_namespaced`` → ``(user_id, "filesystem")``), which is
        # sufficient and works regardless of the connection pool.
    except Exception:
        logger.exception(LogMessage.PERSISTENCE_STORE_FILE_INIT_FAILED)
        from src.infrastructure.deepagent.factory import _get_memory_store
        from src.infrastructure.deepagent.namespace import user_namespaced

        _root.store_file_repository = LangGraphStoreFileRepository(
            store=_get_memory_store(), namespace_provider=lambda: user_namespaced("filesystem")
        )
        logger.info(LogMessage.PERSISTENCE_STORE_FILE_FALLBACK_INMEMORY)

    logger.info(LogMessage.PERSISTENCE_REGISTRY_SET)


async def close_persistence() -> None:
    """Close persistent infrastructure resources.

    Must be called during application shutdown.

    Also resets the deepagent factory's module-level singletons (shared Postgres
    store/checkpointer) so a subsequent process (or test session reusing this
    event loop scope) starts clean instead of encountering connections pinned
    to a shut-down engine/loop.
    """
    if _root.agent_registry and isinstance(_root.agent_registry, PersistentAgentRegistry):
        await _root.agent_registry.close()
        logger.info(LogMessage.PERSISTENT_REGISTRY_CLOSED)

    if _root.async_engine:
        await _root.async_engine.dispose()
        logger.info(LogMessage.SQLALCHEMY_ENGINE_DISPOSED)

    if _root.jwt_adapter is not None:
        await _root.jwt_adapter.close()
        _root.jwt_adapter = None
        logger.info(LogMessage.JWT_ADAPTER_CLOSED)


def reset() -> None:
    """Reset all persisted state. Useful for testing."""
    _root.async_engine = None
    _root.rustfs_store = None
    _root.pg_repository = None
    _root.agent_registry = None
    _root.thread_repository = None
    _root.trace_event_repository = None
    _root.trace_summary_repository = None
    _root.trace_usage_repository = None
    _root.llm_price_repository = None
    _root.store_file_repository = None
    _root.api_key_repository = None
    _root.user_llm_settings_repository = None
    _root.prompt_repository = None
    _root.model_repository = None
    _root.cron_repository = None
    _root.cron_run_repository = None
    _root.runner_freshness = None
    _root.fernet_crypto = None
    _root.jwt_adapter = None


logger.info(LogMessage.DEPENDENCIES_INITIALIZED)


# ============= USE CASE PROVIDERS =============


def _require_thread_repository() -> ThreadRepository:
    """Return thread repository or raise StorageError if not initialized."""
    if _root.thread_repository is None:
        raise StorageError(ErrorMessage.STORAGE_REPO_NOT_INITIALIZED)
    return _root.thread_repository


def _require_trace_event_repository() -> TraceEventRepository:
    """Return trace event repository or raise StorageError if not initialized."""
    if _root.trace_event_repository is None:
        raise StorageError(ErrorMessage.STORAGE_REPO_NOT_INITIALIZED)
    return _root.trace_event_repository


def get_trace_event_repository() -> TraceEventRepository:
    """Provide a TraceEventRepository instance (singleton wired at startup)."""
    return _require_trace_event_repository()


def _require_agent_registry() -> AgentRegistry:
    """Return agent registry or raise StorageError if not initialized."""
    if _root.agent_registry is None:
        raise StorageError(ErrorMessage.STORAGE_REGISTRY_NOT_INITIALIZED)
    return _root.agent_registry


def get_send_message_use_case() -> SendMessageUseCase:
    """Provide a SendMessageUseCase instance."""
    return SendMessageUseCase(
        _require_agent_registry(), _require_thread_repository(), _require_trace_event_repository()
    )


def get_stream_message_use_case() -> StreamMessageUseCase:
    """Provide a StreamMessageUseCase instance."""
    return StreamMessageUseCase(
        _require_agent_registry(), _require_thread_repository(), _require_trace_event_repository()
    )


def get_get_thread_history_use_case() -> GetThreadHistoryUseCase:
    """Provide a GetThreadHistoryUseCase instance."""
    return GetThreadHistoryUseCase(_require_thread_repository(), _require_trace_event_repository())


def get_create_thread_use_case() -> CreateThreadUseCase:
    """Provide a CreateThreadUseCase instance."""
    return CreateThreadUseCase(_require_thread_repository(), _require_agent_registry())


def get_get_thread_use_case() -> GetThreadUseCase:
    """Provide a GetThreadUseCase instance."""
    return GetThreadUseCase(_require_thread_repository())


def get_list_threads_use_case() -> ListThreadsUseCase:
    """Provide a ListThreadsUseCase instance."""
    return ListThreadsUseCase(_require_thread_repository())


def get_delete_thread_use_case() -> DeleteThreadUseCase:
    """Provide a DeleteThreadUseCase instance."""
    return DeleteThreadUseCase(_require_thread_repository())


def get_load_agent_config_use_case() -> LoadAgentConfigUseCase:
    """Provide a LoadAgentConfigUseCase instance."""
    return LoadAgentConfigUseCase(agent_config_loader)


def _require_persistence() -> tuple[RustfsAgentConfigStore, PostgresAgentConfigRepository]:
    """Return persistence adapters or raise StorageError if not initialized."""
    if _root.rustfs_store is None or _root.pg_repository is None:
        raise StorageError(ErrorMessage.STORAGE_PERSISTENCE_NOT_INITIALIZED)
    return _root.rustfs_store, _root.pg_repository


def get_create_agent_config_use_case() -> CreateAgentConfigUseCase:
    """Provide a CreateAgentConfigUseCase instance."""
    store, repo = _require_persistence()
    return CreateAgentConfigUseCase(
        config_loader=agent_config_loader,
        config_store=store,
        config_repository=repo,
        prompt_repository=_require_prompt_repository(),
        model_repository=_require_model_repository(),
    )


def get_update_agent_config_use_case() -> UpdateAgentConfigUseCase:
    """Provide an UpdateAgentConfigUseCase instance."""
    store, repo = _require_persistence()
    return UpdateAgentConfigUseCase(
        config_loader=agent_config_loader,
        config_store=store,
        config_repository=repo,
        agent_registry=_require_agent_registry(),
        prompt_repository=_require_prompt_repository(),
        model_repository=_require_model_repository(),
    )


def get_delete_agent_config_use_case() -> DeleteAgentConfigUseCase:
    """Provide a DeleteAgentConfigUseCase instance."""
    store, repo = _require_persistence()
    return DeleteAgentConfigUseCase(
        config_store=store,
        config_repository=repo,
        agent_registry=_require_agent_registry(),
    )


def get_get_agent_config_use_case() -> GetAgentConfigUseCase:
    """Provide a GetAgentConfigUseCase instance."""
    store, repo = _require_persistence()
    return GetAgentConfigUseCase(
        config_loader=agent_config_loader,
        config_store=store,
        config_repository=repo,
    )


def get_list_agent_configs_use_case() -> ListAgentConfigsUseCase:
    """Provide a ListAgentConfigsUseCase instance."""
    _, repo = _require_persistence()
    return ListAgentConfigsUseCase(config_repository=repo)


def _require_prompt_repository() -> PromptRepository:
    """Return the prompt repository or raise StorageError if not initialized."""
    if _root.prompt_repository is None:
        raise StorageError(ErrorMessage.STORAGE_REPO_NOT_INITIALIZED)
    return _root.prompt_repository


def _require_model_repository() -> ModelRepository:
    """Return the model repository or raise StorageError if not initialized."""
    if _root.model_repository is None:
        raise StorageError(ErrorMessage.STORAGE_REPO_NOT_INITIALIZED)
    return _root.model_repository


def get_prompt_repository() -> PromptRepository:
    """Provide a PromptRepository instance (singleton wired at startup)."""
    return _require_prompt_repository()


def get_model_repository() -> ModelRepository:
    """Provide a ModelRepository instance (singleton wired at startup)."""
    return _require_model_repository()


def _require_cron_repository() -> CronRepository:
    if _root.cron_repository is None:
        raise StorageError(ErrorMessage.STORAGE_REPO_NOT_INITIALIZED)
    return _root.cron_repository


def get_cron_repository() -> CronRepository:
    """Provide a CronRepository instance (singleton wired at startup)."""
    return _require_cron_repository()


def _require_cron_run_repository() -> CronRunRepository:
    if _root.cron_run_repository is None:
        raise StorageError(ErrorMessage.STORAGE_REPO_NOT_INITIALIZED)
    return _root.cron_run_repository


def get_cron_run_repository() -> CronRunRepository:
    """Provide a CronRunRepository instance (singleton wired at startup)."""
    return _require_cron_run_repository()


def get_create_prompt_use_case() -> CreatePromptUseCase:
    """Provide a CreatePromptUseCase instance (local registry, PR-004)."""
    return CreatePromptUseCase(_require_prompt_repository())


def get_get_prompt_use_case() -> GetPromptUseCase:
    """Provide a GetPromptUseCase instance."""
    return GetPromptUseCase(_require_prompt_repository())


def get_list_prompts_use_case() -> ListPromptsUseCase:
    """Provide a ListPromptsUseCase instance."""
    return ListPromptsUseCase(_require_prompt_repository())


def get_update_prompt_use_case() -> UpdatePromptUseCase:
    """Provide a metadata-only UpdatePromptUseCase instance."""
    return UpdatePromptUseCase(_require_prompt_repository())


def get_delete_prompt_use_case() -> DeletePromptUseCase:
    """Provide a DeletePromptUseCase instance (reference-safe, PR-005)."""
    return DeletePromptUseCase(_require_prompt_repository())


def get_add_prompt_version_use_case() -> AddPromptVersionUseCase:
    """Provide an AddPromptVersionUseCase instance (transactional, PR-003)."""
    return AddPromptVersionUseCase(_require_prompt_repository())


def get_list_prompt_versions_use_case() -> ListPromptVersionsUseCase:
    """Provide a ListPromptVersionsUseCase instance."""
    return ListPromptVersionsUseCase(_require_prompt_repository())


def get_get_prompt_version_use_case() -> GetPromptVersionUseCase:
    """Provide a GetPromptVersionUseCase instance (pinned immutable read)."""
    return GetPromptVersionUseCase(_require_prompt_repository())


# ============= MODEL CATALOG PROVIDERS =============


def get_create_model_use_case() -> CreateModelUseCase:
    """Provide a CreateModelUseCase instance (metadata-only, MDL-001)."""
    return CreateModelUseCase(_require_model_repository())


def get_get_model_use_case() -> GetModelUseCase:
    """Provide a GetModelUseCase instance."""
    return GetModelUseCase(_require_model_repository())


def get_list_models_use_case() -> ListModelsUseCase:
    """Provide a ListModelsUseCase instance."""
    return ListModelsUseCase(_require_model_repository())


def get_update_model_use_case() -> UpdateModelUseCase:
    """Provide an UpdateModelUseCase instance."""
    return UpdateModelUseCase(_require_model_repository())


def get_delete_model_use_case() -> DeleteModelUseCase:
    """Provide a DeleteModelUseCase instance (reference-safe, MDL-004)."""
    return DeleteModelUseCase(_require_model_repository())


# ============= CRON PROVIDERS =============


def get_create_cron_use_case() -> CreateCronUseCase:
    """Provide a CreateCronUseCase instance (validated, CRN-002)."""
    store, repo = _require_persistence()
    return CreateCronUseCase(
        cron_repository=_require_cron_repository(),
        agent_config_repository=repo,
        config_store=store,
    )


def get_get_cron_use_case() -> GetCronUseCase:
    """Provide a GetCronUseCase instance."""
    return GetCronUseCase(_require_cron_repository())


def get_list_crons_use_case() -> ListCronsUseCase:
    """Provide a ListCronsUseCase instance."""
    return ListCronsUseCase(_require_cron_repository())


def get_update_cron_use_case() -> UpdateCronUseCase:
    """Provide an UpdateCronUseCase instance."""
    store, repo = _require_persistence()
    return UpdateCronUseCase(
        cron_repository=_require_cron_repository(),
        agent_config_repository=repo,
        config_store=store,
    )


def get_delete_cron_use_case() -> DeleteCronUseCase:
    """Provide a DeleteCronUseCase instance."""
    return DeleteCronUseCase(_require_cron_repository())


def get_execute_cron_run_use_case() -> ExecuteCronRunUseCase:
    """Provide an ExecuteCronRunUseCase instance (single invocation path)."""
    store, _repo = _require_persistence()
    return ExecuteCronRunUseCase(
        cron_repository=_require_cron_repository(),
        run_repository=_require_cron_run_repository(),
        thread_repository=_require_thread_repository(),
        trace_repository=_require_trace_event_repository(),
        agent_registry=_require_agent_registry(),
        config_store=store,
    )


def get_run_cron_now_use_case() -> RunCronNowUseCase:
    """Provide a RunCronNowUseCase instance."""
    return RunCronNowUseCase(
        cron_repository=_require_cron_repository(),
        run_repository=_require_cron_run_repository(),
        executor=get_execute_cron_run_use_case(),
    )


def get_list_cron_runs_use_case() -> ListCronRunsUseCase:
    """Provide a ListCronRunsUseCase instance."""
    return ListCronRunsUseCase(
        cron_repository=_require_cron_repository(),
        run_repository=_require_cron_run_repository(),
    )


def get_scheduler_tick_use_case() -> SchedulerTickUseCase:
    """Provide the production scheduler tick use case (CRN-006 claims)."""
    return SchedulerTickUseCase(
        cron_repository=_require_cron_repository(),
        run_repository=_require_cron_run_repository(),
        executor=get_execute_cron_run_use_case(),
    )


def get_list_traces_use_case() -> ListTracesUseCase:
    """Provide a ListTracesUseCase instance (bounded aggregate, TRC-002)."""
    if _root.trace_summary_repository is None:
        raise StorageError(ErrorMessage.STORAGE_REPO_NOT_INITIALIZED)
    return ListTracesUseCase(_root.trace_summary_repository)


# ============= LLM USAGE DASHBOARD PROVIDERS =============


def _require_trace_usage_repository() -> TraceUsageRepository:
    """Return the usage repository or raise StorageError if not initialized."""
    if _root.trace_usage_repository is None:
        raise StorageError(ErrorMessage.STORAGE_REPO_NOT_INITIALIZED)
    return _root.trace_usage_repository


def get_trace_usage_use_case() -> GetTraceUsageUseCase:
    """Provide a GetTraceUsageUseCase instance (dashboard spec §4)."""
    return GetTraceUsageUseCase(_require_trace_usage_repository())


def get_top_usage_runs_use_case() -> GetTopUsageRunsUseCase:
    """Provide a GetTopUsageRunsUseCase instance (dashboard spec §5)."""
    return GetTopUsageRunsUseCase(_require_trace_usage_repository())


# ============= STORE FILE PROVIDERS =============


def _require_store_file_repository() -> StoreFileRepository:
    """Return store file repository, creating an in-memory fallback if not initialized.

    During tests, ``init_persistence`` is not called, so the repository would
    be ``None``. To keep the API functional without a database, we lazily create
    an :class:`InMemoryStore`-backed adapter on first access.

    The fallback uses a per-user ``namespace_provider`` so authenticated
    requests are isolated, while unauthenticated contexts (tests with no
    ``current_user_id``) fall back to the legacy ``("filesystem",)`` namespace.
    """
    if _root.store_file_repository is None:
        from src.infrastructure.deepagent.factory import _get_memory_store
        from src.infrastructure.deepagent.namespace import user_namespaced

        _root.store_file_repository = LangGraphStoreFileRepository(
            store=_get_memory_store(), namespace_provider=lambda: user_namespaced("filesystem")
        )
    return _root.store_file_repository


def get_store_file_repository() -> StoreFileRepository:
    """Provide a :class:`StoreFileRepository` instance (singleton wired at startup)."""
    return _require_store_file_repository()


def get_list_store_files_use_case() -> ListStoreFilesUseCase:
    """Provide a :class:`ListStoreFilesUseCase` instance."""
    return ListStoreFilesUseCase(_require_store_file_repository())


def get_list_store_file_previews_use_case() -> ListStoreFilePreviewsUseCase:
    """Provide a :class:`ListStoreFilePreviewsUseCase` instance."""
    return ListStoreFilePreviewsUseCase(_require_store_file_repository())


def get_get_store_file_use_case() -> GetStoreFileUseCase:
    """Provide a :class:`GetStoreFileUseCase` instance."""
    return GetStoreFileUseCase(_require_store_file_repository())


def get_put_store_file_use_case() -> PutStoreFileUseCase:
    """Provide a :class:`PutStoreFileUseCase` instance."""
    return PutStoreFileUseCase(_require_store_file_repository())


def get_delete_store_file_use_case() -> DeleteStoreFileUseCase:
    """Provide a :class:`DeleteStoreFileUseCase` instance."""
    return DeleteStoreFileUseCase(_require_store_file_repository())


# ============= API KEY MANAGEMENT PROVIDERS =============


def get_current_user_id() -> str:
    """Provide the authenticated user id from the RLS contextvar.

    Set by :meth:`ComposableAgentsSecurity.verify_credentials` after a
    successful JWT / API-key authentication. Routes that depend on this
    function get a 401 :class:`AuthenticationError` when no user is resolved
    (e.g. the dependency is not overridden and no auth middleware ran).

    Returns:
        The authenticated user id.

    Raises:
        AuthenticationError: If no user id is set in the current context.
    """
    user_id = current_user_id.get()
    if user_id is None:
        raise AuthenticationError(ErrorMessage.AUTH_INVALID_CREDENTIALS)
    return user_id


def get_current_auth_context() -> AuthContext:
    """Provide the full :class:`AuthContext` resolved for the current request.

    Set by :meth:`ComposableAgentsSecurity.verify_credentials` after a
    successful JWT / API-key authentication. Carries the propagated profile
    claims (``email`` / ``name`` / ``username``) on the JWT path, which the
    ``GET /api/v1/users/me`` endpoint exposes.

    Returns:
        The authenticated :class:`AuthContext`.

    Raises:
        AuthenticationError: If no auth context is set in the current context
            (e.g. the dependency is not overridden and no auth middleware ran).
    """
    ctx = current_auth_context.get()
    if ctx is None:
        raise AuthenticationError(ErrorMessage.AUTH_INVALID_CREDENTIALS)
    return ctx


def get_get_current_user_use_case() -> GetCurrentUserUseCase:
    """Provide a :class:`GetCurrentUserUseCase` instance."""
    return GetCurrentUserUseCase()


def _require_api_key_repository() -> ApiKeyRepository:
    """Return the API key repository or raise StorageError if not initialized.

    Returns:
        The wired :class:`ApiKeyRepository` instance.

    Raises:
        StorageError: If the persistence layer is not initialized.
    """
    if _root.api_key_repository is None:
        raise StorageError(ErrorMessage.STORAGE_REPO_NOT_INITIALIZED)
    return _root.api_key_repository


def get_create_api_key_use_case() -> CreateApiKeyUseCase:
    """Provide a :class:`CreateApiKeyUseCase` instance."""
    return CreateApiKeyUseCase(repo=_require_api_key_repository())


def get_list_api_keys_use_case() -> ListApiKeysUseCase:
    """Provide a :class:`ListApiKeysUseCase` instance."""
    return ListApiKeysUseCase(repo=_require_api_key_repository())


def get_revoke_api_key_use_case() -> RevokeApiKeyUseCase:
    """Provide a :class:`RevokeApiKeyUseCase` instance."""
    return RevokeApiKeyUseCase(repo=_require_api_key_repository())


# ============= USER LLM SETTINGS PROVIDERS =============


def _require_user_llm_settings_repository() -> UserLlmSettingsRepository:
    """Return the user-LLM-settings repository or raise StorageError if not initialized.

    Returns:
        The wired :class:`UserLlmSettingsRepository` instance.

    Raises:
        StorageError: If the persistence layer is not initialized.
    """
    if _root.user_llm_settings_repository is None:
        raise StorageError(ErrorMessage.STORAGE_REPO_NOT_INITIALIZED)
    return _root.user_llm_settings_repository


def _require_fernet_crypto() -> FernetCrypto:
    """Return the FernetCrypto instance or raise StorageError if not initialized.

    Returns:
        The wired :class:`FernetCrypto` instance.

    Raises:
        StorageError: If the persistence layer is not initialized.
    """
    if _root.fernet_crypto is None:
        raise StorageError(ErrorMessage.STORAGE_REPO_NOT_INITIALIZED)
    return _root.fernet_crypto


def get_get_user_llm_settings_use_case() -> GetUserLlmSettingsUseCase:
    """Provide a :class:`GetUserLlmSettingsUseCase` instance."""
    return GetUserLlmSettingsUseCase(repo=_require_user_llm_settings_repository())


def get_upsert_user_llm_settings_use_case() -> UpsertUserLlmSettingsUseCase:
    """Provide an :class:`UpsertUserLlmSettingsUseCase` instance."""
    return UpsertUserLlmSettingsUseCase(
        repo=_require_user_llm_settings_repository(),
        crypto=_require_fernet_crypto(),
    )


def get_delete_user_llm_settings_use_case() -> DeleteUserLlmSettingsUseCase:
    """Provide a :class:`DeleteUserLlmSettingsUseCase` instance."""
    return DeleteUserLlmSettingsUseCase(repo=_require_user_llm_settings_repository())


def get_llm_credentials_resolver():
    """Return a closure resolving the current user's LLM credentials.

    Used by the :class:`PersistentAgentRegistry` to build per-user
    :class:`ChatOpenAI` instances. Returns ``None`` when the user has not
    configured a provider, so the factory can raise
    :class:`LlmNotConfiguredError`.

    Returns:
        An async callable ``(user_id: str) -> tuple[str, str] | None``.
    """

    async def _resolve(user_id: str) -> tuple[str, str] | None:
        return await _require_user_llm_settings_repository().get_decrypted(user_id)

    return _resolve


def get_llm_provider_resolver():
    """Return a closure resolving the current user's LLM provider label.

    Used by the :class:`PersistentAgentRegistry` to stamp the serving provider
    on the LLM-usage capture (``metadata.llm_usage.provider``). Returns
    ``None`` when the user has no configured provider (the registry then falls
    back to the OpenAI-compatible serving chain label).

    Returns:
        An async callable ``(user_id: str) -> str | None``.
    """

    async def _resolve(user_id: str) -> str | None:
        settings = await _require_user_llm_settings_repository().get(user_id)
        return settings.provider if settings is not None else None

    return _resolve
