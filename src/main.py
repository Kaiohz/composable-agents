"""Main entry point for the Composable Agents API."""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypeVar, cast

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from src.application.mcp.server import (
    MCP_MOUNT_PATH,
    WELL_KNOWN_PROTECTED_RESOURCE_PATH,
    public_discovery_origin,
)
from src.application.routes.agents import router as agents_router
from src.application.routes.api_keys import router as api_keys_router
from src.application.routes.chat import router as chat_router
from src.application.routes.cron import router as cron_router
from src.application.routes.health import router as health_router
from src.application.routes.model import router as model_router
from src.application.routes.prompt import router as prompt_router
from src.application.routes.store import router as store_router
from src.application.routes.threads import router as threads_router
from src.application.routes.trace import router as trace_router
from src.application.routes.traces import router as traces_router
from src.application.routes.user_llm_settings import router as user_llm_settings_router
from src.application.routes.users import router as users_router
from src.application.routes.websocket import router as websocket_router
from src.config import (
    Settings,  # type: ignore[call-arg]  # noqa: F401  # env-resolved required fields (pydantic-settings)
)
from src.dependencies import (
    close_persistence,
    get_scheduler_tick_use_case,
    init_persistence,
    mcp_asgi_app,
    mcp_session_manager,
    mcp_tool_loader,
    security,
    tracing_provider,
)
from src.domain.errors.agent import (
    AgentConfigAlreadyExistsError,
    AgentConfigReferenceError,
    AgentError,
    AgentNotFoundError,
    BuiltinAgentError,
)
from src.domain.errors.attachment import (
    AttachmentValidationError,
    DocumentsNotAllowedError,
    FilesMcpTemporaryError,
)
from src.domain.errors.base import DomainError
from src.domain.errors.config import ConfigError, ConfigNotFoundError, ConfigValidationError
from src.domain.errors.cron import CronNotFoundError, CronRunConflictError, CronValidationError
from src.domain.errors.hitl import InvalidHitlActionError
from src.domain.errors.llm import LlmNotConfiguredError
from src.domain.errors.mcp import McpError
from src.domain.errors.model import ModelInUseError, ModelNotFoundError
from src.domain.errors.prompt import PromptInUseError, PromptNotFoundError, PromptVersionNotFoundError
from src.domain.errors.security import AuthenticationError, InvalidApiKeyError
from src.domain.errors.storage import StorageError
from src.domain.errors.store_file import StoreFileNotFoundError
from src.domain.errors.thread import ThreadNotFoundError
from src.domain.logging.messages import LogMessage
from src.infrastructure.cron_scheduler import start_scheduler_loop, stop_scheduler_loop
from src.infrastructure.logging import RequestIdMiddleware, configure_logging

settings = Settings()  # type: ignore[call-arg]  # env-resolved required fields (pydantic-settings)

configure_logging(settings)

logger = logging.getLogger(__name__)

E = TypeVar("E", bound=DomainError)


def _add_domain_error_handler(
    error_type: type[E], handler: Callable[[Request, E], Awaitable[JSONResponse] | JSONResponse]
) -> None:
    """Register a typed domain-error handler.

    Starlette keys handlers on the exception type but types the callable
    parameter as the broad ``Exception`` (contravariance), so precisely-typed
    handlers cannot be passed without a cast. The cast is sound: Starlette
    dispatches by ``isinstance`` on the registered type, so the narrowed
    parameter always receives an instance of ``error_type``, and ``E`` is
    bound to ``DomainError`` (every handler reads ``status_code``/``detail``).
    """
    app.add_exception_handler(
        error_type,
        cast("Callable[[Request, Exception], Awaitable[Response] | Response]", handler),
    )


def _run_alembic_upgrade() -> None:
    from alembic.config import Config

    from alembic import command

    alembic_dir = Path(__file__).parent
    cfg = Config(str(alembic_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(alembic_dir / "alembic"))
    command.upgrade(cfg, "head")


_scheduler_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _scheduler_task
    logger.info(LogMessage.APP_STARTUP_INITIATED)
    # MCP authorization-server shape (QA BUG-003): the public metadata document
    # only advertises an issuer a PUBLIC MCP client can reach. An empty value
    # (nothing advertised at all) and a non-https value (the container-side
    # LOGTO_URL fallback) stay non-fatal but are warned at startup — the
    # configmap sets MCP_AUTHORIZATION_SERVER=https://logto.soludev.tech/oidc.
    if not settings.mcp_authorization_server:
        logger.warning(LogMessage.MCP_AUTHORIZATION_SERVER_NOT_ADVERTISED)
    elif not settings.mcp_authorization_server.startswith("https://"):
        logger.warning(LogMessage.MCP_AUTHORIZATION_SERVER_NOT_HTTPS, settings.mcp_authorization_server)
    # MCP session manager lifecycle (feat-composable-mcp): the streamable-HTTP
    # session resources start and stop with the FastAPI lifespan (Technical
    # Notes). Entered FIRST (AsyncExitStack unwinds shutdown in reverse) so a
    # clean startup failure never tears down the shared persistence stack.
    async with contextlib.AsyncExitStack() as stack:
        if mcp_session_manager is not None:
            try:
                await stack.enter_async_context(mcp_session_manager.run())
                logger.info(LogMessage.MCP_SESSION_MANAGER_RUNNING)
            except RuntimeError:
                # The SDK allows ONE run() per session-manager instance. The
                # production boot runs the lifespan once per process (per
                # worker); a repeat lifespan in a long-lived process (unit
                # tests driving lifespan directly) skips MCP session
                # management instead of failing the whole startup.
                logger.warning(LogMessage.MCP_SESSION_MANAGER_ALREADY_RAN)
        try:
            await init_persistence()
            logger.info(LogMessage.APP_PERSISTENCE_INITIALIZED)
        except Exception:
            logger.exception(LogMessage.APP_PERSISTENCE_INIT_FAILED)
        # Cron scheduler background loop (Phase 3 lifecycle). Tests drive
        # ``run_scheduler_tick`` directly, so the loop is intentionally only tied
        # to the ASGI lifespan (never to the import side effects). Startup is
        # NON-FATAL on provider wiring failures — the same resilience as
        # ``init_persistence`` (the tick would crash every cold start without
        # a database, while the REST/SSE/MCP surfaces stay healthy).
        try:
            _scheduler_task = start_scheduler_loop(get_scheduler_tick_use_case().execute)
        except Exception:
            _scheduler_task = None
            logger.exception(LogMessage.APP_SCHEDULER_START_FAILED)
        logger.info(LogMessage.APP_STARTUP_COMPLETE)
        yield
        logger.info(LogMessage.APP_SHUTDOWN_INITIATED)
        await stop_scheduler_loop(_scheduler_task)
        _scheduler_task = None
        try:
            await close_persistence()
            logger.info(LogMessage.APP_PERSISTENCE_CLOSED)
        except Exception:
            logger.exception(LogMessage.APP_PERSISTENCE_CLOSE_FAILED)
        try:
            await mcp_tool_loader.close()
            logger.info(LogMessage.APP_MCP_LOADER_CLOSED)
        except Exception:
            logger.exception(LogMessage.APP_MCP_LOADER_CLOSE_FAILED)
        try:
            await tracing_provider.flush()
            await tracing_provider.shutdown()
            logger.info(LogMessage.APP_TRACING_SHUTDOWN)
        except Exception:
            logger.exception(LogMessage.APP_TRACING_SHUTDOWN_FAILED)
        logger.info(LogMessage.APP_SHUTDOWN_COMPLETE)


app = FastAPI(
    title="composable-agents",
    description="Compose an agent using Deep Agent LangGraph and expose it via FastAPI.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Public OAuth2 discovery document (feat-mcp-oauth2-dual-auth REQ-002). Wired
# BEFORE the protected routers: the route carries NO auth dependency at all —
# it is a static, non-sensitive discovery document consumed by MCP clients
# right after the 401 ``WWW-Authenticate`` challenge (the OPTIONS preflight is
# answered by the CORS middleware without auth). The advertised origin is the
# SAME derivation the 401 challenge uses (public_discovery_origin): the
# configured MCP_PUBLIC_BASE_URL when set, otherwise ONLY an explicitly
# allow-listed request Host (MCP_ALLOWED_HOSTS) — a raw, spoofable Host is
# never echoed into the metadata (M-finding M-2); without a derivable origin
# the route answers 404 (no discovery document at an unverifiable origin).
# ``authorization_servers`` comes from MCP_AUTHORIZATION_SERVER (an explicit
# PUBLIC issuer, QA BUG-003), falling back to ``<logto_url>/oidc``; when
# neither is configured an EMPTY list is advertised — a public document never
# carries an issuer the deployment cannot vouch for.
@app.get(
    WELL_KNOWN_PROTECTED_RESOURCE_PATH,
    summary="OAuth2 protected-resource metadata (public discovery)",
    description=(
        "Unauthenticated RFC 9728-style protected-resource metadata for the "
        "MCP OAuth2 discovery chain: the resource identifier plus the "
        "authorization servers accepting tokens for it."
    ),
    response_description="The static protected-resource metadata document.",
    include_in_schema=False,
)
async def oauth_protected_resource(request: Request) -> Response:
    """Return the static OAuth2 protected-resource metadata (public, REQ-002).

    Args:
        request: The incoming request (its ``Host`` header is used only when
            allow-listed by ``MCP_ALLOWED_HOSTS``).

    Returns:
        ``{"resource": "https://<public-origin>/mcp", "authorization_servers":
        [...]}`` where the advertised issuer is ``MCP_AUTHORIZATION_SERVER``
        (an explicit publicly-reachable issuer, QA BUG-003) falling back to
        ``<logto_url>/oidc`` — the list is EMPTY when neither is configured —
        and 404 (plain, no derivation detail) when no trustworthy origin can
        be derived.
    """
    origin = public_discovery_origin(
        request.headers.get("host", ""),
        configured_base_url=settings.mcp_public_base_url,
        allowed_hosts=settings.mcp_allowed_hosts,
    )
    if origin is None:
        logger.warning(LogMessage.MCP_DISCOVERY_ORIGIN_UNAVAILABLE)
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": "Not Found"})
    return JSONResponse(
        content={
            "resource": f"{origin}{MCP_MOUNT_PATH}",
            "authorization_servers": ([settings.mcp_authorization_server] if settings.mcp_authorization_server else []),
        }
    )


app.include_router(health_router)

# Inbound Composable MCP control plane (feat-composable-mcp): an
# independently-mounted ASGI app that performs its OWN authentication —
# FastAPI router dependencies do NOT protect a mounted ASGI application
# (Technical Notes). The exact-path route avoids Starlette's mount redirect
# (307) for POST /mcp — the auth guard runs FIRST on the original request —
# while the mount serves "/mcp/..." sub-paths. The guard returns HTTP 401
# for missing/invalid credentials before any MCP method runs.
if mcp_asgi_app is not None:
    from starlette.routing import Route

    app.router.routes.append(Route("/mcp", mcp_asgi_app, name="composable_mcp_endpoint"))
    app.mount("/mcp", mcp_asgi_app)

# WebSocket routes manage their own API key validation via verify_api_key_ws
# because FastAPI APIRouter(dependencies=...) does not apply to WebSocket endpoints.
app.include_router(websocket_router)

# All non-WebSocket routes except health are protected behind dual auth
# (JWT bearer token OR per-user API key) via verify_credentials. The
# master-key verify_api_key stays available for backward-compat and is still
# used by the WebSocket router's fallback path.
protected = APIRouter(dependencies=[Depends(security.verify_credentials)])
protected.include_router(threads_router)
protected.include_router(chat_router)
protected.include_router(trace_router)
protected.include_router(traces_router)
protected.include_router(agents_router)
protected.include_router(prompt_router)
protected.include_router(model_router)
protected.include_router(cron_router)
protected.include_router(store_router)
protected.include_router(api_keys_router)
protected.include_router(user_llm_settings_router)
protected.include_router(users_router)
app.include_router(protected)


def _error_response(exc: DomainError) -> JSONResponse:
    """Build a JSON response from any domain error.

    Each domain error carries its own ``status_code`` (an :class:`ErrorCode`)
    and ``detail`` text, so the response shape is uniform for every type.
    :class:`ConfigValidationError` additionally exposes structured ``errors``.
    """
    content: dict = {"detail": exc.detail}
    if isinstance(exc, ConfigValidationError):
        content["errors"] = exc.errors
    return JSONResponse(status_code=int(exc.status_code), content=content)


def agent_config_already_exists_handler(_request: Request, exc: AgentConfigAlreadyExistsError) -> JSONResponse:
    logger.warning(LogMessage.LOG_AGENT_CONFIG_ALREADY_EXISTS, exc.detail)
    return _error_response(exc)


def request_validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """Render a request-validation failure (HTTP 422) without echoing raw input.

    FastAPI's default body embeds the rejected ``input`` value. A malformed
    body may carry credential material (models reject credential-like extra
    fields — QA re-run), so the rendered errors strip the ``input`` member and
    keep only the field location, message and error type.
    """
    errors = [
        {key: value for key, value in error.items() if key not in ("input", "ctx")}
        for error in exc.errors()
        if isinstance(error, dict)
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error.", "errors": errors},
    )


def storage_error_handler(_request: Request, exc: StorageError) -> JSONResponse:
    """Render a storage failure (HTTP 503).

    The full detail (SQL statement, bound parameters, driver class) is logged
    server-side for diagnostics but is NEVER reflected in the response body:
    the client receives a generic, bounded message, so internal schema names,
    bound values or entire failing rows (QA BUG-006/008/009 — the review's
    cross-cutting finding I18) can never be disclosed over HTTP.
    """
    logger.error(LogMessage.LOG_STORAGE_ERROR, exc.detail)
    return JSONResponse(status_code=int(exc.status_code), content={"detail": "A storage operation failed."})


def agent_not_found_handler(_request: Request, exc: AgentNotFoundError) -> JSONResponse:
    logger.warning(LogMessage.LOG_AGENT_NOT_FOUND, exc.detail)
    return _error_response(exc)


def config_not_found_handler(_request: Request, exc: ConfigNotFoundError) -> JSONResponse:
    logger.warning(LogMessage.LOG_CONFIG_NOT_FOUND, exc.detail)
    return _error_response(exc)


def thread_not_found_handler(_request: Request, exc: ThreadNotFoundError) -> JSONResponse:
    logger.warning(LogMessage.LOG_THREAD_NOT_FOUND, exc.detail)
    return _error_response(exc)


def store_file_not_found_handler(_request: Request, exc: StoreFileNotFoundError) -> JSONResponse:
    logger.warning("Store file not found: %s", exc.detail)
    return _error_response(exc)


def prompt_not_found_handler(_request: Request, exc: PromptNotFoundError) -> JSONResponse:
    logger.warning(LogMessage.LOG_PROMPT_NOT_FOUND, exc.detail)
    return _error_response(exc)


def prompt_version_not_found_handler(_request: Request, exc: PromptVersionNotFoundError) -> JSONResponse:
    logger.warning(LogMessage.LOG_PROMPT_NOT_FOUND, exc.detail)
    return _error_response(exc)


def prompt_in_use_handler(_request: Request, exc: PromptInUseError) -> JSONResponse:
    logger.warning(LogMessage.LOG_PROMPT_IN_USE, exc.detail)
    return _error_response(exc)


def model_not_found_handler(_request: Request, exc: ModelNotFoundError) -> JSONResponse:
    logger.warning(LogMessage.LOG_MODEL_NOT_FOUND, exc.detail)
    return _error_response(exc)


def model_in_use_handler(_request: Request, exc: ModelInUseError) -> JSONResponse:
    logger.warning(LogMessage.LOG_MODEL_IN_USE, exc.detail)
    return _error_response(exc)


def cron_not_found_handler(_request: Request, exc: CronNotFoundError) -> JSONResponse:
    logger.warning(LogMessage.LOG_CRON_NOT_FOUND, exc.detail)
    return _error_response(exc)


def cron_validation_handler(_request: Request, exc: CronValidationError) -> JSONResponse:
    logger.warning(LogMessage.LOG_CRON_VALIDATION_ERROR, exc.detail)
    return _error_response(exc)


def cron_run_conflict_handler(_request: Request, exc: CronRunConflictError) -> JSONResponse:
    logger.warning(LogMessage.LOG_CRON_RUN_CONFLICT, exc.detail)
    return _error_response(exc)


def agent_config_reference_handler(_request: Request, exc: AgentConfigReferenceError) -> JSONResponse:
    logger.warning(LogMessage.LOG_AGENT_CONFIG_REFERENCE_ERROR, exc.detail)
    return _error_response(exc)


def builtin_agent_error_handler(_request: Request, exc: BuiltinAgentError) -> JSONResponse:
    """Map read-only built-in-agent violations (REQ-034) to their 403 detail."""
    logger.warning(LogMessage.LOG_BUILTIN_AGENT_READONLY, exc.detail)
    return _error_response(exc)


def invalid_hitl_action_handler(_request: Request, exc: InvalidHitlActionError) -> JSONResponse:
    logger.warning(LogMessage.LOG_INVALID_HITL_ACTION, exc.detail)
    return _error_response(exc)


def attachment_validation_handler(_request: Request, exc: AttachmentValidationError) -> JSONResponse:
    """Map attachment contract violations (REQ-006..008) to their 422 detail."""
    logger.warning(LogMessage.LOG_ATTACHMENT_VALIDATION, exc.detail)
    return _error_response(exc)


def documents_not_allowed_handler(_request: Request, exc: DocumentsNotAllowedError) -> JSONResponse:
    """Map the Files-MCP document policy refusal (REQ-010) to its 422 detail."""
    logger.warning(LogMessage.LOG_DOCUMENTS_NOT_ALLOWED, exc.detail)
    return _error_response(exc)


def files_mcp_temporary_handler(_request: Request, exc: FilesMcpTemporaryError) -> JSONResponse:
    """Map Files MCP temporary tool failures (EC-007) to their 502 detail."""
    logger.error(LogMessage.LOG_FILES_MCP_TEMPORARY_ERROR, exc.detail)
    return _error_response(exc)


def config_validation_handler(_request: Request, exc: ConfigValidationError) -> JSONResponse:
    logger.error(LogMessage.LOG_CONFIG_VALIDATION_ERROR, exc.detail, exc.errors)
    return _error_response(exc)


def config_error_handler(_request: Request, exc: ConfigError) -> JSONResponse:
    logger.error(LogMessage.LOG_CONFIG_ERROR, exc.detail)
    return _error_response(exc)


def agent_error_handler(_request: Request, exc: AgentError) -> JSONResponse:
    logger.error(LogMessage.LOG_AGENT_ERROR, exc.detail)
    return _error_response(exc)


def mcp_error_handler(_request: Request, exc: McpError) -> JSONResponse:
    logger.error(LogMessage.LOG_MCP_ERROR, exc.detail)
    return _error_response(exc)


def domain_error_handler(_request: Request, exc: DomainError) -> JSONResponse:
    logger.error(LogMessage.LOG_UNHANDLED_DOMAIN_ERROR, exc.detail)
    return _error_response(exc)


def invalid_api_key_handler(_request: Request, exc: InvalidApiKeyError) -> JSONResponse:
    logger.warning(LogMessage.LOG_INVALID_API_KEY, exc.detail)
    return _error_response(exc)


def authentication_error_handler(_request: Request, exc: AuthenticationError) -> JSONResponse:
    """Map :class:`AuthenticationError` (raised by verify_credentials) to 401."""
    logger.warning("Authentication failed: %s", exc.detail)
    return _error_response(exc)


def llm_not_configured_handler(_request: Request, exc: LlmNotConfiguredError) -> JSONResponse:
    """Map :class:`LlmNotConfiguredError` (no LLM provider configured) to 422."""
    logger.warning("LLM not configured: %s", exc.detail)
    return _error_response(exc)


# Register one handler per domain error type explicitly. Each handler reads the
# exception's own status_code/detail, so no separate error->HTTP mapping table
# is required. Most-specific types are registered first.
_add_domain_error_handler(ConfigNotFoundError, config_not_found_handler)
_add_domain_error_handler(ConfigValidationError, config_validation_handler)
_add_domain_error_handler(ConfigError, config_error_handler)
_add_domain_error_handler(ThreadNotFoundError, thread_not_found_handler)
_add_domain_error_handler(StoreFileNotFoundError, store_file_not_found_handler)
_add_domain_error_handler(AgentNotFoundError, agent_not_found_handler)
_add_domain_error_handler(AgentConfigAlreadyExistsError, agent_config_already_exists_handler)
_add_domain_error_handler(AgentError, agent_error_handler)
_add_domain_error_handler(InvalidHitlActionError, invalid_hitl_action_handler)
_add_domain_error_handler(AttachmentValidationError, attachment_validation_handler)
_add_domain_error_handler(DocumentsNotAllowedError, documents_not_allowed_handler)
_add_domain_error_handler(FilesMcpTemporaryError, files_mcp_temporary_handler)
# RequestValidationError is a FastAPI boundary error, not a DomainError — the
# same contravariance cast as the typed registrar applies here; mypy's E bound
# does not cover it, hence the direct registration with a narrowed cast.
app.add_exception_handler(
    RequestValidationError,
    cast(
        "Callable[[Request, Exception], Awaitable[Response] | Response]",
        request_validation_error_handler,
    ),
)
_add_domain_error_handler(StorageError, storage_error_handler)
_add_domain_error_handler(McpError, mcp_error_handler)
_add_domain_error_handler(PromptNotFoundError, prompt_not_found_handler)
_add_domain_error_handler(PromptVersionNotFoundError, prompt_version_not_found_handler)
_add_domain_error_handler(PromptInUseError, prompt_in_use_handler)
_add_domain_error_handler(ModelNotFoundError, model_not_found_handler)
_add_domain_error_handler(ModelInUseError, model_in_use_handler)
_add_domain_error_handler(CronNotFoundError, cron_not_found_handler)
_add_domain_error_handler(CronValidationError, cron_validation_handler)
_add_domain_error_handler(CronRunConflictError, cron_run_conflict_handler)
_add_domain_error_handler(AgentConfigReferenceError, agent_config_reference_handler)
_add_domain_error_handler(BuiltinAgentError, builtin_agent_error_handler)
_add_domain_error_handler(InvalidApiKeyError, invalid_api_key_handler)
_add_domain_error_handler(AuthenticationError, authentication_error_handler)
_add_domain_error_handler(LlmNotConfiguredError, llm_not_configured_handler)
_add_domain_error_handler(DomainError, domain_error_handler)


def run_fastapi():
    logger.info(LogMessage.APP_MIGRATIONS_RUNNING)
    _run_alembic_upgrade()
    logger.info(LogMessage.APP_MIGRATIONS_DONE)

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    run_fastapi()
