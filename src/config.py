import os
from typing import Annotated
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv
from pydantic import Field, PrivateAttr, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode

load_dotenv()

CHAT_ATTACHMENT_MAX_COUNT_DEFAULT = 10
CHAT_ATTACHMENT_MAX_TOTAL_BYTES_DEFAULT = 52_428_800
CHAT_ATTACHMENT_MAX_IMAGE_BYTES_DEFAULT = 10_485_760
CHAT_ATTACHMENT_MAX_DOCUMENT_BYTES_DEFAULT = 52_428_800
CREATE_PDF_MAX_MARKDOWN_CHARS_DEFAULT = 1_000_000

#: Spec defaults for the attachment/PDF limits (REQ-092) — read at Settings
#: load time only; business code consumes the Settings attributes / the
#: :class:`AttachmentLimits` VO (no hard-coded duplication elsewhere).
_ATTACHMENT_LIMIT_DEFAULTS: dict[str, int] = {
    "chat_attachment_max_count": CHAT_ATTACHMENT_MAX_COUNT_DEFAULT,
    "chat_attachment_max_total_bytes": CHAT_ATTACHMENT_MAX_TOTAL_BYTES_DEFAULT,
    "chat_attachment_max_image_bytes": CHAT_ATTACHMENT_MAX_IMAGE_BYTES_DEFAULT,
    "chat_attachment_max_document_bytes": CHAT_ATTACHMENT_MAX_DOCUMENT_BYTES_DEFAULT,
    "create_pdf_max_markdown_chars": CREATE_PDF_MAX_MARKDOWN_CHARS_DEFAULT,
}

_ATTACHMENT_LIMIT_FIELDS = tuple(_ATTACHMENT_LIMIT_DEFAULTS)


def resolve_env_int(var: str, default: int) -> int:
    """Resolve a non-negative integer env var with silent default fallback.

    Used by code paths that run OUTSIDE the Settings lifecycle (e.g. the
    builtin ``create_pdf`` tool executing inside the graph). Absent or empty
    values silently fall back to ``default`` (REQ-095); present-but-invalid
    values are caught by the Settings startup validation and would already
    have refused the process startup, so the tool falls back to the default.

    Args:
        var: The environment variable name.
        default: The spec default.

    Returns:
        The resolved positive integer.
    """
    raw = os.environ.get(var)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


class TracingSettings(BaseSettings):
    provider: str = "none"
    enabled: bool = False
    project_name: str = "composable-agents"
    phoenix_collector_endpoint: str | None = None
    phoenix_api_key: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None


class Settings(BaseSettings):
    openai_api_key: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    uvicorn_log_level: str = "info"
    allowed_origins: list[str] = ["http://localhost:8080"]
    api_key: str = ""
    # --- Dual auth (JWT + per-user API keys) ---
    logto_url: str = ""
    jwt_audience: str = ""
    # --- feat-mcp-oauth2-dual-auth (REQ-003 / REQ-013) ---
    # Additional accepted JWT audiences (comma-separated env var
    # ``JWT_AUDIENCES``, e.g. the Logto OpenCode app-id). The legacy single
    # ``JWT_AUDIENCE`` stays honored; the two lists are merged by the JWT
    # adapter (deduped, legacy value first). Empty/absent → legacy-only.
    jwt_audiences: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # MCP transport-security Host allow-list (comma-separated env var
    # ``MCP_ALLOWED_HOSTS``). DNS-rebinding protection stays ENABLED; only the
    # host list is configurable. The default mirrors the mcp SDK's
    # localhost-only auto-enable (dev/QA safe); the production configmap adds
    # the public proxy host. A present-but-empty value REFUSES startup (a
    # allow-empty list would 421 every MCP request — fail fast instead).
    mcp_allowed_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    )
    # Public base origin advertised by the MCP OAuth2 discovery documents
    # (feat-mcp-oauth2 M-2 hardening), e.g.
    # ``https://composables-agent.soludev.tech``. The request ``Host`` header
    # is client-supplied and MUST NOT be echoed unvalidated: when this is
    # configured it always wins; when unset the request Host is trusted ONLY
    # when it matches the ``MCP_ALLOWED_HOSTS`` allow-list. The production
    # configmap sets it.
    mcp_public_base_url: str = ""
    # Public OAuth2 authorization server advertised by the protected-resource
    # metadata document (REQ-002; QA BUG-003): the issuer an MCP client can
    # actually REACH. When unset it falls back to ``<logto_url>/oidc`` (the
    # historical document value); when both are empty NOTHING is advertised
    # (``authorization_servers: []`` — never an unreachable issuer). The
    # production configmap sets it explicitly because LOGTO_URL doubles as
    # the container-side JWKS base and may be an internal address.
    mcp_authorization_server: str = ""
    secret_encryption_key: str = ""
    # --- Composable MCP control plane (feat-composable-mcp) ---
    # Trusted mcp-raganything registry service base URL (REQ-017 / Technical
    # Notes: deployment configuration, never hard-coded, never derived from
    # the frontend config.json).
    mcp_registry_base_url: str = ""
    # Internal base URL of the built-in mcp-raganything MCP mounts
    # (/files/mcp, /classical/mcp, /web/mcp, /memories/mcp) used server-side
    # when assembling the built-in assistant's toolset (REQ-036).
    builtin_mcp_base_url: str = ""
    tracing: TracingSettings = TracingSettings()
    # Agent execution timeouts (seconds). The per-tool timeout isolates a hung
    # MCP tool as a recoverable ToolMessage error (agent continues). The graph
    # idle/invoke timeouts are backstops that only fire on a total stall.
    mcp_tool_timeout: float = 60.0
    agent_stream_idle_timeout: float = 120.0
    agent_invoke_timeout: float = 120.0

    minio_endpoint: str = "localhost:9040"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "composable-agents"
    minio_secure: bool = False

    database_url: str = Field(
        description="PostgreSQL connection string (postgresql://, postgres://, or postgresql+asyncpg://). Required.",
    )

    # --- Chat attachment + create_pdf limits (chat-pdf-attachments, REQ-092..096) ---
    # Read ONCE at Settings load (REQ-096): absent/empty → spec default
    # silently (REQ-095); present-but-invalid or incoherent hierarchy →
    # ValidationError which refuses the process startup (REQ-094).
    chat_attachment_max_count: int = CHAT_ATTACHMENT_MAX_COUNT_DEFAULT
    chat_attachment_max_total_bytes: int = CHAT_ATTACHMENT_MAX_TOTAL_BYTES_DEFAULT
    chat_attachment_max_image_bytes: int = CHAT_ATTACHMENT_MAX_IMAGE_BYTES_DEFAULT
    chat_attachment_max_document_bytes: int = CHAT_ATTACHMENT_MAX_DOCUMENT_BYTES_DEFAULT
    create_pdf_max_markdown_chars: int = CREATE_PDF_MAX_MARKDOWN_CHARS_DEFAULT
    # Sonar S5527/S4830: TLS to PostgreSQL validates certificates by default in
    # every encryption mode; local/self-signed stacks may explicitly opt out.
    # Never set this in production against a public endpoint.
    database_ssl_insecure: bool = False
    postgres_statement_cache_size: int | None = None
    _ssl_mode: str | None = PrivateAttr(default=None)

    @field_validator(*_ATTACHMENT_LIMIT_FIELDS, mode="before")
    @classmethod
    def _coerce_attachment_limit(cls, value: object, info) -> object:
        """Coerce one attachment limit env value (REQ-094/REQ-095).

        Absent/empty values silently fall back to the spec default; only a
        PRESENT but invalid value raises (startup refusal).
        """
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return _ATTACHMENT_LIMIT_DEFAULTS[info.field_name]
        if isinstance(value, str):
            try:
                value = int(value.strip())
            except ValueError as exc:
                raise ValueError(f"{info.field_name} must be a positive integer, got {value!r}") from exc
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{info.field_name} must be a positive integer, got {value!r}")
        if value <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer (> 0), got {value}")
        return value

    @property
    def ssl_mode(self) -> str | None:
        return self._ssl_mode

    @field_validator("jwt_audiences", "mcp_allowed_hosts", mode="before")
    @classmethod
    def _coerce_csv_list(cls, value: object, info: ValidationInfo) -> object:
        """Coerce a comma-separated env string into a list (REQ-003/REQ-013).

        ``JWT_AUDIENCES`` / ``MCP_ALLOWED_HOSTS`` are authored as
        comma-separated env values (never JSON). An already-parsed list
        passes through unchanged; absent values fall back to the field
        default.

        A present-but-empty ``MCP_ALLOWED_HOSTS`` (``""`` or blank-only) is
        INVALID (M-finding: silent total outage): with DNS-rebinding
        protection ENABLED an empty host allow-list rejects every MCP request
        with HTTP 421, so the process refuses to start and the configmap must
        explicitly list the public proxy host plus the localhost defaults.
        ``JWT_AUDIENCES`` tolerates an empty value (it simply means "legacy
        ``JWT_AUDIENCE`` only").
        """
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            if not parts and info.field_name == "mcp_allowed_hosts":
                raise ValueError(
                    "mcp_allowed_hosts must not be empty: with DNS-rebinding protection "
                    "enabled an empty host allow-list rejects every MCP request with "
                    "HTTP 421 — list the public proxy host (MCP_ALLOWED_HOSTS, "
                    "configmap REQ-013) plus the localhost defaults"
                )
            return parts
        return value

    @model_validator(mode="after")
    def normalize_database_url(self) -> "Settings":
        """Normalize DATABASE_URL for asyncpg.

        - postgresql:// or postgres:// → postgresql+asyncpg://
        - Extract sslmode and store it (passed via connect_args, not URL)
        - Strip sslmode and channel_binding (asyncpg doesn't accept them as query params)
        """
        for old, new in (
            ("postgresql://", "postgresql+asyncpg://"),
            ("postgres://", "postgresql+asyncpg://"),
        ):
            if self.database_url.startswith(old):
                self.database_url = self.database_url.replace(old, new, 1)
                break

        parsed = urlsplit(self.database_url)
        params = parse_qs(parsed.query)
        self._ssl_mode = (params.pop("sslmode", [None]) or [None])[0]
        params.pop("channel_binding", None)
        self.database_url = urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urlencode(params, doseq=True),
                parsed.fragment,
            )
        )
        return self

    @field_validator("mcp_authorization_server", mode="before")
    @classmethod
    def _validate_mcp_authorization_server(cls, value: object) -> object:
        """Reject a malformed explicit ``MCP_AUTHORIZATION_SERVER`` (startup).

        A blank value means "unset" (the logto-issuer fallback applies); a
        present-but-NON-absolute value would advertise a malformed issuer an
        MCP client cannot fetch — the same present-but-invalid ⇒ refuse
        startup rule the other configuration validators apply.
        """
        if isinstance(value, str):
            value = value.strip()
            if value and not value.startswith(("http://", "https://")):
                raise ValueError(
                    "mcp_authorization_server must be an absolute URL starting with https:// "
                    "(e.g. 'https://logto.soludev.tech/oidc')"
                )
        return value

    @model_validator(mode="after")
    def resolve_mcp_authorization_server(self) -> "Settings":
        """Fall back to ``<logto_url>/oidc`` when the setting is unset.

        QA BUG-003: the advertised issuer used to be derived inline in the
        well-known route from the raw ``LOGTO_URL`` — an internal (or empty)
        address reached the public discovery document verbatim. The explicit
        setting decouples the PUBLIC issuer from the container-side JWKS
        base; when neither is configured nothing is advertised.
        """
        if not self.mcp_authorization_server and self.logto_url:
            self.mcp_authorization_server = f"{self.logto_url.rstrip('/')}/oidc"
        return self

    @model_validator(mode="after")
    def validate_attachment_limit_hierarchy(self) -> "Settings":
        """Enforce ``image <= document <= total`` coherence at startup (REQ-094)."""
        if self.chat_attachment_max_image_bytes > self.chat_attachment_max_document_bytes:
            raise ValueError(
                "chat_attachment_max_image_bytes must be <= chat_attachment_max_document_bytes "
                f"(image={self.chat_attachment_max_image_bytes}, document={self.chat_attachment_max_document_bytes})"
            )
        if self.chat_attachment_max_document_bytes > self.chat_attachment_max_total_bytes:
            raise ValueError(
                "chat_attachment_max_document_bytes must be <= chat_attachment_max_total_bytes "
                f"(document={self.chat_attachment_max_document_bytes}, total={self.chat_attachment_max_total_bytes})"
            )
        return self
