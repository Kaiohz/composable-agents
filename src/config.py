import os
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv
from pydantic import Field, PrivateAttr, field_validator, model_validator
from pydantic_settings import BaseSettings

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
    secret_encryption_key: str = ""
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
