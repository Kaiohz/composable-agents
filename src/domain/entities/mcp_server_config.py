from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class McpTransportType(StrEnum):
    STDIO = "stdio"
    HTTP = "http"


class McpAuthMode(StrEnum):
    """Authentication mode of a registered MCP server (feat-mcp-oauth2 REQ-004)."""

    #: The configured headers are sent as-is (current behavior, default).
    HEADERS = "headers"
    #: The connected user's credential is injected at load/validate time.
    USER_TOKEN = "user_token"


class McpServerConfig(BaseModel, frozen=True):
    """Configuration d'un serveur MCP."""

    name: str = Field(..., min_length=1)
    transport: McpTransportType
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    auth_token: str | None = None
    # feat-mcp-oauth2 REQ-004: with ``user_token`` the loader injects the
    # connected user's credential (JWT bearer / API key) at connection time.
    auth_mode: McpAuthMode = McpAuthMode.HEADERS

    @model_validator(mode="after")
    def validate_transport_fields(self) -> Self:
        if self.transport == McpTransportType.STDIO and not self.command:
            raise ValueError("'command' is required for stdio transport")
        if self.transport == McpTransportType.HTTP and not self.url:
            raise ValueError("'url' is required for http transport")
        return self
