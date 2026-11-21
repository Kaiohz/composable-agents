"""Registered MCP server entities for the mcp-raganything registry boundary.

The registry (``mcp-raganything``) remains the sole owner of MCP-server
persistence, validation, generation, mounting and rehydration (REQ-017).
``composable-agents`` only carries these projections:

* :class:`RegisteredMcpServer` — the MASKED read projection returned by the
  registry list/get/create/update operations (REQ-018): ``auth_token``,
  ``headers`` and ``env`` are always empty/None here; the registry masks them
  and this adapter never fills them back in.
* :class:`McpServerWrite` — the write-only input for create/update/validate
  (REQ-019): secrets MAY be supplied and are forwarded to the trusted
  registry, but are never logged and never echoed back by this adapter.
* :class:`McpServerValidationReport` — the result of a non-persisting
  validation call.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class RegisteredMcpServer(BaseModel):
    """MASKED projection of one registered MCP server (REQ-018)."""

    name: str
    transport: str = "http"
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    # Masked secrets (REQ-018): the registry masks them; never unmasked here.
    headers: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    auth_token: str | None = None
    tool_count: int = 0
    source_type: str = "external"
    openapi_url: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class McpServerWrite(BaseModel):
    """Write-only input for registry create/update/validate (REQ-019).

    Secret fields (``auth_token`` / ``headers`` / ``env``) are write-only:
    they are forwarded to the trusted registry and are redacted from tool
    results and logs.
    """

    name: str = Field(..., min_length=1)
    transport: Literal["http", "stdio"] = "http"
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    auth_token: str | None = None
    source_type: Literal["external", "openapi"] = "external"
    openapi_url: str | None = None


class McpServerValidationReport(BaseModel):
    """Result of validating a candidate MCP server without persisting it.

    Shaped by the REAL registry contract (``mcp-raganything``'s
    ``McpServerValidateResponse``): the deployed registry answers
    ``{"tool_count": N}`` on a validated candidate and raises an explicit
    error (non-2xx) otherwise. ``name`` and ``valid`` are therefore
    OPTIONAL with sane defaults — a 200 answer is a validated candidate —
    while fuller deployments may still populate the full report shape.
    """

    name: str = ""
    valid: bool = True
    tool_count: int = 0
    errors: list[str] = Field(default_factory=list)
