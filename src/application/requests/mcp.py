"""Input DTOs for the Composable MCP registry tools (REQ-011/REQ-019).

The shapes mirror the mcp-raganything registry REST contract so MCP clients
supply exactly the registry's own fields — no looser MCP-only validation.
Secret fields (``auth_token`` / ``headers`` / ``env``) are write-only: they are
forwarded to the trusted registry and are never echoed back in tool results.

Coherence rules (``validate_server_spec_coherence``) are centralized here and
shared by the create DTO validator AND the update tool's merged-spec check, so
an invalid update is rejected LOCALLY with the stable, actionable message
instead of forwarding the registry's 422 (BUG-002).
"""

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

#: Controlled coherence messages — stable and non-sensitive; they never embed
#: request payloads or secrets, so they are safe to surface to MCP clients.
#: The wording matches the create DTO's established messages verbatim.
_COHERENCE_EXTERNAL_URL_REQUIRED = "url is required for http transport"
_COHERENCE_STDIO_COMMAND_REQUIRED = "command is required for stdio transport"
_COHERENCE_OPENAPI_URL_REQUIRED = "openapi_url is required for openapi source_type"


def validate_server_spec_coherence(
    *,
    transport: str | None = None,
    url: str | None = None,
    command: str | None = None,
    source_type: str = "external",
    openapi_url: str | None = None,
) -> None:
    """Validate the per-source-type coherence of a merged server specification.

    Used verbatim by:
    * :class:`CreateMcpServerRequest` (full create/validate bodies);
    * the ``composable_mcp_servers_update`` tool, which merges the provided
      fields onto the masked existing projection — the visibility-relevant
      fields (``transport`` / ``url`` / ``command`` / ``source_type`` /
      ``openapi_url``) are always present in that projection, so an invalid
      MERGED update is rejected LOCALLY with an actionable, non-sensitive
      message instead of forwarding the registry's opaque 422 (BUG-002).

    Raises:
        ValueError: With the stable coherence message naming the offending
            requirement.
    """
    effective_transport = (transport or "http").lower()
    effective_source_type = (source_type or "external").lower()
    if effective_transport == "http" and not url:
        raise ValueError(_COHERENCE_EXTERNAL_URL_REQUIRED)
    if effective_transport == "stdio" and not command:
        raise ValueError(_COHERENCE_STDIO_COMMAND_REQUIRED)
    if effective_source_type == "openapi" and not openapi_url:
        raise ValueError(_COHERENCE_OPENAPI_URL_REQUIRED)


class CreateMcpServerRequest(BaseModel):
    """Register an external MCP server or generate one from OpenAPI/Swagger."""

    name: str = Field(..., min_length=1, max_length=200)
    transport: Literal["http", "stdio"] = "http"
    url: str | None = Field(default=None, max_length=2000)
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    auth_token: str | None = None
    source_type: Literal["external", "openapi"] = "external"
    openapi_url: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        validate_server_spec_coherence(
            transport=self.transport,
            url=self.url,
            command=self.command,
            source_type=self.source_type,
            openapi_url=self.openapi_url,
        )
        return self


class UpdateMcpServerRequest(BaseModel):
    """Partial update of a registered MCP server (fields optional)."""

    transport: Literal["http", "stdio"] | None = None
    url: str | None = Field(default=None, max_length=2000)
    command: str | None = None
    args: list[str] | None = None
    headers: dict[str, str] | None = None
    env: dict[str, str] | None = None
    auth_token: str | None = None
    source_type: Literal["external", "openapi"] | None = None
    openapi_url: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _at_least_one(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("provide at least one editable field")
        return self


def update_payload(request: UpdateMcpServerRequest) -> dict:
    """Serialize a partial update with only the explicitly provided fields."""
    return request.model_dump(mode="json", exclude_unset=True)
