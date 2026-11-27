import uuid
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.domain.entities.mcp_server_config import McpServerConfig


class BackendType(StrEnum):
    STORE = "store"


class BackendConfig(BaseModel):
    type: BackendType = BackendType.STORE
    checkpoint_backend: Literal["memory", "postgres"] = "postgres"


class InterruptRule(BaseModel):
    allowed_decisions: list[Literal["approve", "edit", "reject"]] = Field(default=["approve", "edit", "reject"])


class HITLConfig(BaseModel):
    rules: dict[str, bool | InterruptRule] = Field(default_factory=dict)


class SubAgentConfig(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    agent_ref: str | None = None
    instructions: str | None = None
    model: str | None = None
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)
    response_format: dict[str, Any] | None = None


class PromptRef(BaseModel):
    """Reference to a prompt registry entry (spec §5.5 AGT-001).

    ``version`` omitted (or ``None``) means *Latest*: the agent resolves the
    prompt's most recent immutable version on every rebuild. A pinned version
    resolves exactly that immutable version, never Latest.
    """

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    version: int | None = Field(default=None, gt=0)


class AgentConfig(BaseModel):
    """Schema principal de configuration d'un Deep Agent via YAML.

    Prompt source (spec §5.5 AGT-002): ``prompt_ref`` is mutually exclusive
    with ``system_prompt`` and ``system_prompt_file``. Exclusivity is enforced
    on field PRESENCE (``is not None``), not truthiness, so an empty inline
    value cannot bypass it.

    Model source (spec §5.5 AGT-003): ``model_ref`` (a catalog model UUID)
    takes precedence over the legacy ``model`` string for model metadata
    selection; the legacy string may remain in the YAML for schema
    compatibility and is ignored when a valid ``model_ref`` exists.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    model: str = Field(default="claude-sonnet-4-5-20250929")
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    prompt_ref: PromptRef | None = None
    model_ref: uuid.UUID | None = None
    tools: list[str] = Field(default_factory=list)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    hitl: HITLConfig = Field(default_factory=HITLConfig)
    memory: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    subagents: list[SubAgentConfig] = Field(default_factory=list)
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)
    # Registry identities of SELECTED user MCP servers (spec agent-resource-
    # import REQ-MCP-003): persisted as additive references (e.g.
    # ``mcp_refs: ["server-name"]``), resolved through the trusted registry
    # execution-config path at runner build time. Connection parameters and
    # secrets stay in the registry; legacy inline ``mcp_servers`` remain only
    # for agents created before the resource-only rule (REQ-LEGACY-001).
    mcp_refs: list[str] = Field(default_factory=list)
    response_format: dict[str, Any] | None = None
    debug: bool = False
    # Overlay-computed marker (REQ-033): ``True`` only when this configuration
    # is the virtual built-in Composable AI Assistant resolved by the built-in
    # overlay. Additive-only: never authored in agent YAML and never persisted
    # (no ``is_builtin`` database column — that column was removed by
    # migration 003 and must NOT be reintroduced).
    is_builtin: bool = False
    # Overlay-computed per-user marker: ``True`` only when this effective
    # built-in definition is the owning user's customization snapshot.
    is_customized: bool = False
    # Server-declared capability flags (built-in assistant ONLY — never
    # authored in YAML, never persisted): the DYNAMIC toolset (built-in
    # mcp-raganything mounts) lives server-side and is FORCED out of
    # ``mcp_servers`` by the invariant projection, so the UI cannot derive
    # document attachments from the config alone. ``files_mcp=True`` lets the
    # chat UI open its document picker (the backend upload policy stays the
    # single authority).
    capabilities: dict[str, bool] | None = None

    @model_validator(mode="after")
    def check_prompt_exclusivity(self) -> Self:
        """Enforce mutually exclusive prompt sources (presence-based)."""
        present = [
            field
            for field, value in (
                ("prompt_ref", self.prompt_ref),
                ("system_prompt", self.system_prompt),
                ("system_prompt_file", self.system_prompt_file),
            )
            if value is not None
        ]
        if len(present) > 1:
            raise ValueError(f"{' and '.join(present)} are mutually exclusive")
        return self
