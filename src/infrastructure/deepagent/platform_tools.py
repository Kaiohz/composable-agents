"""Agent-facing platform toolset (REQ-028/REQ-036).

Builds LangChain-compatible tools over the SAME application services the
REST routes and the MCP control plane already use — one shared behavior, no
logic duplication. Served in two shapes:

* the built-in Composable AI Assistant: platform tools INCLUDING
  ``composable_agents_run`` (self-invocation is excluded by the target guard,
  REQ-030) plus the dynamic MCP capabilities resolved by the agent registry;
* custom agents whose YAML opts in via ``tools: ["mcp"]`` (the declarative
  platform-tools selector): the platform tools INCLUDING one-shot execution
  so agent-to-agent calls ride the same invocation chain/cycle/depth guard
  (REQ-029).

All tools execute inside the caller's RLS context (the runner's task inherits
the contextvars set by authentication), so every capability is scoped to the
originating user exactly like REST and MCP. Domain errors deliberately
propagate as exceptions: the runner's ToolNode converts them into error
ToolMessages the LLM can react to (and the one-shot use case scans for the
invocation rejection marker — see ``run_agent_one_shot``).
"""

import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.application.requests.model import CreateModelRequest, UpdateModelRequest
from src.domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)

#: YAML tools selector that declares "give this agent the platform toolset".
PLATFORM_TOOLS_SELECTOR = "mcp"


def _json(payload: Any) -> str:
    """Render a JSON-safe tool result string (LLM-facing)."""
    return json.dumps(payload, default=str, ensure_ascii=False)


def _data(obj: Any) -> Any:
    """Project a Pydantic object into a JSON-safe payload."""
    if obj is None:
        return None
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    return obj


# ---------------------------------------------------------------------------
# Argument schemas (typed, JSON-schema generating)
# ---------------------------------------------------------------------------
class AgentNameArgs(BaseModel):
    """Single agent-name argument."""

    agent_name: str = Field(..., min_length=1, description="Exact agent name")


class AgentYamlArgs(BaseModel):
    """Create/replace an agent from its name plus YAML text (REQ-012)."""

    agent_name: str = Field(..., min_length=1)
    yaml: str = Field(..., description="Complete agent YAML document")


class OneShotArgs(BaseModel):
    """One-shot execution of an agent."""

    agent_name: str = Field(..., min_length=1)
    message: str = Field(..., description="Non-empty task message")


class RegistryServerNameArgs(BaseModel):
    name: str = Field(..., min_length=1)


class RegistryServerWriteArgs(BaseModel):
    """External MCP server or OpenAPI-generated server definition.

    ``auth_token`` / ``headers`` / ``env`` are WRITE-ONLY inputs (REQ-019):
    they are forwarded to the trusted registry and never echoed back.
    """

    name: str = Field(..., min_length=1)
    transport: str = Field(default="http", description="http or stdio")
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    auth_token: str | None = None
    source_type: str = Field(default="external", description="external or openapi")
    openapi_url: str | None = None


class RegistryServerUpdateArgs(RegistryServerWriteArgs):
    """Partial registry update — name targets the entry, other fields optional."""


class SkillNameArgs(BaseModel):
    skill_name: str = Field(..., min_length=1)


class SkillContentArgs(BaseModel):
    skill_name: str = Field(..., min_length=1)
    content: str = Field(..., description="Full SKILL.md markdown content")


class MemoryNameArgs(BaseModel):
    filename: str = Field(..., min_length=1)


class MemoryContentArgs(BaseModel):
    filename: str = Field(..., min_length=1)
    content: str = Field(..., description="Full memory file content")


class PromptIdArgs(BaseModel):
    prompt_id: str = Field(..., min_length=1)


class PromptCreateArgs(BaseModel):
    name: str = Field(..., min_length=1)
    description: str | None = None
    content: str = Field(default="", description="Prompt content (may be empty)")


class PromptPatchArgs(BaseModel):
    prompt_id: str = Field(..., min_length=1)
    name: str | None = None
    description: str | None = None


class PromptVersionGetArgs(BaseModel):
    prompt_id: str = Field(..., min_length=1)
    version: int = Field(..., ge=1)


class PromptVersionCreateArgs(BaseModel):
    prompt_id: str = Field(..., min_length=1)
    content: str = Field(default="", description="New immutable version content (may be empty)")


class ModelIdArgs(BaseModel):
    model_id: str = Field(..., min_length=1)


class ModelCreateArgs(CreateModelRequest):
    """Agent-facing create args — inherits the REST contract (MODEL-007).

    Deriving from the application DTO (the single source of truth shared with
    REST and the MCP control plane) makes drift structurally impossible:
    ``extra="forbid"`` rejects removed/unknown fields and both values are
    trimmed and non-empty after trim.

    The tool coroutine takes the SCHEMA FIELDS as flat arguments: the runner's
    tool invocation expands the validated ``args_schema`` into kwargs (schemas
    are the validation surface, never the call surface).
    """


class ModelPatchArgs(UpdateModelRequest):
    """Agent-facing patch args — the REST contract plus the ``model_id`` target.

    Inheriting :class:`UpdateModelRequest` keeps the exact PATCH semantics:
    an unknown/removed field is rejected, provided values are non-null and
    non-empty after trim, and an empty patch (only ``model_id``) is rejected.
    The tool coroutine mirrors these fields as flat arguments (see
    :class:`ModelCreateArgs`).
    """

    model_id: str = Field(..., min_length=1)


class CronIdArgs(BaseModel):
    cron_id: str = Field(..., min_length=1)


class CronCreateArgs(BaseModel):
    name: str = Field(..., min_length=1)
    agent_name: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    cron_expression: str = Field(..., min_length=1, description="Five-field cron expression without seconds")
    timezone: str = "UTC"
    enabled: bool = True


class CronPatchArgs(BaseModel):
    cron_id: str = Field(..., min_length=1)
    name: str | None = None
    agent_name: str | None = None
    message: str | None = None
    cron_expression: str | None = None
    timezone: str | None = None
    enabled: bool | None = None


class CronRunsListArgs(BaseModel):
    cron_id: str = Field(..., min_length=1)
    limit: int = Field(default=50, ge=1, le=200)


class ThreadIdArgs(BaseModel):
    thread_id: str = Field(..., min_length=1)


class TracesListArgs(BaseModel):
    agent_name: str | None = None
    limit: int = Field(default=50, ge=1, le=100)
    cursor: str | None = None


class TraceUsageArgs(BaseModel):
    agent_name: str | None = None
    from_ts: str | None = Field(default=None, description="Inclusive ISO lower bound (naive means UTC)")
    to_ts: str | None = Field(default=None, description="Inclusive ISO upper bound (naive means UTC)")


class TraceTopRunsArgs(TraceUsageArgs):
    limit: int = Field(default=10, ge=1, le=100)


class AgentPlatformToolsetBuilder:
    """Assemble the agent-facing platform tools over the shared use cases.

    Each tool resolves its use cases lazily through the composition root at
    CALL time, so a cached runner always calls through the current wiring and
    repositories instead of capturing stale instances.
    """

    def build(self, *, include_run: bool) -> list[StructuredTool]:
        """Return the LangChain platform toolset for an agent.

        Args:
            include_run: When ``True`` the ``composable_agents_run`` one-shot
                execution tool is part of the toolset (agent-to-agent calls,
                guarded by the invocation chain and the built-in target guard).
                Self-invocation remains impossible even when the tool is present
                (REQ-030).
        """
        tools: list[StructuredTool] = [
            *self._agents_family(),
            *self._registry_family(),
            *self._skills_family(),
            *self._memories_family(),
            *self._prompts_family(),
            *self._models_family(),
            *self._crons_family(),
            *self._threads_family(),
            *self._traces_family(),
            self._users_tool(),
        ]
        if include_run:
            tools.append(self._run_tool())
        logger.info(LogMessage.AGENT_TOOLS_TOTAL, "platform-toolset", len(tools))
        return tools

    # ------------------------------------------------------------------
    @staticmethod
    def _tool(coroutine, schema: Any = None) -> StructuredTool:
        """Create one async StructuredTool from a coroutine + typed args schema.

        ``schema`` is a LangChain ``args_schema`` (a pydantic BaseModel class
        or ``None`` for zero-argument tools).
        """
        return StructuredTool(
            name=coroutine.__name__,
            description=(coroutine.__doc__ or "").strip(),
            args_schema=schema,
            coroutine=coroutine,
        )

    # ------------------------------------------------------------------
    def _agents_family(self) -> list[StructuredTool]:
        deps_lazy = self._deps

        async def composable_agents_list() -> str:
            """List the current user's agent metadata (built-in assistant included).

            REQ-011: the projection mirrors REST/MCP exactly — the built-in
            assistant is overlaid first with its effective per-user metadata.
            """
            deps = deps_lazy()
            agents = await deps.get_list_agent_configs_use_case().execute()
            effective_builtin = await deps.get_effective_builtin_agent_service().get_effective_metadata()
            return _json([_data(effective_builtin), *(_data(agent) for agent in agents)])

        async def composable_agents_get(agent_name: str) -> str:
            """Get one agent's validated configuration by exact name."""
            from src.application.services.builtin_agent import is_builtin_agent_name

            deps = deps_lazy()
            if is_builtin_agent_name(agent_name):
                effective = await deps.get_effective_builtin_agent_service().get_effective_config()
                return _json(_data(effective))
            config = await deps.get_get_agent_config_use_case().execute(name=agent_name)
            return _json(_data(config))

        async def composable_agents_create(agent_name: str, yaml: str) -> str:
            """Create an agent from its name plus complete YAML text."""
            from src.application.services.builtin_agent import ensure_custom_agent_name
            from src.application.services.route_validation import read_yaml_bytes, validate_agent_name

            deps = deps_lazy()
            validate_agent_name(agent_name)
            ensure_custom_agent_name(agent_name)
            yaml_content = read_yaml_bytes(yaml.encode("utf-8"))
            config = await deps.get_create_agent_config_use_case().execute(name=agent_name, yaml_content=yaml_content)
            return _json(_data(config))

        async def composable_agents_update(agent_name: str, yaml: str) -> str:
            """Replace a custom agent's YAML, or the user's built-in customization."""
            from src.application.services.builtin_agent import is_builtin_agent_name
            from src.application.services.route_validation import read_yaml_bytes, validate_agent_name

            deps = deps_lazy()
            validate_agent_name(agent_name)
            yaml_content = read_yaml_bytes(yaml.encode("utf-8"))
            if is_builtin_agent_name(agent_name):
                config = await deps.get_effective_builtin_agent_service().update(agent_name, yaml_content)
                return _json(_data(config))
            config = await deps.get_update_agent_config_use_case().execute(name=agent_name, yaml_content=yaml_content)
            return _json(_data(config))

        async def composable_agents_reset_customization(agent_name: str) -> str:
            """Reset the user's built-in assistant customization to its default."""
            from src.application.services.builtin_agent import is_builtin_agent_name
            from src.application.services.route_validation import validate_agent_name
            from src.domain.errors.agent import AgentNotFoundError
            from src.domain.errors.messages import ErrorMessage

            deps = deps_lazy()
            validate_agent_name(agent_name)
            if not is_builtin_agent_name(agent_name):
                raise AgentNotFoundError(ErrorMessage.AGENT_NOT_FOUND.format(name=agent_name))
            await deps.get_effective_builtin_agent_service().reset()
            return _json({"reset": agent_name})

        async def composable_agents_delete(agent_name: str) -> str:
            """Delete an existing custom agent (the built-in assistant is protected)."""
            from src.application.services.builtin_agent import ensure_not_builtin

            deps = deps_lazy()
            ensure_not_builtin(agent_name)
            await deps.get_delete_agent_config_use_case().execute(name=agent_name)
            return _json({"deleted": agent_name})

        return [
            self._tool(composable_agents_list),
            self._tool(composable_agents_get, AgentNameArgs),
            self._tool(composable_agents_create, AgentYamlArgs),
            self._tool(composable_agents_update, AgentYamlArgs),
            self._tool(composable_agents_reset_customization, AgentNameArgs),
            self._tool(composable_agents_delete, AgentNameArgs),
        ]

    # ------------------------------------------------------------------
    def _registry_family(self) -> list[StructuredTool]:
        deps_lazy = self._deps

        def _write(args: RegistryServerWriteArgs):
            from src.domain.entities.registered_mcp_server import McpServerWrite

            return McpServerWrite(
                name=args.name,
                transport="http" if args.transport == "http" else "stdio",
                url=args.url,
                command=args.command,
                args=list(args.args or []),
                headers=dict(args.headers or {}),
                env=dict(args.env or {}),
                auth_token=args.auth_token,
                source_type="openapi" if args.source_type == "openapi" else "external",
                openapi_url=args.openapi_url,
            )

        def _update_changes(args: RegistryServerUpdateArgs) -> dict[str, Any]:
            """Build the partial update payload from the provided fields only."""
            changes: dict[str, Any] = {}
            for key in (
                "transport",
                "url",
                "command",
                "args",
                "headers",
                "env",
                "auth_token",
                "source_type",
                "openapi_url",
            ):
                value = getattr(args, key)
                if value is not None:
                    changes[key] = value
            return changes

        async def composable_mcp_servers_list() -> str:
            """List the user's registered MCP servers (all secrets masked)."""
            servers = await deps_lazy().get_mcp_registry_client().list_servers()
            return _json([_data(server) for server in servers])

        async def composable_mcp_servers_get(name: str) -> str:
            """Get one registered MCP server (all secrets masked)."""
            server = await deps_lazy().get_mcp_registry_client().get_server(name)
            return _json(_data(server))

        async def composable_mcp_servers_validate(args: RegistryServerWriteArgs) -> str:
            """Validate a candidate MCP server WITHOUT registering it."""
            report = await deps_lazy().get_mcp_registry_client().validate_server(_write(args))
            return _json(_data(report))

        async def composable_mcp_servers_create(args: RegistryServerWriteArgs) -> str:
            """Register an external MCP server or generate one from OpenAPI/Swagger."""
            server = await deps_lazy().get_mcp_registry_client().create_server(_write(args))
            return _json(_data(server))

        async def composable_mcp_servers_update(args: RegistryServerUpdateArgs) -> str:
            """Update a registered MCP server (masked response)."""
            server = await deps_lazy().get_mcp_registry_client().update_server(args.name, _update_changes(args))
            return _json(_data(server))

        async def composable_mcp_servers_delete(name: str) -> str:
            """Delete a registered MCP server (destructive, requires care)."""
            await deps_lazy().get_mcp_registry_client().delete_server(name)
            return _json({"deleted": name})

        return [
            self._tool(composable_mcp_servers_list),
            self._tool(composable_mcp_servers_get, RegistryServerNameArgs),
            self._tool(composable_mcp_servers_delete, RegistryServerNameArgs),
            self._tool(composable_mcp_servers_validate, RegistryServerWriteArgs),
            self._tool(composable_mcp_servers_create, RegistryServerWriteArgs),
            self._tool(composable_mcp_servers_update, RegistryServerUpdateArgs),
        ]

    # ------------------------------------------------------------------
    def _skills_family(self) -> list[StructuredTool]:
        deps_lazy = self._deps

        async def composable_skills_list() -> str:
            """List the current user's global skill directories."""
            from src.application.services.route_validation import skills_prefix

            files = await deps_lazy().get_list_store_files_use_case().execute(prefix=skills_prefix())
            return _json({"skills": files})

        async def composable_skills_get(skill_name: str) -> str:
            """Read a skill's path and its SKILL.md content."""
            from src.application.services.route_validation import skill_store_path

            path = skill_store_path(skill_name)
            content = await deps_lazy().get_get_store_file_use_case().execute(path=path)
            if content is None:
                from src.domain.errors.store_file import StoreFileNotFoundError

                raise StoreFileNotFoundError(f"File not found: {path}")
            return _json({"path": path, "content": content})

        async def composable_skills_upsert(skill_name: str, content: str) -> str:
            """Create or replace a global skill's SKILL.md content."""
            from src.application.services.route_validation import skill_store_path

            path = skill_store_path(skill_name)
            await deps_lazy().get_put_store_file_use_case().execute(path=path, content=content)
            return _json({"path": path, "content": content})

        async def composable_skills_delete(skill_name: str) -> str:
            """Delete a global skill directory (destructive, requires care)."""
            from src.application.services.route_validation import skill_directory_path

            directory = skill_directory_path(skill_name)
            deps = deps_lazy()
            files = await deps.get_list_store_files_use_case().execute(prefix=directory)
            for path in files:
                await deps.get_delete_store_file_use_case().execute(path=path)
            return _json({"deleted": skill_name, "files": len(files)})

        async def composable_skills_usage(skill_name: str) -> str:
            """List the agents that reference a named global skill."""
            from src.application.services.route_validation import list_skill_usage_agent_names, skill_directory_path

            skill_directory_path(skill_name)  # validates the plain name (traversal-safe)
            agents = await list_skill_usage_agent_names(
                lambda prefix: deps_lazy().get_list_store_files_use_case().execute(prefix=prefix),
                skill_name,
            )
            return _json({"agents": agents})

        return [
            self._tool(composable_skills_list),
            self._tool(composable_skills_get, SkillNameArgs),
            self._tool(composable_skills_upsert, SkillContentArgs),
            self._tool(composable_skills_delete, SkillNameArgs),
            self._tool(composable_skills_usage, SkillNameArgs),
        ]

    # ------------------------------------------------------------------
    def _memories_family(self) -> list[StructuredTool]:
        deps_lazy = self._deps

        async def composable_memories_list() -> str:
            """List the current user's global memory files."""
            from src.application.services.route_validation import memories_prefix

            files = await deps_lazy().get_list_store_files_use_case().execute(prefix=memories_prefix())
            return _json({"memories": files})

        async def composable_memories_get(filename: str) -> str:
            """Read one global memory file."""
            from src.application.services.route_validation import memory_store_path

            path = memory_store_path(filename)
            content = await deps_lazy().get_get_store_file_use_case().execute(path=path)
            if content is None:
                from src.domain.errors.store_file import StoreFileNotFoundError

                raise StoreFileNotFoundError(f"File not found: {path}")
            return _json({"path": path, "content": content})

        async def composable_memories_upsert(filename: str, content: str) -> str:
            """Create or replace a global memory file."""
            from src.application.services.route_validation import memory_store_path

            path = memory_store_path(filename)
            await deps_lazy().get_put_store_file_use_case().execute(path=path, content=content)
            return _json({"path": path, "content": content})

        async def composable_memories_delete(filename: str) -> str:
            """Delete one global memory file (destructive, requires care)."""
            from src.application.services.route_validation import memory_store_path

            path = memory_store_path(filename)
            await deps_lazy().get_delete_store_file_use_case().execute(path=path)
            return _json({"deleted": path})

        return [
            self._tool(composable_memories_list),
            self._tool(composable_memories_get, MemoryNameArgs),
            self._tool(composable_memories_upsert, MemoryContentArgs),
            self._tool(composable_memories_delete, MemoryNameArgs),
        ]

    # ------------------------------------------------------------------
    def _prompts_family(self) -> list[StructuredTool]:
        deps_lazy = self._deps

        async def composable_prompts_list() -> str:
            """List prompt metadata."""
            prompts = await deps_lazy().get_list_prompts_use_case().execute()
            return _json([_data(prompt) for prompt in prompts])

        async def composable_prompts_get(prompt_id: str) -> str:
            """Get one prompt's metadata."""
            prompt = await deps_lazy().get_get_prompt_use_case().execute(prompt_id)
            return _json(_data(prompt))

        async def composable_prompts_create(args: PromptCreateArgs) -> str:
            """Create prompt metadata and its immutable version 1."""
            prompt = (
                await deps_lazy()
                .get_create_prompt_use_case()
                .execute(name=args.name, description=args.description, content=args.content)
            )
            return _json(_data(prompt))

        async def composable_prompts_update(args: PromptPatchArgs) -> str:
            """Update prompt name and/or description (no version is created)."""
            changes: dict[str, Any] = {}
            if args.name is not None:
                changes["name"] = args.name
            if args.description is not None:
                changes["description"] = args.description
            updated = await deps_lazy().get_update_prompt_use_case().execute(args.prompt_id, changes)
            return _json(_data(updated))

        async def composable_prompts_delete(prompt_id: str) -> str:
            """Delete an unreferenced prompt (destructive, requires care)."""
            await deps_lazy().get_delete_prompt_use_case().execute(prompt_id)
            return _json({"deleted": prompt_id})

        async def composable_prompt_versions_list(prompt_id: str) -> str:
            """List immutable prompt-version summaries."""
            versions = await deps_lazy().get_list_prompt_versions_use_case().execute(prompt_id)
            return _json([_data(version) for version in versions])

        async def composable_prompt_versions_get(args: PromptVersionGetArgs) -> str:
            """Read one exact immutable prompt version."""
            row = await deps_lazy().get_get_prompt_version_use_case().execute(args.prompt_id, args.version)
            return _json(_data(row))

        async def composable_prompt_versions_create(args: PromptVersionCreateArgs) -> str:
            """Atomically append the next immutable version."""
            created = await deps_lazy().get_add_prompt_version_use_case().execute(args.prompt_id, args.content)
            return _json(_data(created))

        return [
            self._tool(composable_prompts_list),
            self._tool(composable_prompts_get, PromptIdArgs),
            self._tool(composable_prompts_create, PromptCreateArgs),
            self._tool(composable_prompts_update, PromptPatchArgs),
            self._tool(composable_prompts_delete, PromptIdArgs),
            self._tool(composable_prompt_versions_list, PromptIdArgs),
            self._tool(composable_prompt_versions_get, PromptVersionGetArgs),
            self._tool(composable_prompt_versions_create, PromptVersionCreateArgs),
        ]

    # ------------------------------------------------------------------
    def _models_family(self) -> list[StructuredTool]:
        deps_lazy = self._deps

        async def composable_models_list() -> str:
            """List identity-only model definitions."""
            models = await deps_lazy().get_list_models_use_case().execute()
            return _json([_data(model) for model in models])

        async def composable_models_get(model_id: str) -> str:
            """Get one model definition."""
            model = await deps_lazy().get_get_model_use_case().execute(model_id)
            return _json(_data(model))

        async def composable_models_create(name: str, provider_model: str) -> str:
            """Create a model identity (name + provider_model only)."""
            model = await deps_lazy().get_create_model_use_case().execute(name=name, provider_model=provider_model)
            return _json(_data(model))

        async def composable_models_update(
            model_id: str,
            name: str | None = None,
            provider_model: str | None = None,
        ) -> str:
            """Update model identity fields (REST-validated presence semantics)."""
            # ``args_schema`` (ModelPatchArgs) already applied the REST rules:
            # unknown/removed fields rejected, explicit nulls rejected and
            # provided values trimmed — an argument of ``None`` here therefore
            # always means "field absent from the call" (presence semantics).
            changes: dict[str, Any] = {}
            if name is not None:
                changes["name"] = name
            if provider_model is not None:
                changes["provider_model"] = provider_model
            model = await deps_lazy().get_update_model_use_case().execute(model_id, changes)
            return _json(_data(model))

        async def composable_models_delete(model_id: str) -> str:
            """Delete an unreferenced model (destructive, requires care)."""
            await deps_lazy().get_delete_model_use_case().execute(model_id)
            return _json({"deleted": model_id})

        return [
            self._tool(composable_models_list),
            self._tool(composable_models_get, ModelIdArgs),
            self._tool(composable_models_create, ModelCreateArgs),
            self._tool(composable_models_update, ModelPatchArgs),
            self._tool(composable_models_delete, ModelIdArgs),
        ]

    # ------------------------------------------------------------------
    def _crons_family(self) -> list[StructuredTool]:
        deps_lazy = self._deps

        async def composable_crons_list() -> str:
            """List the user's cron definitions and scheduling state."""
            crons = await deps_lazy().get_list_crons_use_case().execute()
            return _json([_data(cron) for cron in crons])

        async def composable_crons_get(cron_id: str) -> str:
            """Get one cron definition."""
            cron = await deps_lazy().get_get_cron_use_case().execute(cron_id)
            return _json(_data(cron))

        async def composable_crons_create(args: CronCreateArgs) -> str:
            """Create and validate a cron definition."""
            cron = (
                await deps_lazy()
                .get_create_cron_use_case()
                .execute(
                    name=args.name,
                    agent_name=args.agent_name,
                    message=args.message,
                    cron_expression=args.cron_expression,
                    timezone=args.timezone,
                    enabled=args.enabled,
                )
            )
            return _json(_data(cron))

        async def composable_crons_update(args: CronPatchArgs) -> str:
            """Update an owned cron (frontier recalculates accordingly)."""
            changes: dict[str, Any] = {}
            for key in ("name", "agent_name", "message", "cron_expression", "timezone"):
                value = getattr(args, key)
                if value is not None:
                    changes[key] = value
            if args.enabled is not None:
                changes["enabled"] = args.enabled
            cron = await deps_lazy().get_update_cron_use_case().execute(args.cron_id, changes)
            return _json(_data(cron))

        async def composable_crons_delete(cron_id: str) -> str:
            """Delete an owned cron (destructive, requires care)."""
            await deps_lazy().get_delete_cron_use_case().execute(cron_id)
            return _json({"deleted": cron_id})

        async def composable_crons_run_now(cron_id: str) -> str:
            """Trigger Run now for a cron (works while disabled)."""
            run = await deps_lazy().get_run_cron_now_use_case().execute(cron_id)
            return _json(_data(run))

        async def composable_cron_runs_list(args: CronRunsListArgs) -> str:
            """List bounded cron run history (newest first)."""
            runs = await deps_lazy().get_list_cron_runs_use_case().execute(args.cron_id, args.limit)
            return _json([_data(run) for run in runs])

        return [
            self._tool(composable_crons_list),
            self._tool(composable_crons_get, CronIdArgs),
            self._tool(composable_crons_create, CronCreateArgs),
            self._tool(composable_crons_update, CronPatchArgs),
            self._tool(composable_crons_delete, CronIdArgs),
            self._tool(composable_crons_run_now, CronIdArgs),
            self._tool(composable_cron_runs_list, CronRunsListArgs),
        ]

    # ------------------------------------------------------------------
    def _threads_family(self) -> list[StructuredTool]:
        deps_lazy = self._deps

        async def composable_threads_list() -> str:
            """List the current user's threads with their immutable sources."""
            from src.application.responses.mcp_threads import ThreadSummary

            threads = await deps_lazy().get_list_threads_use_case().execute()
            return _json({"threads": [ThreadSummary.of(thread).model_dump(mode="json") for thread in threads]})

        async def composable_threads_get(thread_id: str) -> str:
            """Get one owned thread."""
            from src.application.responses.mcp_threads import ThreadSummary

            thread = await deps_lazy().get_get_thread_use_case().execute(thread_id)
            return _json(_data(ThreadSummary.of(thread)))

        async def composable_threads_history(thread_id: str) -> str:
            """Get a thread's turn-grouped history."""
            history = await deps_lazy().get_get_thread_history_use_case().execute(thread_id)
            return _json(_data(history))

        async def composable_threads_trace(thread_id: str) -> str:
            """Get a thread's flat trace (ordered by timestamp)."""
            deps = deps_lazy()
            await deps.get_get_thread_use_case().execute(thread_id)
            events = await deps.get_trace_event_repository().list_by_thread(thread_id)
            return _json({"events": [_data(event) for event in events]})

        async def composable_threads_delete(thread_id: str) -> str:
            """Delete an owned thread and its trace history (destructive, requires care)."""
            await deps_lazy().get_delete_thread_use_case().execute(thread_id)
            return _json({"deleted": thread_id})

        return [
            self._tool(composable_threads_list),
            self._tool(composable_threads_get, ThreadIdArgs),
            self._tool(composable_threads_history, ThreadIdArgs),
            self._tool(composable_threads_trace, ThreadIdArgs),
            self._tool(composable_threads_delete, ThreadIdArgs),
        ]

    # ------------------------------------------------------------------
    def _traces_family(self) -> list[StructuredTool]:
        deps_lazy = self._deps

        async def composable_traces_list(args: TracesListArgs) -> str:
            """List trace summaries with the existing filters and pagination."""
            from src.domain.errors.config import ConfigError
            from src.domain.errors.messages import ErrorMessage

            page = (
                await deps_lazy()
                .get_list_traces_use_case()
                .execute(agent_name=args.agent_name, limit=args.limit, cursor=args.cursor)
            )
            if page is None:  # pragma: no cover — use case always returns a page
                raise ConfigError(ErrorMessage.TRACE_CURSOR_INVALID)
            return _json({"items": [_data(item) for item in page.items], "next_cursor": page.next_cursor})

        async def composable_traces_usage(args: TraceUsageArgs) -> str:
            """Get aggregated LLM usage (tokens, cost, runs, latency)."""
            from src.application.services.route_validation import utc_bound

            summary = (
                await deps_lazy()
                .get_trace_usage_use_case()
                .execute(agent_name=args.agent_name, from_ts=utc_bound(args.from_ts), to_ts=utc_bound(args.to_ts))
            )
            return _json(_data(summary))

        async def composable_traces_top_runs(args: TraceTopRunsArgs) -> str:
            """Get the bounded top-runs LLM usage report."""
            from src.application.services.route_validation import utc_bound

            runs = (
                await deps_lazy()
                .get_top_usage_runs_use_case()
                .execute(
                    limit=args.limit,
                    agent_name=args.agent_name,
                    from_ts=utc_bound(args.from_ts),
                    to_ts=utc_bound(args.to_ts),
                )
            )
            return _json([_data(run) for run in runs])

        return [
            self._tool(composable_traces_list, TracesListArgs),
            self._tool(composable_traces_usage, TraceUsageArgs),
            self._tool(composable_traces_top_runs, TraceTopRunsArgs),
        ]

    # ------------------------------------------------------------------
    def _users_tool(self) -> StructuredTool:
        deps_lazy = self._deps

        async def composable_users_me() -> (
            str
        ):  # NOSONAR python:S7503 — the StructuredTool contract wants a coroutine; the use case it delegates to is a sync, non-blocking in-memory query
            """Return the authenticated user's non-sensitive profile."""
            auth_context = deps_lazy().get_current_auth_context()
            profile = deps_lazy().get_get_current_user_use_case().execute(auth_context)
            return _json(_data(profile))

        return self._tool(composable_users_me)

    # ------------------------------------------------------------------
    # one-shot execution platform tool (agent-to-agent, chain-guarded)
    # ------------------------------------------------------------------
    def _run_tool(self) -> StructuredTool:
        deps_lazy = self._deps

        async def composable_agents_run(agent_name: str, message: str) -> str:
            """Run another visible agent as a one-shot task in a new mcp thread.

            Returns the one-shot result (thread_id, agent_name, source and
            the final message). Cycles and invocation chains deeper than four
            nested calls are rejected with a safe error. The built-in
            assistant CANNOT invoke itself (REQ-030): the target check below
            refuses it independently of the invocation chain.
            """
            from src.application.services.builtin_agent import is_builtin_agent_name
            from src.application.use_cases.run_agent_one_shot import get_current_invocation_chain
            from src.domain.errors.agent import AgentInvocationRejectedError
            from src.domain.errors.messages import ErrorMessage

            chain = get_current_invocation_chain()
            # REQ-030: the built-in assistant must never invoke ITSELF — any
            # OTHER target is allowed and rides the same cycle/depth guard.
            if is_builtin_agent_name(agent_name) and agent_name in chain:
                raise AgentInvocationRejectedError(
                    ErrorMessage.AGENT_INVOCATION_REJECTED.format(reason="the built-in assistant cannot invoke itself")
                )
            result = (
                await deps_lazy()
                .get_run_agent_one_shot_use_case()
                .execute(
                    agent_name,
                    message,
                    invocation_chain=chain,
                )
            )
            return _json(
                {
                    "thread_id": result.thread_id,
                    "agent_name": result.agent_name,
                    "source": result.source.value,
                    "message": _data(result.message),
                }
            )

        return self._tool(composable_agents_run, OneShotArgs)

    @staticmethod
    def _deps():
        """Resolve the composition root lazily (call time, no import cycle)."""
        from src import dependencies

        return dependencies
