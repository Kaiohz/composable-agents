"""Typed Composable MCP tools (REQ-008..REQ-020).

One registration module for the whole tool surface. Every tool:

* has a stable name, a concise description and a typed input schema;
* delegates to the SAME application use cases / services as REST (REQ-006) —
  no repository access, scheduling, reference validation, trace persistence
  or runner logic here;
* shares the route-boundary validation (agent-name, YAML size/UTF-8,
  store-path normalization, UTC normalization — REQ-007) via
  ``application/services/route_validation``;
* returns JSON-safe structured outputs reusing the REST/domain contracts;
* maps failures through ``safe_mcp_call`` (REQ-009).

NEVER exposed (REQ-020): LLM settings, API-key administration, credential
revelation (registry ``/reveal``), authentication administration and
cross-user administration. The built-in assistant toolset additionally
excludes self-invocation (REQ-030) — see the agent registry.
"""

import logging
from typing import Annotated, Literal

from src.application.mcp.dto import (
    McpAgentsList,
    McpCronRunsList,
    McpCronsList,
    McpDeleteResult,
    McpModelsList,
    McpNamesList,
    McpPromptsList,
    McpPromptVersionsList,
    McpRegisteredServersList,
    McpSkillUsage,
    McpStoreFile,
    McpThreadsList,
    McpTraceEventsList,
    McpUpsertResult,
)
from src.application.requests.cron import CreateCronRequest, UpdateCronRequest
from src.application.requests.mcp import CreateMcpServerRequest, UpdateMcpServerRequest, update_payload
from src.application.requests.model import CreateModelRequest, UpdateModelRequest
from src.application.requests.prompt import CreatePromptRequest, CreatePromptVersionRequest, UpdatePromptRequest
from src.application.responses.agent_run import RunAgentOneShotResult
from src.application.responses.cron import CronResponse, CronRunResponse
from src.application.responses.mcp_threads import ThreadSummary
from src.application.responses.prompt import PromptResponse, PromptVersionResponse
from src.application.responses.thread_history import ThreadHistory
from src.application.responses.trace_usage import TopRunsResponse, TraceUsageResponse
from src.application.responses.traces import TraceListResponse
from src.application.services.builtin_agent import is_builtin_agent_name
from src.application.services.route_validation import (
    list_skill_usage_agent_names,
    memories_prefix,
    memory_store_path,
    read_yaml_bytes,
    skill_directory_path,
    skill_store_path,
    skills_prefix,
    utc_bound,
    validate_agent_name,
)
from src.application.use_cases.list_traces import MAX_LIMIT
from src.application.use_cases.trace_usage import MAX_TOP_LIMIT
from src.domain.entities.agent_config import AgentConfig
from src.domain.entities.model_metadata import ModelMetadata
from src.domain.entities.registered_mcp_server import McpServerWrite, RegisteredMcpServer
from src.domain.entities.user.user_profile import UserProfile
from src.domain.errors.agent import AgentNotFoundError
from src.domain.errors.config import ConfigError
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage
from src.domain.ports.mcp_registry import McpRegistryClient

logger = logging.getLogger(__name__)

from mcp.types import ToolAnnotations  # noqa: E402


#: MCP tool annotations (REQ-008): supplements only — never a substitute for
#: authorization or HITL.
def _readonly() -> ToolAnnotations:
    """Read-only tool annotations (list/get/validate tools)."""
    return ToolAnnotations(readOnlyHint=True)


def _mutating() -> ToolAnnotations:
    """Mutating tool annotations (create/update/run tools)."""
    return ToolAnnotations(readOnlyHint=False)


def _destructive() -> ToolAnnotations:
    """Destructive tool annotations (every destructive delete tool)."""
    return ToolAnnotations(readOnlyHint=False, destructiveHint=True)


READONLY = _readonly
MUTATING = _mutating
DESTRUCTIVE = _destructive


# ---------------------------------------------------------------------------
# Composition-root access (lazy: the MCP tools are registered at import time,
# the wiring resolves at call time)
# ---------------------------------------------------------------------------
def _deps():
    from src import dependencies

    return dependencies


def register_mcp_tools(mcp) -> None:  # noqa: ANN001 — FastMCP instance
    """Register the complete typed management/execution tool surface."""
    from src.application.mcp.errors import safe_mcp_call

    _register_agents_family(mcp, safe_mcp_call)
    _register_registry_family(mcp, safe_mcp_call)
    _register_skills_family(mcp, safe_mcp_call)
    _register_memories_family(mcp, safe_mcp_call)
    _register_prompts_family(mcp, safe_mcp_call)
    _register_models_family(mcp, safe_mcp_call)
    _register_crons_family(mcp, safe_mcp_call)
    _register_threads_family(mcp, safe_mcp_call)
    _register_traces_family(mcp, safe_mcp_call)
    _register_users_family(mcp, safe_mcp_call)


# ---------------------------------------------------------------------------
# Agents family (REQ-012 + one-shot execution REQ-022..030)
# ---------------------------------------------------------------------------
def _register_agents_family(mcp, safe_mcp_call) -> None:
    @mcp.tool(name="composable_agents_list", annotations=READONLY())
    @safe_mcp_call
    async def composable_agents_list() -> McpAgentsList:
        """List the current user's agents (built-in assistant first, with is_builtin)."""
        agents = await _deps().get_list_agent_configs_use_case().execute()
        effective_builtin = await _deps().get_effective_builtin_agent_service().get_effective_metadata()
        return McpAgentsList(agents=[effective_builtin, *agents])

    @mcp.tool(name="composable_agents_get", annotations=READONLY())
    @safe_mcp_call
    async def composable_agents_get(agent_name: str) -> AgentConfig:
        """Get one visible agent's validated configuration by exact name.

        The built-in assistant resolves to its effective per-user definition
        with ``is_builtin=true`` at the top level.
        """
        validate_agent_name(agent_name)
        if is_builtin_agent_name(agent_name):
            return await _deps().get_effective_builtin_agent_service().get_effective_config()
        return await _deps().get_get_agent_config_use_case().execute(name=agent_name)

    @mcp.tool(name="composable_agents_create", annotations=MUTATING())
    @safe_mcp_call
    async def composable_agents_create(name: str, yaml: str) -> AgentConfig:
        """Create an agent from its name plus a complete YAML document (same rules as REST).

        The reserved built-in assistant name conflicts (409).
        """
        from src.application.services.builtin_agent import ensure_custom_agent_name

        validate_agent_name(name)
        ensure_custom_agent_name(name)
        yaml_content = read_yaml_bytes(yaml.encode("utf-8"))
        logger.info(LogMessage.AGENT_CONFIG_CREATING, name)
        return await _deps().get_create_agent_config_use_case().execute(name=name, yaml_content=yaml_content)

    @mcp.tool(name="composable_agents_update", annotations=MUTATING())
    @safe_mcp_call
    async def composable_agents_update(agent_name: str, yaml: str) -> AgentConfig:
        """Replace a custom agent's YAML, or the user's built-in customization."""
        validate_agent_name(agent_name)
        yaml_content = read_yaml_bytes(yaml.encode("utf-8"))
        if is_builtin_agent_name(agent_name):
            return await _deps().get_effective_builtin_agent_service().update(agent_name, yaml_content)
        logger.info(LogMessage.AGENT_CONFIG_UPDATING, agent_name)
        return await _deps().get_update_agent_config_use_case().execute(name=agent_name, yaml_content=yaml_content)

    @mcp.tool(name="composable_agents_reset_customization", annotations=MUTATING())
    @safe_mcp_call
    async def composable_agents_reset_customization(agent_name: str) -> McpDeleteResult:
        """Reset the current user's built-in assistant customization to its default."""
        validate_agent_name(agent_name)
        if not is_builtin_agent_name(agent_name):
            raise AgentNotFoundError(ErrorMessage.AGENT_NOT_FOUND.format(name=agent_name))
        await _deps().get_effective_builtin_agent_service().reset()
        return McpDeleteResult(deleted=agent_name)

    @mcp.tool(name="composable_agents_delete", annotations=DESTRUCTIVE())
    @safe_mcp_call
    async def composable_agents_delete(agent_name: str) -> McpDeleteResult:
        """Delete an existing custom agent (the built-in assistant is protected)."""
        from src.application.services.builtin_agent import ensure_not_builtin

        validate_agent_name(agent_name)
        ensure_not_builtin(agent_name)
        logger.info(LogMessage.AGENT_CONFIG_DELETING, agent_name)
        await _deps().get_delete_agent_config_use_case().execute(name=agent_name)
        logger.info(LogMessage.AGENT_CONFIG_DELETED, agent_name)
        return McpDeleteResult(deleted=agent_name)

    @mcp.tool(name="composable_agents_run", annotations=MUTATING())
    @safe_mcp_call
    async def composable_agents_run(
        agent_name: str,
        message: str,
    ) -> RunAgentOneShotResult:
        """Run any visible agent as a one-shot task (new source=mcp thread).

        Returns thread_id, agent_name, source, the final message, the bounded
        turn trace and degraded capability info. The message must be
        non-empty; self-invocation, cycles and chains deeper than four nested
        calls are rejected with a safe error.
        """
        if not message.strip():
            raise ConfigError(ErrorMessage.MCP_TOOL_MESSAGE_EMPTY)
        validate_agent_name(agent_name)
        return await _deps().get_run_agent_one_shot_use_case().execute(agent_name, message)


# ---------------------------------------------------------------------------
# MCP registry family (REQ-017..019)
# ---------------------------------------------------------------------------
def _register_registry_family(mcp, safe_mcp_call) -> None:
    def _client() -> McpRegistryClient:
        return _deps().get_mcp_registry_client()

    @mcp.tool(name="composable_mcp_servers_list", annotations=READONLY())
    @safe_mcp_call
    async def composable_mcp_servers_list() -> McpRegisteredServersList:
        """List the user's registered MCP servers (all secrets masked)."""
        servers = await _client().list_servers()
        return McpRegisteredServersList(servers=servers)

    @mcp.tool(name="composable_mcp_servers_get", annotations=READONLY())
    @safe_mcp_call
    async def composable_mcp_servers_get(name: str) -> RegisteredMcpServer:
        """Get one registered MCP server (all secrets masked)."""
        return await _client().get_server(name)

    @mcp.tool(name="composable_mcp_servers_validate", annotations=READONLY())
    @safe_mcp_call
    async def composable_mcp_servers_validate(
        name: str,
        transport: str = "http",
        url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        headers: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        auth_token: str | None = None,
        auth_mode: Literal["headers", "user_token"] | None = None,
        source_type: str = "external",
        openapi_url: str | None = None,
    ) -> dict:
        """Validate an external or OpenAPI-backed MCP server WITHOUT persisting it."""
        request = _create_request(
            name=name,
            transport=transport,
            url=url,
            command=command,
            args=args,
            headers=headers,
            env=env,
            auth_token=auth_token,
            auth_mode=auth_mode,
            source_type=source_type,
            openapi_url=openapi_url,
        )
        report = await _client().validate_server(_as_write(request))
        return report.model_dump(mode="json")

    @mcp.tool(name="composable_mcp_servers_create", annotations=MUTATING())
    @safe_mcp_call
    async def composable_mcp_servers_create(
        name: str,
        transport: str = "http",
        url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        headers: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        auth_token: str | None = None,
        auth_mode: Literal["headers", "user_token"] | None = None,
        source_type: str = "external",
        openapi_url: str | None = None,
    ) -> RegisteredMcpServer:
        """Register an external MCP server or generate one from OpenAPI/Swagger.

        Secrets (auth_token/headers/env) are write-only (REQ-019): the result
        carries the registry's MASKED entry and secrets never reach logs.
        ``auth_mode`` (REQ-004) selects the credential-transmission mode.
        """
        request = _create_request(
            name=name,
            transport=transport,
            url=url,
            command=command,
            args=args,
            headers=headers,
            env=env,
            auth_token=auth_token,
            auth_mode=auth_mode,
            source_type=source_type,
            openapi_url=openapi_url,
        )
        return await _client().create_server(_as_write(request))

    @mcp.tool(name="composable_mcp_servers_update", annotations=MUTATING())
    @safe_mcp_call
    async def composable_mcp_servers_update(
        name: str,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        auth_token: str | None = None,
        auth_mode: Literal["headers", "user_token"] | None = None,
        source_type: str | None = None,
        openapi_url: str | None = None,
    ) -> RegisteredMcpServer:
        """Update a registered MCP server (PARTIAL — provided fields only, masked response).

        Unspecified fields (including the stored auth_token/headers/env) are
        preserved by the TRUSTED server-side registry merge; the response
        carries the registry's MASKED entry, so secrets are never echoed.
        Coherence is validated locally on the MERGED specification (the
        provided fields on top of the existing projection) — an incoherent
        update is a stable, actionable local error, never an opaque
        forwarded 422.
        """
        from src.application.requests.mcp import validate_server_spec_coherence
        from src.domain.errors.config import ConfigError

        request = UpdateMcpServerRequest.model_validate(
            {
                key: value
                for key, value in (
                    ("url", url),
                    ("headers", headers),
                    ("env", env),
                    ("auth_token", auth_token),
                    ("auth_mode", auth_mode),
                    ("source_type", source_type),
                    ("openapi_url", openapi_url),
                )
                if value is not None
            }
        )
        merge_projection = await _client().get_server(name)
        try:
            validate_server_spec_coherence(
                transport=merge_projection.transport,
                url=url if url is not None else merge_projection.url,
                source_type=source_type if source_type is not None else merge_projection.source_type,
                openapi_url=openapi_url if openapi_url is not None else merge_projection.openapi_url,
            )
        except ValueError as exc:
            # Controlled, stable, non-sensitive mapping of the coherence rule
            # to the validation error family (BUG-002) — the raw ValueError is
            # always a centralized, safe message from the shared helper.
            raise ConfigError(str(exc)) from None
        return await _client().update_server(name, update_payload(request))

    @mcp.tool(name="composable_mcp_servers_delete", annotations=DESTRUCTIVE())
    @safe_mcp_call
    async def composable_mcp_servers_delete(name: str) -> McpDeleteResult:
        """Delete a registered MCP server (registry-owned persistence)."""
        await _client().delete_server(name)
        return McpDeleteResult(deleted=name)


def _create_request(**fields) -> CreateMcpServerRequest:
    """Validate the raw MCP create/validate input into the registry request DTO.

    Uses ``model_validate`` so the DTO's Literal/shape rules apply with the
    exact REST contract (invalid input raises the mapped validation error).
    """
    return CreateMcpServerRequest.model_validate(
        {
            **{key: value for key, value in fields.items() if value is not None},
            "args": fields.get("args") or [],
            "headers": fields.get("headers") or {},
            "env": fields.get("env") or {},
        }
    )


def _as_write(request: CreateMcpServerRequest) -> McpServerWrite:
    """Project the MCP create/validate input onto the registry port entity."""
    return McpServerWrite(**request.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Skills family (REQ-013)
# ---------------------------------------------------------------------------
def _register_skills_family(mcp, safe_mcp_call) -> None:
    @mcp.tool(name="composable_skills_list", annotations=READONLY())
    @safe_mcp_call
    async def composable_skills_list() -> McpNamesList:
        """List the current user's global skill directory paths."""
        files = await _deps().get_list_store_files_use_case().execute(prefix=skills_prefix())
        return McpNamesList(items=files)

    @mcp.tool(name="composable_skills_get", annotations=READONLY())
    @safe_mcp_call
    async def composable_skills_get(skill_name: str) -> McpStoreFile:
        """Read a global skill's store path and its SKILL.md content."""
        path = skill_store_path(skill_name)
        content = await _deps().get_get_store_file_use_case().execute(path=path)
        if content is None:
            from src.domain.errors.store_file import StoreFileNotFoundError

            raise StoreFileNotFoundError(f"File not found: {path}")
        return McpStoreFile(path=path, content=content)

    @mcp.tool(name="composable_skills_upsert", annotations=MUTATING())
    @safe_mcp_call
    async def composable_skills_upsert(skill_name: str, content: str) -> McpUpsertResult:
        """Create or replace /skills/{skill_name}/SKILL.md (traversal-safe)."""
        path = skill_store_path(skill_name)
        await _deps().get_put_store_file_use_case().execute(path=path, content=content)
        return McpUpsertResult(path=path)

    @mcp.tool(name="composable_skills_delete", annotations=DESTRUCTIVE())
    @safe_mcp_call
    async def composable_skills_delete(skill_name: str) -> McpDeleteResult:
        """Delete the selected global skill directory (destructive)."""
        directory = skill_directory_path(skill_name)
        deps = _deps()
        files = await deps.get_list_store_files_use_case().execute(prefix=directory)
        for path in files:
            await deps.get_delete_store_file_use_case().execute(path=path)
        return McpDeleteResult(deleted=f"{directory} ({len(files)} file(s))")

    @mcp.tool(name="composable_skills_usage", annotations=READONLY())
    @safe_mcp_call
    async def composable_skills_usage(skill_name: str) -> McpSkillUsage:
        """List the agents using a named global skill."""
        skill_directory_path(skill_name)  # validates the plain name (traversal-safe)
        agents = await list_skill_usage_agent_names(
            lambda prefix: _deps().get_list_store_files_use_case().execute(prefix=prefix),
            skill_name,
        )
        return McpSkillUsage(agents=agents)


# ---------------------------------------------------------------------------
# Memories family (REQ-013)
# ---------------------------------------------------------------------------
def _register_memories_family(mcp, safe_mcp_call) -> None:
    @mcp.tool(name="composable_memories_list", annotations=READONLY())
    @safe_mcp_call
    async def composable_memories_list() -> McpNamesList:
        """List the current user's global memory file paths (all agents)."""
        files = await _deps().get_list_store_files_use_case().execute(prefix=memories_prefix())
        return McpNamesList(items=files)

    @mcp.tool(name="composable_memories_get", annotations=READONLY())
    @safe_mcp_call
    async def composable_memories_get(filename: str) -> McpStoreFile:
        """Read one memory file (agent-scoped shape '<agent_slug>/<filename>')."""
        path = memory_store_path(filename)
        content = await _deps().get_get_store_file_use_case().execute(path=path)
        if content is None:
            from src.domain.errors.store_file import StoreFileNotFoundError

            raise StoreFileNotFoundError(f"File not found: {path}")
        return McpStoreFile(path=path, content=content)

    @mcp.tool(name="composable_memories_upsert", annotations=MUTATING())
    @safe_mcp_call
    async def composable_memories_upsert(filename: str, content: str) -> McpUpsertResult:
        """Create or replace /memories/{agent_slug}/{filename} (prefix-escape-safe, agent-scoped)."""
        path = memory_store_path(filename)
        await _deps().get_put_store_file_use_case().execute(path=path, content=content)
        return McpUpsertResult(path=path)

    @mcp.tool(name="composable_memories_delete", annotations=DESTRUCTIVE())
    @safe_mcp_call
    async def composable_memories_delete(filename: str) -> McpDeleteResult:
        """Delete one agent-scoped memory file (destructive, '<agent_slug>/<filename>')."""
        path = memory_store_path(filename)
        await _deps().get_delete_store_file_use_case().execute(path=path)
        return McpDeleteResult(deleted=path)


# ---------------------------------------------------------------------------
# Prompts family (REQ-014)
# ---------------------------------------------------------------------------
def _register_prompts_family(mcp, safe_mcp_call) -> None:
    @mcp.tool(name="composable_prompts_list", annotations=READONLY())
    @safe_mcp_call
    async def composable_prompts_list() -> McpPromptsList:
        """List prompt metadata."""
        prompts = await _deps().get_list_prompts_use_case().execute()
        return McpPromptsList(prompts=[PromptResponse.model_validate(prompt) for prompt in prompts])

    @mcp.tool(name="composable_prompts_get", annotations=READONLY())
    @safe_mcp_call
    async def composable_prompts_get(prompt_id: str) -> PromptResponse:
        """Get one prompt's metadata."""
        prompt = await _deps().get_get_prompt_use_case().execute(prompt_id)
        return PromptResponse.model_validate(prompt)

    @mcp.tool(name="composable_prompts_create", annotations=MUTATING())
    @safe_mcp_call
    async def composable_prompts_create(args: CreatePromptRequest) -> PromptResponse:
        """Create prompt metadata and its immutable version 1 (content verbatim)."""
        prompt = (
            await _deps()
            .get_create_prompt_use_case()
            .execute(name=args.name, description=args.description, content=args.content)
        )
        return PromptResponse.model_validate(prompt)

    @mcp.tool(name="composable_prompts_update", annotations=MUTATING())
    @safe_mcp_call
    async def composable_prompts_update(prompt_id: str, args: UpdatePromptRequest) -> PromptResponse:
        """Update prompt name/description only (never creates a version)."""
        updated = await _deps().get_update_prompt_use_case().execute(prompt_id, args.model_dump(exclude_unset=True))
        return PromptResponse.model_validate(updated)

    @mcp.tool(name="composable_prompts_delete", annotations=DESTRUCTIVE())
    @safe_mcp_call
    async def composable_prompts_delete(prompt_id: str) -> McpDeleteResult:
        """Delete an unreferenced prompt (in-use prompts conflict)."""
        await _deps().get_delete_prompt_use_case().execute(prompt_id)
        return McpDeleteResult(deleted=prompt_id)

    @mcp.tool(name="composable_prompt_versions_list", annotations=READONLY())
    @safe_mcp_call
    async def composable_prompt_versions_list(prompt_id: str) -> McpPromptVersionsList:
        """List immutable prompt-version summaries."""
        versions = await _deps().get_list_prompt_versions_use_case().execute(prompt_id)
        return McpPromptVersionsList(versions=list(versions))

    @mcp.tool(name="composable_prompt_versions_get", annotations=READONLY())
    @safe_mcp_call
    async def composable_prompt_versions_get(prompt_id: str, version: int) -> PromptVersionResponse:
        """Read one exact immutable prompt version (no Latest fallback)."""
        row = await _deps().get_get_prompt_version_use_case().execute(prompt_id, version)
        return PromptVersionResponse.model_validate(row)

    @mcp.tool(name="composable_prompt_versions_create", annotations=MUTATING())
    @safe_mcp_call
    async def composable_prompt_versions_create(prompt_id: str, args: CreatePromptVersionRequest) -> dict:
        """Atomically append the next immutable version (concurrency-safe)."""
        version = await _deps().get_add_prompt_version_use_case().execute(prompt_id, args.content)
        return PromptVersionResponse.model_validate(version).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Models family (MODEL-007: same identity business contract as REST)
# ---------------------------------------------------------------------------
def _register_models_family(mcp, safe_mcp_call) -> None:
    @mcp.tool(name="composable_models_list", annotations=READONLY())
    @safe_mcp_call
    async def composable_models_list() -> McpModelsList:
        """List identity-only model definitions."""
        models = await _deps().get_list_models_use_case().execute()
        return McpModelsList(models=list(models))

    @mcp.tool(name="composable_models_get", annotations=READONLY())
    @safe_mcp_call
    async def composable_models_get(model_id: str) -> ModelMetadata:
        """Get one model definition (identity only)."""
        return await _deps().get_get_model_use_case().execute(model_id)

    @mcp.tool(name="composable_models_create", annotations=MUTATING())
    @safe_mcp_call
    async def composable_models_create(args: CreateModelRequest) -> ModelMetadata:
        """Create a model identity (provider + name + provider_model)."""
        return (
            await _deps()
            .get_create_model_use_case()
            .execute(
                name=args.name,
                provider=args.provider,
                provider_model=args.provider_model,
            )
        )

    @mcp.tool(name="composable_models_update", annotations=MUTATING())
    @safe_mcp_call
    async def composable_models_update(model_id: str, args: UpdateModelRequest) -> ModelMetadata:
        """Update model identity fields (provided fields only)."""
        return await _deps().get_update_model_use_case().execute(model_id, args.model_dump(exclude_unset=True))

    @mcp.tool(name="composable_models_delete", annotations=DESTRUCTIVE())
    @safe_mcp_call
    async def composable_models_delete(model_id: str) -> McpDeleteResult:
        """Delete an unreferenced model (referenced models conflict)."""
        await _deps().get_delete_model_use_case().execute(model_id)
        return McpDeleteResult(deleted=model_id)


# ---------------------------------------------------------------------------
# Crons family (REQ-016/021)
# ---------------------------------------------------------------------------
def _register_crons_family(mcp, safe_mcp_call) -> None:
    @mcp.tool(name="composable_crons_list", annotations=READONLY())
    @safe_mcp_call
    async def composable_crons_list() -> McpCronsList:
        """List owned cron definitions with their scheduling state."""
        crons = await _deps().get_list_crons_use_case().execute()
        return McpCronsList(crons=[CronResponse.model_validate(cron) for cron in crons])

    @mcp.tool(name="composable_crons_get", annotations=READONLY())
    @safe_mcp_call
    async def composable_crons_get(cron_id: str) -> CronResponse:
        """Get one owned cron."""
        cron = await _deps().get_get_cron_use_case().execute(cron_id)
        return CronResponse.model_validate(cron)

    @mcp.tool(name="composable_crons_create", annotations=MUTATING())
    @safe_mcp_call
    async def composable_crons_create(args: CreateCronRequest) -> CronResponse:
        """Create and validate a cron (five-field expression, IANA timezone)."""
        created = (
            await _deps()
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
        return CronResponse.model_validate(created)

    @mcp.tool(name="composable_crons_update", annotations=MUTATING())
    @safe_mcp_call
    async def composable_crons_update(cron_id: str, args: UpdateCronRequest) -> CronResponse:
        """Update an owned cron (provided fields only; frontier recalculates)."""
        updated = await _deps().get_update_cron_use_case().execute(cron_id, args.model_dump(exclude_unset=True))
        return CronResponse.model_validate(updated)

    @mcp.tool(name="composable_crons_delete", annotations=DESTRUCTIVE())
    @safe_mcp_call
    async def composable_crons_delete(cron_id: str) -> McpDeleteResult:
        """Delete an owned cron (destructive)."""
        await _deps().get_delete_cron_use_case().execute(cron_id)
        return McpDeleteResult(deleted=cron_id)

    @mcp.tool(name="composable_crons_run_now", annotations=MUTATING())
    @safe_mcp_call
    async def composable_crons_run_now(cron_id: str) -> CronRunResponse:
        """Trigger Run now (including for a disabled cron); creates a cron thread."""
        run = await _deps().get_run_cron_now_use_case().execute(cron_id)
        return CronRunResponse.model_validate(run)

    @mcp.tool(name="composable_cron_runs_list", annotations=READONLY())
    @safe_mcp_call
    async def composable_cron_runs_list(
        cron_id: str,
        limit: Annotated[int, "bounded run history page size (1..200)"] = 50,
    ) -> McpCronRunsList:
        """List bounded cron run history (newest first)."""
        if not 1 <= limit <= 200:
            raise ConfigError(ErrorMessage.CRON_RUN_LIMIT_INVALID)
        runs = await _deps().get_list_cron_runs_use_case().execute(cron_id, limit)
        return McpCronRunsList(runs=[CronRunResponse.model_validate(run) for run in runs])


# ---------------------------------------------------------------------------
# Threads family (REQ-023 trace/history visibility)
# ---------------------------------------------------------------------------
def _register_threads_family(mcp, safe_mcp_call) -> None:
    @mcp.tool(name="composable_threads_list", annotations=READONLY())
    @safe_mcp_call
    async def composable_threads_list() -> McpThreadsList:
        """List the current user's threads with their immutable sources."""
        threads = await _deps().get_list_threads_use_case().execute()
        return McpThreadsList(threads=[ThreadSummary.of(thread) for thread in threads])

    @mcp.tool(name="composable_threads_get", annotations=READONLY())
    @safe_mcp_call
    async def composable_threads_get(thread_id: str) -> ThreadSummary:
        """Get one owned thread."""
        thread = await _deps().get_get_thread_use_case().execute(thread_id)
        return ThreadSummary.of(thread)

    @mcp.tool(name="composable_threads_history", annotations=READONLY())
    @safe_mcp_call
    async def composable_threads_history(thread_id: str) -> ThreadHistory:
        """Get a turn-grouped thread history."""
        return await _deps().get_get_thread_history_use_case().execute(thread_id)

    @mcp.tool(name="composable_threads_trace", annotations=READONLY())
    @safe_mcp_call
    async def composable_threads_trace(thread_id: str) -> McpTraceEventsList:
        """Get the thread's flat trace (ordered by timestamp)."""
        deps = _deps()
        await deps.get_get_thread_use_case().execute(thread_id)
        events = await deps.get_trace_event_repository().list_by_thread(thread_id)
        return McpTraceEventsList(events=events)

    @mcp.tool(name="composable_threads_delete", annotations=DESTRUCTIVE())
    @safe_mcp_call
    async def composable_threads_delete(thread_id: str) -> McpDeleteResult:
        """Delete an owned thread with its trace history (destructive)."""
        await _deps().get_delete_thread_use_case().execute(thread_id)
        return McpDeleteResult(deleted=thread_id)


# ---------------------------------------------------------------------------
# Traces / usage family (REQ-007 UTC normalization, bounded lists)
# ---------------------------------------------------------------------------
def _register_traces_family(mcp, safe_mcp_call) -> None:
    @mcp.tool(name="composable_traces_list", annotations=READONLY())
    @safe_mcp_call
    async def composable_traces_list(
        agent_name: str | None = None,
        limit: Annotated[int, "bounded page size (1..100)"] = 50,
        cursor: str | None = None,
    ) -> TraceListResponse:
        """List trace summaries with the existing filters and pagination."""
        if not 1 <= limit <= MAX_LIMIT:
            raise ConfigError(ErrorMessage.TRACE_LIMIT_INVALID)
        page = await _deps().get_list_traces_use_case().execute(agent_name=agent_name, limit=limit, cursor=cursor)
        return TraceListResponse(items=page.items, next_cursor=page.next_cursor)

    @mcp.tool(name="composable_traces_usage", annotations=READONLY())
    @safe_mcp_call
    async def composable_traces_usage(
        agent_name: str | None = None,
        from_ts: Annotated[str, "inclusive ISO lower bound (naive means UTC)"] | None = None,
        to_ts: Annotated[str, "inclusive ISO upper bound (naive means UTC)"] | None = None,
    ) -> TraceUsageResponse:
        """Get aggregated LLM usage with the existing filters."""
        return _usage_response(
            await _deps()
            .get_trace_usage_use_case()
            .execute(agent_name=agent_name, from_ts=utc_bound(from_ts), to_ts=utc_bound(to_ts)),
            group_by=None,
        )

    @mcp.tool(name="composable_traces_top_runs", annotations=READONLY())
    @safe_mcp_call
    async def composable_traces_top_runs(
        limit: Annotated[int, "bounded page size (1..100)"] = 10,
        agent_name: str | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
    ) -> TopRunsResponse:
        """Get the existing bounded top-runs LLM usage report."""
        if not 1 <= limit <= MAX_TOP_LIMIT:
            raise ConfigError(ErrorMessage.TRACE_LIMIT_INVALID)
        runs = (
            await _deps()
            .get_top_usage_runs_use_case()
            .execute(limit=limit, agent_name=agent_name, from_ts=utc_bound(from_ts), to_ts=utc_bound(to_ts))
        )
        return TopRunsResponse(items=runs)


def _usage_response(summary, group_by: Literal["agent", "model"] | None) -> TraceUsageResponse:
    """Build the usage response DTO (shared with the REST route)."""
    from src.application.services.trace_series import usage_series

    return TraceUsageResponse(
        total_input_tokens=summary.total_input_tokens,
        total_output_tokens=summary.total_output_tokens,
        total_tokens=summary.total_tokens,
        total_cost_usd=summary.total_cost_usd,
        run_count=summary.run_count,
        error_count=summary.error_count,
        duration_ms_p50=summary.duration_ms_p50,
        duration_ms_p95=summary.duration_ms_p95,
        series=usage_series(summary, group_by),
    )


# ---------------------------------------------------------------------------
# Users family (REQ-004 identity parity)
# ---------------------------------------------------------------------------
def _register_users_family(mcp, safe_mcp_call) -> None:
    @mcp.tool(name="composable_users_me", annotations=READONLY())
    @safe_mcp_call
    async def composable_users_me() -> UserProfile:
        """Return the authenticated user's non-sensitive profile."""
        deps = _deps()
        ctx = deps.get_current_auth_context()
        return deps.get_get_current_user_use_case().execute(ctx)
