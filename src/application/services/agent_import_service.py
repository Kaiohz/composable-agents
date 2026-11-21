"""Agent YAML import with resource materialization (create-time).

Spec "Configuration des agents par ressources et matérialisation des imports
YAML": every resource actually DEFINED in an imported YAML is created in its
own registry (model catalog, prompt registry, MCP registry, Store files,
sub-agents), then referenced by the persisted agent — the persisted
configuration is RESOURCE-ONLY (REQ-IMPORT-003) while the import input stays
permissive (REQ-IMPORT-001).

Transactional contract (REQ-TX-001..003):

* the agent is persisted LAST, only after a complete import;
* every resource created during the import is recorded in a compensation
  ledger and compensated in reverse order on ANY failure — preexisting
  resources are never touched (their undo actions are never recorded);
* compensation is best-effort and idempotent, shielded from cancellation: an
  already-gone resource never masks the original failure, which is raised
  AFTER compensation completes;
* diagnostics identify the resource type, the resource name and the cause
  with NO secrets (REQ-TX-004).

The orchestrator coordinates ports only; the PURE planning/prevalidation
rules live in :mod:`src.domain.services.agent_import_plan`.
"""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import yaml  # type: ignore[import-untyped]

from src.application.services.builtin_agent import is_builtin_agent_name
from src.application.use_cases._subagent_ref_utils import validate_subagent_refs
from src.domain.entities.agent_config import AgentConfig
from src.domain.entities.agent_config_metadata import AgentConfigMetadata
from src.domain.entities.model_metadata import ModelMetadata
from src.domain.entities.prompt import Prompt
from src.domain.errors.agent import (
    AgentConfigAlreadyExistsError,
    AgentNotFoundError,
)
from src.domain.errors.base import DomainError
from src.domain.errors.config import ConfigError, ConfigValidationError
from src.domain.errors.mcp import McpRegistryNotFoundError
from src.domain.errors.messages import ErrorMessage
from src.domain.errors.model import ModelNotFoundError
from src.domain.errors.prompt import PromptNotFoundError
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_config_repository import AgentConfigRepository
from src.domain.ports.agent_config_store import AgentConfigStore
from src.domain.ports.mcp_registry import McpRegistryClient
from src.domain.ports.model_repository import ModelRepository
from src.domain.ports.prompt_repository import PromptRepository
from src.domain.ports.store_file_repository import StoreFileRepository
from src.domain.services.agent_import_plan import (
    AgentImportPlan,
    McpMaterialization,
    ModelMaterialization,
    PromptMaterialization,
    SubagentMaterialization,
    build_agent_import_plan,
    derived_resource_names,
)
from src.domain.services.agent_registry_refs import registry_ref_metadata_fields, validate_registry_refs
from src.domain.services.mcp_ref_policy import ensure_mcp_refs_registered

logger = logging.getLogger(__name__)

_IMPORT_SOURCE = "<yaml import>"


class _CompensationLedger:
    """Ordered record of every resource created by one import attempt.

    Each entry pairs a human-identifying (kind, name) with an undo callable
    run in REVERSE order on failure. Preexisting resources are never recorded,
    so a rollback can never destroy them (REQ-TX-001).
    """

    def __init__(self) -> None:
        self._entries: list[tuple[str, str, Callable[[], Awaitable[None]]]] = []

    def record(self, kind: str, name: str, undo: Callable[[], Awaitable[None]]) -> None:
        """Record one created resource with its undo action."""
        self._entries.append((kind, name, undo))

    def entries(self) -> list[tuple[str, str, Callable[[], Awaitable[None]]]]:
        """The recorded undoable resources, in creation order."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


class AgentImportService:
    """Orchestrate one import: plan → materialize → persist (all-or-nothing)."""

    def __init__(
        self,
        config_store: AgentConfigStore,
        config_repository: AgentConfigRepository,
        prompt_repository: PromptRepository,
        model_repository: ModelRepository,
        mcp_registry: McpRegistryClient | None,
        store_files: StoreFileRepository,
    ) -> None:
        """Wire the outbound ports (composition-root singletons)."""
        self._config_store = config_store
        self._config_repository = config_repository
        self._prompt_repository = prompt_repository
        self._model_repository = model_repository
        self._mcp_registry = mcp_registry
        self._store_files = store_files

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    async def import_agent(self, name: str, yaml_content: str) -> AgentConfig:
        """Materialize an imported YAML into registries and persist the agent.

        Args:
            name: The requested agent name (must match the YAML ``name``).
            yaml_content: The raw import YAML text.

        Returns:
            The resource-only ``AgentConfig`` as persisted.

        Raises:
            AgentConfigAlreadyExistsError: The name is taken (or reserved).
            ConfigError: Prevalidation failures.
            McpRegistryError: Controlled registry failures (409/502/503).
        """
        raw = self._parse_raw_yaml(yaml_content)
        plan = build_agent_import_plan(raw, name)
        ledger = _CompensationLedger()
        try:
            return await self._execute_import(plan, ledger)
        except BaseException:
            # Shield the rollback: a cancellation must never leave orphans.
            await asyncio.shield(self._compensate(plan.agent_name, ledger))
            raise

    # ------------------------------------------------------------------
    # Materialization run (the agent LAST, REQ-TX-001)
    # ------------------------------------------------------------------
    async def _execute_import(self, plan: AgentImportPlan, ledger: _CompensationLedger) -> AgentConfig:
        """Run the whole import for one validated plan."""
        name = plan.agent_name
        try:
            referenced_model = await self._prevalidate_io(plan)
        except DomainError as exc:
            logger.info(LogMessage.AGENT_IMPORT_PREVALIDATION_FAILED, name, exc.detail)
            raise

        model_id, model_metadata = await self._materialize_model(plan.model, ledger)
        prompt_id = await self._materialize_prompt(plan.prompt, ledger)
        created_mcp_refs = await self._materialize_mcp(plan.mcp_creations, name, ledger)

        subagent_ids: dict[str, str] = {}
        for creation in plan.subagent_creations:
            subagent_ids[creation.entry_name] = await self._create_subagent(name, plan, creation, ledger)

        await self._materialize_files(plan, name, ledger)

        target = self._inject_references(dict(plan.target), plan, model_id, prompt_id, created_mcp_refs, subagent_ids)
        config = _validated_config(target)
        await validate_subagent_refs(config, self._config_repository)
        await self._persist_agent(name, config, target, model_metadata or referenced_model, ledger)

        logger.info(LogMessage.AGENT_IMPORT_COMPLETED, name, len(ledger))
        return config

    @staticmethod
    def _parse_raw_yaml(yaml_content: str) -> dict:
        """Parse the import YAML into its raw mapping (empty/invalid → ConfigError)."""
        if not yaml_content or not yaml_content.strip():
            raise ConfigError(ErrorMessage.YAML_EMPTY.format(source=_IMPORT_SOURCE))
        try:
            raw = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ConfigError(ErrorMessage.YAML_INVALID.format(source=_IMPORT_SOURCE, error=e)) from e
        if not isinstance(raw, dict):
            raise ConfigError(ErrorMessage.YAML_NOT_MAPPING.format(source=_IMPORT_SOURCE, type=type(raw).__name__))
        return raw

    # ------------------------------------------------------------------
    # I/O prevalidation (before ANY write, REQ-IMPORT-006) — one rule family
    # per helper so each stays minimal and fully covered.
    # ------------------------------------------------------------------
    async def _prevalidate_io(self, plan: AgentImportPlan) -> ModelMetadata | None:
        """Run every existence/availability check BEFORE any materialization.

        Returns:
            The referenced catalog model's metadata when the import declares a
            ``model_ref`` (display projection I7), ``None`` otherwise.
        """
        await self._assert_names_available(plan)
        await self._assert_resource_names_available(plan)
        referenced_model = await self._assert_references_resolvable(plan)
        await self._assert_selections_registered(plan)
        await self._assert_files_availability(plan)
        return referenced_model

    async def _assert_names_available(self, plan: AgentImportPlan) -> None:
        """The root and every created sub-agent name must be free."""
        name = plan.agent_name
        if is_builtin_agent_name(name) or await self._config_repository.exists(name):
            raise AgentConfigAlreadyExistsError(ErrorMessage.AGENT_CONFIG_ALREADY_EXISTS.format(name=name))
        for creation in plan.subagent_creations:
            if is_builtin_agent_name(creation.entry_name):
                raise ConfigError(ErrorMessage.AGENT_IMPORT_SUBAGENT_NAME_RESERVED.format(name=creation.entry_name))
            if await self._config_repository.exists(creation.entry_name):
                raise ConfigError(ErrorMessage.AGENT_IMPORT_SUBAGENT_NAME_CONFLICT.format(name=creation.entry_name))

    async def _assert_resource_names_available(self, plan: AgentImportPlan) -> None:
        """Refuse a PREDICTABLE resource-name conflict (REQ-TX-002/006).

        The catalog tables carry no unique constraint on ``name``: without
        this check an import would silently create a SECOND resource named
        ``<agent>-model`` / ``<agent>-prompt`` (also the retry-duplicates
        guard of the reliability NFR).
        """
        derived = derived_resource_names(plan)
        if not derived:
            return
        known_models = {model.name for model in await self._model_repository.list_all()}
        known_prompts = {prompt.name for prompt in await self._prompt_repository.list_all()}
        for resource_name, kind in derived:
            taken = known_models if kind == "model" else known_prompts
            template = ErrorMessage.AGENT_IMPORT_RESOURCE_NAME_CONFLICT
            if resource_name in taken:
                raise ConfigError(template.format(kind=kind, name=resource_name))

    async def _assert_references_resolvable(self, plan: AgentImportPlan) -> ModelMetadata | None:
        """Validate the declared references owner-scoped (non-enumerating).

        Returns:
            The referenced catalog model's metadata when a ``model_ref`` is
            declared (the projection of I7 is resolved from this same read —
            the reference is never fetched a second time).
        """
        referenced_model = (
            await self._model_repository.get(str(_parse_uuid(plan.model_ref)))
            if plan.model_ref is not None
            else None
        )
        # Reuse of the reference-validation rules for the prompt side (the
        # model side above performs exactly the same owner-scoped fetch).
        probe = AgentConfig(name=plan.agent_name, prompt_ref=plan.prompt_ref)
        await validate_registry_refs(probe, self._prompt_repository, None)
        return referenced_model

    async def _assert_selections_registered(self, plan: AgentImportPlan) -> None:
        """MCP: declared references are the owner's; creations validate pre-persist."""
        if not plan.mcp_refs and not plan.mcp_creations:
            return
        registry = self._registry_ready()
        if plan.mcp_refs:
            # Mixed imports may reference already-registered servers; the
            # selection stays owner-scoped and non-enumerating.
            await ensure_mcp_refs_registered(registry, plan.mcp_refs)
        for creation in plan.mcp_creations:
            # REQ-MCP-002: the registry validates the transport/URL semantics
            # BEFORE anything is persisted (its SSRF/connection checks apply).
            report = await registry.validate_server(creation.writer)
            if not report.valid:
                raise ConfigError(ErrorMessage.AGENT_IMPORT_MCP_VALIDATION_FAILED.format(name=creation.writer.name))
            logger.info(LogMessage.AGENT_IMPORT_MCP_VALIDATED, creation.writer.name)

    async def _assert_files_availability(self, plan: AgentImportPlan) -> None:
        """Textual skill/memory references must exist; inline targets must not.

        The reads run CONCURRENTLY (one bounded gather per family — NFR: no
        sequential store round-trips); existence uses the Store File API
        canonical keys, so references resolve against exactly the keys the
        REST surface and the runner both use.
        """
        await self._assert_textual_references_exist(plan)
        await self._assert_inline_targets_absent(plan)

    async def _assert_textual_references_exist(self, plan: AgentImportPlan) -> None:
        """One canonical read per textual reference, gathered (NFR I8)."""
        lookups: dict[str, tuple[str, str]] = {}
        for ref in plan.skill_refs:
            lookups[_skill_store_key(ref)] = ("skill", ref)
        for ref in plan.memory_refs:
            lookups[_memory_store_key(ref)] = ("memory", ref)
        if not lookups:
            return
        contents = await asyncio.gather(
            *(self._store_files.get_file(key) for key in lookups),
            return_exceptions=True,
        )
        for (kind, ref), content in zip(lookups.values(), contents, strict=True):
            if isinstance(content, BaseException):
                raise content
            if content is None:
                raise _missing_reference_error(kind, ref)

    async def _assert_inline_targets_absent(self, plan: AgentImportPlan) -> None:
        """An inline definition must not overwrite an existing file."""
        if not plan.files:
            return
        contents = await asyncio.gather(
            *(self._store_files.get_file(file_definition.file_path) for file_definition in plan.files)
        )
        for file_definition, content in zip(plan.files, contents, strict=True):
            template = (
                ErrorMessage.AGENT_IMPORT_SKILL_EXISTS
                if file_definition.is_skill
                else ErrorMessage.AGENT_IMPORT_MEMORY_EXISTS
            )
            if content is not None:
                raise ConfigError(template.format(path=file_definition.file_path))

    def _registry_ready(self) -> McpRegistryClient:
        """Return the registry client or raise a controlled wiring error (503)."""
        if self._mcp_registry is None:
            raise ConfigError(ErrorMessage.MCP_REGISTRY_UNAVAILABLE)
        return self._mcp_registry

    # ------------------------------------------------------------------
    # Materialization (each step records its undo BEFORE moving on)
    # ------------------------------------------------------------------
    async def _materialize_model(
        self,
        model: ModelMaterialization | None,
        ledger: _CompensationLedger,
    ) -> tuple[str | None, ModelMetadata | None]:
        """Create one Model catalog resource when explicitly declared (REQ-MODEL-001)."""
        if model is None:
            return None, None
        created = await self._model_repository.create(
            ModelMetadata(name=model.resource_name, provider=model.provider, provider_model=model.provider_model)
        )
        ledger.record("model", model.resource_name, self._undo_model(created.id))
        logger.info(LogMessage.AGENT_IMPORT_RESOURCE_CREATED, "model", created.id, model.agent_name)
        return created.id, created

    async def _materialize_prompt(
        self, prompt: PromptMaterialization | None, ledger: _CompensationLedger
    ) -> str | None:
        """Create one Prompt + immutable version 1 (REQ-PROMPT-001), verbatim."""
        if prompt is None:
            return None
        created = await self._prompt_repository.create_with_version(Prompt(name=prompt.resource_name), prompt.content)
        ledger.record("prompt", prompt.resource_name, self._undo_prompt(created.id))
        logger.info(LogMessage.AGENT_IMPORT_RESOURCE_CREATED, "prompt", created.id, prompt.agent_name)
        return created.id

    async def _materialize_mcp(
        self, creations: list[McpMaterialization], root_name: str, ledger: _CompensationLedger
    ) -> list[str]:
        """Create every inline MCP definition in the user's registry (REQ-MCP-001).

        Every declared server (root- or sub-agent-owned) is created
        owner-scoped; ONLY the ROOT-owned identities are returned — a server
        declared by an inline sub-agent belongs to that sub-agent's own
        selection and must never be mounted by the parent (REQ-SUBAGENT-001).
        """
        if not creations:
            return []
        registry = self._registry_ready()
        root_refs: list[str] = []
        for creation in creations:
            await registry.create_server(creation.writer)
            logger.info(LogMessage.AGENT_IMPORT_MCP_CREATED, creation.writer.name, root_name)
            ledger.record("mcp server", creation.writer.name, self._undo_mcp(creation.writer.name))
            if creation.owner_agent == root_name:
                root_refs.append(creation.writer.name)
        return root_refs

    async def _materialize_files(self, plan: AgentImportPlan, agent_name: str, ledger: _CompensationLedger) -> None:
        """Write inline skill/memory files verbatim via the Store File API."""
        for file_definition in plan.files:
            kind = "skill" if file_definition.is_skill else "memory"
            await self._store_files.put_file(file_definition.file_path, file_definition.content)
            logger.info(LogMessage.AGENT_IMPORT_RESOURCE_CREATED, kind, file_definition.file_path, agent_name)
            ledger.record(kind, file_definition.file_path, self._undo_store_file(file_definition.file_path))

    # ------------------------------------------------------------------
    # Inline sub-agents (REQ-SUBAGENT-001)
    # ------------------------------------------------------------------
    async def _create_subagent(
        self,
        parent_name: str,
        plan: AgentImportPlan,
        creation: SubagentMaterialization,
        ledger: _CompensationLedger,
    ) -> str:
        """Materialize one inline sub-agent and persist it as a real agent.

        Its own ``model``/``instructions`` follow the Model/Prompt rules
        (named after the sub-agent); its MCP selection is EXACTLY the set of
        registry identities created under its ownership — never the parent's.
        """
        subagent_name = creation.entry_name
        model_id, model_metadata = await self._materialize_model(creation.model, ledger)
        prompt_id = await self._materialize_prompt(creation.prompt, ledger)
        own_mcp_refs = _owned_mcp_ref_names(plan.mcp_creations, subagent_name)
        target = _subagent_target(creation, model_id, prompt_id, own_mcp_refs)
        config = _validated_config(target)
        await self._persist_agent(subagent_name, config, target, model_metadata, ledger)
        logger.info(LogMessage.AGENT_IMPORT_RESOURCE_CREATED, "agent", subagent_name, parent_name)
        return subagent_name

    # ------------------------------------------------------------------
    # Reference injection + persistence (the agent LAST, REQ-TX-001)
    # ------------------------------------------------------------------
    def _inject_references(
        self,
        target: dict,
        plan: AgentImportPlan,
        model_id: str | None,
        prompt_id: str | None,
        created_mcp_refs: list[str],
        subagent_ids: dict[str, str],
    ) -> dict:
        """Complete the resource-only target with the materialized identities."""
        if model_id is not None:
            target["model_ref"] = model_id
        elif plan.model_ref is not None:
            target["model_ref"] = plan.model_ref
        if prompt_id is not None:
            target["prompt_ref"] = {"id": prompt_id}
        elif plan.prompt_ref is not None:
            target["prompt_ref"] = plan.prompt_ref.model_dump(mode="json", exclude_none=True)
        mcp_refs = [*plan.mcp_refs, *created_mcp_refs]
        if mcp_refs:
            target["mcp_refs"] = mcp_refs
        for creation in plan.subagent_creations:
            position = plan.subagent_entry_index[creation.entry_name]
            target["subagents"][position]["agent_ref"] = subagent_ids[creation.entry_name]
        return target

    async def _persist_agent(
        self,
        name: str,
        config: AgentConfig,
        target: dict,
        model_metadata: ModelMetadata | None,
        ledger: _CompensationLedger,
    ) -> None:
        """Persist one agent (root or created sub-agent): YAML then metadata."""
        await self._config_store.put(name, yaml.safe_dump(target, sort_keys=False))
        ledger.record("agent", name, self._undo_agent(name))
        await self._config_repository.save(_agent_metadata(name, config, model_metadata))

    # ------------------------------------------------------------------
    # Compensation (best-effort, idempotent, reverse order)
    # ------------------------------------------------------------------
    def _undo_agent(self, name: str) -> Callable[[], Awaitable[None]]:
        async def _undo() -> None:
            with contextlib.suppress(AgentNotFoundError):
                await self._config_repository.delete(name)
            with contextlib.suppress(AgentNotFoundError):
                await self._config_store.delete(name)  # idempotent (port contract)

        return _undo

    def _undo_model(self, model_id: str) -> Callable[[], Awaitable[None]]:
        async def _undo() -> None:
            with contextlib.suppress(ModelNotFoundError):
                await self._model_repository.delete(model_id)

        return _undo

    def _undo_prompt(self, prompt_id: str) -> Callable[[], Awaitable[None]]:
        async def _undo() -> None:
            with contextlib.suppress(PromptNotFoundError):
                await self._prompt_repository.delete(prompt_id)

        return _undo

    def _undo_mcp(self, server_name: str) -> Callable[[], Awaitable[None]]:
        async def _undo() -> None:
            registry = self._mcp_registry
            if registry is None:
                return
            with contextlib.suppress(McpRegistryNotFoundError):
                await registry.delete_server(server_name)

        return _undo

    def _undo_store_file(self, path: str) -> Callable[[], Awaitable[None]]:
        async def _undo() -> None:
            await self._store_files.delete_file(path)  # idempotent port contract

        return _undo

    async def _compensate(self, agent_name: str, ledger: _CompensationLedger) -> None:
        """Undo every resource created during the failed import, in reverse order."""
        logger.info(LogMessage.AGENT_IMPORT_COMPENSATION_STARTED, agent_name)
        for kind, resource_name, undo in reversed(ledger.entries()):
            try:
                await undo()
                logger.info(LogMessage.AGENT_IMPORT_COMPENSATED, kind, resource_name, agent_name)
            except Exception as exc:  # noqa: BLE001 — best-effort: keep compensating
                logger.exception(LogMessage.AGENT_IMPORT_COMPENSATION_FAILED, kind, resource_name, agent_name, exc)


def _owned_mcp_ref_names(creations: list[McpMaterialization], owner_agent: str) -> list[str]:
    """The registry identities a sub-agent created under its own ownership."""
    return [creation.writer.name for creation in creations if creation.owner_agent == owner_agent]


def _parse_uuid(model_ref: str) -> uuid.UUID:
    """Validate a declared ``model_ref`` UUID (422 on a non-UUID value)."""
    try:
        return uuid.UUID(model_ref)
    except ValueError as e:
        raise ConfigValidationError(
            [{"loc": ["model_ref"], "msg": "model_ref must be a catalog model UUID", "type": "value_error"}]
        ) from e


def _skill_store_key(ref: str) -> str:
    """The canonical Store File API key of a textual skill reference."""
    ref = ref if ref.startswith("/") else f"/{ref}"
    return f"{ref.rstrip('/')}/SKILL.md"


def _memory_store_key(ref: str) -> str:
    """The canonical Store File API key of a memory reference."""
    return ref if ref.startswith("/") else f"/{ref}"


def _missing_reference_error(kind: str, ref: str):  # noqa: ANN201 — ConfigError
    template = (
        ErrorMessage.AGENT_IMPORT_SKILL_REF_MISSING if kind == "skill" else ErrorMessage.AGENT_IMPORT_MEMORY_REF_MISSING
    )
    return ConfigError(template.format(path=ref))


def _validated_config(target: dict) -> AgentConfig:
    """Validate a resource-only target (final schema gate)."""
    try:
        return AgentConfig.model_validate(target)
    except Exception as e:  # noqa: BLE001 — any validation payload is sanitized below
        if hasattr(e, "errors"):
            raise ConfigValidationError([dict(item) for item in e.errors()]) from e
        raise ConfigError(ErrorMessage.AGENT_IMPORT_TARGET_INVALID.format(reason=e)) from e


def _subagent_target(
    creation: SubagentMaterialization,
    model_id: str | None,
    prompt_id: str | None,
    mcp_refs: list[str],
) -> dict:
    """Build the resource-only YAML mapping of one CREATED sub-agent.

    Its MCP selection is EXACTLY the passed set: a sub-agent never inherits
    the parent's selection and never carries inline ``mcp_servers`` (empty
    when it selected nothing — REQ-MCP-003). The omitted ``debug`` field
    parses to the schema default (``False``).
    """
    entry = creation.entry
    optional_fields = {
        "model_ref": model_id,
        "prompt_ref": {"id": prompt_id} if prompt_id else None,
        "response_format": entry.get("response_format"),
        "skills": list(entry.get("skills") or []) or None,
        "mcp_refs": list(mcp_refs) or None,
    }
    target: dict[str, Any] = {
        "name": creation.entry_name,
        "description": entry["description"],
        "tools": entry.get("tools") or [],
    }
    target.update({field: value for field, value in optional_fields.items() if value})
    return target


def _agent_metadata(name: str, config: AgentConfig, model_metadata: ModelMetadata | None) -> AgentConfigMetadata:
    """Project an imported configuration into its database metadata row."""
    now = datetime.now(UTC)
    return AgentConfigMetadata(
        name=name,
        # Served-identifier display projection: the resolved model's
        # ``provider_model`` — materialized OR (I7) fetched from the catalog
        # reference; ``model_ref`` remains the source of truth.
        model=model_metadata.provider_model if model_metadata is not None else config.model,
        minio_path=f"{name}.yaml",
        description=config.description,
        created_at=now,
        updated_at=now,
        **registry_ref_metadata_fields(config),
    )
