import logging
from datetime import UTC, datetime

import yaml  # type: ignore[import-untyped]

from src.application.services.agent_import_service import AgentImportService
from src.application.use_cases._subagent_ref_utils import validate_subagent_refs
from src.domain.entities.agent_config import AgentConfig
from src.domain.entities.agent_config_metadata import AgentConfigMetadata
from src.domain.errors.agent import AgentConfigAlreadyExistsError
from src.domain.errors.config import ConfigError
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_config_loader import AgentConfigLoader
from src.domain.ports.agent_config_repository import AgentConfigRepository
from src.domain.ports.agent_config_store import AgentConfigStore
from src.domain.ports.model_repository import ModelRepository
from src.domain.ports.prompt_repository import PromptRepository
from src.domain.services.agent_rag_spaces import AgentRagSpaces
from src.domain.services.agent_registry_refs import (
    registry_ref_metadata_fields,
    validate_registry_refs,
)

logger = logging.getLogger(__name__)


class CreateAgentConfigUseCase:
    """Create a new agent configuration in persistent storage.

    When the agent-import materialization service is wired (production
    composition), the creation IS the import: explicitly declared inline
    resources are materialized in their registries and the persisted agent is
    resource-only (spec agent-resource-import). Without the service the
    legacy parse/validate/persist flow applies (degraded wiring and unit
    seams).
    """

    def __init__(
        self,
        config_loader: AgentConfigLoader,
        config_store: AgentConfigStore,
        config_repository: AgentConfigRepository,
        prompt_repository: PromptRepository | None = None,
        model_repository: ModelRepository | None = None,
        import_service: AgentImportService | None = None,
        rag_spaces: AgentRagSpaces | None = None,
    ) -> None:
        self._config_loader = config_loader
        self._config_store = config_store
        self._config_repository = config_repository
        self._prompt_repository = prompt_repository
        self._model_repository = model_repository
        self._import_service = import_service
        self._rag_spaces = rag_spaces

    async def execute(self, name: str, yaml_content: str) -> AgentConfig:
        """Parse YAML, validate, materialize declared resources and persist.

        REQ-018: the agent's RAG space AND every subagent's space are
        idempotently provisioned through the spaces API with the creating
        user's credential — BEST-EFFORT: an unreachable or rejecting spaces
        API only warns and never fails the creation (the space is created
        lazily on first use inside mcp-raganything). The scoping hooks run at
        this level so BOTH creation flows are covered: the imported
        materialization (production wiring) and the legacy
        parse/validate/persist flow (degraded wiring and unit seams) — a
        topology that collides with another agent's space is rejected BEFORE
        anything is persisted, and the provisioning is scheduled only after
        the agent exists.

        Args:
            name: Agent name.
            yaml_content: Raw YAML configuration string.

        Returns:
            The persisted AgentConfig — resource-only when the import
            materialization service is wired.

        Raises:
            AgentConfigAlreadyExistsError: If an agent with this name already exists.
            ConfigError: If the YAML is invalid.
            PromptNotFoundError / ModelNotFoundError: If prompt_ref/model_ref
                cannot be resolved for the owner (§9).
        """
        await self._validate_space_topology(name, yaml_content)
        config = await self._materialize_or_legacy_create(name, yaml_content)
        self._schedule_space_provisioning(config)
        logger.info(LogMessage.AGENT_CONFIG_CREATED_UC, name)
        return config

    async def _validate_space_topology(self, name: str, yaml_content: str) -> None:
        """Fail-fast C-2 ownership guard over the RAW import topology.

        C-2 (option a): reject a topology whose mapped slugs collide with
        another existing agent's slug — shared slugs mean shared RAG spaces.
        Fail-fast BEFORE any flow persists anything. The topology is read
        from the RAW mapping (NOT pydantic): the import's inline
        file-definitions only normalize INSIDE the import flow, so a
        preliminary pydantic load would reject valid imports (main PR #47's
        normalized-target contract).
        """
        if self._rag_spaces is None:
            return
        topology = self._raw_topology(yaml_content)
        if topology is not None:
            yaml_name, subagent_names = topology
            await self._rag_spaces.validate_space_ownership(
                yaml_name if isinstance(yaml_name, str) and yaml_name else name, subagent_names
            )

    async def _materialize_or_legacy_create(self, name: str, yaml_content: str) -> AgentConfig:
        """Route the creation through the wired flow (import or legacy)."""
        if self._import_service is not None:
            return await self._import_service.import_agent(name, yaml_content)
        return await self._legacy_create(name, yaml_content)

    def _schedule_space_provisioning(self, config: AgentConfig) -> None:
        """Schedule the REQ-018 + review C-3 provisioning for ONE created agent.

        Fire-and-forget background provisioning (the task captures the
        caller's contextvars — credential and store namespace remain the
        originating user's); only created AFTER the agent exists.
        """
        if self._rag_spaces is None:
            return
        self._rag_spaces.schedule_provision_for_agent(config.name, [subagent.name for subagent in config.subagents])

    def _raw_topology(self, yaml_content: str) -> tuple[object, list[str]] | None:
        """Best-effort raw read of one import yaml's topology for the guard.

        Returns ``(yaml_name, subagent_names)`` or ``None`` when the document
        is not a mapping (the import itself will produce the proper error).
        Deliberately NOT pydantic: the raw definitions normalize only inside
        the import flow.
        """
        try:
            raw = yaml.safe_load(yaml_content)
        except yaml.YAMLError:
            return None
        if not isinstance(raw, dict):
            return None
        subagent_names = [
            sub["name"]
            for sub in (raw.get("subagents") or [])
            if isinstance(sub, dict) and isinstance(sub.get("name"), str)
        ]
        return raw.get("name"), subagent_names

    async def _legacy_create(self, name: str, yaml_content: str) -> AgentConfig:
        """Parse, validate then store the YAML as-is (pre-materialization flow).

        The RAG-space ownership validation and provisioning run at the
        :meth:`execute` level (shared with the import flow).
        """
        config = self._config_loader.load_from_string(yaml_content)

        if config.name != name:
            raise ConfigError(ErrorMessage.AGENT_NAME_MISMATCH.format(yaml_name=config.name, name=name))

        await validate_subagent_refs(config, self._config_repository)
        await validate_registry_refs(config, self._prompt_repository, self._model_repository)

        if await self._config_repository.exists(name):
            raise AgentConfigAlreadyExistsError(ErrorMessage.AGENT_CONFIG_ALREADY_EXISTS.format(name=name))

        await self._config_store.put(name, yaml_content)

        now = datetime.now(UTC)
        metadata = AgentConfigMetadata(
            name=name,
            model=config.model,
            minio_path=f"{name}.yaml",
            description=config.description,
            created_at=now,
            updated_at=now,
            **registry_ref_metadata_fields(config),
        )
        await self._config_repository.save(metadata)
        return config
