import logging
from datetime import UTC, datetime

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
    ) -> None:
        self._config_loader = config_loader
        self._config_store = config_store
        self._config_repository = config_repository
        self._prompt_repository = prompt_repository
        self._model_repository = model_repository
        self._import_service = import_service

    async def execute(self, name: str, yaml_content: str) -> AgentConfig:
        """Parse YAML, validate, materialize declared resources and persist.

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
        if self._import_service is not None:
            return await self._import_service.import_agent(name, yaml_content)
        return await self._legacy_create(name, yaml_content)

    async def _legacy_create(self, name: str, yaml_content: str) -> AgentConfig:
        """Parse, validate then store the YAML as-is (pre-materialization flow)."""
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

        logger.info(LogMessage.AGENT_CONFIG_CREATED_UC, name)
        return config
