import logging
from datetime import UTC, datetime

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

logger = logging.getLogger(__name__)


async def validate_registry_refs(
    config: AgentConfig,
    prompt_repository: PromptRepository | None,
    model_repository: ModelRepository | None,
) -> None:
    """Validate ``prompt_ref``/``model_ref`` against the local registries.

    Spec §5.5 AGT-002/AGT-003 + §9: missing, inaccessible (foreign), deleted or
    invalid references fail explicitly at save time — there is no fall back to
    inline prompts, Phoenix, the legacy ``model`` string or an agent-name
    lookup. A pinned version that does not exist fails without a Latest
    fallback. Cross-user references are non-enumerating not-founds because the
    repositories are owner-scoped.

    Args:
        config: The parsed agent configuration.
        prompt_repository: Local prompt registry port (when wired).
        model_repository: Model catalog port (when wired).

    Raises:
        PromptNotFoundError: Missing/inaccessible prompt.
        PromptVersionNotFoundError: Missing pinned version.
        ModelNotFoundError: Missing/inaccessible model.
    """
    if config.prompt_ref is not None:
        if prompt_repository is None:
            raise ConfigError(f"Agent '{config.name}' sets prompt_ref but no prompt repository is wired.")
        await prompt_repository.get(str(config.prompt_ref.id))
        if config.prompt_ref.version is not None:
            await prompt_repository.get_version(str(config.prompt_ref.id), config.prompt_ref.version)
    if config.model_ref is not None:
        if model_repository is None:
            raise ConfigError(f"Agent '{config.name}' sets model_ref but no model repository is wired.")
        await model_repository.get(str(config.model_ref))


def registry_ref_metadata_fields(config: AgentConfig) -> dict:
    """Project the YAML's registry references into metadata columns (SEC-002)."""
    return {
        "prompt_ref": str(config.prompt_ref.id) if config.prompt_ref else None,
        "model_ref": str(config.model_ref) if config.model_ref else None,
    }


class CreateAgentConfigUseCase:
    """Create a new agent configuration in persistent storage."""

    def __init__(
        self,
        config_loader: AgentConfigLoader,
        config_store: AgentConfigStore,
        config_repository: AgentConfigRepository,
        prompt_repository: PromptRepository | None = None,
        model_repository: ModelRepository | None = None,
    ) -> None:
        self._config_loader = config_loader
        self._config_store = config_store
        self._config_repository = config_repository
        self._prompt_repository = prompt_repository
        self._model_repository = model_repository

    async def execute(self, name: str, yaml_content: str) -> AgentConfig:
        """Parse YAML, validate, store in MinIO, save metadata in PostgreSQL.

        Args:
            name: Agent name.
            yaml_content: Raw YAML configuration string.

        Returns:
            Validated AgentConfig.

        Raises:
            AgentConfigAlreadyExistsError: If an agent with this name already exists.
            ConfigError: If the YAML is invalid.
            PromptNotFoundError / ModelNotFoundError: If prompt_ref/model_ref
                cannot be resolved for the owner (§9).
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

        logger.info(LogMessage.AGENT_CONFIG_CREATED_UC, name)
        return config
