import logging
from datetime import UTC, datetime

import yaml  # type: ignore[import-untyped]

from src.application.use_cases._subagent_ref_utils import invalidate_dependent_agents, validate_subagent_refs
from src.domain.entities.agent_config import AgentConfig
from src.domain.errors.config import ConfigError
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_config_loader import AgentConfigLoader
from src.domain.ports.agent_config_repository import AgentConfigRepository
from src.domain.ports.agent_config_store import AgentConfigStore
from src.domain.ports.agent_registry import AgentRegistry
from src.domain.ports.mcp_registry import McpRegistryClient
from src.domain.ports.model_repository import ModelRepository
from src.domain.ports.prompt_repository import PromptRepository
from src.domain.services.agent_rag_spaces import AgentRagSpaces, safe_reason
from src.domain.services.agent_registry_refs import registry_ref_metadata_fields, validate_registry_refs
from src.domain.services.agent_resource_write_policy import ensure_resource_only_update
from src.domain.services.mcp_ref_policy import ensure_mcp_refs_registered

logger = logging.getLogger(__name__)


class UpdateAgentConfigUseCase:
    """Update an existing agent configuration in persistent storage."""

    def __init__(
        self,
        config_loader: AgentConfigLoader,
        config_store: AgentConfigStore,
        config_repository: AgentConfigRepository,
        agent_registry: AgentRegistry,
        prompt_repository: PromptRepository | None = None,
        model_repository: ModelRepository | None = None,
        mcp_registry: McpRegistryClient | None = None,
        rag_spaces: AgentRagSpaces | None = None,
    ) -> None:
        self._config_loader = config_loader
        self._config_store = config_store
        self._config_repository = config_repository
        self._agent_registry = agent_registry
        self._prompt_repository = prompt_repository
        self._model_repository = model_repository
        self._mcp_registry = mcp_registry
        self._rag_spaces = rag_spaces

    async def execute(self, name: str, yaml_content: str) -> AgentConfig:
        """Validate, update in MinIO and PostgreSQL, invalidate cache.

        The registry references are revalidated at save time so switching to a
        removed prompt/model or a foreign id fails explicitly (AGT-002/003),
        and the projections keep the in-use integrity mechanism (SEC-002) and
        the freshness fingerprints (CACHE-002) up to date.

        NEW inline definitions are refused on the update boundary
        (REQ-LEGACY-003): materialization applies to creation only, so an
        update YAML introducing an inline model/prompt/MCP definition fails
        explicitly and the stored configuration stays untouched.

        REQ-018: the SUBAGENT spaces are diffed — the spaces of removed
        subagents are deprovisioned and the spaces of added ones provisioned
        (the agent's own space stays intact), best-effort with warnings; the
        new topology's mapped slugs must not collide with another existing
        agent's slug (fail-fast before any write).

        Args:
            name: Agent name to update.
            yaml_content: New raw YAML configuration string.

        Returns:
            Validated AgentConfig.

        Raises:
            AgentNotFoundError: If no agent with this name exists.
            ConfigError: If the name in YAML does not match.
            PromptNotFoundError / ModelNotFoundError: If prompt_ref/model_ref
                cannot be resolved for the owner (§9).
        """
        metadata = await self._config_repository.get(name)

        raw = self._parse_raw_yaml(yaml_content)
        if isinstance(raw, dict):
            try:
                ensure_resource_only_update(raw, metadata)
            except ConfigError as exc:
                logger.info(LogMessage.AGENT_IMPORT_UPDATE_REFUSED, name, exc.detail)
                raise

        config = self._config_loader.load_from_string(yaml_content)

        if config.name != name:
            raise ConfigError(ErrorMessage.AGENT_NAME_MISMATCH_URL.format(yaml_name=config.name, name=name))

        if config.mcp_refs:
            await self._validate_mcp_refs(config)
        await validate_subagent_refs(config, self._config_repository)
        await validate_registry_refs(config, self._prompt_repository, self._model_repository)

        if self._rag_spaces is not None:
            # C-2 (option a): the NEW topology's mapped slugs must not collide
            # with another existing agent's slug — fail fast before any write.
            await self._rag_spaces.validate_space_ownership(
                config.name, [subagent.name for subagent in config.subagents]
            )

        previous_subagent_names = await self._previous_subagent_names(name)

        await self._config_store.put(name, yaml_content)

        now = datetime.now(UTC)
        updated_metadata = metadata.model_copy(
            update={
                "model": config.model,
                "description": config.description,
                **registry_ref_metadata_fields(config),
                "updated_at": now,
            },
        )
        await self._config_repository.save(updated_metadata)

        await self._agent_registry.invalidate(name)
        await invalidate_dependent_agents(self._config_store, self._agent_registry, name)

        if self._rag_spaces is not None:
            # REQ-018 + review C-3: fire-and-forget background diff.
            self._rag_spaces.schedule_subagent_space_diff(
                name,
                previous_subagent_names,
                [subagent.name for subagent in config.subagents],
            )

        logger.info(LogMessage.AGENT_CONFIG_UPDATED_UC, name)
        return config

    @staticmethod
    def _parse_raw_yaml(yaml_content: str):
        """Best-effort raw parse for the resource-only policy gate."""
        try:
            return yaml.safe_load(yaml_content)
        except yaml.YAMLError:
            return None

    async def _validate_mcp_refs(self, config: AgentConfig) -> None:
        """Validate the update's MCP selection against the user's registry."""
        if self._mcp_registry is None:
            raise ConfigError(ErrorMessage.AGENT_UPDATE_REGISTRY_NOT_WIRED.format(name=config.name))
        await ensure_mcp_refs_registered(self._mcp_registry, config.mcp_refs)

    async def _previous_subagent_names(self, name: str) -> list[str]:
        """Read the subagent names of the CURRENTLY stored YAML (diff basis).

        Best-effort: an unreadable stored YAML yields an empty list, which
        degrades the update diff to provision-only instead of failing the
        update (the provisioning itself stays best-effort anyway).

        Args:
            name: The agent name.

        Returns:
            The stored configuration's subagent names (empty on failure).
        """
        try:
            previous_yaml = await self._config_store.get(name)
        except Exception as exc:  # noqa: BLE001
            # The diff basis is best-effort only: an unreadable stored YAML degrades the diff to provision-only.
            logger.debug(LogMessage.RAG_SPACE_DIFF_BASIS_UNAVAILABLE, name, safe_reason(exc))
            return []
        previous_config = self._config_loader.load_from_string(previous_yaml)
        return [subagent.name for subagent in previous_config.subagents]
