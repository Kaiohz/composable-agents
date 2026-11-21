"""Effective resolution and per-user customization of the built-in assistant.

The default built-in definition stays virtual and code-owned. This service
adds ONE user-scoped snapshot on top of it:

* no override → the default definition is served unchanged;
* override present → the snapshot is parsed and the platform invariants are
  re-applied before it is exposed or executed.

Invariants are enforced at write time AND at read time (defense in depth):
the name stays reserved, ``is_builtin`` is computed, the dynamic toolset
fields stay empty, and the destructive-delete HITL floor remains enabled.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from src.application.services.builtin_agent import (
    BUILTIN_AGENT_NAME,
    BUILTIN_HITL_DELETE_TOOLS,
    builtin_agent_config,
    builtin_agent_metadata,
    is_builtin_agent_name,
)
from src.application.use_cases.create_agent_config import (
    registry_ref_metadata_fields,
    validate_registry_refs,
)
from src.domain.entities.agent_config import AgentConfig, HITLConfig
from src.domain.entities.agent_config_metadata import AgentConfigMetadata
from src.domain.entities.builtin_agent_override import BuiltinAgentOverride
from src.domain.errors.config import ConfigError, ConfigValidationError
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_config_loader import AgentConfigLoader
from src.domain.ports.builtin_agent_override_store import BuiltinAgentOverrideStore
from src.domain.ports.model_repository import ModelRepository
from src.domain.ports.prompt_repository import PromptRepository
from src.infrastructure.database.rls_context import current_user_id

logger = logging.getLogger(__name__)

_OVERRIDE_SOURCE = "built-in assistant customization"
_FORBIDDEN_DYNAMIC_FIELDS = ("tools", "subagents", "mcp_servers")


class EffectiveBuiltinAgentService:
    """Resolve, validate and reset the current user's built-in customization."""

    def __init__(
        self,
        config_loader: AgentConfigLoader,
        override_store: BuiltinAgentOverrideStore,
        prompt_repository: PromptRepository | None = None,
        model_repository: ModelRepository | None = None,
    ) -> None:
        self._config_loader = config_loader
        self._override_store = override_store
        self._prompt_repository = prompt_repository
        self._model_repository = model_repository

    async def get_effective_config(self) -> AgentConfig:
        """Return the effective built-in config for the current user."""
        override = await self._current_override()
        if override is None:
            config = builtin_agent_config()
            logger.info(LogMessage.BUILTIN_AGENT_RESOLVED, self._user_label(), False)
            return config

        effective = self._load_override(override)
        logger.info(LogMessage.BUILTIN_AGENT_RESOLVED, self._user_label(), True)
        return effective

    async def get_effective_metadata(self) -> AgentConfigMetadata:
        """Return the effective built-in metadata for the current user."""
        override = await self._current_override()
        if override is None:
            return builtin_agent_metadata()

        config = self._load_override(override)
        return AgentConfigMetadata(
            name=BUILTIN_AGENT_NAME,
            model=config.model,
            minio_path="",
            description=config.description,
            created_at=override.created_at,
            updated_at=override.updated_at,
            **registry_ref_metadata_fields(config),
            is_builtin=True,
            is_customized=True,
        )

    async def update(self, name: str, yaml_content: str) -> AgentConfig:
        """Validate and replace the current user's built-in override."""
        if not is_builtin_agent_name(name):
            raise ConfigError(ErrorMessage.AGENT_NAME_MISMATCH_URL.format(yaml_name=name, name=BUILTIN_AGENT_NAME))

        config = self._config_loader.load_from_string(yaml_content, _OVERRIDE_SOURCE)
        if config.name != BUILTIN_AGENT_NAME:
            raise ConfigError(
                ErrorMessage.AGENT_NAME_MISMATCH_URL.format(yaml_name=config.name, name=BUILTIN_AGENT_NAME)
            )
        self._validate_customizable(config)
        await validate_registry_refs(config, self._prompt_repository, self._model_repository)

        user_id = self._require_user_id()
        now = datetime.now(UTC)
        logger.info(LogMessage.BUILTIN_CUSTOMIZATION_UPDATING, user_id)
        await self._override_store.put(
            BuiltinAgentOverride(
                user_id=user_id,
                agent_name=BUILTIN_AGENT_NAME,
                yaml=yaml_content,
                created_at=now,
                updated_at=now,
            )
        )
        logger.info(LogMessage.BUILTIN_CUSTOMIZATION_UPDATED, user_id)
        return self._apply_invariants(config)

    async def reset(self) -> None:
        """Delete the current user's override without loading its YAML."""
        user_id = self._require_user_id()
        logger.info(LogMessage.BUILTIN_CUSTOMIZATION_RESETTING, user_id)
        await self._override_store.delete(BUILTIN_AGENT_NAME)
        logger.info(LogMessage.BUILTIN_CUSTOMIZATION_RESET, user_id)

    def _load_override(self, override: BuiltinAgentOverride) -> AgentConfig:
        """Parse one stored snapshot and re-apply the platform invariants."""
        config = self._config_loader.load_from_string(override.yaml, _OVERRIDE_SOURCE)
        return self._apply_invariants(config)

    async def _current_override(self) -> BuiltinAgentOverride | None:
        if current_user_id.get() is None:
            return None
        return await self._override_store.get(BUILTIN_AGENT_NAME)

    def _require_user_id(self) -> str:
        user_id = current_user_id.get()
        if user_id is None:
            raise ConfigError(ErrorMessage.BUILTIN_CUSTOMIZATION_USER_REQUIRED)
        return user_id

    @staticmethod
    def _user_label() -> str:
        return current_user_id.get() or "system"

    def _validate_customizable(self, config: AgentConfig) -> None:
        errors: list[dict[str, Any]] = []
        for field in _FORBIDDEN_DYNAMIC_FIELDS:
            if getattr(config, field):
                errors.append(
                    {
                        "loc": ["body", field],
                        "msg": ErrorMessage.BUILTIN_CUSTOMIZATION_FIELD_FORBIDDEN.format(field=field),
                    }
                )
        for tool in BUILTIN_HITL_DELETE_TOOLS:
            rule = config.hitl.rules.get(tool)
            if rule is not None and rule is not True:
                errors.append(
                    {
                        "loc": ["body", "hitl", "rules", tool],
                        "msg": ErrorMessage.BUILTIN_CUSTOMIZATION_HITL_FLOOR.format(tool=tool),
                    }
                )
        if errors:
            raise ConfigValidationError(errors)

    @staticmethod
    def _apply_invariants(config: AgentConfig) -> AgentConfig:
        """Force the platform-managed built-in fields on every read."""
        forced_rules = dict.fromkeys(BUILTIN_HITL_DELETE_TOOLS, True)
        return config.model_copy(
            update={
                "name": BUILTIN_AGENT_NAME,
                "tools": [],
                "subagents": [],
                "mcp_servers": [],
                "hitl": HITLConfig(rules={**config.hitl.rules, **forced_rules}),
                "is_builtin": True,
                "is_customized": True,
            }
        )
