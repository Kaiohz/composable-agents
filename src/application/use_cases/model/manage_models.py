"""Model catalog CRUD use cases (spec §B MODEL-001…008).

Models are identity-only: UUID, display name, ``provider_model`` identifier
and timestamps. No credential or Settings connection field exists and none are
accepted anywhere in this catalog (MODEL-009).
"""

import logging
from typing import Any

from src.domain.entities.model_metadata import ModelMetadata
from src.domain.logging.messages import LogMessage
from src.domain.ports.model_repository import ModelRepository

logger = logging.getLogger(__name__)


class CreateModelUseCase:
    """Create a user-owned model identity."""

    def __init__(self, model_repository: ModelRepository) -> None:
        self._model_repository = model_repository

    async def execute(
        self,
        name: str,
        provider_model: str,
    ) -> ModelMetadata:
        """Persist the identity configuration and return it (MODEL-002)."""
        model = ModelMetadata(name=name, provider_model=provider_model)
        created = await self._model_repository.create(model)
        logger.info(LogMessage.MODEL_CREATED, created.id)
        return created


class GetModelUseCase:
    """Read an owned model metadata configuration."""

    def __init__(self, model_repository: ModelRepository) -> None:
        self._model_repository = model_repository

    async def execute(self, model_id: str) -> ModelMetadata:
        """Return the metadata.

        Raises:
            ModelNotFoundError: Missing or foreign model (§9 non-enumerating).
        """
        model = await self._model_repository.get(model_id)
        logger.info(LogMessage.MODEL_FETCHED, model.id)
        return model


class ListModelsUseCase:
    """List all model identities owned by the current user."""

    def __init__(self, model_repository: ModelRepository) -> None:
        self._model_repository = model_repository

    async def execute(self) -> list[ModelMetadata]:
        return await self._model_repository.list_all()


class UpdateModelUseCase:
    """Edit the display name or the ``provider_model`` identifier.

    Only explicitly-provided fields are applied (``exclude_unset`` semantics),
    so an omitted field keeps its current value (MODEL-006).

    Referencing agents observe the edit on their next invocation.
    """

    def __init__(self, model_repository: ModelRepository) -> None:
        self._model_repository = model_repository

    async def execute(self, model_id: str, changes: dict[str, Any]) -> ModelMetadata:
        """Apply the identity edit and return the updated configuration.

        Raises:
            ModelNotFoundError: Missing or foreign model.
        """
        current = await self._model_repository.get(model_id)
        # Re-validate through the entity: ``model_copy(update=...)`` SKIPS the
        # Pydantic validators, so an unvalidated payload (e.g. an LLM-driven
        # tool surface) could otherwise persist a blank name/provider_model
        # (MODEL-002/004: non-empty after trim is a domain invariant).
        updated = ModelMetadata.model_validate({**current.model_dump(), **changes})
        saved = await self._model_repository.update(updated)
        logger.info(LogMessage.MODEL_UPDATED, saved.id)
        return saved


class DeleteModelUseCase:
    """Delete a model when unreferenced (409 Conflict otherwise)."""

    def __init__(self, model_repository: ModelRepository) -> None:
        self._model_repository = model_repository

    async def execute(self, model_id: str) -> None:
        """Delete the identity row.

        Raises:
            ModelNotFoundError: Missing or foreign model.
            ModelInUseError: Referenced by at least one agent (MODEL-008).
        """
        await self._model_repository.delete(model_id)
        logger.info(LogMessage.MODEL_DELETED, model_id)
