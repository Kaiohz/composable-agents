"""Model catalog CRUD use cases (spec §5.2 MDL-001…MDL-004).

Models are metadata-only: UUID, display name, provider, provider model
identifier, optional base URL and timestamps. No credential fields exist and
none are accepted anywhere in this catalog (MDL-002).
"""

import logging
from typing import Any

from src.domain.entities.model_metadata import ModelMetadata
from src.domain.logging.messages import LogMessage
from src.domain.ports.model_repository import ModelRepository

logger = logging.getLogger(__name__)


class CreateModelUseCase:
    """Create a user-owned model metadata configuration."""

    def __init__(self, model_repository: ModelRepository) -> None:
        self._model_repository = model_repository

    async def execute(
        self,
        name: str,
        provider: str,
        provider_model: str,
        base_url: str | None = None,
    ) -> ModelMetadata:
        """Persist the metadata configuration and return it (MDL-001)."""
        model = ModelMetadata(name=name, provider=provider, provider_model=provider_model, base_url=base_url)
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
    """List all model metadata owned by the current user."""

    def __init__(self, model_repository: ModelRepository) -> None:
        self._model_repository = model_repository

    async def execute(self) -> list[ModelMetadata]:
        return await self._model_repository.list_all()


class UpdateModelUseCase:
    """Edit provider, provider model identifier, base URL or display name.

    Only explicitly-provided fields are applied (``exclude_unset`` semantics),
    so an omitted ``base_url`` leaves it untouched while an explicit ``null``
    clears it.
    """

    def __init__(self, model_repository: ModelRepository) -> None:
        self._model_repository = model_repository

    async def execute(self, model_id: str, changes: dict[str, Any]) -> ModelMetadata:
        """Apply the metadata edit and return the updated configuration.

        Referencing agents observe the edit on their next invocation
        (CACHE-002 fingerprint).

        Raises:
            ModelNotFoundError: Missing or foreign model.
        """
        current = await self._model_repository.get(model_id)
        updated = current.model_copy(update=changes)
        saved = await self._model_repository.update(updated)
        logger.info(LogMessage.MODEL_UPDATED, saved.id)
        return saved


class DeleteModelUseCase:
    """Delete a model when unreferenced (409 Conflict otherwise)."""

    def __init__(self, model_repository: ModelRepository) -> None:
        self._model_repository = model_repository

    async def execute(self, model_id: str) -> None:
        """Delete the metadata row.

        Raises:
            ModelNotFoundError: Missing or foreign model.
            ModelInUseError: Referenced by at least one agent (MDL-004).
        """
        await self._model_repository.delete(model_id)
        logger.info(LogMessage.MODEL_DELETED, model_id)
