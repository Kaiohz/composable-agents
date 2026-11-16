"""Outbound port: ModelRepository (metadata-only model catalog, spec §5.2).

Implementations are per-user (``current_user_id`` contextvar + RLS). Models
carry no credential fields; the port surface mirrors the CRUD routes from
MDL-003.
"""

from abc import ABC, abstractmethod

from src.domain.entities.model_metadata import ModelMetadata


class ModelRepository(ABC):
    """Outbound port for the per-user metadata-only model catalog."""

    @abstractmethod
    async def create(self, model: ModelMetadata) -> ModelMetadata:
        """Create a model metadata configuration owned by the current user."""
        ...

    @abstractmethod
    async def get(self, model_id: str) -> ModelMetadata:
        """Return the model metadata for ``model_id``.

        Raises:
            ModelNotFoundError: If the model is missing or foreign (§9).
        """
        ...

    @abstractmethod
    async def list_all(self) -> list[ModelMetadata]:
        """Return all model metadata owned by the current user."""
        ...

    @abstractmethod
    async def update(self, model: ModelMetadata) -> ModelMetadata:
        """Persist editable metadata changes (name/provider/provider_model/base_url).

        Referencing agents observe the edit on their next invocation
        (CACHE-002 fingerprint).
        """
        ...

    @abstractmethod
    async def delete(self, model_id: str) -> None:
        """Delete the model metadata row.

        Raises:
            ModelNotFoundError: If the model is missing or foreign.
            ModelInUseError: If any agent references the model (MDL-004).
        """
        ...
