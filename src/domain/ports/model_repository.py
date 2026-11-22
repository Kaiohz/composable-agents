"""Outbound port: ModelRepository (identity-only model catalog, spec §B).

Implementations are per-user (``current_user_id`` contextvar + RLS). Models
carry no credential and no Settings connection field; the port surface mirrors
the CRUD routes from MODEL-001…008.
"""

from abc import ABC, abstractmethod

from src.domain.entities.model_metadata import ModelMetadata


class ModelRepository(ABC):
    """Outbound port for the per-user identity-only model catalog."""

    @abstractmethod
    async def create(self, model: ModelMetadata) -> ModelMetadata:
        """Create a model identity owned by the current user."""
        ...

    @abstractmethod
    async def get(self, model_id: str) -> ModelMetadata:
        """Return the model identity for ``model_id``.

        Raises:
            ModelNotFoundError: If the model is missing or foreign (§9).
        """
        ...

    @abstractmethod
    async def list_all(self) -> list[ModelMetadata]:
        """Return all model identities owned by the current user."""
        ...

    @abstractmethod
    async def update(self, model: ModelMetadata) -> ModelMetadata:
        """Persist editable identity changes (name/provider_model).

        Referencing agents observe the edit on their next invocation
        (CACHE-002 fingerprint).
        """
        ...

    @abstractmethod
    async def delete(self, model_id: str) -> None:
        """Delete the model identity row.

        Raises:
            ModelNotFoundError: If the model is missing or foreign.
            ModelInUseError: If any agent references the model (MODEL-008).
        """
        ...
