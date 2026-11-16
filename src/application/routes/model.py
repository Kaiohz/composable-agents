"""Metadata-only model catalog HTTP routes (spec §5.2 MDL-003).

Responses expose exactly the feature-owned metadata fields — no credential
data exists anywhere in the catalog (MDL-002; unknown body fields are rejected
so supplied credential keys are never persisted). Ownership derives from the
authenticated request context; foreign ids are non-enumerating 404s.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, status

from src.application.requests.model import CreateModelRequest, UpdateModelRequest, _serialized_base_url
from src.application.use_cases.model.manage_models import (
    CreateModelUseCase,
    DeleteModelUseCase,
    GetModelUseCase,
    ListModelsUseCase,
    UpdateModelUseCase,
)
from src.dependencies import (
    get_create_model_use_case,
    get_delete_model_use_case,
    get_get_model_use_case,
    get_list_models_use_case,
    get_update_model_use_case,
)
from src.domain.entities.model_metadata import ModelMetadata
from src.domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("")
async def list_models(
    use_case: Annotated[ListModelsUseCase, Depends(get_list_models_use_case)],
) -> list[ModelMetadata]:
    """List owned model metadata (MDL-003)."""
    return await use_case.execute()


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_model(
    body: CreateModelRequest,
    use_case: Annotated[CreateModelUseCase, Depends(get_create_model_use_case)],
) -> ModelMetadata:
    """Create a metadata configuration (no credential fields accepted).

    Args:
        body: ``{name, provider, provider_model, base_url?}``.

    Returns:
        The created model metadata.
    """
    logger.info(LogMessage.MODEL_CREATING, body.name)
    return await use_case.execute(
        name=body.name,
        provider=body.provider,
        provider_model=body.provider_model,
        base_url=_serialized_base_url(body.base_url),
    )


@router.get("/{model_id}")
async def get_model(
    model_id: str,
    use_case: Annotated[GetModelUseCase, Depends(get_get_model_use_case)],
) -> ModelMetadata:
    """Read an owned model metadata configuration.

    Raises:
        ModelNotFoundError: Missing or foreign model (non-enumerating 404).
    """
    return await use_case.execute(model_id)


@router.patch("/{model_id}")
async def update_model(
    model_id: str,
    body: UpdateModelRequest,
    use_case: Annotated[UpdateModelUseCase, Depends(get_update_model_use_case)],
) -> ModelMetadata:
    """Edit metadata fields; referencing agents pick the change up next run.

    Raises:
        ModelNotFoundError: Missing or foreign model.
    """
    changes = body.model_dump(exclude_unset=True)
    if "base_url" in changes:
        changes["base_url"] = _serialized_base_url(changes["base_url"])
    return await use_case.execute(model_id, changes)


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str,
    use_case: Annotated[DeleteModelUseCase, Depends(get_delete_model_use_case)],
) -> None:
    """Delete an unreferenced model (409 Conflict while referenced).

    Raises:
        ModelNotFoundError: Missing or foreign model.
        ModelInUseError: Referenced by at least one agent.
    """
    await use_case.execute(model_id)
