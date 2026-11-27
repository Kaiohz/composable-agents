"""Model catalog HTTP routes (spec §B MODEL-001…006, provider restored).

A Model identity owns its serving ``provider`` (catalog identity derived from
the declared model string's prefix, platform default ``openai``), its display
``name`` and the ``provider_model`` identifier; the provider CONNECTION
(``base_url``/``api_key``) remains Settings-owned (MODEL-009; unknown body
fields are rejected so no Settings or credential data can be smuggled in).
Ownership derives from the authenticated request context; foreign ids are
non-enumerating 404s.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, status

from src.application.requests.model import CreateModelRequest, UpdateModelRequest
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
    """List owned model identities (MODEL-003)."""
    return await use_case.execute()


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_model(
    body: CreateModelRequest,
    use_case: Annotated[CreateModelUseCase, Depends(get_create_model_use_case)],
) -> ModelMetadata:
    """Create a model identity (connection fields are Settings-owned).

    Args:
        body: ``{name, provider? (default ``openai``), provider_model}``.

    Returns:
        The created model identity.
    """
    logger.info(LogMessage.MODEL_CREATING, body.name)
    return await use_case.execute(
        name=body.name,
        provider=body.provider,
        provider_model=body.provider_model,
    )


@router.get("/{model_id}")
async def get_model(
    model_id: str,
    use_case: Annotated[GetModelUseCase, Depends(get_get_model_use_case)],
) -> ModelMetadata:
    """Read an owned model identity configuration.

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
    """Edit the model identity; referencing agents pick the change up next run.

    Raises:
        ModelNotFoundError: Missing or foreign model.
    """
    return await use_case.execute(model_id, body.model_dump(exclude_unset=True))


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
