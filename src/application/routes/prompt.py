"""Local prompt registry HTTP routes (spec §5.1 PR-004).

Prompts live entirely in local PostgreSQL — no route calls Phoenix (PR-001).
Ownership comes from the authenticated request context (dual-auth sets the
RLS contextvars); inaccessible resources use the non-enumerating not-found
behavior (PR-005).
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, status

from src.application.requests.prompt import CreatePromptRequest, CreatePromptVersionRequest, UpdatePromptRequest
from src.application.responses.prompt import PromptResponse, PromptVersionResponse, PromptVersionSummary
from src.application.use_cases.prompt.create_prompt import CreatePromptUseCase
from src.application.use_cases.prompt.manage_prompt_versions import (
    AddPromptVersionUseCase,
    GetPromptVersionUseCase,
    ListPromptVersionsUseCase,
)
from src.application.use_cases.prompt.manage_prompts import (
    DeletePromptUseCase,
    GetPromptUseCase,
    ListPromptsUseCase,
    UpdatePromptUseCase,
)
from src.dependencies import (
    get_add_prompt_version_use_case,
    get_create_prompt_use_case,
    get_delete_prompt_use_case,
    get_get_prompt_use_case,
    get_get_prompt_version_use_case,
    get_list_prompt_versions_use_case,
    get_list_prompts_use_case,
    get_update_prompt_use_case,
)
from src.domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/prompts", tags=["prompts"])


@router.get("")
async def list_prompts(
    use_case: Annotated[ListPromptsUseCase, Depends(get_list_prompts_use_case)],
) -> list[PromptResponse]:
    """List owned prompt metadata (PR-004).

    Returns:
        All prompts owned by the authenticated user.
    """
    prompts = await use_case.execute()
    return [PromptResponse.model_validate(prompt) for prompt in prompts]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_prompt(
    body: CreatePromptRequest,
    use_case: Annotated[CreatePromptUseCase, Depends(get_create_prompt_use_case)],
) -> PromptResponse:
    """Create prompt metadata and immutable version 1 (PR-003).

    Args:
        body: ``{name, description?, content}`` — content stored verbatim.

    Returns:
        The created prompt with ``latest_version == 1``.
    """
    logger.info(LogMessage.PROMPT_REGISTRY_CREATING, body.name)
    created = await use_case.execute(name=body.name, description=body.description, content=body.content)
    return PromptResponse.model_validate(created)


@router.get("/{prompt_id}")
async def get_prompt(
    prompt_id: str,
    use_case: Annotated[GetPromptUseCase, Depends(get_get_prompt_use_case)],
) -> PromptResponse:
    """Return an owned prompt's metadata.

    Raises:
        PromptNotFoundError: Missing or foreign prompt (non-enumerating 404).
    """
    prompt = await use_case.execute(prompt_id)
    return PromptResponse.model_validate(prompt)


@router.patch("/{prompt_id}")
async def update_prompt(
    prompt_id: str,
    body: UpdatePromptRequest,
    use_case: Annotated[UpdatePromptUseCase, Depends(get_update_prompt_use_case)],
) -> PromptResponse:
    """Edit name and/or description (metadata-only, no version created).

    Raises:
        PromptNotFoundError: Missing or foreign prompt.
    """
    updated = await use_case.execute(prompt_id, body.model_dump(exclude_unset=True))
    return PromptResponse.model_validate(updated)


@router.delete("/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prompt(
    prompt_id: str,
    use_case: Annotated[DeletePromptUseCase, Depends(get_delete_prompt_use_case)],
) -> None:
    """Delete an unreferenced prompt (409 Conflict while referenced).

    Raises:
        PromptNotFoundError: Missing or foreign prompt.
        PromptInUseError: Referenced by an agent (Latest or pinned).
    """
    await use_case.execute(prompt_id)


@router.get("/{prompt_id}/versions")
async def list_prompt_versions(
    prompt_id: str,
    use_case: Annotated[ListPromptVersionsUseCase, Depends(get_list_prompt_versions_use_case)],
) -> list[PromptVersionSummary]:
    """List version numbers and creation timestamps, numerically ordered.

    Returns summary projections only (QA BUG-004): the immutable content of a
    version is fetched through ``GET .../versions/{version}``.

    Raises:
        PromptNotFoundError: Missing or foreign prompt.
    """
    return [
        PromptVersionSummary.model_validate(summary) for summary in await use_case.execute(prompt_id)
    ]


@router.post("/{prompt_id}/versions", status_code=status.HTTP_201_CREATED)
async def add_prompt_version(
    prompt_id: str,
    body: CreatePromptVersionRequest,
    use_case: Annotated[AddPromptVersionUseCase, Depends(get_add_prompt_version_use_case)],
) -> PromptVersionSummary:
    """Create the next immutable version (transactional, concurrency-safe).

    Raises:
        PromptNotFoundError: Missing or foreign prompt.
    """
    version = await use_case.execute(prompt_id, body.content)
    return PromptVersionSummary.model_validate(version)


@router.get("/{prompt_id}/versions/{version}")
async def get_prompt_version(
    prompt_id: str,
    version: int,
    use_case: Annotated[GetPromptVersionUseCase, Depends(get_get_prompt_version_use_case)],
) -> PromptVersionResponse:
    """Return one immutable version's full verbatim content.

    Raises:
        PromptNotFoundError: Missing or foreign prompt.
        PromptVersionNotFoundError: The version does not exist (no Latest
            fallback for pinned reads, §9).
    """
    version_row = await use_case.execute(prompt_id, version)
    return PromptVersionResponse.model_validate(version_row)
