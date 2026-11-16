"""Cron subsystem HTTP routes (spec §5.3 CRN-003).

CRUD, Run now and paginated run history. All endpoints are scoped to the
authenticated owner (RLS); a foreign or missing cron is a non-enumerating
404. Manual overlap returns an explicit 409 conflict (CRN-005); failure
summaries in run records are redacted and safe (SCHED-001).
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from src.application.requests.cron import CreateCronRequest, UpdateCronRequest
from src.application.responses.cron import CronResponse, CronRunResponse
from src.application.use_cases.cron.manage_crons import (
    CreateCronUseCase,
    DeleteCronUseCase,
    GetCronUseCase,
    ListCronsUseCase,
    UpdateCronUseCase,
)
from src.application.use_cases.cron.run_crons import ListCronRunsUseCase, RunCronNowUseCase
from src.dependencies import (
    get_create_cron_use_case,
    get_delete_cron_use_case,
    get_get_cron_use_case,
    get_list_cron_runs_use_case,
    get_list_crons_use_case,
    get_run_cron_now_use_case,
    get_update_cron_use_case,
)
from src.domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/crons", tags=["crons"])


@router.get("")
async def list_crons(
    use_case: Annotated[ListCronsUseCase, Depends(get_list_crons_use_case)],
) -> list[CronResponse]:
    """List owned crons with enabled state, next run and last-run summary."""
    crons = await use_case.execute()
    return [CronResponse.model_validate(cron) for cron in crons]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_cron(
    body: CreateCronRequest,
    use_case: Annotated[CreateCronUseCase, Depends(get_create_cron_use_case)],
) -> CronResponse:
    """Create and validate a cron.

    Raises:
        CronValidationError: Bad expression/timezone or an agent requiring
            user runtime placeholders.
        AgentNotFoundError: Missing or foreign agent.
    """
    logger.info(LogMessage.CRON_CREATING, body.name, body.agent_name)
    created = await use_case.execute(
        name=body.name,
        agent_name=body.agent_name,
        message=body.message,
        cron_expression=body.cron_expression,
        timezone=body.timezone,
        enabled=body.enabled,
    )
    return CronResponse.model_validate(created)


@router.get("/{cron_id}")
async def get_cron(
    cron_id: str,
    use_case: Annotated[GetCronUseCase, Depends(get_get_cron_use_case)],
) -> CronResponse:
    """Return the cron definition and scheduling state.

    Raises:
        CronNotFoundError: Missing or foreign cron (non-enumerating 404).
    """
    cron = await use_case.execute(cron_id)
    return CronResponse.model_validate(cron)


@router.patch("/{cron_id}")
async def update_cron(
    cron_id: str,
    body: UpdateCronRequest,
    use_case: Annotated[UpdateCronUseCase, Depends(get_update_cron_use_case)],
) -> CronResponse:
    """Edit fields (including ``enabled``); the frontier recalculates accordingly.

    Raises:
        CronNotFoundError: Missing or foreign cron.
        CronValidationError: Invalid edited fields.
    """
    updated = await use_case.execute(cron_id, body.model_dump(exclude_unset=True))
    return CronResponse.model_validate(updated)


@router.delete("/{cron_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_cron(
    cron_id: str,
    use_case: Annotated[DeleteCronUseCase, Depends(get_delete_cron_use_case)],
) -> None:
    """Delete the cron; future scheduling stops.

    Raises:
        CronNotFoundError: Missing or foreign cron.
    """
    await use_case.execute(cron_id)


@router.post("/{cron_id}/run", status_code=status.HTTP_202_ACCEPTED)
async def run_cron_now(
    cron_id: str,
    use_case: Annotated[RunCronNowUseCase, Depends(get_run_cron_now_use_case)],
) -> CronRunResponse:
    """Trigger Run now (permitted while disabled; still creates a cron thread).

    Raises:
        CronNotFoundError: Missing or foreign cron.
        CronRunConflictError: A run is already active (explicit conflict).
    """
    run = await use_case.execute(cron_id)
    return CronRunResponse.model_validate(run)


@router.get("/{cron_id}/runs")
async def list_cron_runs(
    cron_id: str,
    use_case: Annotated[ListCronRunsUseCase, Depends(get_list_cron_runs_use_case)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[CronRunResponse]:
    """Return paginated run history, newest first.

    ``limit`` is a bounded client input (1..200): out-of-range values are
    rejected with 422 by the parameter validation themselves — never propagated
    into SQL as an internal error (QA BUG-003), and the page can never pull the
    unbounded run history.

    Raises:
        CronNotFoundError: Missing or foreign cron.
    """
    runs = await use_case.execute(cron_id, limit)
    return [CronRunResponse.model_validate(run) for run in runs]
