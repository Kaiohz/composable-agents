import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from src.application.services.builtin_agent import (
    ensure_custom_agent_name,
    ensure_not_builtin,
    is_builtin_agent_name,
)
from src.application.services.route_validation import read_yaml_bytes, validate_agent_name
from src.application.use_cases.create_agent_config import CreateAgentConfigUseCase
from src.application.use_cases.delete_agent_config import DeleteAgentConfigUseCase
from src.application.use_cases.get_agent_config import GetAgentConfigUseCase
from src.application.use_cases.list_agent_configs import ListAgentConfigsUseCase
from src.application.use_cases.update_agent_config import UpdateAgentConfigUseCase
from src.dependencies import (
    get_create_agent_config_use_case,
    get_delete_agent_config_use_case,
    get_effective_builtin_agent_service,
    get_get_agent_config_use_case,
    get_list_agent_configs_use_case,
    get_update_agent_config_use_case,
)
from src.domain.entities.agent_config import AgentConfig
from src.domain.entities.agent_config_metadata import AgentConfigMetadata
from src.domain.errors.agent import AgentNotFoundError
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])


async def _read_yaml_upload(file: UploadFile) -> str:
    """Read, size-check and UTF-8-decode an uploaded YAML (shared with MCP)."""
    data = await file.read()
    return read_yaml_bytes(data)


@router.get("")
async def list_agents(
    use_case: Annotated[ListAgentConfigsUseCase, Depends(get_list_agent_configs_use_case)],
) -> list[AgentConfigMetadata]:
    """List agent configuration metadata — the built-in assistant first.

    REQ-031/033/045: the virtual built-in Composable AI Assistant is overlaid
    before the custom agents with ``is_builtin=True``; custom agents carry
    ``is_builtin=False`` computed by the overlay, never a stored column.
    """
    agents = await use_case.execute()
    effective_builtin = await get_effective_builtin_agent_service().get_effective_metadata()
    logger.info(LogMessage.AGENT_CONFIG_LISTED, len(agents) + 1)
    return [effective_builtin, *agents]


@router.get("/{agent_name}")
async def get_agent(
    agent_name: str,
    use_case: Annotated[GetAgentConfigUseCase, Depends(get_get_agent_config_use_case)],
) -> AgentConfig:
    """Retrieve a single agent configuration by name.

    The built-in assistant resolves to its virtual read-only definition
    (REQ-031/034) with ``is_builtin=True``; custom agents flow through the
    existing use case unchanged.
    """
    validate_agent_name(agent_name)
    logger.info(LogMessage.AGENT_CONFIG_GET, agent_name)
    if is_builtin_agent_name(agent_name):
        return await get_effective_builtin_agent_service().get_effective_config()
    return await use_case.execute(name=agent_name)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_agent(
    use_case: Annotated[CreateAgentConfigUseCase, Depends(get_create_agent_config_use_case)],
    agent_name: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
) -> AgentConfig:
    """Create a new agent configuration from an uploaded YAML file.

    Raises:
        AgentConfigAlreadyExistsError: An agent with this name already exists,
            INCLUDING the reserved built-in assistant name (REQ-034).
    """
    validate_agent_name(agent_name)
    ensure_custom_agent_name(agent_name)
    yaml_content = await _read_yaml_upload(file)
    logger.info(LogMessage.AGENT_CONFIG_CREATING, agent_name)
    result = await use_case.execute(name=agent_name, yaml_content=yaml_content)
    logger.info(LogMessage.AGENT_CONFIG_CREATED, agent_name)
    return result


@router.put("/{agent_name}")
async def update_agent(
    agent_name: str,
    use_case: Annotated[UpdateAgentConfigUseCase, Depends(get_update_agent_config_use_case)],
    file: Annotated[UploadFile, File(...)],
) -> AgentConfig:
    """Update an agent configuration, or the user's built-in customization.

    The built-in assistant accepts a per-user customization snapshot through
    the same multipart contract; its dynamic toolset and HITL floor remain
    server-managed. Custom agents keep the existing stored-YAML update path.
    """
    validate_agent_name(agent_name)
    yaml_content = await _read_yaml_upload(file)
    if is_builtin_agent_name(agent_name):
        return await get_effective_builtin_agent_service().update(agent_name, yaml_content)

    logger.info(LogMessage.AGENT_CONFIG_UPDATING, agent_name)
    result = await use_case.execute(name=agent_name, yaml_content=yaml_content)
    logger.info(LogMessage.AGENT_CONFIG_UPDATED, agent_name)
    return result


@router.delete("/{agent_name}/customization", status_code=status.HTTP_204_NO_CONTENT)
async def reset_builtin_customization(agent_name: str) -> None:
    """Reset the current user's built-in assistant customization.

    The reset deletes the override without loading it, so a corrupt snapshot
    is recoverable. Only the reserved built-in assistant has a customization;
    every other agent name is not found.
    """
    validate_agent_name(agent_name)
    if not is_builtin_agent_name(agent_name):
        raise AgentNotFoundError(ErrorMessage.AGENT_NOT_FOUND.format(name=agent_name))
    await get_effective_builtin_agent_service().reset()


@router.delete("/{agent_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_name: str,
    use_case: Annotated[DeleteAgentConfigUseCase, Depends(get_delete_agent_config_use_case)],
) -> None:
    """Delete an agent configuration.

    Raises:
        BuiltinAgentError: The built-in assistant's definition is read-only
            (REQ-034).
    """
    validate_agent_name(agent_name)
    ensure_not_builtin(agent_name)
    logger.info(LogMessage.AGENT_CONFIG_DELETING, agent_name)
    await use_case.execute(name=agent_name)
    logger.info(LogMessage.AGENT_CONFIG_DELETED, agent_name)
