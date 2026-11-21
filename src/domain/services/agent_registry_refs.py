"""Shared agent registry-reference rules (spec §5.5 AGT-002/AGT-003, §9).

Module-level domain rules reused by every write surface (create import,
update, built-in customization): ``prompt_ref``/``model_ref`` must resolve
against the OWNER's local registries at save time and are projected into the
metadata columns for reference-safe deletion and freshness fingerprints.
"""

from src.domain.entities.agent_config import AgentConfig
from src.domain.errors.config import ConfigError
from src.domain.ports.model_repository import ModelRepository
from src.domain.ports.prompt_repository import PromptRepository


async def validate_registry_refs(
    config: AgentConfig,
    prompt_repository: PromptRepository | None,
    model_repository: ModelRepository | None,
) -> None:
    """Validate ``prompt_ref``/``model_ref`` against the local registries.

    Spec §5.5 AGT-002/AGT-003 + §9: missing, inaccessible (foreign), deleted or
    invalid references fail explicitly at save time — there is no fall back to
    inline prompts, Phoenix, the legacy ``model`` string or an agent-name
    lookup. A pinned version that does not exist fails without a Latest
    fallback. Cross-user references are non-enumerating not-founds because the
    repositories are owner-scoped.

    Args:
        config: The agent configuration carrying the references.
        prompt_repository: Local prompt registry port (when wired).
        model_repository: Model catalog port (when wired).

    Raises:
        PromptNotFoundError: Missing/inaccessible prompt.
        PromptVersionNotFoundError: Missing pinned version.
        ModelNotFoundError: Missing/inaccessible model.
    """
    if config.prompt_ref is not None:
        if prompt_repository is None:
            raise ConfigError(f"Agent '{config.name}' sets prompt_ref but no prompt repository is wired.")
        await prompt_repository.get(str(config.prompt_ref.id))
        if config.prompt_ref.version is not None:
            await prompt_repository.get_version(str(config.prompt_ref.id), config.prompt_ref.version)
    if config.model_ref is not None:
        if model_repository is None:
            raise ConfigError(f"Agent '{config.name}' sets model_ref but no model repository is wired.")
        await model_repository.get(str(config.model_ref))


def registry_ref_metadata_fields(config: AgentConfig) -> dict:
    """Project the YAML's registry references into metadata columns (SEC-002)."""
    return {
        "prompt_ref": str(config.prompt_ref.id) if config.prompt_ref else None,
        "model_ref": str(config.model_ref) if config.model_ref else None,
    }
