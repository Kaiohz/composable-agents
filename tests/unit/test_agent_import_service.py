"""Unit tests for the agent-import orchestrator (port-level fakes).

Every external boundary (MinIO store, PostgreSQL repositories, the trusted
``mcp-raganything`` registry, the LangGraph Store file repository) is a spec'd
``AsyncMock``: the orchestrator's OWN behavior is under test — the
prevalidation rule families, per-construction MCP ownership (the review's C1),
the resource-only sub-agent target (C2), the predictable-name conflict check
(C4) and the reverse-order compensation ledger.
"""

from unittest.mock import AsyncMock

import pytest

from src.application.services.agent_import_service import (
    AgentImportService,
    _CompensationLedger,
    _subagent_target,
)
from src.domain.entities.model_metadata import ModelMetadata
from src.domain.entities.prompt import Prompt
from src.domain.entities.registered_mcp_server import McpServerValidationReport, McpServerWrite
from src.domain.errors.agent import AgentConfigAlreadyExistsError, AgentNotFoundError
from src.domain.errors.base import DomainError
from src.domain.errors.config import ConfigError
from src.domain.errors.model import ModelNotFoundError
from src.domain.ports.agent_config_repository import AgentConfigRepository
from src.domain.ports.agent_config_store import AgentConfigStore
from src.domain.ports.mcp_registry import McpRegistryClient
from src.domain.ports.model_repository import ModelRepository
from src.domain.ports.prompt_repository import PromptRepository
from src.domain.ports.store_file_repository import StoreFileRepository
from src.domain.services.agent_import_plan import (
    McpMaterialization,
    build_agent_import_plan,
)


def _fake_service(
    registry: McpRegistryClient | None = None,
    store_files: StoreFileRepository | None = None,
    model_repo: ModelRepository | None = None,
    prompt_repo: PromptRepository | None = None,
    config_repo: AgentConfigRepository | None = None,
) -> AgentImportService:
    config_repo = config_repo or AsyncMock(spec=AgentConfigRepository)
    config_repo.exists.return_value = False
    return AgentImportService(
        config_store=AsyncMock(spec=AgentConfigStore),
        config_repository=config_repo,
        prompt_repository=prompt_repo or AsyncMock(spec=PromptRepository),
        model_repository=model_repo or AsyncMock(spec=ModelRepository),
        mcp_registry=registry,
        store_files=store_files or AsyncMock(spec=StoreFileRepository),
    )


def _plan(**overrides):
    raw = {"name": "my-agent", "model": "gpt-4o", "system_prompt": "Prompt.", "tools": [], "debug": False}
    for field, value in overrides.items():
        if value is None:
            raw.pop(field, None)
        else:
            raw[field] = value
    return build_agent_import_plan(raw, "my-agent")


def _model(name: str) -> ModelMetadata:
    return ModelMetadata(
        id="0f4c66b4-1c6f-4c05-89f2-cdc6f3277c07",
        name=name,
        provider="openai",
        provider_model="provider-model",
    )


class TestPrevalidationNameFamilies:
    async def test_root_name_conflict_refused_before_everything(self):
        service = _fake_service()
        service._config_repository.exists.return_value = True
        with pytest.raises(AgentConfigAlreadyExistsError):
            await service._prevalidate_io(_plan())

    async def test_builtin_name_is_reserved_at_the_import_boundary(self):
        service = _fake_service()
        raw = {"name": "composable-ai-assistant", "model": "m", "system_prompt": "s", "tools": []}
        with pytest.raises(AgentConfigAlreadyExistsError):
            await service._prevalidate_io(build_agent_import_plan(raw, "composable-ai-assistant"))

    async def test_existing_inline_subagent_name_refused(self):
        repo = AsyncMock(spec=AgentConfigRepository)
        repo.exists.side_effect = [False, True]
        service = _fake_service(config_repo=repo)
        raw = {
            "name": "my-agent",
            "model": "m",
            "system_prompt": "s",
            "subagents": [{"name": "taken", "description": "Child", "instructions": "i", "model": "m", "tools": []}],
        }
        with pytest.raises(ConfigError) as exc:
            await service._prevalidate_io(build_agent_import_plan(raw, "my-agent"))
        assert "taken" in str(exc.value)


class TestPrevalidationResourceNameConflicts:
    async def test_derived_model_name_conflict_refused(self):
        model_repo = AsyncMock(spec=ModelRepository)
        model_repo.list_all.return_value = [_model("my-agent-model")]
        service = _fake_service(model_repo=model_repo)
        with pytest.raises(ConfigError) as exc:
            await service._prevalidate_io(_plan())
        assert "my-agent-model" in str(exc.value)

    async def test_derived_prompt_name_conflict_refused(self):
        prompt_repo = AsyncMock(spec=PromptRepository)
        prompt_repo.list_all.return_value = [Prompt(name="my-agent-prompt")]
        service = _fake_service(prompt_repo=prompt_repo)
        with pytest.raises(ConfigError) as exc:
            await service._prevalidate_io(_plan())
        assert "my-agent-prompt" in str(exc.value)

    async def test_subagent_derived_names_are_conflict_checked_too(self):
        raw = {
            "name": "my-agent",
            "model": "m",
            "system_prompt": "s",
            "subagents": [{"name": "helper", "description": "Child", "instructions": "i", "model": "m", "tools": []}],
        }
        model_repo = AsyncMock(spec=ModelRepository)
        model_repo.list_all.return_value = [_model("helper-model")]
        service = _fake_service(model_repo=model_repo)
        with pytest.raises(ConfigError):
            await service._prevalidate_io(build_agent_import_plan(raw, "my-agent"))

    async def test_available_names_pass_the_conflict_check(self):
        model_repo = AsyncMock(spec=ModelRepository)
        prompt_repo = AsyncMock(spec=PromptRepository)
        model_repo.list_all.return_value = []
        prompt_repo.list_all.return_value = []
        service = _fake_service(model_repo=model_repo, prompt_repo=prompt_repo)
        assert await service._prevalidate_io(_plan()) is None


class TestPrevalidationReferencesAndSelections:
    async def test_missing_model_reference_fails_controlled(self):
        model_repo = AsyncMock(spec=ModelRepository)
        model_repo.get.side_effect = ModelNotFoundError("not found")
        service = _fake_service(model_repo=model_repo)
        with pytest.raises(ModelNotFoundError):
            await service._prevalidate_io(
                _plan(model=None, model_ref="0f4c66b4-1c6f-4c05-89f2-cdc6f3277c07")
            )

    async def test_model_ref_metadata_is_returned_for_the_display_projection(self):
        model_repo = AsyncMock(spec=ModelRepository)
        model_repo.get.return_value = _model("referenced-model")
        service = _fake_service(model_repo=model_repo)
        referenced = await service._prevalidate_io(
            _plan(model=None, model_ref="0f4c66b4-1c6f-4c05-89f2-cdc6f3277c07")
        )
        assert referenced is not None and referenced.name == "referenced-model"

    async def test_declared_mcp_refs_are_validated_against_the_registry(self):
        registry = AsyncMock(spec=McpRegistryClient)
        registry.list_servers.return_value = []
        service = _fake_service(registry=registry)
        with pytest.raises(ConfigError):
            await service._prevalidate_io(_plan(mcp_refs=["ghost"]))

    async def test_registry_wiring_failure_is_controlled(self):
        service = _fake_service(registry=None)
        with pytest.raises(ConfigError):
            await service._prevalidate_io(_plan(mcp_servers=[{"name": "s", "transport": "http", "url": "http://x"}]))

    async def test_inline_creations_are_registry_validated_before_any_write(self):
        registry = AsyncMock(spec=McpRegistryClient)
        registry.validate_server.return_value = McpServerValidationReport(name="s", valid=False, errors=["rejected"])
        service = _fake_service(registry=registry)
        with pytest.raises(ConfigError) as exc:
            await service._prevalidate_io(_plan(mcp_servers=[{"name": "s", "transport": "http", "url": "http://x"}]))
        assert "s" in str(exc.value)
        registry.validate_server.assert_awaited_once()
        registry.create_server.assert_not_awaited()


class TestPrevalidationFileReferences:
    async def test_missing_textual_skill_reference_refused(self):
        store = AsyncMock(spec=StoreFileRepository)
        store.get_file.return_value = None
        service = _fake_service(store_files=store)
        with pytest.raises(ConfigError) as exc:
            await service._prevalidate_io(_plan(skills=["skills/missing/"]))
        assert "skills/missing/" in str(exc.value)

    async def test_missing_textual_memory_reference_refused(self):
        store = AsyncMock(spec=StoreFileRepository)
        store.get_file.return_value = None
        service = _fake_service(store_files=store)
        with pytest.raises(ConfigError) as exc:
            await service._prevalidate_io(_plan(memory=["memories/missing.md"]))
        assert "memories/missing.md" in str(exc.value)

    async def test_inline_target_that_already_exists_is_refused_before_any_write(self):
        store = AsyncMock(spec=StoreFileRepository)
        store.get_file.return_value = "# existing\n"
        service = _fake_service(store_files=store)
        with pytest.raises(ConfigError) as exc:
            await service._prevalidate_io(_plan(skills=[{"path": "skills/exists/", "content": "c"}]))
        assert "exists" in str(exc.value)
        store.put_file.assert_not_awaited()

    async def test_store_read_failure_propagates_as_controlled_error(self):
        store = AsyncMock(spec=StoreFileRepository)
        store.get_file.side_effect = DomainError("store unavailable")
        service = _fake_service(store_files=store)
        with pytest.raises(DomainError):
            await service._prevalidate_io(_plan(skills=["skills/x/"]))


class TestMcpOwnershipAtMaterialization:
    async def test_subagent_owned_servers_do_not_reach_the_root_selection(self):
        registry = AsyncMock(spec=McpRegistryClient)
        service = _fake_service(registry=registry)
        creations = [
            McpMaterialization(writer=McpServerWrite(name="root-srv", transport="http", url="http://a/mcp"), owner_agent="my-agent"),
            McpMaterialization(writer=McpServerWrite(name="child-srv", transport="http", url="http://b/mcp"), owner_agent="helper"),
        ]
        refs = await service._materialize_mcp(creations, "my-agent", _CompensationLedger())
        assert refs == ["root-srv"], "only the ROOT-owned identities are returned (review C1)"
        assert registry.create_server.await_count == 2, "every declared server is still created owner-scoped"


class TestSubagentTarget:
    @staticmethod
    def _creation(raw: dict):
        return build_agent_import_plan(
            {"name": "root", "model": "m", "system_prompt": "s", "tools": [], "subagents": [raw]},
            "root",
        ).subagent_creations[0]

    def test_target_is_resource_only_with_an_explicit_selection(self):
        raw = {
            "name": "helper",
            "description": "Child",
            "instructions": "i",
            "model": "m",
            "tools": ["mod:tool"],
            "skills": ["skills/docs/"],
            "mcp_servers": [{"name": "child-srv", "transport": "http", "url": "http://b/mcp"}],
        }
        target = _subagent_target(self._creation(raw), "m1", "p1", ["child-srv"])
        assert target == {
            "name": "helper",
            "description": "Child",
            "tools": ["mod:tool"],
            "model_ref": "m1",
            "prompt_ref": {"id": "p1"},
            "skills": ["skills/docs/"],
            "mcp_refs": ["child-srv"],
        }

    def test_empty_selection_is_represented_by_an_empty_set(self):
        raw = {"name": "helper", "description": "Child", "instructions": "i", "model": "m", "tools": []}
        target = _subagent_target(self._creation(raw), "m1", "p1", [])
        assert "mcp_refs" not in target
        assert set(target) <= {"name", "description", "tools", "model_ref", "prompt_ref"}


class TestCompensation:
    async def test_undo_agent_tolerates_missing_everything(self):
        service = _fake_service()
        service._config_repository.delete.side_effect = AgentNotFoundError("gone")
        service._config_store.delete.side_effect = AgentNotFoundError("gone")
        await service._undo_agent("missing")()

    async def test_compensation_continues_after_a_failing_undo(self):
        service = _fake_service()
        ledger = _CompensationLedger()
        ledger.record("model", "m1", _raising_undo)
        ledger.record("agent", "a1", AsyncMock())
        await service._compensate("my-agent", ledger)

    async def test_failed_import_compensates_created_resources(self):
        model_repo = AsyncMock(spec=ModelRepository)
        model_repo.create.return_value = _model("my-agent-model")
        prompt_repo = AsyncMock(spec=PromptRepository)
        prompt_repo.create_with_version.side_effect = ConfigError("registry outage")
        service = _fake_service(model_repo=model_repo, prompt_repo=prompt_repo)
        service._config_repository.exists.return_value = False
        service._model_repository.list_all.return_value = []
        service._prompt_repository.list_all.return_value = []

        with pytest.raises(DomainError):
            await service.import_agent("my-agent", "name: my-agent\nmodel: gpt-4o\nsystem_prompt: s\ntools: []\n")

        model_repo.delete.assert_awaited_once_with("0f4c66b4-1c6f-4c05-89f2-cdc6f3277c07")
        service._config_store.put.assert_not_awaited(), "the agent must never outlive a failed import"


async def _raising_undo() -> None:
    raise ConfigError("already deleted elsewhere")
