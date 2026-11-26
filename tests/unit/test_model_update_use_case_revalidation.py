"""Unit tests — ``UpdateModelUseCase`` re-validates through the domain entity.

Regression for review C1 (model-catalog-settings-only): the use case used to
apply identity patches with ``model_copy(update=...)``, which SKIPS the Pydantic
validators, letting an unvalidated caller (the LLM-driven tool surface) persist
a whitespace-only ``name``/``provider_model``. The entity invariant "non-empty
after trim" (MODEL-002/004) must be enforced at the domain layer, whatever the
entry point validates.

Entry: the REAL :class:`PostgresModelRepository` over the shared in-memory
SQLite engine (no internal component mocked).
"""

import pytest

import src.infrastructure.database.models.prompt  # noqa: F401 — registers the tables agent_configs' FKs reference
from src.application.use_cases.model.manage_models import CreateModelUseCase, UpdateModelUseCase
from src.infrastructure.database.rls_context import current_user_id
from src.infrastructure.postgres_model.adapter import PostgresModelRepository

pytestmark = pytest.mark.asyncio

_OWNER = "user-a"


@pytest.fixture(autouse=True)
def _owner_context():
    """Scope the adapter's owner contextvar like a real request."""
    token = current_user_id.set(_OWNER)
    yield
    current_user_id.reset(token)


@pytest.fixture
async def repo(db_engine) -> PostgresModelRepository:
    return PostgresModelRepository(engine=db_engine)


@pytest.fixture
def create_use_case(repo) -> CreateModelUseCase:
    return CreateModelUseCase(model_repository=repo)


@pytest.fixture
def update_use_case(repo) -> UpdateModelUseCase:
    return UpdateModelUseCase(model_repository=repo)


class TestUpdateModelUseCaseRevalidation:
    async def test_whitespace_only_name_is_rejected_by_the_entity(self, repo, create_use_case, update_use_case):
        """C1: a patch that bypassed the boundary DTOs can no longer persist a
        blank value — the merged entity is re-validated (MODEL-004)."""
        created = await create_use_case.execute(name="Valid", provider_model="pm-x")
        with pytest.raises(ValueError, match="must be a non-empty string"):
            await update_use_case.execute(created.id, {"name": "   "})

        fetched = await repo.get(created.id)
        assert fetched.name == "Valid", "the rejected patch must not be persisted"
        assert fetched.provider_model == "pm-x"

    async def test_whitespace_only_provider_model_is_rejected_by_the_entity(
        self, repo, create_use_case, update_use_case
    ):
        created = await create_use_case.execute(name="Valid", provider_model="pm-x")
        with pytest.raises(ValueError, match="must be a non-empty string"):
            await update_use_case.execute(created.id, {"provider_model": "  "})

        fetched = await repo.get(created.id)
        assert fetched.provider_model == "pm-x", "the rejected patch must not be persisted"
        assert fetched.name == "Valid"

    async def test_valid_change_is_trimmed_and_applied(self, create_use_case, update_use_case):
        """The re-validated entity keeps the create path behavior: values are
        trimmed by the entity validators."""
        created = await create_use_case.execute(name="Valid", provider_model="pm-x")
        saved = await update_use_case.execute(created.id, {"name": "  Renamed  "})
        assert saved.name == "Renamed"
        assert saved.provider_model == "pm-x"
