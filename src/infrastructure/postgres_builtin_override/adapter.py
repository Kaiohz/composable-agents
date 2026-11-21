"""PostgreSQL adapter for per-user built-in assistant overrides.

Every operation is scoped to ``current_user_id`` and therefore to the same RLS
principal as the rest of the request. The table is never enumerated globally
and never appears in the custom-agent catalog.
"""

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.domain.entities.builtin_agent_override import BuiltinAgentOverride
from src.domain.errors.messages import ErrorMessage
from src.domain.errors.storage import StorageError
from src.domain.ports.builtin_agent_override_store import BuiltinAgentOverrideStore
from src.infrastructure.database.models.builtin_agent_override import BuiltinAgentOverrideModel
from src.infrastructure.database.rls_context import current_user_id


class PostgresBuiltinAgentOverrideStore(BuiltinAgentOverrideStore):
    """Store one customization snapshot per authenticated user."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @staticmethod
    def _current_user_id() -> str:
        user_id = current_user_id.get()
        if user_id is None:
            raise StorageError(ErrorMessage.BUILTIN_CUSTOMIZATION_USER_REQUIRED)
        return user_id

    async def get(self, agent_name: str) -> BuiltinAgentOverride | None:
        try:
            async with AsyncSession(self._engine, expire_on_commit=False) as session:
                model = await self._current_row(session, agent_name)
                return self._to_entity(model) if model is not None else None
        except SQLAlchemyError as exc:
            raise StorageError(ErrorMessage.STORAGE_FAILED_GET_BUILTIN_OVERRIDE.format(error=exc)) from exc

    async def put(self, override: BuiltinAgentOverride) -> None:
        user_id = self._current_user_id()
        if override.user_id != user_id:
            raise StorageError(ErrorMessage.BUILTIN_CUSTOMIZATION_USER_REQUIRED)
        try:
            async with AsyncSession(self._engine, expire_on_commit=False) as session:
                model = await self._current_row(session, override.agent_name)
                if model is None:
                    session.add(
                        BuiltinAgentOverrideModel(
                            user_id=user_id,
                            agent_name=override.agent_name,
                            yaml=override.yaml,
                            created_at=override.created_at,
                            updated_at=override.updated_at,
                        )
                    )
                else:
                    model.yaml = override.yaml
                    model.updated_at = override.updated_at
                await session.commit()
        except SQLAlchemyError as exc:
            raise StorageError(ErrorMessage.STORAGE_FAILED_SAVE_BUILTIN_OVERRIDE.format(error=exc)) from exc

    async def delete(self, agent_name: str) -> None:
        try:
            async with AsyncSession(self._engine, expire_on_commit=False) as session:
                model = await self._current_row(session, agent_name)
                if model is not None:
                    await session.delete(model)
                    await session.commit()
        except SQLAlchemyError as exc:
            raise StorageError(ErrorMessage.STORAGE_FAILED_DELETE_BUILTIN_OVERRIDE.format(error=exc)) from exc

    async def _current_row(self, session: AsyncSession, agent_name: str) -> BuiltinAgentOverrideModel | None:
        user_id = self._current_user_id()
        result = await session.execute(
            select(BuiltinAgentOverrideModel).where(
                BuiltinAgentOverrideModel.user_id == user_id,
                BuiltinAgentOverrideModel.agent_name == agent_name,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    def _to_entity(model: BuiltinAgentOverrideModel) -> BuiltinAgentOverride:
        return BuiltinAgentOverride(
            user_id=model.user_id,
            agent_name=model.agent_name,
            yaml=model.yaml,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
