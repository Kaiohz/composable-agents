"""PostgreSQL adapter for the metadata-only model catalog (spec §5.2).

Ownership derives from the ``current_user_id`` contextvar (writes) plus
Row-Level Security (reads), exactly like the other per-user repositories.
Deletion (MDL-004/SEC-002) checks agent references in the same transaction
with the ``agent_configs.model_ref`` foreign key as the race backstop. No
credential fields exist anywhere in this adapter (MDL-002).
"""

import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.domain.entities.model_metadata import ModelMetadata
from src.domain.errors.messages import ErrorMessage
from src.domain.errors.model import ModelInUseError, ModelNotFoundError
from src.domain.errors.storage import StorageError
from src.domain.logging.messages import LogMessage
from src.domain.ports.model_repository import ModelRepository
from src.infrastructure.database.models.agent_config import AgentConfigModel
from src.infrastructure.database.models.model_config import ModelConfigModel
from src.infrastructure.database.rls_context import current_user_id

logger = logging.getLogger(__name__)


def _to_metadata(model: ModelConfigModel) -> ModelMetadata:
    """Direct transformation: ORM row → domain entity (metadata only)."""
    return ModelMetadata(
        id=str(model.id),
        name=model.name,
        provider=model.provider,
        provider_model=model.provider_model,
        base_url=model.base_url,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


class PostgresModelRepository(ModelRepository):
    """Adapter persisting per-user model metadata in PostgreSQL."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @staticmethod
    def _current_user_id() -> str | None:
        return current_user_id.get()

    async def create(self, model: ModelMetadata) -> ModelMetadata:
        uid = self._current_user_id() or ""
        now = datetime.now(UTC)
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                row = ModelConfigModel(
                    id=UUID(model.id),
                    name=model.name,
                    provider=model.provider,
                    provider_model=model.provider_model,
                    base_url=model.base_url,
                    created_at=now,
                    updated_at=now,
                    user_id=uid,
                )
                session.add(row)
                await session.commit()
                logger.info(LogMessage.MODEL_CREATED, row.id)
                return _to_metadata(row)
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.MODEL_FAILED_CREATE.format(error=e)) from e

    async def get(self, model_id: str) -> ModelMetadata:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                row = await self._get_owned(session, model_id)
                if row is None:
                    raise ModelNotFoundError(ErrorMessage.MODEL_NOT_FOUND.format(model_id=model_id))
                logger.info(LogMessage.MODEL_FETCHED, model_id)
                return _to_metadata(row)
            except ModelNotFoundError:
                raise
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.MODEL_FAILED_GET.format(model_id=model_id, error=e)) from e

    async def list_all(self) -> list[ModelMetadata]:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                stmt = select(ModelConfigModel).order_by(ModelConfigModel.created_at.desc())
                uid = self._current_user_id()
                if uid is not None:
                    stmt = stmt.where(ModelConfigModel.user_id == uid)
                rows = (await session.execute(stmt)).scalars().all()
                return [_to_metadata(m) for m in rows]
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.MODEL_FAILED_LIST.format(error=e)) from e

    async def update(self, model: ModelMetadata) -> ModelMetadata:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                row = await self._get_owned(session, model.id)
                if row is None:
                    raise ModelNotFoundError(ErrorMessage.MODEL_NOT_FOUND.format(model_id=model.id))
                row.name = model.name
                row.provider = model.provider
                row.provider_model = model.provider_model
                row.base_url = model.base_url
                row.updated_at = datetime.now(UTC)
                await session.commit()
                logger.info(LogMessage.MODEL_UPDATED, row.id)
                return _to_metadata(row)
            except ModelNotFoundError:
                raise
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.MODEL_FAILED_GET.format(model_id=model.id, error=e)) from e

    async def delete(self, model_id: str) -> None:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                row = await self._get_owned(session, model_id)
                if row is None:
                    raise ModelNotFoundError(ErrorMessage.MODEL_NOT_FOUND.format(model_id=model_id))
                # Server-side in-use check (MDL-004/SEC-002).
                ref_count = (
                    await session.execute(
                        select(func.count())
                        .select_from(AgentConfigModel)
                        .where(AgentConfigModel.model_ref == row.id)
                    )
                ).scalar_one()
                if ref_count:
                    logger.warning(LogMessage.MODEL_DELETED_IN_USE, model_id)
                    raise ModelInUseError(ErrorMessage.MODEL_IN_USE.format(model_id=model_id))
                await session.delete(row)
                await session.commit()
                logger.info(LogMessage.MODEL_DELETED, model_id)
            except ModelNotFoundError:
                raise
            except ModelInUseError:
                raise
            except IntegrityError as e:
                # Race backstop via agent_configs.model_ref FK (SEC-002).
                raise ModelInUseError(ErrorMessage.MODEL_IN_USE.format(model_id=model_id)) from e
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.MODEL_FAILED_DELETE.format(model_id=model_id, error=e)) from e

    async def _get_owned(self, session: AsyncSession, model_id: str) -> ModelConfigModel | None:
        """Fetch the model row scoped to the current user (RLS owner)."""
        stmt = select(ModelConfigModel).where(ModelConfigModel.id == UUID(model_id))
        uid = self._current_user_id()
        if uid is not None:
            stmt = stmt.where(ModelConfigModel.user_id == uid)
        return (await session.execute(stmt)).scalar_one_or_none()
