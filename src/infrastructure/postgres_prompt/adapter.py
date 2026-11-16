"""PostgreSQL adapter for the local prompt registry (spec §5.1).

Per-user isolation mirrors the other repositories: the ``current_user_id``
contextvar scopes writes (row ownership) and reads (``WHERE user_id``), and
PostgreSQL Row-Level Security policies enforce the same scope at the database
level (SEC-001).

Version allocation (PR-003) locks the parent prompt row (``FOR UPDATE``) and
derives the next version from ``MAX(version)`` inside the same transaction, so
concurrent content submissions allocate distinct sequential versions and never
overwrite one another. The database additionally enforces ``(prompt_id,
version)`` uniqueness.

Deletion (PR-005/SEC-002) checks agent references in the same transaction and
relies on the ``agent_configs.prompt_ref`` foreign key as the race backstop.
"""

import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.domain.entities.prompt import Prompt, PromptVersion
from src.domain.entities.prompt_version_summary import PromptVersionSummary
from src.domain.errors.messages import ErrorMessage
from src.domain.errors.prompt import PromptInUseError, PromptNotFoundError, PromptVersionNotFoundError
from src.domain.errors.storage import StorageError
from src.domain.logging.messages import LogMessage
from src.domain.ports.prompt_repository import PromptRepository
from src.infrastructure.database.models.agent_config import AgentConfigModel
from src.infrastructure.database.models.prompt import PromptModel, PromptVersionModel
from src.infrastructure.database.rls_context import current_user_id

logger = logging.getLogger(__name__)


def _to_prompt(model: PromptModel) -> Prompt:
    """Direct transformation: ORM row → domain entity."""
    return Prompt(
        id=str(model.id),
        name=model.name,
        description=model.description,
        latest_version=model.latest_version,
        created_at=model.created_at,
        updated_at=model.updated_at,
        user_id=model.user_id,
    )


def _to_version(model: PromptVersionModel) -> PromptVersion:
    """Direct transformation: version row → domain entity (content verbatim)."""
    return PromptVersion(
        prompt_id=str(model.prompt_id),
        version=model.version,
        content=model.content,
        created_at=model.created_at,
        user_id=model.user_id,
    )


class PostgresPromptRepository(PromptRepository):
    """Adapter persisting prompts and immutable versions in PostgreSQL."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @staticmethod
    def _current_user_id() -> str | None:
        return current_user_id.get()

    async def create_with_version(self, prompt: Prompt, content: str) -> Prompt:
        uid = self._current_user_id() or ""
        now = datetime.now(UTC)
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                model = PromptModel(
                    id=UUID(prompt.id),
                    name=prompt.name,
                    description=prompt.description,
                    latest_version=1,
                    created_at=now,
                    updated_at=now,
                    user_id=uid,
                )
                session.add(model)
                session.add(
                    PromptVersionModel(
                        prompt_id=model.id,
                        version=1,
                        content=content,
                        created_at=now,
                        user_id=uid,
                    )
                )
                await session.commit()
                logger.info(LogMessage.PROMPT_REGISTRY_CREATED, model.id, 1)
                return Prompt(
                    id=str(model.id),
                    name=model.name,
                    description=model.description,
                    latest_version=1,
                    created_at=model.created_at,
                    updated_at=model.updated_at,
                    user_id=model.user_id,
                )
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.PROMPT_FAILED_CREATE.format(error=e)) from e

    async def get(self, prompt_id: str) -> Prompt:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                uid = self._current_user_id()
                if uid is None:
                    model = await session.get(PromptModel, UUID(prompt_id))
                else:
                    model = (
                        await session.execute(
                            select(PromptModel).where(PromptModel.id == UUID(prompt_id), PromptModel.user_id == uid)
                        )
                    ).scalar_one_or_none()
                if model is None:
                    raise PromptNotFoundError(ErrorMessage.PROMPT_NOT_FOUND.format(prompt_id=prompt_id))
                return _to_prompt(model)
            except PromptNotFoundError:
                raise
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.PROMPT_FAILED_GET.format(prompt_id=prompt_id, error=e)) from e

    async def list_all(self) -> list[Prompt]:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                stmt = select(PromptModel).order_by(PromptModel.created_at.desc())
                uid = self._current_user_id()
                if uid is not None:
                    stmt = stmt.where(PromptModel.user_id == uid)
                rows = (await session.execute(stmt)).scalars().all()
                return [_to_prompt(m) for m in rows]
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.PROMPT_FAILED_LIST.format(error=e)) from e

    async def update_metadata(self, prompt: Prompt) -> Prompt:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                uid = self._current_user_id()
                stmt = select(PromptModel).where(PromptModel.id == UUID(prompt.id))
                if uid is not None:
                    stmt = stmt.where(PromptModel.user_id == uid)
                model = (await session.execute(stmt)).scalar_one_or_none()
                if model is None:
                    raise PromptNotFoundError(ErrorMessage.PROMPT_NOT_FOUND.format(prompt_id=prompt.id))
                model.name = prompt.name
                model.description = prompt.description
                model.updated_at = datetime.now(UTC)
                await session.commit()
                logger.info(LogMessage.PROMPT_REGISTRY_METADATA_UPDATED, model.id)
                return _to_prompt(model)
            except PromptNotFoundError:
                raise
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.PROMPT_FAILED_GET.format(prompt_id=prompt.id, error=e)) from e

    async def delete(self, prompt_id: str) -> None:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                uid = self._current_user_id()
                stmt = select(PromptModel).where(PromptModel.id == UUID(prompt_id))
                if uid is not None:
                    stmt = stmt.where(PromptModel.user_id == uid)
                model = (await session.execute(stmt)).scalar_one_or_none()
                if model is None:
                    raise PromptNotFoundError(ErrorMessage.PROMPT_NOT_FOUND.format(prompt_id=prompt_id))
                # Server-side in-use check (PR-005/SEC-002): any agent of this
                # user referencing the prompt (Latest or pinned) blocks deletion.
                ref_count = (
                    await session.execute(
                        select(func.count())
                        .select_from(AgentConfigModel)
                        .where(AgentConfigModel.prompt_ref == model.id)
                    )
                ).scalar_one()
                if ref_count:
                    raise PromptInUseError(ErrorMessage.PROMPT_IN_USE.format(prompt_id=prompt_id))
                await session.delete(model)
                await session.commit()
                logger.info(LogMessage.PROMPT_REGISTRY_DELETED, model.id)
            except PromptNotFoundError:
                raise
            except PromptInUseError:
                logger.warning(LogMessage.PROMPT_DELETED_IN_USE, prompt_id)
                raise
            except IntegrityError as e:
                # Race backstop: agent_configs.prompt_ref FK — an agent saved
                # between the check and the delete blocks the deletion.
                raise PromptInUseError(ErrorMessage.PROMPT_IN_USE.format(prompt_id=prompt_id)) from e
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.PROMPT_FAILED_DELETE.format(prompt_id=prompt_id, error=e)) from e

    async def add_version(self, prompt_id: str, content: str) -> PromptVersion:
        uid = self._current_user_id() or ""
        now = datetime.now(UTC)
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                stmt = select(PromptModel).where(PromptModel.id == UUID(prompt_id)).with_for_update()
                if uid is not None:
                    stmt = stmt.where(PromptModel.user_id == uid)
                model = (await session.execute(stmt)).scalar_one_or_none()
                if model is None:
                    raise PromptNotFoundError(ErrorMessage.PROMPT_NOT_FOUND.format(prompt_id=prompt_id))
                # Transactional, race-safe allocation: MAX(version) inside the
                # locked transaction (PR-003).
                current_max = (
                    await session.execute(
                        select(func.coalesce(func.max(PromptVersionModel.version), 0)).where(
                            PromptVersionModel.prompt_id == model.id
                        )
                    )
                ).scalar_one()
                next_version = current_max + 1
                version_model = PromptVersionModel(
                    prompt_id=model.id,
                    version=next_version,
                    content=content,
                    created_at=now,
                    user_id=model.user_id,
                )
                session.add(version_model)
                model.latest_version = next_version
                model.updated_at = now
                await session.commit()
                logger.info(LogMessage.PROMPT_REGISTRY_VERSION_ADDED, model.id, next_version)
                return _to_version(version_model)
            except PromptNotFoundError:
                raise
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.PROMPT_FAILED_ADD_VERSION.format(prompt_id=prompt_id, error=e)) from e

    async def list_versions(self, prompt_id: str) -> list[PromptVersionSummary]:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                uid = self._current_user_id()
                prompt_stmt = select(PromptModel.id).where(PromptModel.id == UUID(prompt_id))
                if uid is not None:
                    prompt_stmt = prompt_stmt.where(PromptModel.user_id == uid)
                exists = (await session.execute(prompt_stmt)).scalar_one_or_none()
                if exists is None:
                    raise PromptNotFoundError(ErrorMessage.PROMPT_NOT_FOUND.format(prompt_id=prompt_id))
                # SQL-level projection (BUG-004): the version LIST carries only
                # number + creation timestamp — no content is loaded from the
                # immutable version rows (a 50-version × 20 KB prompt was
                # shipping ~1 MB per page).
                rows = (
                    await session.execute(
                        select(
                            PromptVersionModel.prompt_id,
                            PromptVersionModel.version,
                            PromptVersionModel.created_at,
                        )
                        .where(PromptVersionModel.prompt_id == exists)
                        .order_by(PromptVersionModel.version.asc())
                    )
                ).all()
                logger.info(LogMessage.PROMPT_REGISTRY_VERSIONS_LISTED, len(rows), prompt_id)
                return [
                    PromptVersionSummary(
                        prompt_id=str(prompt_id_value),
                        version=version_value,
                        created_at=created_at_value,
                    )
                    for prompt_id_value, version_value, created_at_value in rows
                ]
            except PromptNotFoundError:
                raise
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.PROMPT_FAILED_LIST_VERSIONS.format(prompt_id=prompt_id, error=e)) from e

    async def get_version(self, prompt_id: str, version: int | None = None) -> PromptVersion:
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            try:
                uid = self._current_user_id()
                prompt_stmt = select(PromptModel.id).where(PromptModel.id == UUID(prompt_id))
                if uid is not None:
                    prompt_stmt = prompt_stmt.where(PromptModel.user_id == uid)
                exists = (await session.execute(prompt_stmt)).scalar_one_or_none()
                if exists is None:
                    raise PromptNotFoundError(ErrorMessage.PROMPT_NOT_FOUND.format(prompt_id=prompt_id))
                if version is None:
                    # Latest: highest existing version (never a fallback guess).
                    version_stmt = (
                        select(PromptVersionModel)
                        .where(PromptVersionModel.prompt_id == exists)
                        .order_by(PromptVersionModel.version.desc())
                        .limit(1)
                    )
                else:
                    version_stmt = select(PromptVersionModel).where(
                        PromptVersionModel.prompt_id == exists,
                        PromptVersionModel.version == version,
                    )
                model = (await session.execute(version_stmt)).scalar_one_or_none()
                if model is None:
                    raise PromptVersionNotFoundError(
                        ErrorMessage.PROMPT_VERSION_NOT_FOUND.format(version=version, prompt_id=prompt_id)
                    )
                logger.info(LogMessage.PROMPT_REGISTRY_FETCHED, prompt_id)
                return _to_version(model)
            except (PromptNotFoundError, PromptVersionNotFoundError):
                raise
            except SQLAlchemyError as e:
                raise StorageError(ErrorMessage.PROMPT_FAILED_GET.format(prompt_id=prompt_id, error=e)) from e
