"""SQLAlchemy ORM models for the local prompt registry (spec §5.1).

``prompts`` holds per-user prompt metadata; ``prompt_versions`` holds the
immutable content versions. ``(prompt_id, version)`` is the primary key so the
database enforces version uniqueness (PR-003). ``user_id`` is denormalized on
versions so owner-scoped RLS policies protect direct access by id (SEC-001).
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, Integer, PrimaryKeyConstraint, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from src.infrastructure.database.models.base import Base


class PromptModel(Base):
    """ORM model for a per-user prompt (metadata only)."""

    __tablename__ = "prompts"
    __table_args__ = (Index("ix_prompts_user_id", "user_id"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    latest_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)


class PromptVersionModel(Base):
    """ORM model for one immutable prompt content version."""

    __tablename__ = "prompt_versions"
    __table_args__ = (
        PrimaryKeyConstraint("prompt_id", "version", name="pk_prompt_versions"),
        Index("ix_prompt_versions_user_id", "user_id"),
    )

    prompt_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("prompts.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    # Stored and returned verbatim, including whitespace; empty string allowed.
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
