"""SQLAlchemy ORM model for the metadata-only model catalog (spec §5.2).

Models carry NO credential columns — no API keys, JWTs, encryption material
or secret fields of any kind (MDL-002). ``user_id`` drives per-user RLS
policies and is not part of the API-facing entity.
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Index, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from src.infrastructure.database.models.base import Base


class ModelConfigModel(Base):
    """ORM model for one user-owned model metadata configuration."""

    __tablename__ = "models"
    __table_args__ = (Index("ix_models_user_id", "user_id"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    provider_model: Mapped[str] = mapped_column(Text, nullable=False)
    base_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
