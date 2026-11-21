"""SQLAlchemy model for per-user built-in assistant customization snapshots."""

from datetime import datetime

from sqlalchemy import DateTime, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.infrastructure.database.models.base import Base


class BuiltinAgentOverrideModel(Base):
    """One customization snapshot per ``(user_id, agent_name)``.

    The table is deliberately separate from ``agent_configs`` and from the
    shared agent YAML bucket: the built-in assistant stays a virtual overlay
    and this row only shadows it for its owning user.
    """

    __tablename__ = "builtin_agent_overrides"
    __table_args__ = (Index("ix_builtin_agent_overrides_user_id", "user_id"),)

    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(100), primary_key=True)
    yaml: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
