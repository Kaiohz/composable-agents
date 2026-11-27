"""SQLAlchemy ORM model for agent configuration metadata.

Beyond the descriptive metadata, the row projects the agent's registry
references (``prompt_ref`` / ``model_ref``, spec §5.5). These relational
columns are the integrity mechanism for SEC-002: foreign keys to
``prompts``/``models`` prevent a prompt/model deletion from racing an agent
save into a dangling reference, and power the server-side in-use deletion
check.
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from src.infrastructure.database.models.base import Base


class AgentConfigModel(Base):
    __tablename__ = "agent_configs"
    __table_args__ = (
        Index("ix_agent_configs_user_id", "user_id"),
        Index("ix_agent_configs_prompt_ref", "prompt_ref"),
        Index("ix_agent_configs_model_ref", "model_ref"),
    )

    name: Mapped[str] = mapped_column(String(100), primary_key=True)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    minio_path: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # Registry references projected from the agent YAML (None for legacy
    # agents without prompt_ref/model_ref — they remain valid untouched).
    prompt_ref: Mapped[UUID | None] = mapped_column(Uuid, ForeignKey("prompts.id", ondelete="RESTRICT"), nullable=True)
    model_ref: Mapped[UUID | None] = mapped_column(Uuid, ForeignKey("models.id", ondelete="RESTRICT"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Owner of the configuration — NOT NULL with default '' (RLS plumbing).
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, server_default="", default="")
