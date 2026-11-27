"""SQLAlchemy ORM models for the cron subsystem (spec §5.3).

``crons`` holds per-user cron definitions with their scheduling frontier
(``next_run_at``). ``cron_runs`` holds durable run records; a partial unique
index on ``(cron_id, occurrence_at)`` over non-skipped rows gives every
scheduled occurrence a deterministic, cross-worker-safe identity (CRN-006).
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    Uuid,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.infrastructure.database.models.base import Base


class CronModel(Base):
    """ORM model for a per-user cron definition."""

    __tablename__ = "crons"
    __table_args__ = (
        Index("ix_crons_user_id", "user_id"),
        Index("ix_crons_due_scan", "next_run_at", postgresql_where=text("enabled IS TRUE")),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    cron_expression: Mapped[str] = mapped_column(String(100), nullable=False)
    timezone: Mapped[str] = mapped_column(String(100), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_run_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)


class CronRunModel(Base):
    """ORM model for one durable cron run record."""

    __tablename__ = "cron_runs"
    __table_args__ = (
        Index(
            "uq_cron_runs_occurrence_claim",
            "cron_id",
            "occurrence_at",
            unique=True,
            postgresql_where=text("status <> 'skipped'"),
        ),
        # CRN-005: at most one active (queued/running) run per cron — the
        # DB-level overlap guard for manual AND scheduled runs alike (the
        # occurrence-claim index cannot protect manual runs: NULL occurrence
        # timestamps are distinct in unique indexes).
        Index(
            "uq_cron_runs_single_active",
            "cron_id",
            unique=True,
            postgresql_where=text("status IN ('queued', 'running')"),
        ),
        # Concurrent scheduler ticks record at most one skipped row per
        # (cron, occurrence).
        Index(
            "uq_cron_runs_skipped",
            "cron_id",
            "occurrence_at",
            unique=True,
            postgresql_where=text("status = 'skipped'"),
        ),
        Index("ix_cron_runs_history", "cron_id", "created_at"),
        Index("ix_cron_runs_user_id", "user_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    cron_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("crons.id", ondelete="CASCADE"), nullable=False)
    trigger: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    occurrence_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    thread_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
