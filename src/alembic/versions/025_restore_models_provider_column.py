"""Restore the ``provider`` identity column on the models catalog.

Revision ID: 025
Revises: 024
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

The model-catalog slice evolved: a Model resource carries its serving
PROVIDER as part of the identity (derived from the declared model string's
prefix — ``anthropic`` from ``anthropic:claude-3-7-sonnet``, the default
OpenAI-compatible chain otherwise). The provider CONNECTION (``base_url`` /
``api_key``) remains exclusively Settings-owned (MODEL-009) — this column is
an identity attribute only.

Existing rows default to the OpenAI-compatible serving chain (the platform
default), which matches every pre-restore row's actual serving chain; the
column-level ``DEFAULT 'openai'`` keeps the same semantics for raw seeders
and future inserts that omit the identity provider.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "025"
down_revision: str | None = "024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "models"


def upgrade() -> None:
    op.execute(f'ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS "provider" TEXT;')
    op.execute(f'UPDATE {_TABLE} SET "provider" = \'openai\' WHERE "provider" IS NULL;')
    op.execute(f"ALTER TABLE {_TABLE} ALTER COLUMN \"provider\" SET DEFAULT 'openai';")
    op.execute(f'ALTER TABLE {_TABLE} ALTER COLUMN "provider" SET NOT NULL;')


def downgrade() -> None:
    op.execute(f'ALTER TABLE {_TABLE} DROP COLUMN IF EXISTS "provider";')
