"""Drop the Settings-owned connection columns from the models table.

Revision ID: 024
Revises: 023

Spec §B/§D (model-catalog-settings-only.md): Settings owns the provider
connection (``provider``, ``base_url``, ``api_key``); a Model resource owns
ONLY its display name and the ``provider_model`` identifier to execute. The
``models.provider`` / ``models.base_url`` columns are redundant duplicates of
user LLM Settings and are dropped (DB-002). Historic values are definitively
lost (DB-005) — no backup is expected.

The downgrade recreates the columns at their original nullability (``base_url``
nullable, ``provider`` NOT NULL); existing rows are given the technical
non-empty value ``unknown`` before restoring the NOT NULL constraint (DB-004).
No original value is recoverable. Other columns, indexes, constraints and the
RLS policies of ``models`` are untouched (DB-003).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "024"
down_revision: str | None = "023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "models"


def upgrade() -> None:
    op.execute(f'ALTER TABLE {_TABLE} DROP COLUMN IF EXISTS "base_url";')
    op.execute(f'ALTER TABLE {_TABLE} DROP COLUMN IF EXISTS "provider";')


def downgrade() -> None:
    op.execute(f'ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS "base_url" TEXT NULL;')
    # Recreate at the original NOT NULL with a technical non-empty value: the
    # historic values are unrecoverable (DB-004/DB-005).
    op.execute(f'ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS "provider" TEXT;')
    op.execute(f"UPDATE {_TABLE} SET provider = 'unknown' WHERE provider IS NULL;")
    op.execute(f'ALTER TABLE {_TABLE} ALTER COLUMN "provider" SET NOT NULL;')
