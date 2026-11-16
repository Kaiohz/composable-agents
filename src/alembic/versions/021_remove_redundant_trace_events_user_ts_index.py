"""Remove the redundant ix_trace_events_user_id_ts (LLM usage dashboard, I-A).

Revision ID: 021
Revises: 020

Fix pass 1 created ``ix_trace_events_user_id_ts ON trace_events (user_id,
timestamp)`` as a "support index" for the ``/traces/usage`` window pushdown.
Verification of the executed statement (rendered from
``PostgresTraceUsageRepository``) shows it **cannot serve that query**:

* the window subquery and the outer ``events`` CTE carry their owner equality
  on ``threads.user_id`` (``t2.user_id = :user_id``) — the denormalized
  ``trace_events.user_id`` is deliberately NOT used because legacy/system event
  rows carry ``user_id = ''`` and the thread is the durable owner;
* PostgreSQL cannot propagate the ``:user_id`` parameter across the
  ``t2.id = te2.thread_id`` join onto ``te2.user_id``, so the composite index
  gets no equality condition on its leading column → it is not usable here.
  The index-driven, window-proportional plan comes from the EXISTING pair
  ``ix_threads_user_id`` (migration 012) → ``ix_trace_events_thread_ts``
  (migration 005, probed per thread with the timestamp range) plus
  ``ix_trace_events_thread_turn`` (migration 005) for the turn semi-join;
* worse, the composite duplicates the leading column of
  ``ix_trace_events_user_id`` (migration 012) on the hottest write table
  (trace_events — several rows per LLM turn): double index maintenance and
  storage for zero reads.

The windowed turn set is therefore defined WITHOUT a new index: the turn
opener anchor is ``trace_events.type IN ('human_message', 'hitl_decision')``
(a HITL-resume turn owns NO human message — C2) and the scan is bounded by the
existing thread/timestamp indexes. The upgrade is an idempotent
``DROP INDEX IF EXISTS``: a no-op on databases that never received the pass-1
index, a corrective drop where it was already created. The downgrade
re-creates the original pass-1 index to keep the revision chain symmetric.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "021"
down_revision: str | None = "020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_trace_events_user_id_ts;")


def downgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_trace_events_user_id_ts
        ON trace_events (user_id, timestamp);
        """
    )
