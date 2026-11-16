"""PostgreSQL adapter for the TraceUsageRepository port (dashboard spec §4-§6).

One SQL statement per endpoint; the query count stays constant regardless of
the window size or run count (no per-run / per-model fetching):

* One run = one turn (``thread_id`` + ``turn_id``) of the user's threads. Runs
  are classified by the presence of a terminal ``AI_MESSAGE`` event: present →
  COMPLETED run, absent → ERROR run (turns that never produced an AI answer).
* Tokens come from the terminal AI event's ``metadata.llm_usage`` (the capture
  contract in ``DeepAgentRunner``); the model name falls back to the agent's
  configured model for runs whose LLM returned no usage metadata.
* Cost resolves against the global ``llm_prices`` table (USD per 1M tokens,
  exact NUMERIC arithmetic). The price row is resolved by LONGEST-PREFIX match
  (spec §3 "known model names and common prefixes", BUG-002): a dated id such
  as ``gpt-4o-2024-08-06`` resolves the seeded ``gpt-4o`` row's prices, and an
  EXACT id naturally wins as the longest prefix. The anchor is a version
  boundary — the prefix must be followed by a non-alphanumeric character (or
  the end of the id) so unrelated families never collide. Models with no
  prefix match contribute ``cost=0`` and are marked ``priced=false`` — the
  pipeline never breaks on them.
* ``duration_ms`` is derived from the timestamps (run opener → AI_MESSAGE)
  rather than trusting stored metadata; percentiles run over COMPLETED runs
  only — error runs carry no terminal timestamp and never distort them.
* The window bounds are pushed down into the ``events`` CTE: only the turns
  whose OPENING event starts inside ``[from, to]`` are aggregated, so the
  per-request aggregation cost grows with the window rather than with the
  full history set. The plan is thread-level: it starts from the
  user's threads (``ix_threads_user_id``) and walks each thread's events via
  the existing ``ix_trace_events_thread_ts`` /
  ``ix_trace_events_thread_turn`` pair (migrations 012 / 005) — no extra
  index is needed or created. The turn opener is the HUMAN_MESSAGE (conversational
  and cron turns) or, for a HITL-resume turn, the leading HITL_DECISION
  (a resume turn owns NO human message — dropping it would silently hide the
  resumed LLM call's tokens and cost from every windowed refresh, C2).
* The tokens of one turn are SUMmed over its ``ai_message`` events. Today a
  turn owns exactly ONE ``ai_message`` (the terminal one — usage metadata is
  attached to no other event), so ``SUM ≡ MAX``; ``SUM`` is kept because it
  stays correct if a turn ever carries several ``ai_message`` events (a
  HITL-resume turn is a SEPARATE turn — the resume mints a new turn id, it
  does not extend the old one).
* A run whose model name cannot be resolved (no ``llm_usage`` metadata AND a
  deleted/unknown agent configuration) is labeled ``"unknown"`` — never NULL —
  so it stays counted everywhere, is priced as unpriced (``cost=0`` / priced
  ``false``) and can never crash the endpoints (C1).
* The ``GET /usage`` statement returns the overall totals plus both series
  breakdowns in a single result set (UNION ALL over shared CTE branches).

Authorization is two-layered (SEC-001, BUG-001) with the ownership boundary on
the PARENT THREAD row: the query runs under the bypass-aware RLS scope
(``system_rls_context``) AND carries an explicit ``threads.user_id`` ownership
predicate, so only events of the authenticated user's own threads can be
aggregated — never another user's. Why thread-level ownership rather than the
denormalized ``trace_events.user_id``: system-seeded/legacy event rows are
written without an owner (``user_id=''``) and are invisible to their own
thread owner under the per-row event policy — the durable owner of a run is
its thread. Unlike the sibling ``PostgresTraceSummaryRepository`` — which keeps
RLS active — this adapter opens a bypass-aware scope by design: the aggregated
event rows may carry the legacy empty owner (``user_id=''``), so the explicit
thread predicate (not the denormalized event owner) is the authoritative
boundary. Without an authenticated principal the adapter returns an empty
result instead of a query (non-enumeration by construction).
"""

import logging
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from src.domain.entities.llm_usage_agent_breakdown import LlmUsageAgentBreakdown
from src.domain.entities.llm_usage_model_breakdown import LlmUsageModelBreakdown
from src.domain.entities.llm_usage_run import LlmUsageRun
from src.domain.entities.llm_usage_summary import LlmUsageSummary
from src.domain.errors.messages import ErrorMessage
from src.domain.errors.storage import StorageError
from src.domain.logging.messages import LogMessage
from src.domain.ports.trace_usage_repository import TraceUsageRepository
from src.infrastructure.database.rls_context import current_user_id, system_rls_context

logger = logging.getLogger(__name__)


def _guarded_bigint(expr: str) -> str:
    """Build a NULL-safe, non-numeric-safe cast of a JSONB text accessor.

    A corrupted ``llm_usage`` payload must never break the aggregation: values
    that are not plain non-negative integers contribute 0.

    Args:
        expr: SQL expression extracting a TEXT value (``->>``).

    Returns:
        A guarded ``bigint`` cast SQL fragment.
    """
    return f"CASE WHEN COALESCE({expr}, '') ~ '^[0-9]+$' THEN ({expr})::bigint ELSE 0 END"


def _runs_ctes_sql(agent_clause: str, window_clause: str) -> str:
    """Compose the shared events → runs → priced CTE body.

    Args:
        agent_clause: Optional ``AND t.agent_name = :agent_name`` fragment
            (empty when no agent filter).
        window_clause: Optional ``AND te.turn_id IN (...)`` fragment applied
            INSIDE the ``events`` CTE — bounds the scan to the turns whose
            opener (HUMAN_MESSAGE or HITL_DECISION) is inside the window
            (empty when the request has no window bounds).

    Returns:
        The WITH-clause SQL (up to and including the ``priced`` CTE).
    """
    input_tokens = _guarded_bigint("(e.metadata -> 'llm_usage' ->> 'input_tokens')")
    output_tokens = _guarded_bigint("(e.metadata -> 'llm_usage' ->> 'output_tokens')")
    total_tokens = _guarded_bigint("(e.metadata -> 'llm_usage' ->> 'total_tokens')")
    model_name = "COALESCE(r.usage_model_name, r.agent_model, 'unknown')"
    return f"""
WITH events AS (
    SELECT te.thread_id AS thread_id,
           te.turn_id AS turn_id,
           t.agent_name AS agent_name,
           ac.model AS agent_model,
           te.type AS type,
           te.metadata AS metadata,
           te.timestamp AS ts
    FROM trace_events te
    JOIN threads t ON t.id = te.thread_id
    LEFT JOIN agent_configs ac ON ac.name = t.agent_name
    WHERE t.user_id = :user_id{agent_clause}{window_clause}
),
runs AS (
    SELECT e.thread_id AS thread_id,
           e.turn_id AS turn_id,
           e.agent_name AS agent_name,
           e.agent_model AS agent_model,
           -- Run start = the turn OPENER event's timestamp: the HUMAN_MESSAGE
           -- (conversational + cron turns); for a HITL-resume turn (which has
           -- NO human message — the leading HITL_DECISION is its lowest ts)
           -- this falls back to MIN(ts).
           COALESCE(MIN(e.ts) FILTER (WHERE e.type = 'human_message'), MIN(e.ts)) AS started_at,
           MAX(e.ts) FILTER (WHERE e.type = 'ai_message') AS ai_ts,
           -- Tokens are SUMmed over the turn's ai_message events. Today a turn
           -- owns exactly ONE ai_message (the terminal one — usage metadata is
           -- attached to no other event), so SUM ≡ MAX; SUM is kept because it
           -- stays correct if a turn ever carries several ai_message events
           -- (a HITL-resume turn is a SEPARATE turn: the resume mints a new
           -- turn id, it does not extend the old one).
           COALESCE(SUM({input_tokens}) FILTER (WHERE e.type = 'ai_message'), 0) AS input_tokens,
           COALESCE(SUM({output_tokens}) FILTER (WHERE e.type = 'ai_message'), 0) AS output_tokens,
           COALESCE(SUM({total_tokens}) FILTER (WHERE e.type = 'ai_message'), 0) AS total_tokens,
           MAX(e.metadata -> 'llm_usage' ->> 'model_name') AS usage_model_name
    FROM events e
    GROUP BY e.thread_id, e.turn_id, e.agent_name, e.agent_model
),
priced AS (
    SELECT r.thread_id AS thread_id,
           r.turn_id AS turn_id,
           r.agent_name AS agent_name,
           r.started_at AS started_at,
           r.ai_ts AS ai_ts,
           r.input_tokens AS input_tokens,
           r.output_tokens AS output_tokens,
           r.total_tokens AS total_tokens,
           {model_name} AS model_name,
           CASE WHEN r.ai_ts IS NULL OR r.started_at IS NULL THEN NULL
                ELSE EXTRACT(EPOCH FROM (r.ai_ts - r.started_at)) * 1000 END AS duration_ms,
           pm.matched_model IS NOT NULL AS priced,
           CASE WHEN pm.matched_model IS NULL THEN 0.0
                ELSE (pm.input_price_per_million * r.input_tokens
                      + pm.output_price_per_million * r.output_tokens) / 1000000.0 END AS cost_usd
    FROM runs r
    LEFT JOIN LATERAL (
        -- Longest-prefix price resolution (spec §3 "known model names AND
        -- common prefixes", BUG-002): a dated id (gpt-4o-2024-08-06) resolves
        -- its base model's price row; an EXACT id is the longest prefix and
        -- wins. The match is anchored on a version boundary — the prefix must
        -- be followed by a non-alphanumeric character (the '-' of a dated
        -- suffix, or the end of the id) so unrelated families never collide
        -- (gpt-4ox… is not gpt-4o). llm_prices is a tiny global reference
        -- table (~20 rows): the per-run top-1 scan is negligible, the
        -- statement stays a single query.
        SELECT p.model_name AS matched_model,
               p.input_price_per_million,
               p.output_price_per_million
        FROM llm_prices p
        WHERE position(p.model_name IN {model_name}) = 1
          AND NOT (substr({model_name}, length(p.model_name) + 1, 1) ~ '[A-Za-z0-9]')
        ORDER BY length(p.model_name) DESC
        LIMIT 1
    ) pm ON TRUE
)"""


def _usage_sql(agent_name: str | None, from_ts: datetime | None, to_ts: datetime | None) -> str:
    """Compose the single usage-aggregation statement (totals + both series).

    Args:
        agent_name: EXACT agent filter (or ``None``).
        from_ts: Inclusive window lower bound (or ``None``).
        to_ts: Inclusive window upper bound (or ``None``).

    Returns:
        The full SQL statement returning branch 0 (overall totals), branch 1
        (per-agent series) and branch 2 (per-model series) rows.
    """
    agent_clause, window_clause = _clauses_sql(agent_name, from_ts, to_ts)
    return (
        _runs_ctes_sql(agent_clause, window_clause)
        + """
SELECT 0 AS branch, NULL::text AS name,
       COUNT(*) FILTER (WHERE ai_ts IS NOT NULL) AS run_count,
       COUNT(*) FILTER (WHERE ai_ts IS NULL) AS error_count,
       COALESCE(SUM(input_tokens) FILTER (WHERE ai_ts IS NOT NULL), 0)::bigint AS total_input_tokens,
       COALESCE(SUM(output_tokens) FILTER (WHERE ai_ts IS NOT NULL), 0)::bigint AS total_output_tokens,
       COALESCE(SUM(total_tokens) FILTER (WHERE ai_ts IS NOT NULL), 0)::bigint AS total_tokens,
       COALESCE(SUM(cost_usd), 0) AS total_cost_usd,
       COALESCE(bool_or(priced), FALSE) AS priced,
       COALESCE(percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms::float8)
                FILTER (WHERE duration_ms IS NOT NULL), 0.0) AS duration_ms_p50,
       COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms::float8)
                FILTER (WHERE duration_ms IS NOT NULL), 0.0) AS duration_ms_p95
FROM priced
UNION ALL
SELECT 1, agent_name,
       COUNT(*) FILTER (WHERE ai_ts IS NOT NULL),
       COUNT(*) FILTER (WHERE ai_ts IS NULL),
       COALESCE(SUM(input_tokens) FILTER (WHERE ai_ts IS NOT NULL), 0)::bigint,
       COALESCE(SUM(output_tokens) FILTER (WHERE ai_ts IS NOT NULL), 0)::bigint,
       COALESCE(SUM(total_tokens) FILTER (WHERE ai_ts IS NOT NULL), 0)::bigint,
       COALESCE(SUM(cost_usd), 0),
       COALESCE(bool_or(priced), FALSE),
       0.0, 0.0
FROM priced
GROUP BY agent_name
UNION ALL
SELECT 2, model_name,
       COUNT(*) FILTER (WHERE ai_ts IS NOT NULL),
       COUNT(*) FILTER (WHERE ai_ts IS NULL),
       COALESCE(SUM(input_tokens) FILTER (WHERE ai_ts IS NOT NULL), 0)::bigint,
       COALESCE(SUM(output_tokens) FILTER (WHERE ai_ts IS NOT NULL), 0)::bigint,
       COALESCE(SUM(total_tokens) FILTER (WHERE ai_ts IS NOT NULL), 0)::bigint,
       COALESCE(SUM(cost_usd), 0),
       COALESCE(bool_or(priced), FALSE),
       0.0, 0.0
FROM priced
-- Belt-and-braces: model_name is COALESCEd to 'unknown' (never NULL), so this
-- filter can no longer drop rows from the series — a guard against a future
-- regression of the C1 fix, not a live filter.
WHERE model_name IS NOT NULL
GROUP BY model_name
ORDER BY branch, total_tokens DESC, name"""
    )


def _top_sql(agent_name: str | None, from_ts: datetime | None, to_ts: datetime | None) -> str:
    """Compose the single top-N most-expensive-runs statement.

    Args:
        agent_name: EXACT agent filter (or ``None``).
        from_ts: Inclusive window lower bound (or ``None``).
        to_ts: Inclusive window upper bound (or ``None``).

    Returns:
        The full SQL statement returning the completed runs sorted by
        estimated cost descending, capped by ``:limit``.
    """
    agent_clause, window_clause = _clauses_sql(agent_name, from_ts, to_ts)
    return (
        _runs_ctes_sql(agent_clause, window_clause)
        + """
SELECT r.thread_id AS thread_id,
       r.turn_id AS turn_id,
       r.agent_name AS agent_name,
       r.model_name AS model_name,
       r.total_tokens AS total_tokens,
       r.cost_usd AS cost_usd,
       COALESCE(r.duration_ms, 0)::float8 AS duration_ms,
       r.started_at AS started_at
FROM priced r
WHERE r.ai_ts IS NOT NULL
ORDER BY r.cost_usd DESC, r.total_tokens DESC, r.turn_id
LIMIT :limit"""
    )


def _clauses_sql(agent_name: str | None, from_ts: datetime | None, to_ts: datetime | None) -> tuple[str, str]:
    """Build the optional agent/window WHERE fragments for the shared CTEs.

    The agent filter is an EXACT match on the parent thread's agent name. The
    window filter is pushed down INSIDE the ``events`` CTE: only the turns
    whose OPENING event starts inside ``[from_ts, to_ts]`` are ever scanned, so
    the statement cost stays proportional to the window size instead of the
    user's whole history. A turn's opener is its HUMAN_MESSAGE (conversational
    AND cron turns) or its leading HITL_DECISION for a HITL-resume turn — the
    resume mints a NEW turn id (``send_message``) and owns NO human message,
    and its resumed LLM call MUST stay in the windowed result set (C2). The
    anchor reproduces exactly the ``started_at`` semantics of the ``runs`` CTE
    (``COALESCE(MIN(human_ts), MIN(ts))``): a resume turn was — and stays —
    counted with the pre-pushdown ``MIN(ts)`` fallback. The ownership predicate
    inside the window subquery is thread-level (same boundary as the outer
    CTE) — the denormalized ``trace_events.user_id`` is unreliable for
    legacy/system rows.

    Args:
        agent_name: EXACT agent filter (or ``None``).
        from_ts: Inclusive window lower bound (or ``None``).
        to_ts: Inclusive window upper bound (or ``None``).

    Returns:
        Tuple ``(agent_clause, window_clause)`` — both fragments are applied
        inside the ``events`` CTE; empty fragments mean no filter.
    """
    agent_clause = "\n      AND t.agent_name = :agent_name" if agent_name is not None else ""
    if from_ts is None and to_ts is None:
        return agent_clause, ""
    bounds = ""
    if from_ts is not None:
        bounds += "\n              AND te2.timestamp >= :from_ts"
    if to_ts is not None:
        bounds += "\n              AND te2.timestamp <= :to_ts"
    window_clause = (
        "\n      -- Window pushdown: bound the scanned turns to the window."
        "\n      -- A turn's opener is its HUMAN_MESSAGE (conversational/cron"
        "\n      -- turn) or its leading HITL_DECISION (a HITL-resume turn has"
        "\n      -- NO human message; its resumed LLM call must stay counted, C2)."
        "\n      AND te.turn_id IN ("
        "\n          SELECT te2.turn_id"
        "\n          FROM trace_events te2"
        "\n          JOIN threads t2 ON t2.id = te2.thread_id"
        "\n          WHERE t2.user_id = :user_id"
        "\n            AND te2.type IN ('human_message', 'hitl_decision')"
        f"{bounds}"
        "\n      )"
    )
    return agent_clause, window_clause


class PostgresTraceUsageRepository(TraceUsageRepository):
    """Adapter aggregating LLM usage for the authenticated user in PostgreSQL."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @staticmethod
    def _current_user_id() -> str | None:
        """Return the ``current_user_id`` contextvar value or ``None``."""
        return current_user_id.get()

    @staticmethod
    def _params(
        uid: str, agent_name: str | None, from_ts: datetime | None, to_ts: datetime | None, limit: int | None = None
    ) -> dict:
        """Build the bound-parameter dict from the request filters.

        Args:
            uid: Authenticated owner (RLS + explicit predicates).
            agent_name: EXACT agent filter (or ``None``).
            from_ts: Window lower bound (or ``None``).
            to_ts: Window upper bound (or ``None``).
            limit: Top-N cap (only bound by the top-runs statement).

        Returns:
            The parameter dict containing only bound values.
        """
        params: dict = {"user_id": uid}
        if agent_name is not None:
            params["agent_name"] = agent_name
        if from_ts is not None:
            params["from_ts"] = from_ts
        if to_ts is not None:
            params["to_ts"] = to_ts
        if limit is not None:
            params["limit"] = limit
        return params

    async def summarize_usage(
        self,
        *,
        agent_name: str | None = None,
        from_ts: datetime | None = None,
        to_ts: datetime | None = None,
    ) -> LlmUsageSummary:
        """Aggregate tokens / cost / runs / latency for the user's window.

        Args:
            agent_name: Optional EXACT (case-sensitive) agent filter.
            from_ts: Inclusive lower bound on the runs' start timestamp.
            to_ts: Inclusive upper bound on the runs' start timestamp.

        Returns:
            The :class:`LlmUsageSummary` with both series breakdowns.

        Raises:
            StorageError: On infrastructure failure.
        """
        uid = self._current_user_id()
        if not uid:
            # Defense-in-depth (BUG-001): no ownership scope to honour — return
            # a bounded empty summary instead of an unscoped query. ``not uid``
            # also covers the empty string, which would match legacy/system
            # rows whose user_id = '' (never another tenant's data).
            logger.warning(LogMessage.TRACE_USAGE_UNSCOPED)
            return LlmUsageSummary()

        stmt = text(_usage_sql(agent_name, from_ts, to_ts))
        try:
            # Bypass-aware scope: with the bypass GUC set the per-row event
            # policy admits rows, and the explicit ``threads.user_id`` bound
            # parameter is the hard ownership boundary (see module docstring —
            # system-seeded/legacy events carry no denormalized owner). The
            # engine still sets ``app.bypass_rls`` transaction-scoped per
            # statement; the predicate scopes every row the query touches.
            async with system_rls_context(), self._engine.connect() as conn:
                rows = (await conn.execute(stmt, self._params(uid, agent_name, from_ts, to_ts))).all()
        except (SQLAlchemyError, OSError) as e:
            # asyncpg raises RAW OSError (refused/DNS — nothing listens) or
            # TimeoutError (connect timeout, itself an OSError subclass) on a
            # hard outage; SQLAlchemy does
            # not wrap connect-time OS errors — still an infrastructure
            # failure: StorageError → HTTP 503, never a raw 500.
            raise StorageError(ErrorMessage.TRACE_USAGE_AGGREGATE_FAILED.format(error=e)) from e

        summary: LlmUsageSummary | None = None
        series_by_agent: list[LlmUsageAgentBreakdown] = []
        series_by_model: list[LlmUsageModelBreakdown] = []
        for row in rows:
            match row.branch:
                case 0:
                    summary = LlmUsageSummary(
                        total_input_tokens=int(row.total_input_tokens),
                        total_output_tokens=int(row.total_output_tokens),
                        total_tokens=int(row.total_tokens),
                        total_cost_usd=float(row.total_cost_usd),
                        run_count=int(row.run_count),
                        error_count=int(row.error_count),
                        duration_ms_p50=float(row.duration_ms_p50),
                        duration_ms_p95=float(row.duration_ms_p95),
                    )
                    logger.info(LogMessage.TRACE_USAGE_LISTED, agent_name, int(row.run_count), int(row.error_count))
                case 1:
                    series_by_agent.append(
                        LlmUsageAgentBreakdown(
                            agent_name=row.name,
                            input_tokens=int(row.total_input_tokens),
                            output_tokens=int(row.total_output_tokens),
                            total_tokens=int(row.total_tokens),
                            cost_usd=float(row.total_cost_usd),
                            priced=bool(row.priced),
                        )
                    )
                case _:
                    series_by_model.append(
                        LlmUsageModelBreakdown(
                            model_name=row.name,
                            input_tokens=int(row.total_input_tokens),
                            output_tokens=int(row.total_output_tokens),
                            total_tokens=int(row.total_tokens),
                            cost_usd=float(row.total_cost_usd),
                            priced=bool(row.priced),
                        )
                    )
        if summary is None:
            # Defensive: the statement always returns the totals branch; an
            # unexpected shape degrades to a bounded empty summary. Distinct
            # from the no-principal guard above — this is a SQL-shape bug, not
            # an authorization condition (I-C: never reuse UNSCOPED here).
            logger.warning(LogMessage.TRACE_USAGE_UNEXPECTED_SHAPE)
            return LlmUsageSummary()
        summary.series_by_agent = series_by_agent
        summary.series_by_model = series_by_model
        return summary

    async def top_runs(
        self,
        *,
        limit: int,
        agent_name: str | None = None,
        from_ts: datetime | None = None,
        to_ts: datetime | None = None,
    ) -> list[LlmUsageRun]:
        """Return the most expensive completed runs of the window.

        Args:
            limit: Maximum number of runs to return (caller validates bounds).
            agent_name: Optional EXACT (case-sensitive) agent filter.
            from_ts: Inclusive lower bound on the runs' start timestamp.
            to_ts: Inclusive upper bound on the runs' start timestamp.

        Returns:
            Runs sorted by estimated cost descending, capped to ``limit``.

        Raises:
            StorageError: On infrastructure failure.
        """
        uid = self._current_user_id()
        if not uid:
            # Same guard as summarize_usage: covers None AND the empty string.
            logger.warning(LogMessage.TRACE_USAGE_UNSCOPED)
            return []

        stmt = text(_top_sql(agent_name, from_ts, to_ts))
        params = self._params(uid, agent_name, from_ts, to_ts, limit=limit)
        try:
            # Bypass-aware scope with the explicit thread-ownership predicate —
            # same mechanism as :meth:`summarize_usage` (module docstring).
            async with system_rls_context(), self._engine.connect() as conn:
                rows = (await conn.execute(stmt, params)).all()
        except (SQLAlchemyError, OSError) as e:
            # Same mapping as summarize_usage: a hard outage (raw asyncpg
            # OSError, incl. its TimeoutError subclass) is still StorageError
            # → HTTP 503.
            raise StorageError(ErrorMessage.TRACE_USAGE_TOP_FAILED.format(error=e)) from e
        runs = [
            LlmUsageRun(
                thread_id=row.thread_id,
                turn_id=row.turn_id,
                agent_name=row.agent_name,
                # Belt-and-braces (C1): the SQL already COALESCEs to 'unknown';
                # a future SQL change can never 500 the endpoint with a NULL.
                model_name=row.model_name or "unknown",
                total_tokens=int(row.total_tokens),
                cost_usd=float(row.cost_usd),
                duration_ms=float(row.duration_ms),
                started_at=row.started_at,
            )
            for row in rows
        ]
        logger.info(LogMessage.TRACE_USAGE_TOP_LISTED, limit, len(runs))
        return runs
