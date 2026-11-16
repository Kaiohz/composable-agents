"""Deterministic usage-data seeder for the /traces/usage behavioral tests.

Seeds, through the REAL production engine under ``system_rls_context``:

* ``threads`` + ``trace_events`` rows carrying ``metadata.llm_usage`` in the
  exact shape the real chat path persists (see the capture contract in
  ``test_llm_usage_capture.py``), with controlled token counts, timestamps and
  HUMAN→AI durations so the aggregation math (totals, cost, percentiles,
  top-N ordering, error-run semantics) is exactly assertable through the REAL
  HTTP endpoints. This mirrors the established raw-seed pattern of
  ``TestSummaryRepositoryOwnershipPredicate`` (test_traces_routes.py):
  trace/threads seeding is adapter-specific SQL state, the endpoints under
  test remain real.
* one ``llm_prices`` row for the priced model — inserted through the REAL
  system-only write path (bypass-aware RLS policy), with extreme per-million
  prices chosen so token→USD math lands on exact integers
  (cost = input_tokens * 1 + output_tokens * 2 USD).

Fixture data (2 agents × 2 models — completed turns):

  key  agent             model               in    out   duration_ms
  t1   UsageAgentPriced  usage-priced-model    1000   500    100
  t2   UsageAgentPriced  usage-priced-model     500  1000    200
  t3   UsageAgentPriced  usage-priced-model    2000   500    200
  t4   UsageAgentUnpric. usage-unpriced-model  2000  1000    300
  t5   UsageAgentUnpric. usage-unpriced-model  2000  1000    300

* costs: t1=2000, t2=2500, t3=3000 (priced), t4/t5=0 (unpriced) — strictly
  ordered so the top-N run list has a deterministic cost ordering.
* percentiles over completed-turn durations [100, 200, 200, 300, 300]:
  p50=200 and p95=300 under EVERY reasonable percentile method (cont AND
  nearest-rank both agree on these values).
* ``with_errors=True`` adds 4 error runs (HUMAN_MESSAGE only, no AI_MESSAGE,
  zero duration) on the priced agent: if any implementation counted them as
  0-duration runs, p50 would drop to 100 — so an exact p50=200 proves error
  runs are excluded from duration percentiles. Error runs also contribute no
  tokens/cost and never appear in the top-N run list.
"""

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

from src.infrastructure.database.rls_context import system_rls_context

AGENT_PRICED = "UsageAgentPriced"
AGENT_UNPRICED = "UsageAgentUnpriced"
PRICED_MODEL = "usage-priced-model"
UNPRICED_MODEL = "usage-unpriced-model"

# $1,000,000 per 1M input tokens (=$1/token) and $2,000,000 per 1M output
# tokens (=$2/token) → per-run USD = input_tokens * 1 + output_tokens * 2
# (exact integer arithmetic; compared via pytest.approx anyway).
PRICE_INPUT_PER_MILLION = 1_000_000.0
PRICE_OUTPUT_PER_MILLION = 2_000_000.0

_ERROR_TURN_COUNT = 4  # zero-duration error runs (see module docstring)


def _turn(key: str, agent: str, model: str, tin: int, tout: int, dur: int, is_error: bool) -> tuple:
    return (key, agent, model, tin, tout, dur, is_error)


_TURNS: list[tuple] = [
    _turn("t1", AGENT_PRICED, PRICED_MODEL, 1000, 500, 100, False),
    _turn("t2", AGENT_PRICED, PRICED_MODEL, 500, 1000, 200, False),
    _turn("t3", AGENT_PRICED, PRICED_MODEL, 2000, 500, 200, False),
    _turn("t4", AGENT_UNPRICED, UNPRICED_MODEL, 2000, 1000, 300, False),
    _turn("t5", AGENT_UNPRICED, UNPRICED_MODEL, 2000, 1000, 300, False),
]

# Error runs sit on the priced agent (whose other turns are completed), so
# every entry of the agent/model series keeps at least one completed run and
# the series contract stays fully deterministic either way.
_ERROR_TURNS: list[tuple] = [
    _turn(f"e{i}", AGENT_PRICED, PRICED_MODEL, 0, 0, 0, True) for i in range(1, _ERROR_TURN_COUNT + 1)
]


def llm_usage_metadata(model_name: str, input_tokens: int, output_tokens: int) -> dict[str, Any]:
    """The persisted ``llm_usage`` metadata shape (capture contract)."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "model_name": model_name,
        "provider": "test",
    }


def parse_ts(value: datetime | str) -> datetime:
    """Tolerant timestamp parse for HTTP JSON payloads (Z, +00:00, naive)."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _expected(with_errors: bool) -> dict[str, Any]:
    """Compute the expected aggregates independently from the data table."""
    priced = [t for t in _TURNS if t[2] == PRICED_MODEL]
    unpriced = [t for t in _TURNS if t[2] == UNPRICED_MODEL]

    def sums(rows) -> tuple[int, int]:
        return sum(r[3] for r in rows), sum(r[4] for r in rows)

    def cost_usd(rows) -> float:
        return sum(r[3] / 1e6 * PRICE_INPUT_PER_MILLION + r[4] / 1e6 * PRICE_OUTPUT_PER_MILLION for r in rows)

    pin, pout = sums(priced)
    uin, uout = sums(unpriced)
    pcost, ucost = cost_usd(priced), 0.0

    agent_series = [
        {
            "agent_name": AGENT_PRICED,
            "input_tokens": pin,
            "output_tokens": pout,
            "total_tokens": pin + pout,
            "cost_usd": pcost,
            "priced": True,
        },
        {
            "agent_name": AGENT_UNPRICED,
            "input_tokens": uin,
            "output_tokens": uout,
            "total_tokens": uin + uout,
            "cost_usd": ucost,
            "priced": False,
        },
    ]
    model_series = [
        {
            "model_name": PRICED_MODEL,
            "input_tokens": pin,
            "output_tokens": pout,
            "total_tokens": pin + pout,
            "cost_usd": pcost,
            "priced": True,
        },
        {
            "model_name": UNPRICED_MODEL,
            "input_tokens": uin,
            "output_tokens": uout,
            "total_tokens": uin + uout,
            "cost_usd": ucost,
            "priced": False,
        },
    ]
    # Expected top runs, cost descending (strict ordering by data design:
    # t3=3000 > t2=2500 > t1=2000 > t4=t5=0 — ties t4/t5 order not asserted).
    runs_sorted = ["t3", "t2", "t1", "t4", "t5"]
    return {
        "totals": {
            "input_tokens": pin + uin,
            "output_tokens": pout + uout,
            "total_tokens": pin + pout + uin + uout,
            "total_cost_usd": pcost + ucost,
            "run_count": len(_TURNS),
            "error_count": _ERROR_TURN_COUNT if with_errors else 0,
            "duration_ms_p50": 200,
            "duration_ms_p95": 300,
        },
        "series_by_agent": agent_series,
        "series_by_model": model_series,
        "top_order": runs_sorted,
    }


async def seed_usage_data(
    stack, *, user_id: str, base: datetime | None = None, with_errors: bool = False
) -> dict[str, Any]:
    """Seed deterministic usage data + prices and return the expected fixture.

    Args:
        stack: the behavioral ``stack`` fixture dict (real engine).
        user_id: owner of the seeded threads/events (per-user RLS visibility).
        base: anchor datetime for the seeded timestamps (defaults to now).
        with_errors: also seed the 4 error runs (HUMAN_MESSAGE only turns).

    Returns:
        A fixture dict: ``expected`` (totals / series / top order),
        ``turn_ids`` (key → persisted turn id), ``thread_ids``,
        ``started_at`` (key → HUMAN timestamp datetime) and the
        ``window_from``/``window_to`` ISO strings covering all seeded runs.
    """
    base = base or datetime.now(UTC).replace(microsecond=0)
    turns = _TURNS + (_ERROR_TURNS if with_errors else [])
    expected = _expected(with_errors)

    thread_priced = str(uuid.uuid4())
    thread_unpriced = str(uuid.uuid4())
    thread_id_by_agent = {AGENT_PRICED: thread_priced, AGENT_UNPRICED: thread_unpriced}

    turn_ids: dict[str, str] = {}
    started_at: dict[str, datetime] = {}
    event_rows: list[dict[str, Any]] = []
    for i, (key, agent, model, tin, tout, dur, is_error) in enumerate(turns):
        turn_id = str(uuid.uuid4())
        turn_ids[key] = turn_id
        human_ts = base + timedelta(minutes=i + 1)
        started_at[key] = human_ts
        event_rows.append(
            {
                "id": str(uuid.uuid4()),
                "thread_id": thread_id_by_agent[agent],
                "turn_id": turn_id,
                "type": "human_message",
                "meta": None,
                "ts": human_ts,
                "seq": 0,
                "content": f"seed prompt {key}",
            }
        )
        if not is_error:
            event_rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "thread_id": thread_id_by_agent[agent],
                    "turn_id": turn_id,
                    "type": "ai_message",
                    "meta": json.dumps({"llm_usage": llm_usage_metadata(model, tin, tout)}),
                    "ts": human_ts + timedelta(milliseconds=dur),
                    "seq": 1,
                    "content": json.dumps({"role": "ai", "content": "seed-ok"}),
                }
            )

    async with system_rls_context(), stack["engine"].begin() as conn:
        for tid, agent in ((thread_priced, AGENT_PRICED), (thread_unpriced, AGENT_UNPRICED)):
            await conn.execute(
                text(
                    "INSERT INTO threads (id, agent_name, user_id, created_at, updated_at) "
                    "VALUES (:id, :agent, :user_id, :created, :created)"
                ),
                {"id": tid, "agent": agent, "user_id": user_id, "created": base},
            )
        for row in event_rows:
            await conn.execute(
                text(
                    "INSERT INTO trace_events (id, thread_id, turn_id, type, source, name, content, "
                    "metadata, timestamp, sequence) VALUES (:id, :thread_id, :turn_id, :type, "
                    "NULL, NULL, :content, CAST(:meta AS jsonb), :ts, :seq)"
                ),
                row,
            )
        # Priced-model reference row through the REAL system-only write path
        # (idempotent upsert keyed on model_name — enforces single-row-per-model).
        await conn.execute(
            text(
                "INSERT INTO llm_prices (model_name, provider, input_price_per_million, output_price_per_million) "
                "VALUES (:m, 'test', :pi, :po) "
                "ON CONFLICT (model_name) DO UPDATE SET "
                "input_price_per_million = :pi, output_price_per_million = :po"
            ),
            {"m": PRICED_MODEL, "pi": PRICE_INPUT_PER_MILLION, "po": PRICE_OUTPUT_PER_MILLION},
        )

    return {
        "expected": expected,
        "turn_ids": turn_ids,
        "thread_ids": {"priced": thread_priced, "unpriced": thread_unpriced},
        "started_at": started_at,
        "window_from": (base - timedelta(minutes=1)).isoformat(),
        "window_to": (base + timedelta(minutes=len(turns) + 10)).isoformat(),
    }


ORPHAN_AGENT = "OrphanAgent"


async def seed_orphan_run(stack, *, user_id: str, base: datetime | None = None) -> dict[str, Any]:
    """Seed one completed run whose model name is NOT resolvable (C1 regression).

    The run's agent configuration was deleted (no ``agent_configs`` row for
    ``threads.agent_name``) AND the turn's ``ai_message`` event carries NO
    ``metadata.llm_usage``: both the usage-model and the agent-model
    resolutions are NULL, which used to propagate NULL into
    ``LlmUsageRun.model_name`` and crash ``GET /usage/top``. After the C1 fix
    the run is labeled model ``"unknown"`` — counted everywhere, unpriced
    (``cost=0`` / ``priced=false``) — exactly the spec's "the pipeline never
    breaks" contract.

    Args:
        stack: the behavioral ``stack`` fixture dict (real engine).
        user_id: owner of the seeded thread/events (per-user RLS visibility).
        base: anchor datetime for the seeded timestamps (defaults to now).

    Returns:
        ``thread_id`` / ``turn_id`` / ``started_at`` of the seeded orphan run
        (the AI event sits 50ms after the HUMAN event).
    """
    base = base or datetime.now(UTC).replace(microsecond=0)
    thread_id = str(uuid.uuid4())
    turn_id = str(uuid.uuid4())
    human_ts = base
    ai_ts = base + timedelta(milliseconds=50)
    async with system_rls_context(), stack["engine"].begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO threads (id, agent_name, user_id, created_at, updated_at) "
                "VALUES (:id, :agent, :user_id, :created, :created)"
            ),
            {"id": thread_id, "agent": ORPHAN_AGENT, "user_id": user_id, "created": base},
        )
        for event in (
            {
                "id": str(uuid.uuid4()),
                "type": "human_message",
                "meta": None,
                "ts": human_ts,
                "seq": 0,
                "content": "orphan prompt",
            },
            {
                "id": str(uuid.uuid4()),
                "type": "ai_message",
                "meta": None,
                "ts": ai_ts,
                "seq": 1,
                "content": json.dumps({"role": "ai", "content": "orphan-ok"}),
            },
        ):
            await conn.execute(
                text(
                    "INSERT INTO trace_events (id, thread_id, turn_id, type, source, name, content, "
                    "metadata, timestamp, sequence) VALUES (:id, :thread_id, :turn_id, :type, "
                    "NULL, NULL, :content, CAST(:meta AS jsonb), :ts, :seq)"
                ),
                {
                    "id": event["id"],
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "type": event["type"],
                    "content": event["content"],
                    "meta": event["meta"],
                    "ts": event["ts"],
                    "seq": event["seq"],
                },
            )
    return {
        "thread_id": thread_id,
        "turn_id": turn_id,
        "agent": ORPHAN_AGENT,
        "started_at": human_ts,
        "ai_ts": ai_ts,
    }


HITL_RESUME_AGENT = "HITLResumeAgent"
MULTI_AI_AGENT = "MultiAiAgent"


async def _seed_agent_turn(
    stack,
    *,
    user_id: str,
    base: datetime,
    agent: str,
    events: list[dict[str, Any]],
) -> tuple[str, str]:
    """Persist one agent-config-free thread with raw ``trace_events`` rows.

    Same raw-seed pattern as :func:`seed_orphan_run` (mirrors what the real
    capture path persists; the endpoints under test stay real).

    Args:
        stack: the behavioral ``stack`` fixture dict (real engine).
        user_id: owner of the thread (per-user RLS visibility).
        base: anchor datetime for the thread timestamps.
        agent: thread's ``agent_name`` (no ``agent_configs`` row is created).
        events: pre-built event dicts (``type``/``name``/``meta``/``ts``/``seq``/``content``).

    Returns:
        The persisted ``(thread_id, turn_id)`` pair (one fresh turn id).
    """
    thread_id = str(uuid.uuid4())
    turn_id = str(uuid.uuid4())
    async with system_rls_context(), stack["engine"].begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO threads (id, agent_name, user_id, created_at, updated_at) "
                "VALUES (:id, :agent, :user_id, :created, :created)"
            ),
            {"id": thread_id, "agent": agent, "user_id": user_id, "created": base},
        )
        for event in events:
            await conn.execute(
                text(
                    "INSERT INTO trace_events (id, thread_id, turn_id, type, source, name, content, "
                    "metadata, timestamp, sequence) VALUES (:id, :thread_id, :turn_id, :type, "
                    "NULL, :name, :content, CAST(:meta AS jsonb), :ts, :seq)"
                ),
                {
                    "id": str(uuid.uuid4()),
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "type": event["type"],
                    "name": event["name"],
                    "content": event["content"],
                    "meta": event["meta"],
                    "ts": event["ts"],
                    "seq": event["seq"],
                },
            )
    return thread_id, turn_id


async def seed_hitl_resume_run(stack, *, user_id: str, base: datetime | None = None) -> dict[str, Any]:
    """Seed one completed HITL-RESUME turn: HITL_DECISION + AI_MESSAGE, NO human.

    The resume path mints a NEW turn id (``send_message``) whose leading event
    is a ``hitl_decision`` (``DeepAgentRunner._build_resume_decisions``) — the
    turn owns NO ``human_message``. The window pushdown anchors a turn on
    whichever opener it owns (C2), so windowed requests MUST keep counting
    this turn: its ``llm_usage`` (1 in / 1 out on the unpriced model → 2
    tokens, cost 0) and its duration (85ms from the HITL_DECISION opener to
    the AI_MESSAGE).

    Args:
        stack: the behavioral ``stack`` fixture dict (real engine).
        user_id: owner of the thread (per-user RLS visibility).
        base: anchor datetime (defaults to now).

    Returns:
        ``thread_id`` / ``turn_id`` / ``agent`` / ``model`` / ``started_at``
        (the HITL_DECISION timestamp) / ``ai_ts`` plus ``window_from`` /
        ``window_to`` covering the run.
    """
    base = base or datetime.now(UTC).replace(microsecond=0)
    decision_ts = base
    ai_ts = base + timedelta(milliseconds=85)
    thread_id, turn_id = await _seed_agent_turn(
        stack,
        user_id=user_id,
        base=base,
        agent=HITL_RESUME_AGENT,
        events=[
            {
                "type": "hitl_decision",
                "name": "approve",
                "meta": json.dumps({"tool_call_id": "call-1", "edits": None}),
                "ts": decision_ts,
                "seq": 0,
                "content": "resume reason",
            },
            {
                "type": "ai_message",
                "name": None,
                "meta": json.dumps({"llm_usage": llm_usage_metadata(UNPRICED_MODEL, 1, 1)}),
                "ts": ai_ts,
                "seq": 1,
                "content": json.dumps({"role": "ai", "content": "resumed"}),
            },
        ],
    )
    return {
        "thread_id": thread_id,
        "turn_id": turn_id,
        "agent": HITL_RESUME_AGENT,
        "model": UNPRICED_MODEL,
        "started_at": decision_ts,
        "ai_ts": ai_ts,
        "window_from": (base - timedelta(minutes=1)).isoformat(),
        "window_to": (base + timedelta(minutes=10)).isoformat(),
    }


async def seed_multi_ai_turn(stack, *, user_id: str, base: datetime | None = None) -> dict[str, Any]:
    """Seed one turn carrying TWO ``ai_message`` events, each with its own usage.

    Pins the SUM aggregation (I2): the endpoint must report the SUM of both
    calls' tokens (130 in / 420 out / 550 total — a MAX implementation would
    report 100 / 400 / 400) and derive the duration from the LAST
    ``ai_message`` timestamp (150ms after the human opener).

    Args:
        stack: the behavioral ``stack`` fixture dict (real engine).
        user_id: owner of the thread (per-user RLS visibility).
        base: anchor datetime (defaults to now).

    Returns:
        Same shape as :func:`seed_hitl_resume_run` (``started_at`` = the
        HUMAN_MESSAGE opener).
    """
    base = base or datetime.now(UTC).replace(microsecond=0)
    human_ts = base
    ai1_ts = base + timedelta(milliseconds=40)
    ai2_ts = base + timedelta(milliseconds=150)
    thread_id, turn_id = await _seed_agent_turn(
        stack,
        user_id=user_id,
        base=base,
        agent=MULTI_AI_AGENT,
        events=[
            {
                "type": "human_message",
                "name": None,
                "meta": None,
                "ts": human_ts,
                "seq": 0,
                "content": "multi-ai prompt",
            },
            {
                "type": "ai_message",
                "name": None,
                "meta": json.dumps({"llm_usage": llm_usage_metadata(UNPRICED_MODEL, 100, 20)}),
                "ts": ai1_ts,
                "seq": 1,
                "content": json.dumps({"role": "ai", "content": "call-1"}),
            },
            {
                "type": "ai_message",
                "name": None,
                "meta": json.dumps({"llm_usage": llm_usage_metadata(UNPRICED_MODEL, 30, 400)}),
                "ts": ai2_ts,
                "seq": 2,
                "content": json.dumps({"role": "ai", "content": "call-2"}),
            },
        ],
    )
    return {
        "thread_id": thread_id,
        "turn_id": turn_id,
        "agent": MULTI_AI_AGENT,
        "model": UNPRICED_MODEL,
        "started_at": human_ts,
        "ai_ts": ai2_ts,
        "window_from": (base - timedelta(minutes=1)).isoformat(),
        "window_to": (base + timedelta(minutes=10)).isoformat(),
    }


async def seed_usage_turn(
    stack,
    *,
    user_id: str,
    agent: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    base: datetime | None = None,
) -> dict[str, Any]:
    """Seed ONE completed turn carrying ``llm_usage`` for the given model.

    Generic single-turn seeder for the price-resolution tests (BUG-002): no
    ``agent_configs`` row is created (the aggregation falls back to the usage
    model name) and NO ``llm_prices`` row is added — the resolution exercises
    exactly the rows the startup seeder (or a previous fixture) left behind.

    Args:
        stack: the behavioral ``stack`` fixture dict (real engine).
        user_id: owner of the thread (per-user RLS visibility).
        agent: thread's ``agent_name`` (exact-match filter key if needed).
        model: the persisted ``llm_usage.model_name``.
        input_tokens / output_tokens: deterministic token counts.
        base: anchor datetime (defaults to now).

    Returns:
        ``thread_id`` / ``turn_id`` / ``agent`` / ``model`` / ``started_at``
        (the HUMAN opener) and the ``window`` dict for the usage endpoints.
    """
    started = base or datetime.now(UTC).replace(microsecond=0)
    thread_id, turn_id = await _seed_agent_turn(
        stack,
        user_id=user_id,
        base=started,
        agent=agent,
        events=[
            {
                "type": "human_message",
                "name": None,
                "meta": None,
                "ts": started,
                "seq": 0,
                "content": f"pricing prompt {model}",
            },
            {
                "type": "ai_message",
                "name": None,
                "meta": json.dumps({"llm_usage": llm_usage_metadata(model, input_tokens, output_tokens)}),
                "ts": started + timedelta(milliseconds=60),
                "seq": 1,
                "content": json.dumps({"role": "ai", "content": "priced"}),
            },
        ],
    )
    return {
        "thread_id": thread_id,
        "turn_id": turn_id,
        "agent": agent,
        "model": model,
        "started_at": started,
        "window": {
            "from": (started - timedelta(minutes=1)).isoformat(),
            "to": (started + timedelta(minutes=10)).isoformat(),
        },
    }
