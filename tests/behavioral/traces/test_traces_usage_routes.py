"""Behavioral tests — GET /api/v1/traces/usage aggregation (dashboard §4, §6, §7).

Route under test: ``GET /api/v1/traces/usage?agent_name=<exact>&from=<iso>&to=<iso>&group_by=agent|model``

Aggregation contract fixed here for the implementation (Step 2):

* response body (exact keys):
  ``total_input_tokens``, ``total_output_tokens``, ``total_tokens``,
  ``total_cost_usd``, ``run_count``, ``error_count``, ``duration_ms_p50``,
  ``duration_ms_p95``, ``series``.
* ``run_count`` counts COMPLETED runs (turns with a terminal AI_MESSAGE);
  ``error_count`` counts ERROR runs (turns missing the terminal AI_MESSAGE).
  The two sets are disjoint — every run in the window is exactly one of them.
* ``duration_ms_p50``/``p95`` are computed over COMPLETED runs ONLY
  (HUMAN_MESSAGE → AI_MESSAGE per turn). Error runs are excluded: they carry
  no AI timestamp, contribute no duration and never distort the percentiles.
* Cost = per-run tokens × the ``llm_prices`` row for ``llm_usage.model_name``
  (input_price_per_million / output_price_per_million, USD). Models with no
  price row are ``cost=0`` + ``priced=false`` — the pipeline never breaks.
* ``series`` (when ``group_by`` is absent) holds BOTH breakdowns:
  ``series.by_agent`` entries ``{agent_name, input_tokens, output_tokens,
  total_tokens, cost_usd, priced}`` and ``series.by_model`` entries
  ``{model_name, input_tokens, output_tokens, total_tokens, cost_usd,
  priced}``. ``group_by=agent`` returns only ``{"by_agent": [...]}``,
  ``group_by=model`` only ``{"by_model": [...]}``.
* Filters: ``agent_name`` is an EXACT match (same TRC-001 semantics as the
  existing trace list), ``from``/``to`` ISO bounds on the window.
* Per-user RLS isolation: usage aggregates see ONLY the authenticated user's
  threads/trace_events (user-a never sees user-b's usage, filter probing
  included). Both aggregation endpoints require authentication (401).

Data sources: real chat-path invocations via StubLlmServer for the capture→
aggregate e2e leg, and the deterministic deterministic seeder
(``usage_fixture.seed_usage_data``) for the exact-math legs (real
PostgreSQL, real HTTP endpoints — the seeded rows mirror exactly what the
capture persists, so both legs exercise the same aggregate SQL).
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from src.application.use_cases.trace_usage import GetTopUsageRunsUseCase, GetTraceUsageUseCase
from src.dependencies import get_top_usage_runs_use_case, get_trace_usage_use_case
from src.infrastructure.database.rls_context import current_user_id
from src.infrastructure.postgres_trace.usage_repository import PostgresTraceUsageRepository
from tests.behavioral.conftest import configure_llm_credentials, create_thread, invoke_agent
from tests.behavioral.traces.usage_fixture import (
    AGENT_PRICED,
    AGENT_UNPRICED,
    ORPHAN_AGENT,
    UNPRICED_MODEL,
    parse_ts,
    seed_hitl_resume_run,
    seed_multi_ai_turn,
    seed_orphan_run,
    seed_usage_data,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

USAGE_KEYS = {
    "total_input_tokens",
    "total_output_tokens",
    "total_tokens",
    "total_cost_usd",
    "run_count",
    "error_count",
    "duration_ms_p50",
    "duration_ms_p95",
    "series",
}


async def get_usage(client, headers, **params):
    return await client.get("/api/v1/traces/usage", headers=headers, params=params)


def _agent_yaml(name: str) -> str:
    return f'name: {name}\nmodel: stub-model\nsystem_prompt: "Usage routes agent."\ntools: []\ndebug: false\n'


async def _create_agent(client, headers, name: str) -> str:
    resp = await client.post(
        "/api/v1/agents",
        headers=headers,
        data={"agent_name": name},
        files={"file": ("agent.yaml", _agent_yaml(name).encode(), "text/yaml")},
    )
    assert resp.status_code == 201, resp.text
    return name


@pytest.fixture
def as_user():
    from contextlib import contextmanager

    @contextmanager
    def _ctx(user_id: str | None):
        token = current_user_id.set(user_id)
        try:
            yield
        finally:
            current_user_id.reset(token)

    return _ctx


class TestUsageRealChatPath:
    async def test_real_invocations_aggregate_into_usage_totals(self, client, users, stub_llm):
        """Capture → aggregate e2e: two real turns (stub usage 1/1/2) through
        the chat path must aggregate to run_count=2, 2 in / 2 out / 4 total
        tokens. 'stub-model' is unpriced → cost 0.0 and priced=false."""
        stub_llm.reset()
        agent_name = await _create_agent(client, users["a"]["headers"], f"UsageE2E{uuid.uuid4().hex[:6]}")
        thread = await create_thread(client, users["a"]["headers"], agent_name)
        assert thread.status_code == 201, thread.text
        await configure_llm_credentials(client, users["a"]["headers"], api_key="sk-cred-usage-e2e")

        resp = await invoke_agent(client, users["a"]["headers"], thread.json()["id"])
        assert resp.status_code == 200, resp.text
        resp = await invoke_agent(client, users["a"]["headers"], thread.json()["id"], message="second turn")
        assert resp.status_code == 200, resp.text

        body = await get_usage(client, users["a"]["headers"], agent_name=agent_name)
        assert body.status_code == 200, body.text
        data = body.json()
        assert set(data) == USAGE_KEYS, f"usage response contract: {set(data) ^ USAGE_KEYS}"
        assert data["run_count"] == 2, data
        assert data["error_count"] == 0, data
        assert data["total_input_tokens"] == 2, data
        assert data["total_output_tokens"] == 2, data
        assert data["total_tokens"] == 4, data
        assert data["total_cost_usd"] == pytest.approx(0.0), "unpriced stub-model must cost zero"
        assert isinstance(data["duration_ms_p50"], (int, float)) and data["duration_ms_p50"] >= 0
        assert isinstance(data["duration_ms_p95"], (int, float)) and data["duration_ms_p95"] >= 0
        assert data["duration_ms_p50"] <= data["duration_ms_p95"], data

        # Unpriced model breakdown: tokens present, cost 0, priced=false.
        by_model = data["series"]["by_model"]
        model_entry = next(e for e in by_model if e["model_name"] == "stub-model")
        assert set(model_entry) == {"model_name", "input_tokens", "output_tokens", "total_tokens", "cost_usd", "priced"}
        assert model_entry["total_tokens"] == 4
        assert model_entry["cost_usd"] == pytest.approx(0.0)
        assert model_entry["priced"] is False

        # Agent breakdown mirrors the same run set.
        by_agent = data["series"]["by_agent"]
        agent_entry = next(e for e in by_agent if e["agent_name"] == agent_name)
        assert set(agent_entry) == {"agent_name", "input_tokens", "output_tokens", "total_tokens", "cost_usd", "priced"}
        assert agent_entry["total_tokens"] == 4
        assert agent_entry["priced"] is False

    async def test_group_by_narrows_the_series_breakdowns(self, client, users, stub_llm):
        stub_llm.reset()
        agent_name = await _create_agent(client, users["a"]["headers"], f"UsageGB{uuid.uuid4().hex[:6]}")
        thread = await create_thread(client, users["a"]["headers"], agent_name)
        await configure_llm_credentials(client, users["a"]["headers"], api_key="sk-cred-usage-gb")
        resp = await invoke_agent(client, users["a"]["headers"], thread.json()["id"])
        assert resp.status_code == 200, resp.text

        by_agent = (await get_usage(client, users["a"]["headers"], agent_name=agent_name, group_by="agent")).json()
        assert set(by_agent["series"]) == {"by_agent"}, by_agent["series"]
        assert [e["agent_name"] for e in by_agent["series"]["by_agent"]] == [agent_name]

        by_model = (await get_usage(client, users["a"]["headers"], agent_name=agent_name, group_by="model")).json()
        assert set(by_model["series"]) == {"by_model"}, by_model["series"]
        assert [e["model_name"] for e in by_model["series"]["by_model"]] == ["stub-model"]

    async def test_run_without_usage_metadata_reports_zero_tokens(self, client, users, stub_llm):
        """Dashboard §2: a turn whose LLM returned no usage_metadata still
        counts as a completed run with zero tokens, cost=0, priced=false."""
        stub_llm.reset(usage=None)
        agent_name = await _create_agent(client, users["a"]["headers"], f"UsageNoMeta{uuid.uuid4().hex[:6]}")
        thread = await create_thread(client, users["a"]["headers"], agent_name)
        await configure_llm_credentials(client, users["a"]["headers"], api_key="sk-cred-usage-nometa")
        resp = await invoke_agent(client, users["a"]["headers"], thread.json()["id"])
        assert resp.status_code == 200, resp.text

        data = (await get_usage(client, users["a"]["headers"], agent_name=agent_name)).json()
        assert data["run_count"] == 1, data
        assert data["error_count"] == 0, data
        assert data["total_input_tokens"] == 0
        assert data["total_output_tokens"] == 0
        assert data["total_tokens"] == 0
        assert data["total_cost_usd"] == pytest.approx(0.0)
        model_entry = next(e for e in data["series"]["by_model"] if e["model_name"] == "stub-model")
        assert model_entry["priced"] is False and model_entry["total_tokens"] == 0

    async def test_error_run_from_real_stream_path_is_counted(self, client, users, stack, stub_llm):
        """Dashboard §6: a failing LLM turn (real stream path persists the
        HUMAN_MESSAGE then errors with no AI_MESSAGE) counts as an error run
        — and a subsequent successful turn counts as a completed run."""
        agent_name = await _create_agent(client, users["a"]["headers"], f"UsageErr{uuid.uuid4().hex[:6]}")
        thread = await create_thread(client, users["a"]["headers"], agent_name)
        thread_id = thread.json()["id"]
        await configure_llm_credentials(client, users["a"]["headers"], api_key="sk-cred-usage-err")

        stub_llm.reset(mode="fail")
        resp = await client.post(
            f"/api/v1/chat/{thread_id}/stream", headers=users["a"]["headers"], json={"message": "boom"}
        )
        assert resp.status_code == 200, resp.text
        assert '"error"' in resp.text, "the stream must surface the LLM failure payload"

        stub_llm.reset()
        ok = await invoke_agent(client, users["a"]["headers"], thread_id)
        assert ok.status_code == 200, ok.text

        data = (await get_usage(client, users["a"]["headers"], agent_name=agent_name)).json()
        assert data["run_count"] == 1, f"only the successful turn is a completed run: {data}"
        assert data["error_count"] == 1, f"the failed turn must be counted in error_count: {data}"
        assert data["total_tokens"] == 2  # only the successful stub turn (1 in / 1 out)


class TestUsageAggregationMath:
    async def test_exact_totals_cost_and_series_on_multi_agent_multi_model(self, stack, client, users):
        """Dashboard §4 — deterministic seeded fixture (2 agents × 2 models,
        priced + unpriced): the endpoint returns the exact totals, exact USD
        cost math and exact agent/model breakdowns."""
        fixture = await seed_usage_data(stack, user_id=users["a"]["id"])
        expected = fixture["expected"]

        data = (
            await get_usage(
                client,
                users["a"]["headers"],
                **{"from": fixture["window_from"], "to": fixture["window_to"]},
            )
        ).json()
        assert set(data) == USAGE_KEYS
        assert data["total_input_tokens"] == expected["totals"]["input_tokens"] == 7500
        assert data["total_output_tokens"] == expected["totals"]["output_tokens"] == 4000
        assert data["total_tokens"] == expected["totals"]["total_tokens"] == 11500
        assert data["total_cost_usd"] == pytest.approx(expected["totals"]["total_cost_usd"]) == pytest.approx(7500.0)
        assert data["run_count"] == 5
        assert data["error_count"] == 0
        assert data["duration_ms_p50"] == 200
        assert data["duration_ms_p95"] == 300

        def _norm(entries, key_field):
            return sorted(
                ({**e, "cost_usd": round(e["cost_usd"], 6)} for e in entries),
                key=lambda e: e[key_field],
            )

        def _norm_expected(entries, key_field):
            return sorted(
                ({**e, "cost_usd": round(e["cost_usd"], 6)} for e in entries),
                key=lambda e: e[key_field],
            )

        assert _norm(data["series"]["by_agent"], "agent_name") == _norm_expected(
            expected["series_by_agent"], "agent_name"
        )
        assert _norm(data["series"]["by_model"], "model_name") == _norm_expected(
            expected["series_by_model"], "model_name"
        )

    async def test_agent_filter_is_exact_and_scopes_all_metrics(self, stack, client, users):
        await seed_usage_data(stack, user_id=users["a"]["id"])

        data = (await get_usage(client, users["a"]["headers"], agent_name=AGENT_PRICED)).json()
        assert data["run_count"] == 3
        assert data["error_count"] == 0
        assert data["total_input_tokens"] == 3500
        assert data["total_output_tokens"] == 2000
        assert data["total_tokens"] == 5500
        assert data["total_cost_usd"] == pytest.approx(7500.0)
        agent_entries = data["series"]["by_agent"]
        assert len(agent_entries) == 1 and agent_entries[0]["agent_name"] == AGENT_PRICED

        # EXACT match — never partial, never case-insensitive (TRC-001 parity).
        for wrong in (AGENT_PRICED.lower(), AGENT_PRICED[:5], f"x{AGENT_PRICED}"):
            empty = (await get_usage(client, users["a"]["headers"], agent_name=wrong)).json()
            assert empty["run_count"] == 0 and empty["total_tokens"] == 0
            assert empty["series"]["by_agent"] == []

    async def test_priced_and_unpriced_models_in_series(self, stack, client, users):
        """Dashboard §2/§7: the priced model resolves USD from llm_prices; the
        unknown model reports cost=0 + priced=false without breaking the
        aggregate math."""
        await seed_usage_data(stack, user_id=users["a"]["id"])
        data = (await get_usage(client, users["a"]["headers"])).json()

        priced_entry = next(e for e in data["series"]["by_model"] if e["model_name"] != UNPRICED_MODEL)
        assert priced_entry["model_name"] == "usage-priced-model"
        assert priced_entry["priced"] is True
        assert priced_entry["cost_usd"] == pytest.approx(7500.0)
        assert priced_entry["total_tokens"] == 5500

        unpriced_entry = next(e for e in data["series"]["by_model"] if e["priced"] is False)
        assert unpriced_entry["model_name"] == UNPRICED_MODEL
        assert unpriced_entry["cost_usd"] == pytest.approx(0.0)
        assert unpriced_entry["total_tokens"] == 6000
        assert unpriced_entry["input_tokens"] == 4000 and unpriced_entry["output_tokens"] == 2000

        # The global total carries both models: priced cost only.
        assert data["total_cost_usd"] == pytest.approx(7500.0)


class TestUsageWindowAndValidation:
    async def test_future_window_yields_zero_usage(self, stack, client, users):
        from datetime import UTC, datetime, timedelta

        await seed_usage_data(stack, user_id=users["a"]["id"])
        now = datetime.now(UTC).replace(microsecond=0)
        frm = (now + timedelta(days=1)).isoformat()
        to = (now + timedelta(days=2)).isoformat()

        data = (await get_usage(client, users["a"]["headers"], **{"from": frm, "to": to})).json()
        assert data["run_count"] == 0 and data["error_count"] == 0
        assert data["total_tokens"] == 0 and data["total_cost_usd"] == pytest.approx(0.0)
        assert data["series"]["by_agent"] == [] and data["series"]["by_model"] == []

    @pytest.mark.parametrize(
        ("frm", "to"),
        [
            ("not-a-date", "2027-01-01T00:00:00Z"),
            ("2026-13-45T00:00:00Z", "2027-01-01T00:00:00Z"),
            ("2026-01-01T00:00:00Z", "not-a-date"),
            ("2026-01-01T00:00:00Z", "2026-13-45T00:00:00Z"),
        ],
    )
    async def test_malformed_bounds_are_client_errors(self, stack, client, users, frm, to):
        await seed_usage_data(stack, user_id=users["a"]["id"])
        resp = await get_usage(client, users["a"]["headers"], **{"from": frm, "to": to})
        assert resp.status_code in {400, 422}, f"{frm!r}/{to!r} must be rejected, got {resp.status_code}"


class TestUsageAuthorizationAndIsolation:
    async def test_unauthenticated_401(self, client):
        resp = await get_usage(client, {})
        assert resp.status_code == 401

    async def test_user_b_cannot_see_user_a_usage(self, stack, client, users):
        """Dashboard §8 — RLS isolation: user-a's seeded usage is invisible to
        user-b, including via the agent filter (no enumeration)."""
        fixture = await seed_usage_data(stack, user_id=users["a"]["id"])

        other = (await get_usage(client, users["b"]["headers"])).json()
        assert other["run_count"] == 0 and other["error_count"] == 0
        assert other["total_tokens"] == 0 and other["total_cost_usd"] == pytest.approx(0.0)
        assert other["series"]["by_agent"] == [] and other["series"]["by_model"] == []

        # Filtering by user-a's agent must not disclose its existence either.
        probed = (await get_usage(client, users["b"]["headers"], agent_name=AGENT_PRICED)).json()
        assert probed["run_count"] == 0 and probed["series"]["by_agent"] == []

        probed_model = (await get_usage(client, users["b"]["headers"], agent_name=AGENT_UNPRICED)).json()
        assert probed_model["run_count"] == 0

        # And owner-visible within the seeded window only for user-a.
        own = (
            await get_usage(
                client,
                users["a"]["headers"],
                **{"from": fixture["window_from"], "to": fixture["window_to"]},
            )
        ).json()
        assert own["run_count"] == 5


class TestMissingModelName:
    async def test_orphan_completed_run_is_labeled_unknown_not_500(self, stack, client, users):
        """C1 regression (aggregation leg): a completed run whose agent was
        deleted and whose LLM emitted no usage metadata resolves model_name
        NULL — the endpoint must answer 200 and COUNT the run everywhere under
        the model label ``"unknown"`` (never crash, never drop the run)."""
        await seed_orphan_run(stack, user_id=users["a"]["id"])

        resp = await get_usage(client, users["a"]["headers"])
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["run_count"] == 1, "the orphan run is COMPLETED (has an ai_message)"
        assert data["error_count"] == 0
        assert data["total_tokens"] == 0
        assert data["total_cost_usd"] == pytest.approx(0.0)  # 'unknown' is unpriced

        unknown = next(e for e in data["series"]["by_model"] if e["model_name"] == "unknown")
        assert unknown["priced"] is False
        assert unknown["cost_usd"] == pytest.approx(0.0)
        assert unknown["total_tokens"] == 0

        orphan_agent = next(e for e in data["series"]["by_agent"] if e["agent_name"] == ORPHAN_AGENT)
        assert orphan_agent["total_tokens"] == 0


class TestHitlResumeTurn:
    async def test_hitl_resume_turn_is_counted_inside_the_window(self, stack, client, users):
        """C2 regression: a HITL-RESUME turn (HITL_DECISION + AI_MESSAGE and
        NO human_message — the resume mints a NEW turn id) must stay counted
        in EVERY windowed request: run_count, tokens, both series and the
        top-N list. Its opener anchor is the HITL_DECISION timestamp."""
        fixture = await seed_hitl_resume_run(stack, user_id=users["a"]["id"])
        window = {"from": fixture["window_from"], "to": fixture["window_to"]}

        data = (await get_usage(client, users["a"]["headers"], **window)).json()
        assert data["run_count"] == 1, "the resumed LLM call must be counted (C2)"
        assert data["error_count"] == 0
        assert data["total_input_tokens"] == 1
        assert data["total_output_tokens"] == 1
        assert data["total_tokens"] == 2

        agent_entry = next(e for e in data["series"]["by_agent"] if e["agent_name"] == fixture["agent"])
        assert agent_entry["total_tokens"] == 2
        assert agent_entry["cost_usd"] == pytest.approx(0.0)

        model_entry = next(e for e in data["series"]["by_model"] if e["model_name"] == fixture["model"])
        assert model_entry["total_tokens"] == 2
        assert model_entry["priced"] is False

        top = await client.get("/api/v1/traces/usage/top", headers=users["a"]["headers"], params={**window, "limit": 5})
        assert top.status_code == 200, top.text
        items = top.json()["items"]
        assert len(items) == 1, f"the resumed turn must be ranked too: {items}"
        assert items[0]["turn_id"] == fixture["turn_id"]
        assert items[0]["thread_id"] == fixture["thread_id"]
        assert items[0]["model_name"] == fixture["model"]
        # started_at = the run OPENER (the HITL_DECISION timestamp); duration
        # spans the seeded DECISION→AI gap (85ms).
        assert parse_ts(items[0]["started_at"]) == fixture["started_at"]
        assert items[0]["duration_ms"] == pytest.approx(85.0)

    async def test_hitl_resume_turn_outside_the_window_is_excluded(self, stack, client, users):
        """The anchor IS the opener timestamp — the pushdown still filters: a
        window strictly after the resume turn excludes it."""
        await seed_hitl_resume_run(stack, user_id=users["a"]["id"])
        now = datetime.now(UTC).replace(microsecond=0)

        future = {"from": (now + timedelta(days=1)).isoformat(), "to": (now + timedelta(days=2)).isoformat()}
        data = (await get_usage(client, users["a"]["headers"], **future)).json()
        assert data["run_count"] == 0 and data["error_count"] == 0
        assert data["total_tokens"] == 0
        assert data["series"]["by_agent"] == [] and data["series"]["by_model"] == []


class TestMultiAiMessageTurn:
    async def test_multiple_ai_messages_are_summed_with_last_ai_duration(self, stack, client, users):
        """I2 pin: one turn with TWO ``ai_message`` events (each with its own
        ``llm_usage``) aggregates the SUM of both calls (130 in / 420 out /
        550 total — a MAX implementation would report 100 / 400 / 400) and
        derives the duration from the LAST ai_message timestamp (150ms)."""
        fixture = await seed_multi_ai_turn(stack, user_id=users["a"]["id"])
        window = {"from": fixture["window_from"], "to": fixture["window_to"]}

        data = (await get_usage(client, users["a"]["headers"], **window)).json()
        assert data["run_count"] == 1
        assert data["error_count"] == 0
        assert data["total_input_tokens"] == 130
        assert data["total_output_tokens"] == 420
        assert data["total_tokens"] == 550
        assert data["total_cost_usd"] == pytest.approx(0.0)  # unpriced model
        # Percentiles over the single completed run: duration from the LAST ts.
        assert data["duration_ms_p50"] == pytest.approx(150.0)
        assert data["duration_ms_p95"] == pytest.approx(150.0)

        # The series carry the same summed tokens — never per-call values.
        model_entry = next(e for e in data["series"]["by_model"] if e["model_name"] == fixture["model"])
        assert model_entry["input_tokens"] == 130 and model_entry["output_tokens"] == 420
        assert model_entry["total_tokens"] == 550

        top = await client.get("/api/v1/traces/usage/top", headers=users["a"]["headers"], params={**window, "limit": 5})
        assert top.status_code == 200, top.text
        items = top.json()["items"]
        assert len(items) == 1
        assert items[0]["turn_id"] == fixture["turn_id"]
        assert items[0]["total_tokens"] == 550
        assert items[0]["duration_ms"] == pytest.approx(150.0)


class TestNaiveWindowBounds:
    async def test_naive_bounds_are_interpreted_as_utc(self, stack, client, users):
        """I4 — naive (offset-less) ``from``/``to`` must be treated as UTC at
        the route boundary: identical aggregates to the equivalent
        offset-aware bounds (never the DB session timezone)."""

        def naive(value: str) -> str:
            return parse_ts(value).replace(tzinfo=None).isoformat()

        fixture = await seed_usage_data(stack, user_id=users["a"]["id"])
        window = {"from": fixture["window_from"], "to": fixture["window_to"]}

        aware = (await get_usage(client, users["a"]["headers"], **window)).json()
        shifted = (await get_usage(client, users["a"]["headers"], **{k: naive(v) for k, v in window.items()})).json()
        assert aware["run_count"] == 5
        assert shifted["run_count"] == 5, f"naive bounds must mean UTC: {shifted}"
        assert shifted["total_tokens"] == aware["total_tokens"]
        assert shifted["total_cost_usd"] == pytest.approx(aware["total_cost_usd"])

        top = await client.get(
            "/api/v1/traces/usage/top",
            headers=users["a"]["headers"],
            params={**{k: naive(v) for k, v in window.items()}, "limit": 10},
        )
        assert top.status_code == 200, top.text
        assert len(top.json()["items"]) == 5, "naive bounds filter the same runs on /usage/top"


class TestStreamPathCapture:
    async def test_stream_path_persists_llm_usage(self, client, users, stub_llm):
        """I7-4: ``POST /api/v1/chat/{id}/stream`` must ALSO persist the
        ``llm_usage`` — the streamed turn counts with the same token contract
        as the non-streaming invoke path (stub emits 1 in / 1 out / 2 total)."""
        stub_llm.reset()
        agent_name = await _create_agent(client, users["a"]["headers"], f"UsageStream{uuid.uuid4().hex[:6]}")
        thread = await create_thread(client, users["a"]["headers"], agent_name)
        thread_id = thread.json()["id"]
        await configure_llm_credentials(client, users["a"]["headers"], api_key="sk-cred-usage-stream")

        resp = await client.post(
            f"/api/v1/chat/{thread_id}/stream", headers=users["a"]["headers"], json={"message": "hello"}
        )
        assert resp.status_code == 200, resp.text

        data = (await get_usage(client, users["a"]["headers"], agent_name=agent_name)).json()
        assert data["run_count"] == 1, data
        assert data["error_count"] == 0, data
        assert data["total_input_tokens"] == 1 and data["total_output_tokens"] == 1
        assert data["total_tokens"] == 2
        model_entry = next(e for e in data["series"]["by_model"] if e["model_name"] == "stub-model")
        assert model_entry["total_tokens"] == 2 and model_entry["priced"] is False


class TestStorageFailure:
    async def test_db_outage_answers_503_not_500(self, stack, client, users):
        """Adapter-failure case (I7-1): a PostgreSQL outage behind the usage
        repository surfaces as 503 (StorageError → SERVICE_UNAVAILABLE) —
        never a raw 500 — on BOTH usage endpoints.

        Fault injection: the REAL ``PostgresTraceUsageRepository`` pointed at
        a dead DSN — no internal component is mocked; the golden-rule
        boundary (the external LLM API) stays the only stub."""
        from sqlalchemy.ext.asyncio import create_async_engine

        app = stack["app"]
        dead = create_async_engine("postgresql+asyncpg://test:test@127.0.0.1:59919/dead", connect_args={"timeout": 5})
        broken = PostgresTraceUsageRepository(engine=dead)
        app.dependency_overrides[get_trace_usage_use_case] = lambda: GetTraceUsageUseCase(broken)
        app.dependency_overrides[get_top_usage_runs_use_case] = lambda: GetTopUsageRunsUseCase(broken)
        try:
            usage = await get_usage(client, users["a"]["headers"])
            assert usage.status_code == 503, usage.text

            top = await client.get("/api/v1/traces/usage/top", headers=users["a"]["headers"], params={"limit": 5})
            assert top.status_code == 503, top.text
        finally:
            app.dependency_overrides.pop(get_trace_usage_use_case, None)
            app.dependency_overrides.pop(get_top_usage_runs_use_case, None)
            await dead.dispose()
