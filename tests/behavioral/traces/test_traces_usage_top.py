"""Behavioral tests — GET /api/v1/traces/usage/top (dashboard §5, §6, §8).

Route under test: ``GET /api/v1/traces/usage/top?limit=<n>&from=<iso>&to=<iso>&agent_name=<exact>``

Contract fixed here for the implementation (Step 2):

* response body: ``{"items": [...]}`` — the top N most expensive runs in the
  window, sorted by cost descending.
* each item (exact keys — spec §29): ``thread_id``, ``turn_id``,
  ``agent_name``, ``model_name``, ``total_tokens``, ``cost_usd``,
  ``duration_ms``, ``started_at`` (the run's HUMAN_MESSAGE timestamp).
* ``limit`` is a positive int (0/negative rejected with a 4xx); default
  leaves room for the dashboard table.
* error runs (no terminal AI_MESSAGE — no tokens, no cost) NEVER appear
  among the top items.
* ``agent_name`` filters to the exact agent (TRC-001 semantics).
* per-user RLS isolation: only the authenticated user's runs are eligible.
* unauthenticated requests get 401.

Data: deterministic seeded fixture (``usage_fixture.seed_usage_data``) with
strictly descending per-run costs (3000 > 2500 > 2000 > 0 > 0) so the exact
ordering is assertable, mirroring exactly the rows the real chat path
persists.
"""

import pytest

from src.infrastructure.database.rls_context import current_user_id
from tests.behavioral.traces.usage_fixture import (
    AGENT_PRICED,
    ORPHAN_AGENT,
    parse_ts,
    seed_orphan_run,
    seed_usage_data,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

TOP_ITEM_KEYS = {
    "thread_id",
    "turn_id",
    "agent_name",
    "model_name",
    "total_tokens",
    "cost_usd",
    "duration_ms",
    "started_at",
}


async def get_top(client, headers, **params):
    return await client.get("/api/v1/traces/usage/top", headers=headers, params=params)


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


class TestTopRunsContract:
    async def test_item_shape_is_exact(self, stack, client, users):
        await seed_usage_data(stack, user_id=users["a"]["id"])

        body = await get_top(client, users["a"]["headers"], limit=3)
        assert body.status_code == 200, body.text
        data = body.json()
        assert set(data) == {"items"}, f"top response contract: {set(data)}"
        assert len(data["items"]) == 3
        for item in data["items"]:
            assert set(item) == TOP_ITEM_KEYS, f"top item contract: {set(item) ^ TOP_ITEM_KEYS}"

    async def test_top_n_is_sorted_by_cost_descending(self, stack, client, users):
        """Dashboard §5 — deterministic costs 3000 > 2500 > 2000 → exact
        top-3 ordering with exact token/cost/duration/started_at values."""
        fixture = await seed_usage_data(stack, user_id=users["a"]["id"])
        expected = fixture["expected"]
        turns = fixture["turn_ids"]

        data = (await get_top(client, users["a"]["headers"], limit=3)).json()
        top3 = data["items"]
        assert [i["turn_id"] for i in top3] == [turns[k] for k in expected["top_order"][:3]]
        assert [i["cost_usd"] for i in top3] == pytest.approx([3000.0, 2500.0, 2000.0])
        assert [i["total_tokens"] for i in top3] == [2500, 1500, 1500]
        assert [i["duration_ms"] for i in top3] == [200, 200, 100]
        assert all(i["agent_name"] == AGENT_PRICED for i in top3)
        assert all(i["model_name"] == "usage-priced-model" for i in top3)
        # started_at = the run's HUMAN_MESSAGE timestamp (deterministic seed).
        for item, key in zip(top3, expected["top_order"][:3], strict=True):
            assert parse_ts(item["started_at"]) == fixture["started_at"][key]
        # thread linkage for dashboard navigation to /traces/:threadId.
        priced_thread = str(fixture["thread_ids"]["priced"])
        assert all(i["thread_id"] == priced_thread for i in top3)

    async def test_limit_truncates_and_validates(self, stack, client, users):
        fixture = await seed_usage_data(stack, user_id=users["a"]["id"])
        expected = fixture["expected"]

        # Default limit keeps room but never exceeds the completed runs.
        default = (await get_top(client, users["a"]["headers"])).json()
        assert len(default["items"]) <= expected["totals"]["run_count"]

        one = (await get_top(client, users["a"]["headers"], limit=1)).json()
        assert [i["cost_usd"] for i in one["items"]] == pytest.approx([3000.0])

        for bad_limit in ("0", "-1"):
            resp = await get_top(client, users["a"]["headers"], limit=bad_limit)
            assert resp.status_code in {400, 422}, f"limit={bad_limit} must be rejected, got {resp.status_code}"

    async def test_agent_filter_scopes_top_runs_exactly(self, stack, client, users):
        await seed_usage_data(stack, user_id=users["a"]["id"])

        data = (await get_top(client, users["a"]["headers"], agent_name=AGENT_PRICED, limit=10)).json()
        assert len(data["items"]) == 3  # only the three priced-agent runs
        assert all(i["agent_name"] == AGENT_PRICED for i in data["items"])
        assert [i["cost_usd"] for i in data["items"]] == pytest.approx([3000.0, 2500.0, 2000.0])

        # EXACT match: foreign/partial agent yields nothing.
        empty = (await get_top(client, users["a"]["headers"], agent_name=AGENT_PRICED.lower())).json()
        assert empty["items"] == []

    async def test_window_bounds_apply(self, stack, client, users):
        from datetime import UTC, datetime, timedelta

        fixture = await seed_usage_data(stack, user_id=users["a"]["id"])
        now = datetime.now(UTC).replace(microsecond=0)
        future = (
            await get_top(
                client,
                users["a"]["headers"],
                **{"from": (now + timedelta(days=1)).isoformat(), "to": (now + timedelta(days=2)).isoformat()},
            )
        ).json()
        assert future["items"] == []

        covered = (
            await get_top(
                client,
                users["a"]["headers"],
                **{"from": fixture["window_from"], "to": fixture["window_to"]},
                limit=5,
            )
        ).json()
        assert len(covered["items"]) == 5


class TestTopRunsErrorExclusion:
    async def test_error_runs_never_appear_in_top_items(self, stack, client, users):
        """Dashboard §6 — the 4 seeded error runs (HUMAN_MESSAGE only) carry
        no cost/tokens and must be absent from the top list; the percentile
        exclusion side of their semantics is proven on the usage endpoint."""
        fixture = await seed_usage_data(stack, user_id=users["a"]["id"], with_errors=True)

        data = (await get_top(client, users["a"]["headers"], limit=10)).json()
        completed_ids = {fixture["turn_ids"][k] for k in ("t1", "t2", "t3", "t4", "t5")}
        error_ids = {fixture["turn_ids"][f"e{i}"] for i in range(1, 5)}
        returned_ids = {i["turn_id"] for i in data["items"]}
        assert returned_ids == completed_ids, f"error runs leaked into top items: {returned_ids & error_ids}"
        assert len(data["items"]) == 5
        # The error runs have no cost — the top costs are unchanged.
        assert [i["cost_usd"] for i in data["items"][:3]] == pytest.approx([3000.0, 2500.0, 2000.0])


class TestTopRunsAuthorizationAndIsolation:
    async def test_unauthenticated_401(self, client):
        resp = await get_top(client, {})
        assert resp.status_code == 401

    async def test_user_b_cannot_see_user_a_top_runs(self, stack, client, users):
        """Dashboard §8 — user-b gets an empty top list over user-a's usage
        (no leak, no enumeration via filters)."""
        await seed_usage_data(stack, user_id=users["a"]["id"])

        other = (await get_top(client, users["b"]["headers"], limit=10)).json()
        assert other["items"] == []

        probed = (await get_top(client, users["b"]["headers"], agent_name=AGENT_PRICED)).json()
        assert probed["items"] == []


class TestMissingModelName:
    async def test_completed_run_without_usage_and_deleted_agent_is_reported_not_500(self, stack, client, users):
        """C1 regression: a completed run whose agent configuration was
        DELETED and whose LLM returned no usage metadata has an unresolvable
        model name — it must NOT break ``/usage/top`` (no 500 from
        ``LlmUsageRun(model_name=None)``): the run is reported with the model
        labeled ``"unknown"``, zero tokens and zero cost (unpriced)."""
        fixture = await seed_orphan_run(stack, user_id=users["a"]["id"])

        resp = await get_top(client, users["a"]["headers"], limit=10)
        assert resp.status_code == 200, resp.text
        items = resp.json()["items"]
        assert len(items) == 1, f"the orphan run must be reported, not dropped: {items}"
        item = items[0]
        assert item["model_name"] == "unknown"
        assert item["turn_id"] == fixture["turn_id"]
        assert item["agent_name"] == ORPHAN_AGENT
        assert item["thread_id"] == fixture["thread_id"]
        assert item["total_tokens"] == 0
        assert item["cost_usd"] == pytest.approx(0.0)
        # started_at = the run's HUMAN_MESSAGE timestamp; duration spans the
        # seeded HUMAN→AI gap (50ms).
        assert parse_ts(item["started_at"]) == fixture["started_at"]
        assert item["duration_ms"] == pytest.approx(50.0)
