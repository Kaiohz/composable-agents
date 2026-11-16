"""Behavioral tests — global trace list (spec §5.4 TRC-001..TRC-004).

Route under test: GET /api/v1/traces?agent_name=<exact>&limit=<n>&cursor=<c>
plus the reused detail endpoint GET /api/v1/threads/{id}/history.

Coverage: exact (case-sensitive) agent filtering, cursor pagination
stability, summary correctness (event_count, turn_count, last_event_at,
started_at), empty threads, owner-only visibility, deleted-agent history, and
the CHAT source label on summaries and detail payloads.
"""

import uuid

import pytest

from tests.behavioral.conftest import configure_llm_credentials, create_thread, invoke_agent

pytestmark = pytest.mark.asyncio(loop_scope="session")

SUMMARY_FIELDS = {"thread_id", "agent_name", "source", "started_at", "last_event_at", "event_count", "turn_count"}


def agent_yaml(name: str) -> str:
    return f'name: {name}\nmodel: stub-model\nsystem_prompt: "Traces agent."\ntools: []\ndebug: false\n'


async def _agent(client, headers, name: str) -> str:
    resp = await client.post(
        "/api/v1/agents",
        headers=headers,
        data={"agent_name": name},
        files={"file": ("agent.yaml", agent_yaml(name).encode(), "text/yaml")},
    )
    assert resp.status_code == 201, resp.text
    return name


async def _traces(client, headers, **params):
    resp = await client.get("/api/v1/traces", headers=headers, params=params)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"items", "next_cursor"}, f"Trace list contract: {body!r}"
    return body


async def _create(client, headers, agent_name: str):
    resp = await create_thread(client, headers, agent_name)
    assert resp.status_code == 201, resp.text
    return resp.json()


class TestListContract:
    async def test_items_have_exact_summary_fields_with_source(self, client, users):
        agent_name = await _agent(client, users["a"]["headers"], f"AlphaTrace{uuid.uuid4().hex[:6]}")
        thread = await _create(client, users["a"]["headers"], agent_name)

        body = await _traces(client, users["a"]["headers"])
        item = next(i for i in body["items"] if i["thread_id"] == thread["id"])
        assert set(item) == SUMMARY_FIELDS, f"Trace summary exposes unexpected fields: {set(item) ^ SUMMARY_FIELDS}"
        assert item["agent_name"] == agent_name
        assert item["source"] == "chat"
        assert item["started_at"] is not None

    async def test_empty_thread_summary_zero_counts_and_null_last_event(self, client, users):
        """§9: thread with no events → zero counts and null last-event stamp."""
        agent_name = await _agent(client, users["a"]["headers"], f"EmptyTrace{uuid.uuid4().hex[:6]}")
        thread = await _create(client, users["a"]["headers"], agent_name)

        body = await _traces(client, users["a"]["headers"], agent_name=agent_name)
        assert [i["thread_id"] for i in body["items"]] == [thread["id"]]
        item = body["items"][0]
        assert item["event_count"] == 0
        assert item["turn_count"] == 0
        assert item["last_event_at"] is None

    async def test_summary_counts_reflect_real_invocations(self, client, users, stub_llm):
        """event_count = total trace events; turn_count = distinct non-null
        turn ids; last_event_at = latest durable event timestamp."""
        stub_llm.reset()
        agent_name = await _agent(client, users["a"]["headers"], f"Counted{uuid.uuid4().hex[:6]}")
        thread = await _create(client, users["a"]["headers"], agent_name)
        await configure_llm_credentials(client, users["a"]["headers"], api_key="sk-cred-traces")

        resp = await invoke_agent(client, users["a"]["headers"], thread["id"])
        assert resp.status_code == 200, resp.text
        resp = await invoke_agent(client, users["a"]["headers"], thread["id"], message="second turn")
        assert resp.status_code == 200, resp.text

        body = await _traces(client, users["a"]["headers"], agent_name=agent_name)
        item = next(i for i in body["items"] if i["thread_id"] == thread["id"])
        # Two turns × (HUMAN_MESSAGE + AI_MESSAGE) events:
        assert item["event_count"] == 4
        assert item["turn_count"] == 2
        assert item["last_event_at"] is not None
        assert item["last_event_at"] >= item["started_at"]


class TestExactAgentFilter:
    async def test_filters_by_exact_agent_name(self, client, users):
        agent_x = await _agent(client, users["a"]["headers"], "ExactX")
        agent_y = await _agent(client, users["a"]["headers"], "ExactY")
        tx = await _create(client, users["a"]["headers"], agent_x)
        ty = await _create(client, users["a"]["headers"], agent_y)

        body = await _traces(client, users["a"]["headers"], agent_name=agent_x)
        ids = [i["thread_id"] for i in body["items"]]
        assert tx["id"] in ids and ty["id"] not in ids
        assert all(i["agent_name"] == agent_x for i in body["items"])

    async def test_filter_is_case_sensitive_and_never_partial(self, client, users):
        agent_x = await _agent(client, users["a"]["headers"], "CaseSensitive")
        thread = await _create(client, users["a"]["headers"], agent_x)

        for wrong in ("casesensitive", "Case", "ensitive", "casesensitivex"):
            body = await _traces(client, users["a"]["headers"], agent_name=wrong)
            assert thread["id"] not in [i["thread_id"] for i in body["items"]]
            assert body["items"] == []

        body = await _traces(client, users["a"]["headers"], agent_name="CaseSensitive")
        assert [i["thread_id"] for i in body["items"]] == [thread["id"]]

    async def test_no_match_is_empty_list_not_error(self, client, users):
        body = await _traces(client, users["a"]["headers"], agent_name=f"ghost-{uuid.uuid4().hex[:8]}")
        assert body["items"] == []
        assert body["next_cursor"] is None

    async def test_historical_agent_name_survives_agent_deletion(self, client, users):
        """TRC-004: stored agent names remain visible/filterable for
        historical threads even after the agent is deleted."""
        agent_name = await _agent(client, users["a"]["headers"], f"GoneAgent{uuid.uuid4().hex[:6]}")
        thread = await _create(client, users["a"]["headers"], agent_name)

        removed = await client.delete(f"/api/v1/agents/{agent_name}", headers=users["a"]["headers"])
        assert removed.status_code == 204

        body = await _traces(client, users["a"]["headers"], agent_name=agent_name)
        ids = [i["thread_id"] for i in body["items"]]
        assert thread["id"] in ids
        assert all(i["agent_name"] == agent_name for i in body["items"])


class TestPagination:
    async def _five_threads(self, client, users, agent_name: str):
        return [await _create(client, users["a"]["headers"], agent_name) for _ in range(5)]

    async def test_limit_pages_cover_all_threads_without_overlap(self, client, users):
        agent_name = await _agent(client, users["a"]["headers"], f"Paged{uuid.uuid4().hex[:6]}")
        threads = await self._five_threads(client, users, agent_name)
        all_ids = {t["id"] for t in threads}

        seen: list[str] = []
        cursors: list[str | None] = [None]
        pages: list[list[dict]] = []
        for _ in range(6):  # 5 threads / limit 2 → 3 pages
            cursor = cursors[-1]
            params = {"agent_name": agent_name, "limit": 2}
            if cursor is not None:
                params["cursor"] = cursor
            body = await _traces(client, users["a"]["headers"], **params)
            pages.append(body["items"])
            seen.extend(i["thread_id"] for i in body["items"])
            cursors.append(body["next_cursor"])
            if body["next_cursor"] is None:
                break

        assert len(pages) == 3, f"Expected 3 pages for 5 threads with limit 2, got {len(pages)}"
        assert len(seen) == len(set(seen)), "Cursor pages must not overlap"
        assert set(seen) == all_ids

    async def test_cursor_is_stable_for_repeated_requests(self, client, users):
        agent_name = await _agent(client, users["a"]["headers"], f"Stable{uuid.uuid4().hex[:6]}")
        for _ in range(4):
            await _create(client, users["a"]["headers"], agent_name)

        page1 = await _traces(client, users["a"]["headers"], agent_name=agent_name, limit=2)
        assert page1["next_cursor"] is not None
        again = await _traces(
            client, users["a"]["headers"], agent_name=agent_name, limit=2, cursor=page1["next_cursor"]
        )
        first_from_cursor = [i["thread_id"] for i in again["items"]]
        repeat = await _traces(
            client, users["a"]["headers"], agent_name=agent_name, limit=2, cursor=page1["next_cursor"]
        )
        assert [i["thread_id"] for i in repeat["items"]] == first_from_cursor

    async def test_limit_boundary_validation(self, client, users):
        agent_name = await _agent(client, users["a"]["headers"], f"Bounds{uuid.uuid4().hex[:6]}")
        await _create(client, users["a"]["headers"], agent_name)

        # Maximum is 100 (TRC-001):
        resp = await client.get("/api/v1/traces", headers=users["a"]["headers"], params={"limit": 101})
        assert resp.status_code in {400, 422}, f"limit>100 must be rejected, got {resp.status_code}"
        # Positive limit:
        resp = await client.get("/api/v1/traces", headers=users["a"]["headers"], params={"limit": 0})
        assert resp.status_code in {400, 422}, f"limit=0 must be rejected, got {resp.status_code}"

        # Within bounds works and defaults leave room:
        body = await _traces(client, users["a"]["headers"], limit=100)
        assert isinstance(body["items"], list)

    async def test_sorted_by_latest_activity_descending(self, client, users, stub_llm):
        stub_llm.reset()
        agent_name = await _agent(client, users["a"]["headers"], f"Ordered{uuid.uuid4().hex[:6]}")
        empty = await _create(client, users["a"]["headers"], agent_name)
        await configure_llm_credentials(client, users["a"]["headers"], api_key="sk-cred-order")
        active = await _create(client, users["a"]["headers"], agent_name)
        resp = await invoke_agent(client, users["a"]["headers"], active["id"])
        assert resp.status_code == 200

        body = await _traces(client, users["a"]["headers"], agent_name=agent_name)

        def activity(item):
            return item["last_event_at"] or item["started_at"]

        values = [activity(i) for i in body["items"]]
        assert values == sorted(values, reverse=True), f"Trace list must sort by latest activity desc: {values}"
        assert body["items"][0]["thread_id"] == active["id"]
        assert empty["id"] in [i["thread_id"] for i in body["items"]]


class TestAuthorization:
    async def test_only_owner_threads_are_included(self, client, users):
        agent_name = await _agent(client, users["a"]["headers"], f"Private{uuid.uuid4().hex[:6]}")
        a_thread = await _create(client, users["a"]["headers"], agent_name)

        b_body = await _traces(client, users["b"]["headers"])
        assert a_thread["id"] not in [i["thread_id"] for i in b_body["items"]]

        b_filtered = await _traces(client, users["b"]["headers"], agent_name=agent_name)
        assert b_filtered["items"] == []  # no metadata disclosure via filter either

    async def test_unauthenticated_401(self, client):
        resp = await client.get("/api/v1/traces")
        assert resp.status_code == 401


class TestCursorRobustness:
    async def test_malformed_cursor_returns_400_not_500(self, client, users):
        """C3: any malformed cursor shape must surface as the safe invalid-cursor
        client error — never an unhandled 500 — including payloads that decode
        to non-objects, non-ISO timestamps and timezone-naive timestamps."""
        import base64

        for payload in (
            "[]",
            "null",
            '"x"',
            '{"v":"v1","activity":"not-a-date","thread_id":"x"}',
            '{"v":"v1","activity":"2026-01-01T00:00:00","thread_id":"x"}',  # naive tz
            '{"v":"v9","activity":"2026-01-01T00:00:00+00:00","thread_id":"x"}',  # bad version
            "not base64 at all",
        ):
            cursor = base64.urlsafe_b64encode(payload.encode()).decode()
            resp = await client.get("/api/v1/traces", headers=users["a"]["headers"], params={"cursor": cursor})
            assert resp.status_code == 400, f"{payload!r} -> {resp.status_code} {resp.text}"

            raw = await client.get("/api/v1/traces", headers=users["a"]["headers"], params={"cursor": payload})
            assert raw.status_code == 400, f"raw cursor {payload!r} -> {raw.status_code}"


class TestTraceDetailReusesThreadHistory:
    async def test_history_includes_source_and_is_read_only_data(self, client, users, stub_llm):
        """TRC-003: detail reuses GET /api/v1/threads/{id}/history with
        `source` in the embedded thread summary."""
        stub_llm.reset()
        agent_name = await _agent(client, users["a"]["headers"], f"Detail{uuid.uuid4().hex[:6]}")
        thread = await _create(client, users["a"]["headers"], agent_name)
        await configure_llm_credentials(client, users["a"]["headers"], api_key="sk-cred-detail")
        resp = await invoke_agent(client, users["a"]["headers"], thread["id"])
        assert resp.status_code == 200

        history = await client.get(f"/api/v1/threads/{thread['id']}/history", headers=users["a"]["headers"])
        assert history.status_code == 200
        payload = history.json()
        assert payload["thread"]["source"] == "chat"
        assert payload["thread"]["id"] == thread["id"]
        assert payload["turns"], "History must keep its existing timeline shape (turns)"

    async def test_detail_unknown_or_foreign_thread_404(self, client, users):
        response = await client.get(f"/api/v1/threads/{uuid.uuid4()}/history", headers=users["a"]["headers"])
        assert response.status_code == 404

        agent_name = await _agent(client, users["a"]["headers"], f"Foreign{uuid.uuid4().hex[:6]}")
        thread = await _create(client, users["a"]["headers"], agent_name)
        foreign = await client.get(f"/api/v1/threads/{thread['id']}/history", headers=users["b"]["headers"])
        assert foreign.status_code == 404


class TestSummaryRepositoryOwnershipPredicate:
    """QA BUG-001: the trace-summary query must carry an explicit
    application-level ownership predicate — isolation may not depend on the
    database role's RLS privileges alone (a superuser/BYPASSRLS/table-owner
    application role bypasses RLS unconditionally). Verified directly against
    the adapter: with the ownership-scoped contextvar the summaries are
    owner-filtered at the SQL level; with no authenticated principal the
    adapter returns a bounded empty page (non-enumerating by construction)."""

    async def test_ownership_predicate_filters_by_authenticated_user(self, stack, client, users):
        from datetime import UTC, datetime

        from sqlalchemy import text

        from src.infrastructure.database import rls_context
        from src.infrastructure.database.rls_context import system_rls_context
        from src.infrastructure.postgres_trace.summary_repository import PostgresTraceSummaryRepository

        # Seed cross-tenant rows directly under the bypass scope — the state a
        # superuser/BYPASSRLS deployment role would have (every user's rows are
        # visible to the application connection).
        row_user_a, row_user_b = str(uuid.uuid4()), str(uuid.uuid4())
        agent_a, agent_b = f"RawOwnerA{uuid.uuid4().hex[:6]}", f"RawOwnerB{uuid.uuid4().hex[:6]}"
        now = datetime.now(UTC)
        async with system_rls_context(), stack["engine"].begin() as conn:
            assert (
                await conn.execute(
                    text(
                        "INSERT INTO threads (id, agent_name, user_id, created_at, updated_at) "
                        "VALUES (:id, :agent, 'user-a', :created, :created)"
                    ),
                    {"id": row_user_a, "agent": agent_a, "created": now},
                )
            ).rowcount == 1
            assert (
                await conn.execute(
                    text(
                        "INSERT INTO threads (id, agent_name, user_id, created_at, updated_at) "
                        "VALUES (:id, :agent, 'user-b', :created, :created)"
                    ),
                    {"id": row_user_b, "agent": agent_b, "created": now},
                )
            ).rowcount == 1

        # Query the adapter INSIDE the bypass scope: RLS is disabled at the DB
        # level for this connection (row_security=off + app.bypass_rls=on), so
        # — exactly like a superuser/BYPASSRLS/table-owner application role in
        # the QA deployment — ONLY the application-level ownership predicate
        # can scope the result.
        repo = PostgresTraceSummaryRepository(stack["engine"])
        token = rls_context.current_user_id.set("user-a")
        try:
            async with system_rls_context():
                summaries, has_more = await repo.list_summaries()
                assert has_more is False
                ids = {s.thread_id for s in summaries}
                assert row_user_a in ids, f"user-a's thread must be visible to user-a: {ids}"
                assert row_user_b not in ids, f"cross-user leak through the summary adapter: {ids}"
                assert all(s.agent_name != agent_b for s in summaries)

                # The exact agent filter operates within the owner's scope only —
                # a foreign agent name yields an empty page (no enumeration).
                summaries_filtered, _ = await repo.list_summaries(agent_name=agent_b)
                assert summaries_filtered == [], "a foreign agent name must not be resolvable (no enumeration)"

                summaries_back, _ = await repo.list_summaries(agent_name=agent_a)
                assert [s.thread_id for s in summaries_back] == [row_user_a]

            # No authenticated principal → bounded empty page, never an unscoped
            # scan (even though the connection could see every row).
            async with system_rls_context():
                rls_context.current_user_id.set(None)
                summaries, has_more = await repo.list_summaries()
                assert summaries == [] and has_more is False
        finally:
            rls_context.current_user_id.reset(token)
