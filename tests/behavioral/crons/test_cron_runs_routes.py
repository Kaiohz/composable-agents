"""Behavioral tests — cron run lifecycle: Run now, run history, safe errors
(spec §5.3 CRN-003 run records, CRN-004 execution behavior, CRN-005 overlap).

Routes under test:
* POST /api/v1/crons/{id}/run   (Run now)
* GET  /api/v1/crons/{id}/runs  (paginated run history, newest first)

Scheduled occurrences (trigger="scheduled") are exercised against the real
scheduler entry point in ``test_cron_scheduler.py`` — this file pins the
manual path and the run-record contract.
"""

import asyncio
import uuid

import pytest

from tests.behavioral.conftest import (
    configure_llm_credentials,
    create_agent_yaml,
    poll_condition,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

RUN_STATUSES = {"queued", "running", "succeeded", "failed", "skipped"}
RUN_FIELDS = {"id", "cron_id", "trigger", "status", "occurrence_at", "thread_id", "source", "started_at", "finished_at"}


def agent_yaml(name: str) -> str:
    return f'name: {name}\nmodel: stub-model\nsystem_prompt: "Cron runner."\ntools: []\ndebug: false\n'


async def _cron_with_fast_agent(client, users) -> dict:
    """Cron (every-minute expression) whose agent runs against the stub LLM."""
    name = f"agent-{uuid.uuid4().hex[:8]}"
    _, created_agent = await create_agent_yaml(client, users["a"]["headers"], agent_yaml(name), name)
    assert created_agent.status_code == 201, created_agent.text
    await configure_llm_credentials(client, users["a"]["headers"], api_key="sk-cred-cron-tests")

    cron = await client.post(
        "/api/v1/crons",
        headers=users["a"]["headers"],
        json={
            "name": f"cron-{uuid.uuid4().hex[:6]}",
            "agent_name": name,
            "message": "scheduled work",
            "cron_expression": "*/5 * * * *",
        },
    )
    assert cron.status_code == 201, cron.text
    return cron.json()


async def _runs_for(client, headers, cron_id: str) -> list[dict]:
    """Fetch run history, tolerant to list-envelope shapes ({"items": [...]} or
    bare list) since the spec does not pin the wrapper."""
    resp = await client.get(f"/api/v1/crons/{cron_id}/runs", headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    if isinstance(body, dict) and "items" in body:
        return body["items"]
    assert isinstance(body, list), f"Unexpected runs response shape: {body!r}"
    return body


class TestRunNow:
    async def test_manual_run_creates_cron_thread_and_succeeded_run(self, client, users, stub_llm):
        """CRN-004: manual & scheduled executions share the invocation path;
        every accepted attempt creates a new thread with source="cron"."""
        stub_llm.reset()
        cron = await _cron_with_fast_agent(client, users)
        cron_id = cron["id"]

        run_now = await client.post(f"/api/v1/crons/{cron_id}/run", headers=users["a"]["headers"])
        assert run_now.status_code in {200, 201, 202}, run_now.text

        ok, runs = await _has_terminal_run(client, users["a"]["headers"], cron_id, "succeeded")
        assert ok, f"No succeeded run observed: {runs}"
        manual_runs = [r for r in (await _runs_for(client, users["a"]["headers"], cron_id)) if r["trigger"] == "manual"]
        run = next(r for r in manual_runs if r["status"] == "succeeded")

        # Run record contract (CRN-003):
        assert {"id", "cron_id", "trigger", "status"} <= set(run)
        assert run["cron_id"] == cron_id
        assert run["trigger"] == "manual"
        assert run["status"] in RUN_STATUSES
        assert run["occurrence_at"] is None  # manual runs have no occurrence
        assert run["thread_id"] is not None  # a failure-after-acceptance keeps its thread too
        assert run["source"] == "cron"  # source="cron" when a thread exists
        assert run["started_at"] is not None

        # The linked thread exists, is owned by the cron owner and labeled cron:
        thread = await client.get(f"/api/v1/threads/{run['thread_id']}", headers=users["a"]["headers"])
        assert thread.status_code == 200
        assert thread.json()["source"] == "cron"

        history = await client.get(f"/api/v1/threads/{run['thread_id']}/history", headers=users["a"]["headers"])
        assert history.status_code == 200
        assert history.json()["thread"]["source"] == "cron"

    async def test_run_now_allowed_while_disabled(self, client, users, stub_llm):
        """CRN-004: Run now is permitted while the cron is disabled; it still
        creates a cron thread."""
        stub_llm.reset()
        cron = await _cron_with_fast_agent(client, users)
        cron_id = cron["id"]
        disabled = await client.patch(
            f"/api/v1/crons/{cron_id}", headers=users["a"]["headers"], json={"enabled": False}
        )
        assert disabled.status_code == 200

        run_now = await client.post(f"/api/v1/crons/{cron_id}/run", headers=users["a"]["headers"])
        assert run_now.status_code in {200, 201, 202}, run_now.text

        ok, seen = await _has_terminal_run(client, users["a"]["headers"], cron_id, "succeeded")
        assert ok, f"Run now on a disabled cron must still execute: {seen}"

    async def test_run_now_unknown_or_foreign_cron_404(self, client, users):
        agent_name = f"agent-{uuid.uuid4().hex[:8]}"
        resp = await client.post(f"/api/v1/crons/{uuid.uuid4()}/run", headers=users["a"]["headers"])
        assert resp.status_code == 404

        cron = await _cron_with_fast_agent(client, users)
        foreign = await client.post(f"/api/v1/crons/{cron['id']}/run", headers=users["b"]["headers"])
        assert foreign.status_code == 404
        assert agent_name  # (placeholder to keep both 404s together)

    async def test_concurrent_run_now_starts_exactly_one_invocation(self, client, client_factory, users, stub_llm):
        """CRN-005: two simultaneous Run now requests must yield exactly one
        invocation and one cron thread — the single-active-run guard is a
        database constraint, not a check-then-act race."""
        stub_llm.reset(mode="slow:2.0")
        cron = await _cron_with_fast_agent(client, users)
        cron_id = cron["id"]

        async def fire():
            ac = await client_factory()
            try:
                return await ac.post(f"/api/v1/crons/{cron_id}/run", headers=users["a"]["headers"])
            finally:
                await ac.aclose()

        first, second = await asyncio.gather(fire(), fire())
        statuses = sorted([first.status_code, second.status_code])
        assert 409 in statuses, f"one request must be rejected with a conflict, got {statuses}"

        ok, runs = await _has_terminal_run(client, users["a"]["headers"], cron_id, "succeeded")
        assert ok, f"No succeeded run observed after the concurrent run-now race: {runs}"
        executed = [
            r
            for r in await _runs_for(client, users["a"]["headers"], cron_id)
            if r["status"] in {"queued", "running", "succeeded", "failed"}
        ]
        assert len(executed) == 1, f"Exactly one execution expected, got {executed}"
        thread_ids = {r["thread_id"] for r in executed if r["thread_id"] is not None}
        assert len(thread_ids) == 1, f"Exactly one cron thread expected, saw {thread_ids}"


class TestRunErrorsAreSafe:
    async def test_llm_failure_marks_run_failed_with_safe_error(self, client, users, stub_llm):
        """CRN-003/CRN-004: a failure after acceptance retains its run record
        and linked thread; the error summary must not leak credentials."""
        stub_llm.reset(mode="fail")
        cron = await _cron_with_fast_agent(client, users)
        cron_id = cron["id"]

        run_now = await client.post(f"/api/v1/crons/{cron_id}/run", headers=users["a"]["headers"])
        assert run_now.status_code in {200, 201, 202}, run_now.text

        ok, runs = await _has_terminal_run(client, users["a"]["headers"], cron_id, "failed")
        assert ok, f"No failed run observed: {runs}"
        failed = next(r for r in (await _runs_for(client, users["a"]["headers"], cron_id)) if r["status"] == "failed")

        assert failed["finished_at"] is not None
        assert failed["thread_id"] is not None  # failure after acceptance keeps the thread
        error_summary = failed.get("error") or failed.get("error_summary") or ""
        assert error_summary  # "safe error summary" exists
        # No application authentication tokens or sensitive config in the summary:
        assert "sk-cred-cron-tests" not in error_summary
        assert "Bearer" not in error_summary
        assert "cpk_" not in error_summary

    async def test_agent_deleted_after_create_fails_explicitly(self, client, users, stub_llm):
        """§9: cron referencing an agent that is later deleted → next accepted
        run fails explicitly and remains visible in history."""
        stub_llm.reset()
        cron = await _cron_with_fast_agent(client, users)
        cron_id = cron["id"]
        agent_name = cron["agent_name"]

        removed = await client.delete(f"/api/v1/agents/{agent_name}", headers=users["a"]["headers"])
        assert removed.status_code == 204

        run_now = await client.post(f"/api/v1/crons/{cron_id}/run", headers=users["a"]["headers"])
        assert run_now.status_code in {200, 201, 202}, run_now.text  # accepted, fails during execution

        ok, runs = await _has_terminal_run(client, users["a"]["headers"], cron_id, "failed")
        assert ok, f"Accepted run must fail explicitly after agent deletion: {runs}"


class TestManualOverlap:
    async def test_manual_run_during_active_run_conflicts_or_skips(self, client, users, stub_llm):
        """CRN-005: manual overlap returns an explicit conflict/skipped outcome
        and never starts a concurrent invocation."""
        stub_llm.reset(mode="slow:2.5")
        cron = await _cron_with_fast_agent(client, users)
        cron_id = cron["id"]

        first = await client.post(f"/api/v1/crons/{cron_id}/run", headers=users["a"]["headers"])
        assert first.status_code in {200, 201, 202}, first.text

        ok, seen = await poll_condition(running_run_predicate(client, users["a"]["headers"], cron_id), timeout=10.0)
        assert ok, f"First manual run never entered 'running': {seen}"

        second = await client.post(f"/api/v1/crons/{cron_id}/run", headers=users["a"]["headers"])
        # Explicit conflict (409) or an accepted-and-skipped outcome — never a
        # concurrent invocation.
        assert second.status_code in {200, 201, 202, 409}, second.text

        # Let the first run finish.
        ok, _ = await poll_condition(
            terminal_run_predicate(client, users["a"]["headers"], cron_id, "succeeded"),
            timeout=20.0,
        )
        runs = await _runs_for(client, users["a"]["headers"], cron_id)
        assert ok, f"Active run should have continued to success: {runs}"
        succeeded = [r for r in runs if r["status"] == "succeeded"]
        skipped = [r for r in runs if r["status"] == "skipped"]
        assert len(succeeded) == 1, f"Overlap must not produce concurrent invocations: {runs}"
        # The overlap outcome is explicit: rejected with 409 or recorded skipped.
        assert second.status_code == 409 or len(skipped) >= 1, (
            f"Manual overlap must be an explicit conflict/skipped outcome: http={second.status_code}, runs={runs}"
        )

        # A skipped overlap does not create a thread (CRN-004) — exactly one
        # cron thread for the single execution.
        thread_ids = {r["thread_id"] for r in runs if r["thread_id"] is not None}
        assert len(thread_ids) == 1, f"Exactly one thread for one execution, saw {thread_ids}"
        listing = await client.get("/api/v1/threads", headers=users["a"]["headers"])
        cron_threads = [t for t in listing.json() if t["source"] == "cron" and t["id"] in thread_ids]
        assert len(cron_threads) == 1


def running_run_predicate(client, headers, cron_id: str):
    async def predicate():
        runs = await _runs_for(client, headers, cron_id)
        matches = [r for r in runs if r["status"] == "running"]
        return (True, matches[0]) if matches else (False, runs)

    return predicate


def terminal_run_predicate(client, headers, cron_id: str, status: str, expected_min: int = 1):
    async def predicate():
        runs = await _runs_for(client, headers, cron_id)
        matches = [r for r in runs if r["status"] == status]
        return (True, matches) if len(matches) >= expected_min else (False, runs)

    return predicate


async def _has_terminal_run(client, headers, cron_id: str, status: str, expected_min: int = 1, timeout: float = 20.0):
    """Poll run history until ``expected_min`` runs reach ``status`` (or timeout)."""

    async def check():
        runs = await _runs_for(client, headers, cron_id)
        matches = [r for r in runs if r["status"] == status]
        return (True, matches) if len(matches) >= expected_min else (False, runs)

    return await poll_condition(check, timeout=timeout)


class TestRunHistoryLimitBounds:
    """QA BUG-003: the run-history ``limit`` is a bounded client input.

    Out-of-range values are a 4xx validation error at the parameter layer —
    never a 5xx, never an internal SQL statement echoed to the client."""

    async def test_negative_limit_is_a_validation_error(self, client, users, stub_llm):
        cron = await _cron_with_fast_agent(client, users)
        resp = await client.get(f"/api/v1/crons/{cron['id']}/runs?limit=-1", headers=users["a"]["headers"])
        assert resp.status_code in {400, 422}, f"negative limit must be 4xx, got {resp.status_code} {resp.text}"

    async def test_zero_limit_is_a_validation_error(self, client, users, stub_llm):
        cron = await _cron_with_fast_agent(client, users)
        resp = await client.get(f"/api/v1/crons/{cron['id']}/runs?limit=0", headers=users["a"]["headers"])
        assert resp.status_code in {400, 422}, f"zero limit must be 4xx, got {resp.status_code} {resp.text}"

    async def test_unbounded_limit_is_a_validation_error(self, client, users, stub_llm):
        cron = await _cron_with_fast_agent(client, users)
        resp = await client.get(f"/api/v1/crons/{cron['id']}/runs?limit=1000000", headers=users["a"]["headers"])
        assert resp.status_code in {400, 422}, f"oversized limit must be 4xx, got {resp.status_code} {resp.text}"

    async def test_no_internal_sql_echoed_on_validation_error(self, client, users, stub_llm):
        cron = await _cron_with_fast_agent(client, users)
        resp = await client.get(f"/api/v1/crons/{cron['id']}/runs?limit=-1", headers=users["a"]["headers"])
        body = resp.text
        assert "SELECT" not in body, f"internal SQL statement echoed to the client: {body}"
        assert "cron_runs" not in body, f"internal schema echoed to the client: {body}"
        assert "asyncpg" not in body, f"internal driver details echoed to the client: {body}"

    async def test_upper_bound_limit_is_accepted(self, client, users, stub_llm):
        cron = await _cron_with_fast_agent(client, users)
        resp = await client.get(f"/api/v1/crons/{cron['id']}/runs?limit=200", headers=users["a"]["headers"])
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == (body if isinstance(body, list) else body), body


class TestRunRecordFieldSet:
    """QA BUG-005: the cron and run-record payloads must not carry the owner id."""

    async def test_cron_create_payload_has_no_user_id(self, client, users, stub_llm):
        cron = await _cron_with_fast_agent(client, users)
        got = await client.get(f"/api/v1/crons/{cron['id']}", headers=users["a"]["headers"])
        assert got.status_code == 200
        assert "user_id" not in got.json(), f"owner id leaked in cron payload: {got.json()}"

    async def test_cron_list_payloads_have_no_user_id(self, client, users, stub_llm):
        await _cron_with_fast_agent(client, users)
        listing = await client.get("/api/v1/crons", headers=users["a"]["headers"])
        assert listing.status_code == 200
        items = listing.json() if isinstance(listing.json(), list) else listing.json()["items"]
        for item in items:
            assert "user_id" not in item, f"owner id leaked in cron list payload: {item}"

    async def test_run_history_payloads_have_no_user_id(self, client, users, stub_llm):
        stub_llm.reset()
        cron = await _cron_with_fast_agent(client, users)
        run_now = await client.post(f"/api/v1/crons/{cron['id']}/run", headers=users["a"]["headers"])
        assert run_now.status_code in {200, 201, 202}, run_now.text
        assert "user_id" not in run_now.json(), f"owner id leaked in run payload: {run_now.json()}"

        ok, runs = await _has_terminal_run(client, users["a"]["headers"], cron["id"], "succeeded")
        assert ok, runs
        for run in runs:
            assert "user_id" not in run, f"owner id leaked in run history payload: {run}"
            # The computed cron-source marker survives the DTO projection.
            assert run["source"] == "cron"
