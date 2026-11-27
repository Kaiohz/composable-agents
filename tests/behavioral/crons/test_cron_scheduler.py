"""Behavioral tests — scheduler correctness (spec §5.3 CRN-005/CRN-006, §7 SEC-001/SCHED-001).

Entry point: the REAL production scheduler tick, awaited the way the
production background loop invokes it (see ``run_scheduler_tick`` in the
behavioral conftest). Competing workers are simulated by awaiting the same
entry point from two tasks — the DB constraints must make the outcome
deterministic (in-memory locks alone are not enough, CRN-006).

Downtime/overlap timelines are simulated by writing ``crons.next_run_at`` in
the past (the state the scheduler consumes), instead of sleeping through
real minutes — see ``set_cron_next_run``.
"""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from tests.behavioral.conftest import (
    configure_llm_credentials,
    create_agent_yaml,
    poll_condition,
    run_scheduler_tick,
    run_scheduler_tick_concurrently,
    set_cron_next_run,
)
from tests.behavioral.crons.test_cron_runs_routes import _runs_for

pytestmark = pytest.mark.asyncio(loop_scope="session")


class TestStaleRunRecovery:
    async def test_stale_active_run_is_reclaimed(self, stack, client, users, stub_llm):
        """C2: a run left 'running' by a terminated worker must not wedge the
        cron forever — the scheduler reclaims lease-expired active runs
        (terminal failure, safe error summary, no automatic retry) so later
        occurrences execute again."""
        stub_llm.reset()
        cron = await _scheduled_cron(client, users)
        cron_id = cron["id"]

        # Simulate a worker killed mid-run: an OLD active row with no terminal
        # state (created_at/started_at far beyond the 30-minute lease).
        await _insert_stale_running_run(stack, cron_id, minutes_old=45)
        # Make an occurrence due so the same pass reclaims and then executes.
        frontier = datetime.now(UTC).replace(second=0, microsecond=0)
        await set_cron_next_run(stack, cron_id, frontier.isoformat())

        await run_scheduler_tick()

        ok, runs = await poll_condition(
            terminal_run_predicate(client, users["a"]["headers"], cron_id, "failed"), timeout=15.0
        )
        assert ok, f"The stale run must be reclaimed as a failed record: {runs}"
        stale = next(r for r in runs if r["status"] == "failed")
        assert stale["error"], "the reclaimed run must carry a safe error summary"
        assert "Bearer" not in stale["error"] and "cpk_" not in stale["error"]

        # The cron is usable again: the due occurrence of the same pass executes.
        ok2, succeeded = await poll_condition(
            terminal_run_predicate(client, users["a"]["headers"], cron_id, "succeeded"), timeout=30.0
        )
        assert ok2, f"The cron must execute again after reclamation: {succeeded}"
        runs_after = await _scheduled_runs(client, users["a"]["headers"], cron_id)
        # The reclaimed run stays failed (no retry) and was never invoked —
        # exactly ONE real execution (a row with an actual thread) happened.
        invoked = [
            r
            for r in runs_after
            if r["status"] in {"queued", "running", "succeeded", "failed"} and r["thread_id"] is not None
        ]
        assert len(invoked) == 1, f"Exactly one real execution after reclamation: {runs_after}"


def agent_yaml(name: str) -> str:
    return f'name: {name}\nmodel: stub-model\nsystem_prompt: "Scheduled agent."\ntools: []\ndebug: false\n'


async def _scheduled_cron(client, users, expression="* * * * *") -> dict:
    """Enabled cron owned by user-a whose agent resolves via the stub LLM."""
    name = f"agent-{uuid.uuid4().hex[:8]}"
    _, created_agent = await create_agent_yaml(client, users["a"]["headers"], agent_yaml(name), name)
    assert created_agent.status_code == 201, created_agent.text
    await configure_llm_credentials(client, users["a"]["headers"], api_key="sk-cred-scheduler")

    cron = await client.post(
        "/api/v1/crons",
        headers=users["a"]["headers"],
        json={
            "name": f"sched-{uuid.uuid4().hex[:6]}",
            "agent_name": name,
            "message": "tick work",
            "cron_expression": expression,
        },
    )
    assert cron.status_code == 201, cron.text
    return cron.json()


async def _scheduled_runs(client, headers, cron_id: str) -> list[dict]:
    runs = await _runs_for(client, headers, cron_id)
    return [r for r in runs if r["trigger"] == "scheduled"]


class TestAtomicOccurrenceClaim:
    async def test_two_competing_workers_execute_occurrence_once(self, stack, client, users):
        """CRN-006 / success criterion 2.8: two scheduler workers competing for
        the same occurrence create at most one execution and one cron thread
        (deterministic identity + DB uniqueness + atomic claim)."""
        cron = await _scheduled_cron(client, users)
        cron_id = cron["id"]

        # The occurrence that just became due: the exact minute now.
        now = datetime.now(UTC).replace(second=0, microsecond=0)
        await set_cron_next_run(stack, cron_id, (now - timedelta(minutes=1)).isoformat())

        await run_scheduler_tick_concurrently(worker_count=2)

        runs = await _scheduled_runs(client, users["a"]["headers"], cron_id)
        executed = [r for r in runs if r["status"] in {"queued", "running", "succeeded", "failed"}]
        assert len(executed) <= 1, f"Occurrence executed more than once: {runs}"
        assert len(runs) <= 1, f"No extra run records may be created for one occurrence: {runs}"

        # Exactly one cron thread per execution:
        listing = await client.get("/api/v1/threads", headers=users["a"]["headers"])
        cron_threads = [t for t in listing.json() if t["source"] == "cron"]
        assert len(cron_threads) <= 1, f"Expected at most one cron thread, saw {len(cron_threads)}"

    async def test_coalesced_missed_occurrences_execute_at_most_once(self, stack, client, users):
        """CRN-005: multiple missed occurrences coalesce into at most ONE
        execution (for the latest due occurrence) — never a catch-up burst."""
        cron = await _scheduled_cron(client, users)
        cron_id = cron["id"]

        # Simulate a worker that was down for ~5 minutes of every-minute
        # occurrences: five individual occurrences were missed.
        now = datetime.now(UTC).replace(second=0, microsecond=0)
        await set_cron_next_run(stack, cron_id, (now - timedelta(minutes=5)).isoformat())

        await run_scheduler_tick()

        runs = await _scheduled_runs(client, users["a"]["headers"], cron_id)
        executed = [r for r in runs if r["status"] in {"queued", "running", "succeeded", "failed"}]
        assert len(executed) <= 1, f"Catch-up burst detected: {len(executed)} executions for 5 missed occurrences"

    async def test_edit_prevents_stale_occurrence_execution(self, stack, client, users):
        """CRN-006: schedule edits must prevent stale, unclaimed occurrences
        from the previous schedule from executing."""
        cron = await _scheduled_cron(client, users, expression="*/5 * * * *")
        cron_id = cron["id"]

        # Stale occurrence under the OLD schedule becomes due…
        await set_cron_next_run(stack, cron_id, "2020-01-01T00:00:00+00:00")
        # …but the schedule is edited before the scheduler next runs.
        patched = await client.patch(
            f"/api/v1/crons/{cron_id}", headers=users["a"]["headers"], json={"cron_expression": "0 12 * * *"}
        )
        assert patched.status_code == 200

        await run_scheduler_tick()

        runs = await _scheduled_runs(client, users["a"]["headers"], cron_id)
        assert len(runs) == 0, f"Stale occurrences from the previous schedule must not execute: {runs}"


class TestOverlapPolicy:
    async def test_occurrence_during_active_run_is_skipped(self, stack, client, users, stub_llm):
        """CRN-005: if another occurrence becomes due while a run is active it
        is skipped and recorded; the active run continues."""
        stub_llm.reset(mode="slow:3.0")
        cron = await _scheduled_cron(client, users)
        cron_id = cron["id"]

        now = datetime.now(UTC).replace(second=0, microsecond=0)
        await set_cron_next_run(stack, cron_id, (now - timedelta(minutes=1)).isoformat())

        worker = asyncio.create_task(run_scheduler_tick())

        ok, seen = await poll_condition(
            running_or_queued_run_predicate(client, users["a"]["headers"], cron_id), timeout=15.0
        )
        assert ok, f"Scheduled run never became active: {seen}"

        # A second occurrence becomes due while the first is active.
        await run_scheduler_tick()

        await worker

        runs = await _scheduled_runs(client, users["a"]["headers"], cron_id)
        statuses = sorted(r["status"] for r in runs)
        assert "skipped" in statuses, f"The overlapped occurrence must be recorded as skipped: {runs}"
        executed = [r for r in runs if r["status"] in {"queued", "running", "succeeded", "failed"}]
        assert len(executed) == 1, f"Active run must continue; exactly one execution allowed: {runs}"

        ok, _ = await poll_condition(
            terminal_run_predicate(client, users["a"]["headers"], cron_id, "succeeded"), timeout=20.0
        )
        runs = await _scheduled_runs(client, users["a"]["headers"], cron_id)
        assert ok and any(r["status"] == "succeeded" for r in runs), f"Active run should complete: {runs}"


class TestDisabledPolicy:
    async def test_disabled_cron_runs_nothing_and_no_replay_after_enable(self, stack, client, users):
        """CRN-005: occurrences while disabled are not executed and are not
        replayed after re-enable."""
        cron = await _scheduled_cron(client, users)
        cron_id = cron["id"]

        # An occurrence that was due just before the cron got disabled:
        now = datetime.now(UTC).replace(second=0, microsecond=0)
        await set_cron_next_run(stack, cron_id, (now - timedelta(minutes=1)).isoformat())
        disabled = await client.patch(
            f"/api/v1/crons/{cron_id}", headers=users["a"]["headers"], json={"enabled": False}
        )
        assert disabled.status_code == 200

        await run_scheduler_tick()
        await run_scheduler_tick()

        runs = await _scheduled_runs(client, users["a"]["headers"], cron_id)
        assert len(runs) == 0, f"Disabled cron must not run scheduled occurrences: {runs}"

        # Re-enabling recalculates the next occurrence into the FUTURE.
        enabled = await client.patch(f"/api/v1/crons/{cron_id}", headers=users["a"]["headers"], json={"enabled": True})
        assert enabled.status_code == 200
        body = enabled.json()
        if body.get("next_run_at") is not None:
            assert datetime.fromisoformat(body["next_run_at"]) > datetime.now(UTC), (
                "Re-enable must recalibrate next_run_at into the future (no replay of missed occurrences)"
            )

        await run_scheduler_tick()
        runs = await _scheduled_runs(client, users["a"]["headers"], cron_id)
        assert len(runs) == 0, f"Missed occurrences while disabled must never be replayed: {runs}"


class TestFailurePolicy:
    async def test_failed_occurrence_is_terminal_never_retried(self, stack, client, users, stub_llm):
        """CRN-005: a failed run is terminal — the same occurrence is never
        automatically retried on a subsequent tick."""
        stub_llm.reset(mode="fail")
        cron = await _scheduled_cron(client, users)
        cron_id = cron["id"]

        now = datetime.now(UTC).replace(second=0, microsecond=0)
        await set_cron_next_run(stack, cron_id, (now - timedelta(minutes=1)).isoformat())

        await run_scheduler_tick()

        ok, failed = await poll_condition(
            terminal_run_predicate(client, users["a"]["headers"], cron_id, "failed"), timeout=15.0
        )
        assert ok, f"Scheduled run should fail explicitly (stub LLM in fail mode): {failed}"
        runs_after_first = await _scheduled_runs(client, users["a"]["headers"], cron_id)
        assert len(runs_after_first) == 1
        # Scheduled runs carry their deterministic occurrence timestamp.
        assert runs_after_first[0]["occurrence_at"] is not None, "Scheduled runs must expose occurrence_at"

        # A second tick while nothing new is due: no retry, no new run.
        await run_scheduler_tick()
        runs_after_second = await _scheduled_runs(client, users["a"]["headers"], cron_id)
        assert len(runs_after_second) == 1, f"Failed occurrence must not be retried: {runs_after_second}"

    async def test_agent_gaining_user_placeholder_fails_run_explicitly(self, stack, client, users, stub_llm, stub_registry):
        """§10.13 / §9: an agent edited LATER to require ${USER_JWT} must fail
        its next accepted run explicitly — never borrowing the interactive
        user's JWT/API key. Under the resource-only regime the edit registers
        the credential-hungry MCP server in the registry and selects it via
        ``mcp_refs``; the unattended-safety scan resolves the selection through
        the trusted path and fails the run BEFORE any LLM call."""
        async with httpx.AsyncClient() as http:
            registered = await http.post(
                f"{stub_registry.base_url}/api/v1/mcp/servers",
                headers=users["a"]["headers"],
                json={
                    "name": "cred-server",
                    "transport": "http",
                    "url": "http://127.0.0.1:59998/mcp",
                    "env": {"CREDENTIAL": "${USER_JWT}"},
                },
            )
        assert registered.status_code == 201, registered.text

        cron = await _scheduled_cron(client, users)
        cron_id = cron["id"]
        agent_name = cron["agent_name"]

        # Later edit: the agent now requires the interactive user's JWT via
        # the selected registry server (connection data stays in the registry).
        # The resource-only update boundary (REQ-LEGACY-003) means model and
        # prompt must keep their registered references.
        current = await client.get(f"/api/v1/agents/{agent_name}", headers=users["a"]["headers"])
        assert current.status_code == 200, current.text
        model_id = current.json()["model_ref"]
        prompt_id = current.json()["prompt_ref"]["id"]
        yaml_with_placeholder = (
            f"name: {agent_name}\n"
            "description: \"Now credential-hungry.\"\n"
            "tools: []\n"
            "debug: false\n"
            "mcp_refs:\n"
            "  - cred-server\n"
            "prompt_ref:\n"
            f"  id: {prompt_id}\n"
            "model_ref:\n"
            f"  {model_id}\n"
        )
        resp = await client.put(
            f"/api/v1/agents/{agent_name}",
            headers=users["a"]["headers"],
            files={"file": ("agent.yaml", yaml_with_placeholder.encode(), "text/yaml")},
        )
        assert resp.status_code == 200, resp.text

        await set_cron_next_run(stack, cron_id, (datetime.now(UTC) - timedelta(minutes=1)).isoformat())
        stub_llm.reset()  # no LLM call may even happen for the failing run

        await run_scheduler_tick()

        ok, runs = await poll_condition(
            terminal_run_predicate(client, users["a"]["headers"], cron_id, "failed"), timeout=15.0
        )
        assert ok, f"Unattended run must fail explicitly once the agent requires user placeholders: {runs}"
        # No credential was ever forwarded: the stub saw no LLM request from
        # this failing invocation.
        assert not any("cred-server" in str(r.body) for r in stub_llm.app.requests)
        error_summary = next(r.get("error") or r.get("error_summary") or "" for r in runs if r["status"] == "failed")
        assert "Bearer" not in error_summary and "cpk_" not in error_summary

    async def test_scheduler_never_leaks_credentials_in_run_errors(self, stack, client, users, stub_llm):
        """SCHED-001: scheduler errors must not expose JWTs/API keys or other
        sensitive configuration."""
        stub_llm.reset(mode="fail")
        cron = await _scheduled_cron(client, users)
        await set_cron_next_run(stack, cron["id"], (datetime.now(UTC) - timedelta(minutes=1)).isoformat())
        await run_scheduler_tick()

        ok, _ = await poll_condition(
            terminal_run_predicate(client, users["a"]["headers"], cron["id"], "failed"), timeout=15.0
        )
        assert ok
        runs = await _runs_for(client, users["a"]["headers"], cron["id"])
        for run in runs:
            blob = str(run)
            assert "sk-cred-scheduler" not in blob
            assert "cpk_" not in blob
            assert "Bearer" not in blob


class TestOwnerRlsContext:
    async def test_scheduled_execution_runs_under_owner_context(self, stack, client, users, stub_llm):
        """CRN-004/SEC-001: runs execute under the cron owner's user/RLS
        context — created threads belong to the owner, never to another user,
        and no cross-user dependency may be resolved."""
        stub_llm.reset()
        cron = await _scheduled_cron(client, users)
        cron_id = cron["id"]

        await set_cron_next_run(stack, cron_id, (datetime.now(UTC) - timedelta(minutes=1)).isoformat())
        await run_scheduler_tick()

        ok, runs = await poll_condition(
            terminal_run_predicate(client, users["a"]["headers"], cron_id, "succeeded"), timeout=20.0
        )
        assert ok, f"Scheduled run should succeed: {runs}"
        thread_id = next(r["thread_id"] for r in runs if r["thread_id"])

        # Owner sees the thread… (RLS context was the owner's)
        got = await client.get(f"/api/v1/threads/{thread_id}", headers=users["a"]["headers"])
        assert got.status_code == 200
        assert got.json()["source"] == "cron"
        # …a foreign user does not.
        foreign = await client.get(f"/api/v1/threads/{thread_id}", headers=users["b"]["headers"])
        assert foreign.status_code == 404

        # The owner's resolved credentials reached the LLM (owner context).
        assert stub_llm.app.requests, "Scheduled run must invoke the agent under the owner's credentials"
        assert all(r.auth_header == "Bearer sk-cred-scheduler" for r in stub_llm.app.requests)


class TestDeletedAgentRun:
    async def test_scheduled_run_after_agent_deleted_fails_explicitly(self, stack, client, users, stub_llm):
        """§9: cron referencing an agent that is later deleted → the next
        accepted run fails explicitly and remains visible in run history."""
        cron = await _scheduled_cron(client, users)
        cron_id = cron["id"]

        removed = await client.delete(f"/api/v1/agents/{cron['agent_name']}", headers=users["a"]["headers"])
        assert removed.status_code == 204

        await set_cron_next_run(stack, cron_id, (datetime.now(UTC) - timedelta(minutes=1)).isoformat())
        await run_scheduler_tick()

        ok, runs = await poll_condition(
            terminal_run_predicate(client, users["a"]["headers"], cron_id, "failed"), timeout=15.0
        )
        assert ok, f"Run must fail explicitly when the agent vanished: {runs}"
        failed = next(r for r in runs if r["status"] == "failed")
        # A failure after acceptance retains its run record and linked thread.
        assert failed["thread_id"] is not None


def running_or_queued_run_predicate(client, headers, cron_id: str):
    async def predicate():
        runs = await _runs_for(client, headers, cron_id)
        matches = [r for r in runs if r["status"] in {"running", "queued"}]
        return (True, matches[0]) if matches else (False, runs)

    return predicate


def terminal_run_predicate(client, headers, cron_id: str, status: str, expected_min: int = 1):
    async def predicate():
        runs = await _runs_for(client, headers, cron_id)
        matches = [r for r in runs if r["status"] == status]
        return (True, matches) if len(matches) >= expected_min else (False, runs)

    return predicate


async def _insert_stale_running_run(stack, cron_id: str, minutes_old: int) -> None:
    """Simulate a worker killed mid-run: insert an OLD active (running) row.

    Uses the system (RLS bypass) context — the equivalent of a server-side
    maintenance path — because no authenticated principal exists for it. The
    row's created_at/started_at are back-dated beyond the stale-run lease.
    """
    from datetime import UTC, datetime, timedelta
    from uuid import uuid4

    from sqlalchemy import text

    from src.infrastructure.database.rls_context import system_rls_context

    stale_time = datetime.now(UTC) - timedelta(minutes=minutes_old)
    async with system_rls_context(), stack["engine"].begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO cron_runs (id, cron_id, user_id, trigger, status, occurrence_at, "
                "thread_id, started_at, created_at) "
                "VALUES (:id, :cron_id, (SELECT user_id FROM crons WHERE id = :cron_id), "
                "'scheduled', 'running', :occurrence_at, NULL, :started_at, :created_at)"
            ),
            {
                "id": str(uuid4()),
                "cron_id": cron_id,
                "occurrence_at": stale_time.replace(second=0, microsecond=0),
                "started_at": stale_time,
                "created_at": stale_time,
            },
        )
