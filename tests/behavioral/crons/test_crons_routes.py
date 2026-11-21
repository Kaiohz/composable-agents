"""Behavioral tests — cron definitions CRUD and validation (spec §5.3 CRN-001…CRN-003).

Routes under test (real HTTP entry points on the real app):
* GET    /api/v1/crons
* POST   /api/v1/crons
* GET    /api/v1/crons/{id}
* PATCH  /api/v1/crons/{id}
* DELETE /api/v1/crons/{id}

Validated: five-field cron expressions (no seconds), IANA timezone, agent
existence/accessibility, unattended-run placeholder rejection (${USER_JWT} /
${USER_API_KEY}), next-run calculation, and owner RLS scoping.
"""

import uuid
from datetime import UTC, datetime

import pytest

from tests.behavioral.conftest import create_agent_yaml

pytestmark = pytest.mark.asyncio(loop_scope="session")

VALIDATION_CODES = {400, 422}
# Spec leaves the exact code unpinned for "agent must exist and be accessible"
# and "unattended runtime placeholders" — document families:
NOT_FOUND_CODES = {404}
PLACEHOLDER_REJECT_CODES = {400, 403, 422}


def agent_yaml(name: str, extra: str = "") -> str:
    return f'name: {name}\nmodel: stub-model\nsystem_prompt: "Cron agent."\ntools: []\ndebug: false\n{extra}'


async def _agent(client, headers, extra: str = "") -> str:
    name = f"cron-agent-{uuid.uuid4().hex[:8]}"
    _, resp = await create_agent_yaml(client, headers, agent_yaml(name, extra), name)
    assert resp.status_code == 201, resp.text
    return name


def cron_payload(agent_name: str, **overrides) -> dict:
    payload = {
        "name": f"Nightly {uuid.uuid4().hex[:6]}",
        "agent_name": agent_name,
        "message": "Run the daily digest",
        "cron_expression": "0 9 * * *",
    }
    payload.update(overrides)
    return payload


async def _create_cron(client, headers, agent_name, **overrides):
    resp = await client.post("/api/v1/crons", headers=headers, json=cron_payload(agent_name, **overrides))
    return resp


class TestCreateCron:
    async def test_create_returns_201_with_definition_and_next_run(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        resp = await _create_cron(client, users["a"]["headers"], agent)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        uuid.UUID(body["id"])
        assert body["name"].startswith("Nightly")
        assert body["agent_name"] == agent
        assert body["message"] == "Run the daily digest"
        assert body["cron_expression"] == "0 9 * * *"  # original retained (CRN-001)
        assert body["timezone"] == "UTC"  # defaults to UTC
        assert body["enabled"] is True  # defaults to true
        assert body["next_run_at"] is not None
        assert body["created_at"] is not None and body["updated_at"] is not None

    async def test_create_with_explicit_timezone_echoed(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        resp = await _create_cron(client, users["a"]["headers"], agent, timezone="Europe/Paris")
        assert resp.status_code == 201
        assert resp.json()["timezone"] == "Europe/Paris"

    async def test_create_created_disabled_allowed(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        resp = await _create_cron(client, users["a"]["headers"], agent, enabled=False)
        assert resp.status_code == 201
        body = resp.json()
        assert body["enabled"] is False
        # No next scheduled occurrence for a disabled cron (or at least none in the past).
        if body["next_run_at"] is not None:
            assert datetime.fromisoformat(body["next_run_at"]) > datetime.now(UTC)

    async def test_create_requires_agent_to_exist_and_be_accessible(self, client, users):
        for headers, expectation in ((users["a"]["headers"], False), (users["b"]["headers"], True)):
            resp = await _create_cron(client, headers, "no-such-agent-anywhere")
            if expectation:
                assert resp.status_code in NOT_FOUND_CODES
            else:
                assert resp.status_code in NOT_FOUND_CODES

    async def test_cross_user_agent_not_usable(self, client, users):
        """CRN-002: agent exists AND is accessible to the cron owner — user B
        cannot schedule user A's agent."""
        a_agent = await _agent(client, users["a"]["headers"])
        resp = await _create_cron(client, users["b"]["headers"], a_agent)
        assert resp.status_code in NOT_FOUND_CODES

    async def test_valid_five_field_expressions_accepted(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        for expression in ("*/15 * * * *", "0 0 * * 1-5", "30 4 1 * *"):
            resp = await _create_cron(client, users["a"]["headers"], agent, cron_expression=expression)
            assert resp.status_code == 201, (expression, resp.text)
            await client.delete(f"/api/v1/crons/{resp.json()['id']}", headers=users["a"]["headers"])

    async def test_six_field_or_seconds_expression_rejected(self, client, users):
        """§9: expression with six fields / seconds → validation error."""
        agent = await _agent(client, users["a"]["headers"])
        for expression in ("0 0 * * * *", "*/30 * * * * *", "0 9 * * * 2026"):
            resp = await _create_cron(client, users["a"]["headers"], agent, cron_expression=expression)
            assert resp.status_code in VALIDATION_CODES, (expression, resp.status_code)

    async def test_invalid_cron_syntax_rejected(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        for expression in ("not-a-cron", "* * *", "61 99 * * *", ""):
            resp = await _create_cron(client, users["a"]["headers"], agent, cron_expression=expression)
            assert resp.status_code in VALIDATION_CODES, (expression, resp.status_code)

    async def test_invalid_timezone_rejected(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        for timezone in ("Mars/Olympus", "Not/AZone", ""):
            resp = await _create_cron(client, users["a"]["headers"], agent, timezone=timezone)
            assert resp.status_code in VALIDATION_CODES, (timezone, resp.status_code)

    async def test_empty_message_or_name_rejected(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        for payload_override in ({"message": ""}, {"message": None}, {"name": ""}):
            resp = await _create_cron(client, users["a"]["headers"], agent, **payload_override)
            assert resp.status_code in VALIDATION_CODES, payload_override

    async def test_agent_requiring_user_jwt_rejected(self, client, users, stub_registry):
        """CRN-002: agent config requiring ${USER_JWT} is unsuitable for
        unattended execution — creation must fail explicitly. Under the
        resource-only regime the placeholder lives in the REGISTRY entry
        materialized by the import (mcp_refs), so the unattended-safety scan
        resolves the agent's MCP selection through the trusted path."""
        placeholder_yaml_extra = (
            "mcp_servers:\n"
            "  - name: cred-server\n"
            "    transport: http\n"
            "    url: http://127.0.0.1:59998/mcp\n"
            "    env:\n"
            "      CREDENTIAL: \"${USER_JWT}\"\n"
        )
        agent = await _agent(client, users["a"]["headers"], extra=placeholder_yaml_extra)
        resp = await _create_cron(client, users["a"]["headers"], agent)
        assert resp.status_code in PLACEHOLDER_REJECT_CODES, resp.text

    async def test_agent_requiring_user_api_key_rejected(self, client, users, stub_registry):
        placeholder_yaml_extra = (
            "mcp_servers:\n"
            "  - name: cred-server\n"
            "    transport: http\n"
            "    url: http://127.0.0.1:59998/mcp\n"
            "    env:\n"
            "      CREDENTIAL: \"${USER_API_KEY}\"\n"
        )
        agent = await _agent(client, users["a"]["headers"], extra=placeholder_yaml_extra)
        resp = await _create_cron(client, users["a"]["headers"], agent)
        assert resp.status_code in PLACEHOLDER_REJECT_CODES, resp.text

    async def test_unauthenticated_401(self, client):
        resp = await client.post("/api/v1/crons", json=cron_payload("whatever"))
        assert resp.status_code == 401


class TestListAndGet:
    async def test_list_owned_with_state(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        created = await _create_cron(client, users["a"]["headers"], agent)
        assert created.status_code == 201

        listing = await client.get("/api/v1/crons", headers=users["a"]["headers"])
        assert listing.status_code == 200
        items = listing.json() if isinstance(listing.json(), list) else listing.json()["items"]
        mine = [c for c in items if c["id"] == created.json()["id"]]
        assert len(mine) == 1
        crons = mine[0]
        assert {"id", "name", "agent_name", "cron_expression", "timezone", "enabled", "next_run_at"} <= set(crons)

    async def test_foreign_cron_not_listed(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        created = await _create_cron(client, users["a"]["headers"], agent)
        listing = await client.get("/api/v1/crons", headers=users["b"]["headers"])
        items = listing.json() if isinstance(listing.json(), list) else listing.json()["items"]
        assert created.json()["id"] not in [c["id"] for c in items]

    async def test_get_owned_definition(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        created = await _create_cron(client, users["a"]["headers"], agent)
        got = await client.get(f"/api/v1/crons/{created.json()['id']}", headers=users["a"]["headers"])
        assert got.status_code == 200
        assert got.json()["agent_name"] == agent

    async def test_get_unknown_404(self, client, users):
        resp = await client.get(f"/api/v1/crons/{uuid.uuid4()}", headers=users["a"]["headers"])
        assert resp.status_code == 404

    async def test_get_foreign_404_non_enumerating(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        created = await _create_cron(client, users["a"]["headers"], agent)
        resp = await client.get(f"/api/v1/crons/{created.json()['id']}", headers=users["b"]["headers"])
        assert resp.status_code == 404
        assert created.json()["name"] not in resp.text


class TestPatchCron:
    async def test_patch_enabled_false_recalculates_next_run(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        created = await _create_cron(client, users["a"]["headers"], agent)
        cron_id = created.json()["id"]

        patched = await client.patch(
            f"/api/v1/crons/{cron_id}", headers=users["a"]["headers"], json={"enabled": False}
        )
        assert patched.status_code == 200
        body = patched.json()
        assert body["enabled"] is False
        if body["next_run_at"] is not None:
            assert datetime.fromisoformat(body["next_run_at"]) > datetime.now(UTC)

    async def test_patch_expression_recalculates_next_run(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        created = await _create_cron(client, users["a"]["headers"], agent)
        cron_id = created.json()["id"]

        patched = await client.patch(
            f"/api/v1/crons/{cron_id}",
            headers=users["a"]["headers"],
            json={"cron_expression": "30 23 * * *", "timezone": "Asia/Tokyo"},
        )
        assert patched.status_code == 200
        body = patched.json()
        assert body["cron_expression"] == "30 23 * * *"
        assert body["timezone"] == "Asia/Tokyo"
        assert body["next_run_at"] is not None
        assert datetime.fromisoformat(body["next_run_at"]) > datetime.now(UTC)

    async def test_patch_invalid_expression_rejected(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        created = await _create_cron(client, users["a"]["headers"], agent)
        resp = await client.patch(
            f"/api/v1/crons/{created.json()['id']}",
            headers=users["a"]["headers"],
            json={"cron_expression": "0 0 * * * *"},
        )
        assert resp.status_code in VALIDATION_CODES

    async def test_patch_invalid_timezone_rejected(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        created = await _create_cron(client, users["a"]["headers"], agent)
        resp = await client.patch(
            f"/api/v1/crons/{created.json()['id']}", headers=users["a"]["headers"], json={"timezone": "Nowhere/Land"}
        )
        assert resp.status_code in VALIDATION_CODES

    async def test_patch_unknown_404(self, client, users):
        resp = await client.patch(f"/api/v1/crons/{uuid.uuid4()}", headers=users["a"]["headers"], json={"enabled": False})
        assert resp.status_code == 404

    async def test_patch_foreign_404(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        created = await _create_cron(client, users["a"]["headers"], agent)
        resp = await client.patch(
            f"/api/v1/crons/{created.json()['id']}", headers=users["b"]["headers"], json={"enabled": False}
        )
        assert resp.status_code == 404


class TestDeleteCron:
    async def test_delete_stops_future_scheduling(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        created = await _create_cron(client, users["a"]["headers"], agent)
        cron_id = created.json()["id"]

        deleted = await client.delete(f"/api/v1/crons/{cron_id}", headers=users["a"]["headers"])
        assert deleted.status_code == 204
        gone = await client.get(f"/api/v1/crons/{cron_id}", headers=users["a"]["headers"])
        assert gone.status_code == 404

    async def test_delete_unknown_404(self, client, users):
        resp = await client.delete(f"/api/v1/crons/{uuid.uuid4()}", headers=users["a"]["headers"])
        assert resp.status_code == 404

    async def test_delete_foreign_404(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        created = await _create_cron(client, users["a"]["headers"], agent)
        resp = await client.delete(f"/api/v1/crons/{created.json()['id']}", headers=users["b"]["headers"])
        assert resp.status_code == 404


class TestExplicitNullNonNullableBoundary:
    """QA PASS-3 (BUG-007/008/010): an explicit ``null`` for the non-nullable
    cron fields is a clean 4xx validation error at the request boundary —
    never a 500/503, never internal SQL echoed, never a ``None`` repr in a 404."""

    async def test_patch_null_name_is_a_client_error(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        cron = await _create_cron(client, users["a"]["headers"], agent)
        resp = await client.patch(f"/api/v1/crons/{cron.json()['id']}", headers=users["a"]["headers"], json={"name": None})
        assert resp.status_code in {400, 422}, f"null name must be 4xx, got {resp.status_code} {resp.text}"

    async def test_patch_null_message_is_a_client_error(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        cron = await _create_cron(client, users["a"]["headers"], agent)
        resp = await client.patch(f"/api/v1/crons/{cron.json()['id']}", headers=users["a"]["headers"], json={"message": None})
        assert resp.status_code in {400, 422}, f"null message must be 4xx, got {resp.status_code} {resp.text}"

    async def test_patch_null_cron_expression_is_a_client_error(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        cron = await _create_cron(client, users["a"]["headers"], agent)
        resp = await client.patch(
            f"/api/v1/crons/{cron.json()['id']}", headers=users["a"]["headers"], json={"cron_expression": None}
        )
        assert resp.status_code in {400, 422}, f"null cron_expression must be 4xx, got {resp.status_code} {resp.text}"

    async def test_patch_null_enabled_is_a_client_error(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        cron = await _create_cron(client, users["a"]["headers"], agent)
        resp = await client.patch(f"/api/v1/crons/{cron.json()['id']}", headers=users["a"]["headers"], json={"enabled": None})
        assert resp.status_code in {400, 422}, f"null enabled must be 4xx, got {resp.status_code} {resp.text}"

    async def test_patch_null_agent_name_is_a_validation_error_not_a_404_for_none(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        cron = await _create_cron(client, users["a"]["headers"], agent)
        resp = await client.patch(f"/api/v1/crons/{cron.json()['id']}", headers=users["a"]["headers"], json={"agent_name": None})
        assert resp.status_code in {400, 422}, f"null agent_name must be 4xx, got {resp.status_code} {resp.text}"
        assert "None" not in resp.text, f"a Python repr leaked into the payload: {resp.text}"
        assert "not found" not in resp.text

    async def test_patch_null_timezone_still_a_clean_client_error(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        cron = await _create_cron(client, users["a"]["headers"], agent)
        resp = await client.patch(f"/api/v1/crons/{cron.json()['id']}", headers=users["a"]["headers"], json={"timezone": None})
        assert resp.status_code in {400, 422}

    async def test_null_patches_never_echo_internal_sql(self, client, users):
        agent = await _agent(client, users["a"]["headers"])
        cron = await _create_cron(client, users["a"]["headers"], agent)
        cid = cron.json()["id"]
        for field in ("name", "agent_name", "message", "cron_expression", "timezone", "enabled"):
            resp = await client.patch(f"/api/v1/crons/{cid}", headers=users["a"]["headers"], json={field: None})
            body = resp.text
            assert "SELECT" not in body and "cron_runs" not in body.lower() or "crons/{" in body, body
            assert "asyncpg" not in body and "NOT NULL" not in body.upper(), body
            assert "internal" not in body.lower() or "schema" not in body.lower(), body
            # The row must be untouched after each rejected patch.
            got = await client.get(f"/api/v1/crons/{cid}", headers=users["a"]["headers"])
            assert got.status_code == 200
            assert got.json()["cron_expression"] == "0 9 * * *"
            assert got.json()["enabled"] is True
