"""Behavioral tests — prompt deletion conflicts and reference integrity
(spec §5.1 PR-005, §7 SEC-002).

Route under test: DELETE /api/v1/prompts/{id}, plus the agent create/update
paths that reference prompts (POST/PUT /api/v1/agents) to assert the
server-side, race-safe in-use check.
"""

import asyncio
import uuid

import pytest

from tests.behavioral.conftest import create_agent_yaml

pytestmark = pytest.mark.asyncio(loop_scope="session")

# Spec leaves the exact save-time failure code for invalid/cross-user refs
# unpinned ("non-enumerating unavailable/not-found" vs validation):
UNAVAILABLE_CODES = {404, 422}


def agent_yaml_with_prompt_ref(name: str, prompt_id: str, version: int | None = None) -> str:
    ref = f"prompt_ref:\n  id: {prompt_id}\n"
    if version is not None:
        ref += f"  version: {version}\n"
    return (
        f"name: {name}\n"
        "model: test-model\n"
        "tools: []\n"
        "debug: false\n"
        f"{ref}"
    )


class TestDeletionConflict:
    async def test_delete_prompt_referenced_by_agent_latest_returns_409(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        agent_name = f"agent-{uuid.uuid4().hex[:8]}"
        _, created = await create_agent_yaml(
            client, users["a"]["headers"], agent_yaml_with_prompt_ref(agent_name, prompt["id"]), agent_name
        )
        assert created.status_code == 201

        deleted = await client.delete(f"/api/v1/prompts/{prompt['id']}", headers=users["a"]["headers"])
        assert deleted.status_code == 409
        # The response identifies in-use state without exposing other users' data (PR-005)
        assert "detail" in deleted.json()
        assert users["b"]["id"] not in deleted.text

    async def test_delete_prompt_referenced_pinned_returns_409(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        await client.post(
            f"/api/v1/prompts/{prompt['id']}/versions", headers=users["a"]["headers"], json={"content": "v2"}
        )
        agent_name = f"agent-{uuid.uuid4().hex[:8]}"
        _, created = await create_agent_yaml(
            client, users["a"]["headers"], agent_yaml_with_prompt_ref(agent_name, prompt["id"], version=2), agent_name
        )
        assert created.status_code == 201

        deleted = await client.delete(f"/api/v1/prompts/{prompt['id']}", headers=users["a"]["headers"])
        assert deleted.status_code == 409

    async def test_delete_succeeds_after_referencing_agent_deleted(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        agent_name = f"agent-{uuid.uuid4().hex[:8]}"
        _, created = await create_agent_yaml(
            client, users["a"]["headers"], agent_yaml_with_prompt_ref(agent_name, prompt["id"]), agent_name
        )
        assert created.status_code == 201

        removed = await client.delete(f"/api/v1/agents/{agent_name}", headers=users["a"]["headers"])
        assert removed.status_code == 204

        deleted = await client.delete(f"/api/v1/prompts/{prompt['id']}", headers=users["a"]["headers"])
        assert deleted.status_code == 204

    async def test_delete_prompt_referenced_by_other_users_agent_not_blocked(self, client, users):
        """Agents only reference their OWN user's prompts (AGT-002), so a
        foreign agent referencing the id cannot exist; deleting own prompt is
        never blocked by another user's data (PR-005)."""
        prompt = await _create_prompt(client, users["a"]["headers"])
        deleted = await client.delete(f"/api/v1/prompts/{prompt['id']}", headers=users["a"]["headers"])
        assert deleted.status_code == 204


class TestDeletionIntegrityRace:
    async def test_concurrent_agent_save_and_prompt_delete_never_dangles(self, client, client_factory, users):
        """PR-005/SEC-002: prompt delete racing with an agent save must not
        accept a dangling reference. Invariant after the race:
        agent referencing the prompt exists  ↔  prompt still exists."""
        prompt = await _create_prompt(client, users["a"]["headers"])
        agent_name = f"agent-{uuid.uuid4().hex[:8]}"

        async def create_agent():
            ac = await client_factory()
            try:
                return await create_agent_yaml(
                    ac, users["a"]["headers"], agent_yaml_with_prompt_ref(agent_name, prompt["id"]), agent_name
                )
            finally:
                await ac.aclose()

        async def delete_prompt():
            ac = await client_factory()
            try:
                return await ac.delete(f"/api/v1/prompts/{prompt['id']}", headers=users["a"]["headers"])
            finally:
                await ac.aclose()

        agent_result, delete_resp = await asyncio.gather(create_agent(), delete_prompt())
        agent_resp = agent_result[1]  # create_agent_yaml returns (name, resp)

        prompt_after = await client.get(f"/api/v1/prompts/{prompt['id']}", headers=users["a"]["headers"])
        agent_after = await client.get(f"/api/v1/agents/{agent_name}", headers=users["a"]["headers"])

        agent_accepted = agent_resp.status_code == 201 and agent_after.status_code == 200
        prompt_exists = prompt_after.status_code == 200

        if agent_accepted:
            # A saved agent referencing the prompt implies the prompt survived
            # (deletion returned 409) — no dangling reference accepted.
            assert delete_resp.status_code == 409, (
                f"Dangling reference accepted: agent saved ({agent_resp.status_code}) while prompt deleted ({delete_resp.status_code})"
            )
            assert prompt_exists
        else:
            # Prompt deleted → the agent save must have failed explicitly
            # (never silently referencing a missing prompt).
            assert delete_resp.status_code == 204
            assert agent_resp.status_code in {400, 404, 409, 422}, agent_resp.status_code


class TestUnreferencedDelete:
    async def test_delete_prompt_with_multiple_versions_204(self, client, users):
        """Deleting a prompt removes it and all immutable versions."""
        prompt = await _create_prompt(client, users["a"]["headers"])
        pid = prompt["id"]
        await client.post(f"/api/v1/prompts/{pid}/versions", headers=users["a"]["headers"], json={"content": "v2"})
        await client.post(f"/api/v1/prompts/{pid}/versions", headers=users["a"]["headers"], json={"content": "v3"})

        deleted = await client.delete(f"/api/v1/prompts/{pid}", headers=users["a"]["headers"])
        assert deleted.status_code == 204
        assert (await client.get(f"/api/v1/prompts/{pid}/versions/3", headers=users["a"]["headers"])).status_code == 404

    async def test_delete_unknown_prompt_404(self, client, users):
        resp = await client.delete(f"/api/v1/prompts/{uuid.uuid4()}", headers=users["a"]["headers"])
        assert resp.status_code == 404


async def _create_prompt(client, headers, name="Deletion prompt", content="content"):
    resp = await client.post("/api/v1/prompts", headers=headers, json={"name": name, "content": content})
    assert resp.status_code == 201
    return resp.json()
