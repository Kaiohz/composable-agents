"""Behavioral tests — per-user customization of the built-in assistant.

Entry point: the REAL REST agent routes over the REAL PostgreSQL/MinIO stack.
The feature keeps the built-in assistant virtual: its default definition stays
in code, while a per-user override is stored in a dedicated user-scoped table
(never in the shared agent YAML bucket, never as an ``agent_configs`` row).
"""

import uuid

import pytest
from sqlalchemy import text

from src.infrastructure.database.rls_context import system_rls_context
from tests.behavioral.conftest import configure_llm_credentials, create_thread, invoke_agent

pytestmark = pytest.mark.asyncio(loop_scope="session")

BUILTIN_NAME = "composable-ai-assistant"
VALIDATION_CODES = {400, 422}
UNAVAILABLE_CODES = {404, 422}

FORCED_HITL_TOOLS = (
    "composable_agents_delete",
    "composable_mcp_servers_delete",
    "composable_skills_delete",
    "composable_memories_delete",
    "composable_prompts_delete",
    "composable_models_delete",
    "composable_crons_delete",
    "composable_threads_delete",
)


def builtin_yaml(
    *,
    name: str = BUILTIN_NAME,
    system_prompt: str = "Personalized built-in prompt.",
    model: str = "personalized-model",
    description: str = "Personalized built-in assistant.",
    extra: str = "",
) -> str:
    return (
        f"name: {name}\n"
        f"model: {model}\n"
        f'description: "{description}"\n'
        f'system_prompt: "{system_prompt}"\n'
        "tools: []\n"
        "debug: false\n" + extra
    )


async def put_override(client, headers: dict[str, str], yaml_text: str):
    return await client.put(
        f"/api/v1/agents/{BUILTIN_NAME}",
        headers=headers,
        files={"file": ("builtin.yaml", yaml_text.encode(), "text/yaml")},
    )


async def count_rows(stack: dict, table: str, name: str) -> int:
    async with system_rls_context(), stack["engine"].begin() as conn:
        return (
            await conn.execute(
                text(f"SELECT count(*) FROM {table} WHERE name = :name"),  # noqa: S608
                {"name": name},
            )
        ).scalar_one()


async def count_override_rows(stack: dict, user_id: str) -> int:
    async with system_rls_context(), stack["engine"].begin() as conn:
        return (
            await conn.execute(
                text("SELECT count(*) FROM builtin_agent_overrides WHERE user_id = :user_id"),
                {"user_id": user_id},
            )
        ).scalar_one()


class TestDefaultAndReadResolution:
    async def test_default_definition_is_served_without_customization(self, client, users, stack):
        got = await client.get(f"/api/v1/agents/{BUILTIN_NAME}", headers=users["a"]["headers"])
        assert got.status_code == 200, got.text
        body = got.json()
        assert body["is_builtin"] is True
        assert body["is_customized"] is False
        assert body["system_prompt"].startswith("You are the Composable AI Assistant")
        assert set(FORCED_HITL_TOOLS).issubset(body["hitl"]["rules"])

        listing = await client.get("/api/v1/agents", headers=users["a"]["headers"])
        assert listing.status_code == 200
        entries = listing.json()
        assert entries[0]["name"] == BUILTIN_NAME
        assert entries[0]["is_builtin"] is True
        assert entries[0]["is_customized"] is False
        assert count(await client.get("/api/v1/agents", headers=users["a"]["headers"]), BUILTIN_NAME) == 1

    async def test_override_is_visible_only_to_its_owner(self, client, users, stack):
        created = await put_override(client, users["a"]["headers"], builtin_yaml())
        assert created.status_code == 200, created.text
        assert created.json()["is_builtin"] is True
        assert created.json()["is_customized"] is True

        got_a = await client.get(f"/api/v1/agents/{BUILTIN_NAME}", headers=users["a"]["headers"])
        assert got_a.status_code == 200
        assert got_a.json()["system_prompt"] == "Personalized built-in prompt."
        assert got_a.json()["model"] == "personalized-model"
        assert got_a.json()["is_customized"] is True

        got_b = await client.get(f"/api/v1/agents/{BUILTIN_NAME}", headers=users["b"]["headers"])
        assert got_b.status_code == 200
        assert got_b.json()["system_prompt"].startswith("You are the Composable AI Assistant")
        assert got_b.json()["is_customized"] is False

        listing_a = await client.get("/api/v1/agents", headers=users["a"]["headers"])
        listing_b = await client.get("/api/v1/agents", headers=users["b"]["headers"])
        assert built_in_entry(listing_a.json())["is_customized"] is True
        assert built_in_entry(listing_b.json())["is_customized"] is False
        assert count(listing_a, BUILTIN_NAME) == 1
        assert count(listing_b, BUILTIN_NAME) == 1

    async def test_override_never_pollutes_agent_persistence(self, client, users, stack):
        created = await put_override(client, users["a"]["headers"], builtin_yaml())
        assert created.status_code == 200

        assert await count_rows(stack, "agent_configs", BUILTIN_NAME) == 0
        assert await count_override_rows(stack, users["a"]["id"]) == 1

    async def test_second_update_replaces_the_override_snapshot(self, client, users):
        first = await put_override(client, users["a"]["headers"], builtin_yaml(system_prompt="First revision."))
        assert first.status_code == 200, first.text

        second = await put_override(
            client,
            users["a"]["headers"],
            builtin_yaml(system_prompt="Second revision.", description="Second description."),
        )
        assert second.status_code == 200, second.text

        got = await client.get(f"/api/v1/agents/{BUILTIN_NAME}", headers=users["a"]["headers"])
        assert got.json()["system_prompt"] == "Second revision."
        assert got.json()["description"] == "Second description."
        assert "First revision." not in got.json()["system_prompt"]


class TestUpdateValidation:
    async def test_put_without_yaml_returns_standard_validation_error(self, client, users):
        got = await client.put(f"/api/v1/agents/{BUILTIN_NAME}", headers=users["a"]["headers"])
        assert got.status_code == 422, got.text

    async def test_rename_is_rejected(self, client, users):
        got = await put_override(
            client,
            users["a"]["headers"],
            builtin_yaml(name="another-agent"),
        )
        assert got.status_code == 400, got.text
        assert "does not match" in got.json()["detail"]

    async def test_dynamic_tools_cannot_be_injected(self, client, users):
        got = await put_override(
            client,
            users["a"]["headers"],
            builtin_yaml(extra='tools:\n  - "search"\n'),
        )
        assert got.status_code == 422, got.text
        assert "tools" in got.json()["detail"]

    async def test_static_subagents_are_rejected(self, client, users):
        got = await put_override(
            client,
            users["a"]["headers"],
            builtin_yaml(extra="subagents:\n  - name: helper\n    description: Helper\n"),
        )
        assert got.status_code == 422, got.text
        assert "subagents" in got.json()["detail"]

    async def test_static_mcp_servers_are_rejected(self, client, users):
        got = await put_override(
            client,
            users["a"]["headers"],
            builtin_yaml(
                extra="mcp_servers:\n  - name: injected\n    transport: http\n    url: http://127.0.0.1/mcp\n"
            ),
        )
        assert got.status_code == 422, got.text
        assert "mcp_servers" in got.json()["detail"]

    async def test_hitl_delete_floor_cannot_be_disabled(self, client, users):
        got = await put_override(
            client,
            users["a"]["headers"],
            builtin_yaml(extra="hitl:\n  rules:\n    composable_agents_delete: false\n"),
        )
        assert got.status_code == 422, got.text
        assert "composable_agents_delete" in got.json()["detail"]

    async def test_hitl_floor_and_additions_round_trip(self, client, users):
        rules = "\n".join(f"    {tool}: true" for tool in FORCED_HITL_TOOLS)
        yaml_text = builtin_yaml(extra=f"hitl:\n  rules:\n{rules}\n    my_custom_tool: true\n")
        got = await put_override(client, users["a"]["headers"], yaml_text)
        assert got.status_code == 200, got.text

        effective = await client.get(f"/api/v1/agents/{BUILTIN_NAME}", headers=users["a"]["headers"])
        rules_effective = effective.json()["hitl"]["rules"]
        assert all(rules_effective[tool] is True for tool in FORCED_HITL_TOOLS)
        assert rules_effective["my_custom_tool"] is True

    async def test_invalid_prompt_ref_is_rejected_without_fallback(self, client, users):
        got = await put_override(
            client,
            users["a"]["headers"],
            builtin_yaml(extra=f"prompt_ref:\n  id: {uuid.uuid4()}\n").replace(
                'system_prompt: "Personalized built-in prompt."\n', ""
            ),
        )
        assert got.status_code in UNAVAILABLE_CODES, got.text

    async def test_invalid_model_ref_is_rejected_without_fallback(self, client, users):
        got = await put_override(
            client,
            users["a"]["headers"],
            builtin_yaml(extra=f"model_ref: {uuid.uuid4()}\n"),
        )
        assert got.status_code in UNAVAILABLE_CODES, got.text


class TestReset:
    async def test_reset_restores_default_and_is_idempotent(self, client, users):
        created = await put_override(client, users["a"]["headers"], builtin_yaml())
        assert created.status_code == 200

        first = await client.delete(f"/api/v1/agents/{BUILTIN_NAME}/customization", headers=users["a"]["headers"])
        assert first.status_code == 204, first.text

        got_a = await client.get(f"/api/v1/agents/{BUILTIN_NAME}", headers=users["a"]["headers"])
        assert got_a.json()["is_customized"] is False
        assert got_a.json()["system_prompt"].startswith("You are the Composable AI Assistant")

        second = await client.delete(f"/api/v1/agents/{BUILTIN_NAME}/customization", headers=users["a"]["headers"])
        assert second.status_code == 204

    async def test_reset_for_custom_agent_name_is_not_found(self, client, users):
        got = await client.delete("/api/v1/agents/my-agent/customization", headers=users["a"]["headers"])
        assert got.status_code == 404, got.text

    async def test_corrupt_override_is_recoverable_by_reset(self, client, users, stack):
        async with system_rls_context(), stack["engine"].begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO builtin_agent_overrides "
                    "(user_id, agent_name, yaml, created_at, updated_at) "
                    "VALUES (:user_id, :agent_name, :yaml, now(), now())"
                ),
                {"user_id": users["a"]["id"], "agent_name": BUILTIN_NAME, "yaml": "not: [valid"},
            )

        reset = await client.delete(f"/api/v1/agents/{BUILTIN_NAME}/customization", headers=users["a"]["headers"])
        assert reset.status_code == 204, reset.text

        got = await client.get(f"/api/v1/agents/{BUILTIN_NAME}", headers=users["a"]["headers"])
        assert got.status_code == 200
        assert got.json()["is_customized"] is False


class TestEffectiveRunner:
    async def test_invocation_uses_the_owner_override(self, client, users, stub_llm):
        created = await put_override(
            client,
            users["a"]["headers"],
            builtin_yaml(system_prompt="CUSTOMIZED_RUNNER_PROMPT", model="custom-runner-model"),
        )
        assert created.status_code == 200, created.text

        await configure_llm_credentials(client, users["a"]["headers"])
        thread = await create_thread(client, users["a"]["headers"], BUILTIN_NAME)
        assert thread.status_code == 201, thread.text

        response = await invoke_agent(client, users["a"]["headers"], thread.json()["id"])
        assert response.status_code == 200, response.text

        record = stub_llm.app.requests[-1]
        assert "CUSTOMIZED_RUNNER_PROMPT" in "".join(record.system_messages), (
            f"the built-in runner ignored the user's override: {record.system_messages}"
        )
        assert record.model == "custom-runner-model", (
            f"the built-in runner ignored the override's model: {record.model}"
        )


def built_in_entry(entries: list[dict]) -> dict:
    matches = [entry for entry in entries if entry["name"] == BUILTIN_NAME]
    assert len(matches) == 1, f"the built-in assistant must appear exactly once, got {matches}"
    return matches[0]


def count(response, name: str) -> int:
    return sum(1 for entry in response.json() if entry["name"] == name)
