"""Behavioral tests — per-user customization of the built-in assistant.

Entry point: the REAL REST agent routes over the REAL PostgreSQL/MinIO stack.
The feature keeps the built-in assistant virtual: its default definition stays
in code, while a per-user override is stored in a dedicated user-scoped table
(never in the shared agent YAML bucket, never as an ``agent_configs`` row).
"""

import uuid
from typing import Any

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


def builtin_yaml_model_ref(
    *,
    model_ref: str,
    name: str = BUILTIN_NAME,
    system_prompt: str = "Personalized built-in prompt.",
    description: str = "Personalized built-in assistant.",
    legacy_model: str | None = None,
) -> str:
    """A built-in override that selects its model through ``model_ref`` only.

    The legacy ``model`` field is absent (the customization path the frontend
    produces) unless ``legacy_model`` is explicitly supplied — the edge case
    where both fields coexist.
    """
    legacy = f"model: {legacy_model}\n" if legacy_model is not None else ""
    return (
        f"name: {name}\n"
        f"{legacy}"
        f"model_ref: {model_ref}\n"
        f'description: "{description}"\n'
        f'system_prompt: "{system_prompt}"\n'
        "tools: []\n"
        "debug: false\n"
    )


async def create_model(client, headers: dict[str, str], name: str, provider_model: str) -> dict:
    """Create a Model through the REAL POST /api/v1/models route.

    The new contract (spec §B) accepts exactly ``name`` and ``provider_model``.
    """
    resp = await client.post("/api/v1/models", headers=headers, json={"name": name, "provider_model": provider_model})
    assert resp.status_code == 201, resp.text
    return resp.json()


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


class TestModelRefProjection:
    """Spec §A (BUILTIN-META-002/004/005/007): the listing projection of a
    built-in customized with ``model_ref`` must expose the referenced Model's
    CURRENT ``provider_model`` — never the legacy Pydantic default that sits in
    ``AgentConfig.model`` (``claude-sonnet-4-5-20250929``) when the YAML omits
    ``model``."""

    async def test_listing_projects_the_referenced_provider_model(self, client, users):
        model = await create_model(client, users["a"]["headers"], "GLM Nitro", "z-ai/glm-5.3-flash:nitro")
        created = await put_override(client, users["a"]["headers"], builtin_yaml_model_ref(model_ref=model["id"]))
        assert created.status_code == 200, created.text

        listing = await client.get("/api/v1/agents", headers=users["a"]["headers"])
        assert listing.status_code == 200, listing.text
        entry = built_in_entry(listing.json())
        assert entry["model"] == "z-ai/glm-5.3-flash:nitro", (
            f"the built-in card must project the referenced provider_model, got: {entry['model']}"
        )
        # The legacy Pydantic default must never leak into the projection.
        assert entry["model"] != "claude-sonnet-4-5-20250929"
        # BUILTIN-META-004/007: the other fields stay unchanged.
        assert entry["model_ref"] == model["id"]
        assert entry["is_builtin"] is True
        assert entry["is_customized"] is True
        assert entry["description"] == "Personalized built-in assistant."
        assert entry["name"] == BUILTIN_NAME

    async def test_projection_refreshes_after_the_model_is_edited(self, client, users):
        """BUILTIN-META-005: the projection reads the CURRENT resource at each
        listing, so an edited ``provider_model`` is visible on the next call."""
        model = await create_model(client, users["a"]["headers"], "Editable", "catalog-v1-model")
        created = await put_override(client, users["a"]["headers"], builtin_yaml_model_ref(model_ref=model["id"]))
        assert created.status_code == 200

        first = built_in_entry((await client.get("/api/v1/agents", headers=users["a"]["headers"])).json())
        assert first["model"] == "catalog-v1-model"

        patched = await client.patch(
            f"/api/v1/models/{model['id']}", headers=users["a"]["headers"], json={"provider_model": "catalog-v2-model"}
        )
        assert patched.status_code == 200, patched.text

        second = built_in_entry((await client.get("/api/v1/agents", headers=users["a"]["headers"])).json())
        assert second["model"] == "catalog-v2-model", (
            "the projection must read the current Model resource on each listing"
        )

    async def test_yaml_with_model_ref_and_legacy_model_still_projects_provider_model(self, client, users):
        """Edge case: a YAML containing BOTH ``model_ref`` and a legacy ``model``
        projects (and the runtime executes) the referenced ``provider_model``."""
        model = await create_model(client, users["a"]["headers"], "Both Fields", "winning-provider-model")
        created = await put_override(
            client,
            users["a"]["headers"],
            builtin_yaml_model_ref(model_ref=model["id"], legacy_model="legacy-string"),
        )
        assert created.status_code == 200, created.text

        entry = built_in_entry((await client.get("/api/v1/agents", headers=users["a"]["headers"])).json())
        assert entry["model"] == "winning-provider-model"
        assert entry["model_ref"] == model["id"]

    async def test_customized_builtin_without_model_ref_keeps_config_model(self, client, users):
        """BUILTIN-META-003: without ``model_ref``, the projection conserves
        ``config.model``, including an explicitly supplied legacy value."""
        created = await put_override(client, users["a"]["headers"], builtin_yaml(model="personalized-model"))
        assert created.status_code == 200

        entry = built_in_entry((await client.get("/api/v1/agents", headers=users["a"]["headers"])).json())
        assert entry["model"] == "personalized-model"
        assert entry["model_ref"] is None

    async def test_missing_model_ref_at_projection_errors_without_fallback(self, client, users, stack):
        """Reliability: a stored override whose ``model_ref`` can no longer be
        resolved makes the listing fail with the existing resolution error — it
        must never fall back to the Pydantic default ``model``."""
        missing_id = str(uuid.uuid4())
        async with system_rls_context(), stack["engine"].begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO builtin_agent_overrides "
                    "(user_id, agent_name, yaml, created_at, updated_at) "
                    "VALUES (:user_id, :agent_name, :yaml, now(), now())"
                ),
                {
                    "user_id": users["a"]["id"],
                    "agent_name": BUILTIN_NAME,
                    "yaml": builtin_yaml_model_ref(model_ref=missing_id),
                },
            )

        listing = await client.get("/api/v1/agents", headers=users["a"]["headers"])
        assert listing.status_code == 404, (
            f"an unresolvable model_ref must surface the existing resolution error, got: {listing.text}"
        )
        assert "Model not found" in listing.json()["detail"]

    async def test_builtin_projections_are_per_user(self, client, users):
        """Edge case: two users personalize their built-in with different
        Models; each listing projects exclusively the owner's resource."""
        model_a = await create_model(client, users["a"]["headers"], "Model A", "user-a-provider-model")
        model_b = await create_model(client, users["b"]["headers"], "Model B", "user-b-provider-model")
        got_a = await put_override(client, users["a"]["headers"], builtin_yaml_model_ref(model_ref=model_a["id"]))
        got_b = await put_override(client, users["b"]["headers"], builtin_yaml_model_ref(model_ref=model_b["id"]))
        assert got_a.status_code == 200, got_a.text
        assert got_b.status_code == 200, got_b.text

        entry_a = built_in_entry((await client.get("/api/v1/agents", headers=users["a"]["headers"])).json())
        entry_b = built_in_entry((await client.get("/api/v1/agents", headers=users["b"]["headers"])).json())
        assert entry_a["model"] == "user-a-provider-model"
        assert entry_a["model_ref"] == model_a["id"]
        assert entry_b["model"] == "user-b-provider-model"
        assert entry_b["model_ref"] == model_b["id"]

    async def test_projection_resolves_the_model_once_per_listing(self, client, users, stack):
        """BUILTIN-META-006 (SQL-level, like the RLS tests): a listing of a
        built-in customized with ``model_ref`` performs EXACTLY one Model read
        (the ``at most one resolution`` NFR — no hidden second lookup), while a
        user without an override performs none (BUILTIN-META-001)."""
        model = await create_model(client, users["a"]["headers"], "Counted", "counted-provider-model")
        created = await put_override(client, users["a"]["headers"], builtin_yaml_model_ref(model_ref=model["id"]))
        assert created.status_code == 200, created.text

        from sqlalchemy import event

        engine = stack["engine"]
        model_queries: list[str] = []

        def _count_model_reads(*args: Any) -> None:
            """SQLAlchemy ``before_cursor_execute`` hook — positional signature."""
            statement = args[2]
            if "FROM models" in statement:
                model_queries.append(statement)

        event.listens_for(engine.sync_engine, "before_cursor_execute")(_count_model_reads)
        try:
            # BUILTIN-META-001: no override → no Model read at all.
            listing_b = await client.get("/api/v1/agents", headers=users["b"]["headers"])
            assert listing_b.status_code == 200, listing_b.text
            assert model_queries == [], "a user without override must not trigger any Model read"

            model_queries.clear()
            listing_a = await client.get("/api/v1/agents", headers=users["a"]["headers"])
            assert listing_a.status_code == 200, listing_a.text
            assert built_in_entry(listing_a.json())["model"] == "counted-provider-model"
            assert len(model_queries) == 1, f"exactly one resolution per listing, got: {len(model_queries)}"

            model_queries.clear()
            again = await client.get("/api/v1/agents", headers=users["a"]["headers"])
            assert again.status_code == 200, again.text
            assert built_in_entry(again.json())["model"] == "counted-provider-model"
            assert len(model_queries) == 1, "the resolution is per-listing, never accumulated or cached"
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", _count_model_reads)
