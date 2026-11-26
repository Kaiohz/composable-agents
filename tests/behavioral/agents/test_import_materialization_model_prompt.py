"""Behavioral tests — YAML import materializes inline model & prompt resources
(spec §4.D REQ-MODEL-001..003, §4.E REQ-PROMPT-001..005, §4.A REQ-FORM-005).

Entry point: the REAL ``POST /api/v1/agents`` multipart YAML create route
over the real PostgreSQL/MinIO stack. Materialization is observed through
the prompt/model registry REST routes (side effects) and the strict-schema
agent read-back (``GET /api/v1/agents/{name}``).

Target contract:
* an explicitly declared inline ``model`` creates a Model catalog resource
  named ``<agent-name>-model`` (provider from the prefix before the first
  ``, ``openai`` otherwise; ``provider_model`` = the exact YAML value) and
  the agent persists ``model_ref``;
* an explicitly declared inline ``system_prompt`` creates a Prompt registry
  resource named ``<agent-name>-prompt`` with immutable version 1 holding
  the EXACT text, and the agent persists ``prompt_ref`` in Latest mode;
* existing ``model_ref`` / ``prompt_ref`` references are reused without
  creating any resource;
* for a NEW custom agent, a missing model (and a missing prompt) is a
  validation failure — the Pydantic legacy default must not silently
  materialize or mask an incomplete configuration.
"""

import uuid

import pytest
import yaml

from tests.behavioral.conftest import create_agent_yaml

pytestmark = pytest.mark.asyncio(loop_scope="session")

VALIDATION_CODES = {400, 422}
# Name conflicts are explicit errors; the registry contract decides 409 vs
# validation-level wording ({400,409,422} accepted).
CONFLICT_CODES = {400, 409, 422}


def _name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def import_yaml(name: str, *, model: str | None = None, system_prompt: str | None = None, **extra) -> str:
    """Build a minimal import YAML containing ONLY explicitly declared fields."""
    data: dict = {"name": name, "tools": [], "debug": False}
    if model is not None:
        data["model"] = model
    if system_prompt is not None:
        data["system_prompt"] = system_prompt
    data.update(extra)
    return yaml.safe_dump(data, sort_keys=False)


async def create_model(client, headers, provider="openai", provider_model="catalog-model") -> dict:
    resp = await client.post(
        "/api/v1/models",
        headers=headers,
        json={"name": f"m-{uuid.uuid4().hex[:8]}", "provider": provider, "provider_model": provider_model},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def create_prompt(client, headers, content="registry prompt content") -> dict:
    resp = await client.post(
        "/api/v1/prompts", headers=headers, json={"name": f"p-{uuid.uuid4().hex[:8]}", "content": content}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def get_agent(client, headers: dict[str, str], name: str):
    return await client.get(f"/api/v1/agents/{name}", headers=headers)


async def models_named(client, headers: dict[str, str], name: str) -> list[dict]:
    listing = await client.get("/api/v1/models", headers=headers)
    assert listing.status_code == 200, listing.text
    return [m for m in listing.json() if m.get("name") == name]


async def prompts_named(client, headers: dict[str, str], name: str) -> list[dict]:
    listing = await client.get("/api/v1/prompts", headers=headers)
    assert listing.status_code == 200, listing.text
    return [p for p in listing.json() if p.get("name") == name]


class TestModelMaterialization:
    async def test_inline_model_is_materialized_and_agent_uses_its_ref(self, client, users):
        """REQ-MODEL-001: inline `model` creates a Model resource; the agent
        persists `model_ref` and no longer carries the inline value."""
        agent = _name("imp-model")
        _, resp = await create_agent_yaml(
            client,
            users["a"]["headers"],
            import_yaml(agent, model="anthropic:claude-3-7-sonnet", system_prompt="You are helpful."),
            agent,
        )
        assert resp.status_code == 201, resp.text

        created = await models_named(client, users["a"]["headers"], f"{agent}-model")
        assert len(created) == 1, f"a Model resource named '{agent}-model' must be materialized: {created}"
        assert created[0]["provider"] == "anthropic"
        assert created[0]["provider_model"] == "anthropic:claude-3-7-sonnet"

        got = await get_agent(client, users["a"]["headers"], agent)
        assert got.status_code == 200
        assert got.json()["model_ref"] == created[0]["id"]

    async def test_inline_model_without_prefix_defaults_to_openai_provider(self, client, users):
        """REQ-MODEL-001: no ``:`` prefix in the model value → provider from the
        OpenAI-compatible serving chain (``openai``), exact provider_model."""
        agent = _name("imp-noprefix")
        _, resp = await create_agent_yaml(
            client,
            users["a"]["headers"],
            import_yaml(agent, model="gpt-4o-mini", system_prompt="You are helpful."),
            agent,
        )
        assert resp.status_code == 201, resp.text

        created = await models_named(client, users["a"]["headers"], f"{agent}-model")
        assert len(created) == 1, created
        assert created[0]["provider"] == "openai"
        assert created[0]["provider_model"] == "gpt-4o-mini"

    async def test_existing_model_ref_is_reused_without_new_resource(self, client, users):
        """REQ-IMPORT-004 / REQ-MODEL-002: a valid `model_ref` is validated and
        reused — no additional Model resource is created for it."""
        model = await create_model(client, users["a"]["headers"])
        agent = _name("imp-refonly")
        _, resp = await create_agent_yaml(
            client,
            users["a"]["headers"],
            import_yaml(agent, model_ref=model["id"], system_prompt="You are helpful."),
            agent,
        )
        assert resp.status_code == 201, resp.text

        assert await models_named(client, users["a"]["headers"], f"{agent}-model") == [], (
            "a reference to an existing model must not materialize a duplicate resource"
        )
        got = await get_agent(client, users["a"]["headers"], agent)
        assert got.json()["model_ref"] == model["id"]


class TestPromptMaterialization:
    async def test_inline_system_prompt_is_materialized_as_prompt_version_one(self, client, users):
        """REQ-PROMPT-001: inline `system_prompt` creates a Prompt resource +
        immutable version 1 holding the EXACT text; the agent persists
        `prompt_ref` in Latest mode (no pinned version)."""
        agent = _name("imp-prompt")
        text = "Line one.\nLine two.\n"
        _, resp = await create_agent_yaml(
            client,
            users["a"]["headers"],
            import_yaml(agent, model="gpt-4o", system_prompt=text),
            agent,
        )
        assert resp.status_code == 201, resp.text

        created = await prompts_named(client, users["a"]["headers"], f"{agent}-prompt")
        assert len(created) == 1, f"a Prompt resource named '{agent}-prompt' must be materialized: {created}"
        assert created[0]["latest_version"] == 1

        version = await client.get(
            f"/api/v1/prompts/{created[0]['id']}/versions/1", headers=users["a"]["headers"]
        )
        assert version.status_code == 200, version.text
        assert version.json()["content"] == text, "version 1 must hold the exact verbatim content"

        got = await get_agent(client, users["a"]["headers"], agent)
        assert got.json()["prompt_ref"]["id"] == created[0]["id"]
        assert got.json()["prompt_ref"].get("version") in (None,), (
            "the imported prompt must be attached in Latest mode, not pinned"
        )

    async def test_explicitly_empty_inline_prompt_is_preserved(self, client, users):
        """REQ-PROMPT-002: an explicitly empty inline prompt is valid content —
        it must be materialized verbatim, never replaced by a platform prompt."""
        agent = _name("imp-emptyprompt")
        _, resp = await create_agent_yaml(
            client,
            users["a"]["headers"],
            import_yaml(agent, model="gpt-4o", system_prompt=""),
            agent,
        )
        assert resp.status_code == 201, resp.text

        created = await prompts_named(client, users["a"]["headers"], f"{agent}-prompt")
        assert len(created) == 1, created
        version = await client.get(
            f"/api/v1/prompts/{created[0]['id']}/versions/1", headers=users["a"]["headers"]
        )
        assert version.json()["content"] == ""


class TestRequiredResources:
    async def test_new_custom_agent_without_model_or_model_ref_is_rejected(self, client, users):
        """REQ-MODEL-003: a new custom agent must declare a model — neither the
        Pydantic legacy default nor a silent resource creation may substitute
        for the missing declaration."""
        agent = _name("imp-nomodel")
        _, resp = await create_agent_yaml(
            client,
            users["a"]["headers"],
            import_yaml(agent, system_prompt="You are helpful."),
            agent,
        )
        assert resp.status_code in VALIDATION_CODES, resp.text
        assert await models_named(client, users["a"]["headers"], f"{agent}-model") == [], (
            "the legacy Pydantic default model must not materialize a resource"
        )

    async def test_new_custom_agent_without_prompt_or_prompt_ref_is_rejected(self, client, users):
        """REQ-PROMPT-005: a new custom agent must declare a prompt."""
        agent = _name("imp-noprompt")
        _, resp = await create_agent_yaml(
            client,
            users["a"]["headers"],
            import_yaml(agent, model="gpt-4o"),
            agent,
        )
        assert resp.status_code in VALIDATION_CODES, resp.text
        assert await prompts_named(client, users["a"]["headers"], f"{agent}-prompt") == []


class TestReferencedModelPrecedence:
    async def test_model_ref_with_legacy_model_string_keeps_precedence_and_creates_no_unused_resource(
        self, client, users
    ):
        """REQ-MODEL-002: legacy `model` next to a valid `model_ref` —
        precedence kept, and NO unused Model resource is materialized."""
        model = await create_model(client, users["a"]["headers"])
        agent = _name("imp-prec")
        _, resp = await create_agent_yaml(
            client,
            users["a"]["headers"],
            import_yaml(agent, model="legacy-unused-string", model_ref=model["id"], system_prompt="You are helpful."),
            agent,
        )
        assert resp.status_code == 201, resp.text

        assert await models_named(client, users["a"]["headers"], f"{agent}-model") == []
        got = await get_agent(client, users["a"]["headers"], agent)
        assert got.json()["model_ref"] == model["id"]


class TestResourceNameConflicts:
    """REQ-TX-002 / REQ-IMPORT-006: the deterministic derived names are
    conflict-checked against the owner's catalogs BEFORE any resource is
    created — a retry or a pre-existing name can never silently duplicate."""

    async def test_preexisting_model_name_conflict_refuses_the_import(self, client, users):
        agent = _name("imp-nameclash")
        taken = await client.post(
            "/api/v1/models",
            headers=users["a"]["headers"],
            json={"name": f"{agent}-model", "provider": "catalog", "provider_model": "catalog-model"},
        )
        assert taken.status_code == 201, taken.text

        _, resp = await create_agent_yaml(
            client,
            users["a"]["headers"],
            import_yaml(agent, model="gpt-4o", system_prompt="You are helpful."),
            agent,
        )
        assert resp.status_code in CONFLICT_CODES, resp.text

        created = await models_named(client, users["a"]["headers"], f"{agent}-model")
        assert len(created) == 1, "the conflicting import must not duplicate the existing resource"
        assert (await get_agent(client, users["a"]["headers"], agent)).status_code == 404
        assert await prompts_named(client, users["a"]["headers"], f"{agent}-prompt") == [], (
            "no orphan prompt may survive a refused import"
        )

    async def test_preexisting_prompt_name_conflict_refuses_the_import(self, client, users):
        agent = _name("imp-promptclash")
        taken = await client.post(
            "/api/v1/prompts",
            headers=users["a"]["headers"],
            json={"name": f"{agent}-prompt", "content": "Existing content."},
        )
        assert taken.status_code == 201, taken.text

        _, resp = await create_agent_yaml(
            client,
            users["a"]["headers"],
            import_yaml(agent, model="gpt-4o", system_prompt="You are helpful."),
            agent,
        )
        assert resp.status_code in CONFLICT_CODES, resp.text

        assert len(await prompts_named(client, users["a"]["headers"], f"{agent}-prompt")) == 1, (
            "the conflicting import must not duplicate the existing prompt"
        )
        assert (await get_agent(client, users["a"]["headers"], agent)).status_code == 404
        assert await models_named(client, users["a"]["headers"], f"{agent}-model") == []
