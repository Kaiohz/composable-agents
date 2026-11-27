"""Behavioral tests — manual / YAML editing stays resource-only
(spec §4.I REQ-LEGACY-002/003, §4.D/E update semantics).

Entry point: the REAL ``PUT /api/v1/agents/{name}`` multipart YAML route.

Target contract:
* materialization applies to CREATION only — a YAML update carrying NEW
  inline model/prompt definitions is refused with an explicit error and the
  stored configuration stays untouched;
* an update must leave no partially materialized state behind.
"""

import uuid

import pytest
import yaml

from tests.behavioral.conftest import create_agent_yaml

pytestmark = pytest.mark.asyncio(loop_scope="session")

VALIDATION_CODES = {400, 422}


def _name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def yaml_from(data: dict) -> str:
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


async def put_agent(client, headers: dict[str, str], name: str, yaml_text: str):
    return await client.put(
        f"/api/v1/agents/{name}",
        headers=headers,
        files={"file": ("agent.yaml", yaml_text.encode(), "text/yaml")},
    )


async def models_named(client, headers: dict[str, str], name: str) -> list[dict]:
    listing = await client.get("/api/v1/models", headers=headers)
    assert listing.status_code == 200, listing.text
    return [m for m in listing.json() if m.get("name") == name]


async def prompts_named(client, headers: dict[str, str], name: str) -> list[dict]:
    listing = await client.get("/api/v1/prompts", headers=headers)
    assert listing.status_code == 200, listing.text
    return [p for p in listing.json() if p.get("name") == name]


class TestUpdateRefusesNewInlineDefinitions:
    async def test_update_with_new_inline_system_prompt_is_rejected_and_stored_config_untouched(self, client, users):
        """REQ-LEGACY-003: an update YAML containing a NEW inline prompt
        definition is refused until update-with-conversion exists; the stored
        configuration (and the prompt registry) is unchanged."""
        prompt = await create_prompt(client, users["a"]["headers"])
        agent = _name("upd-inlineprompt")
        _, created = await create_agent_yaml(
            client,
            users["a"]["headers"],
            yaml_from(
                {"name": agent, "tools": [], "debug": False, "model": "gpt-4o", "prompt_ref": {"id": prompt["id"]}}
            ),
            agent,
        )
        assert created.status_code == 201, created.text

        resp = await put_agent(
            client,
            users["a"]["headers"],
            agent,
            yaml_from(
                {
                    "name": agent,
                    "model": "gpt-4o",
                    "system_prompt": "Brand new inline prompt.",
                    "tools": [],
                    "debug": False,
                }
            ),
        )
        assert resp.status_code in VALIDATION_CODES, resp.text

        got = await get_agent(client, users["a"]["headers"], agent)
        assert got.status_code == 200
        assert got.json()["prompt_ref"]["id"] == prompt["id"], "the stored configuration must be untouched"
        assert got.json().get("system_prompt") in (None, ""), "the new inline value must not be silently used"
        new_prompts = await prompts_named(client, users["a"]["headers"], f"{agent}-prompt")
        assert new_prompts == [], "a rejected update must not materialize resources"

    async def test_update_with_new_inline_model_is_rejected_and_stored_config_untouched(self, client, users):
        """REQ-LEGACY-003: same rule for a NEW inline model definition on update."""
        model = await create_model(client, users["a"]["headers"])
        agent = _name("upd-inlinemodel")
        _, created = await create_agent_yaml(
            client,
            users["a"]["headers"],
            yaml_from(
                {
                    "name": agent,
                    "tools": [],
                    "debug": False,
                    "model_ref": model["id"],
                    "system_prompt": "Existing prompt.",
                }
            ),
            agent,
        )
        assert created.status_code == 201, created.text

        resp = await put_agent(
            client,
            users["a"]["headers"],
            agent,
            yaml_from(
                {
                    "name": agent,
                    "model": "brand-new-inline-model",
                    "system_prompt": "Existing prompt.",
                    "tools": [],
                    "debug": False,
                }
            ),
        )
        assert resp.status_code in VALIDATION_CODES, resp.text

        got = await get_agent(client, users["a"]["headers"], agent)
        assert got.status_code == 200
        assert got.json()["model_ref"] == model["id"], "the stored configuration must be untouched"
        assert await models_named(client, users["a"]["headers"], f"{agent}-model") == [], (
            "a rejected update must not materialize resources"
        )


class TestUpdateRefusesNestedInlineDefinitions:
    async def test_nested_inline_mcp_servers_in_subagents_are_rejected_and_stored_config_untouched(self, client, users):
        """BUG-002: an inline MCP definition must never enter through the
        update boundary at ANY depth — a `subagents[i]` entry carrying
        `mcp_servers` (with its auth token) is refused before any write; the
        stored configuration and its secret-free read-back stay untouched."""
        model = await create_model(client, users["a"]["headers"])
        prompt = await create_prompt(client, users["a"]["headers"])
        agent = _name("upd-nestedmcp")
        _, created = await create_agent_yaml(
            client,
            users["a"]["headers"],
            yaml_from(
                {
                    "name": agent,
                    "tools": [],
                    "debug": False,
                    "model_ref": model["id"],
                    "prompt_ref": {"id": prompt["id"]},
                }
            ),
            agent,
        )
        assert created.status_code == 201, created.text

        nested_inline = {
            "name": "nested-inline-srv",
            "transport": "http",
            "url": "http://127.0.0.1:18999/mcp",
            "auth_token": "sekrit-nested-token",
        }
        resp = await put_agent(
            client,
            users["a"]["headers"],
            agent,
            yaml_from(
                {
                    "name": agent,
                    "tools": [],
                    "debug": False,
                    "model_ref": model["id"],
                    "prompt_ref": {"id": prompt["id"]},
                    "subagents": [
                        {
                            "name": "nested-child",
                            "description": "Nested child.",
                            "instructions": "Nested child instructions.",
                            "model": "gpt-4o",
                            "tools": [],
                            "mcp_servers": [nested_inline],
                        }
                    ],
                }
            ),
        )
        assert resp.status_code in VALIDATION_CODES, resp.text
        assert "subagents" in resp.text, "the diagnostic must identify the nested field path"
        assert "sekrit-nested-token" not in resp.text, "no secret may be echoed in the refusal"

        got = await get_agent(client, users["a"]["headers"], agent)
        assert got.status_code == 200
        assert got.json()["model_ref"] == model["id"], "the stored configuration must be untouched"
        assert "sekrit-nested-token" not in got.text, "no secret may be persisted or echoed"
        assert "nested-inline-srv" not in got.text
        assert got.json().get("subagents") in (None, []), "the refused nested body must not be persisted"


class TestUpdateRefusesInlineModelNextToRef:
    async def test_update_with_model_ref_and_new_inline_model_is_rejected(self, client, users):
        """BUG-003: an update YAML carrying BOTH the existing `model_ref` AND
        a NEW inline `model` is refused, exactly like the prompt-side rule —
        the raw YAML is persisted verbatim, so the inline value would stay in
        double with the reference and override the display projection."""
        model = await create_model(client, users["a"]["headers"])
        prompt = await create_prompt(client, users["a"]["headers"])
        agent = _name("upd-dblmodel")
        _, created = await create_agent_yaml(
            client,
            users["a"]["headers"],
            yaml_from(
                {
                    "name": agent,
                    "tools": [],
                    "debug": False,
                    "model_ref": model["id"],
                    "prompt_ref": {"id": prompt["id"]},
                }
            ),
            agent,
        )
        assert created.status_code == 201, created.text

        resp = await put_agent(
            client,
            users["a"]["headers"],
            agent,
            yaml_from(
                {
                    "name": agent,
                    "model": "brand-new-inline-model",
                    "model_ref": model["id"],
                    "prompt_ref": {"id": prompt["id"]},
                    "tools": [],
                    "debug": False,
                }
            ),
        )
        assert resp.status_code in VALIDATION_CODES, resp.text
        assert "model" in resp.text, "the diagnostic must identify the inline model definition"

        got = await get_agent(client, users["a"]["headers"], agent)
        assert got.status_code == 200
        assert got.json()["model_ref"] == model["id"], "the stored configuration must be untouched"
        assert "brand-new-inline-model" not in got.text, (
            "the inline value must not be persisted in double with its reference"
        )
        assert got.json().get("system_prompt") in (None, "")
