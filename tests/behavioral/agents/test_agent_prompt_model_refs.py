"""Behavioral tests — agent YAML prompt_ref/model_ref contracts
(spec §5.5 AGT-001…AGT-003).

Routes under test:
* POST /api/v1/agents  (multipart YAML create)
* PUT  /api/v1/agents/{name} (multipart YAML update)
* GET  /api/v1/agents/{name} (strict schema read-back)

All save-time validation is server-side and authoritative (AGT-004.8).
Cross-user references are treated as non-enumerating unavailable resources.
"""

import uuid

import pytest

from tests.behavioral.conftest import create_agent_yaml

pytestmark = pytest.mark.asyncio(loop_scope="session")

VALIDATION_CODES = {400, 422}
# Spec §9: cross-user references are "non-enumerating unavailable/not-found".
# The exact code among {404 not-found, 422 validation} is left unpinned:
UNAVAILABLE_CODES = {404, 422}


def _uuid() -> str:
    return str(uuid.uuid4())


def yaml_prompt_ref(name: str, prompt_id: str, version: int | None = None, extra: str = "") -> str:
    # REQ-MODEL-003: a new custom agent must declare a model source as well.
    body = f"name: {name}\nmodel: test-model\ntools: []\ndebug: false\n"
    if version is not None:
        body += f"prompt_ref:\n  id: {prompt_id}\n  version: {version}\n"
    else:
        body += f"prompt_ref:\n  id: {prompt_id}\n"
    body += extra
    return body


def yaml_model_ref(name: str, model_id: str, legacy_model: str | None = None) -> str:
    # REQ-PROMPT-005: a new custom agent must declare a prompt source as well.
    body = f"name: {name}\ntools: []\ndebug: false\nmodel_ref: {model_id}\n"
    if legacy_model is not None:
        body += f"model: {legacy_model}\n"
    body += 'system_prompt: "You are helpful."\n'
    return body


async def _create_prompt(client, headers, content="prompt content"):
    resp = await client.post("/api/v1/prompts", headers=headers, json={"name": f"p-{_uuid()[:8]}", "content": content})
    assert resp.status_code == 201
    return resp.json()


class TestPromptRefAccepted:
    async def test_latest_ref_saved_and_preserved(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        name = f"agent-{uuid.uuid4().hex[:8]}"
        _, resp = await create_agent_yaml(client, users["a"]["headers"], yaml_prompt_ref(name, prompt["id"]), name)
        assert resp.status_code == 201, resp.text

        got = await client.get(f"/api/v1/agents/{name}", headers=users["a"]["headers"])
        assert got.status_code == 200
        # §8.12: prompt_ref preserved (not stripped) in the strict schema.
        assert got.json()["prompt_ref"]["id"] == prompt["id"]
        assert "version" not in got.json()["prompt_ref"] or got.json()["prompt_ref"]["version"] is None

    async def test_pinned_ref_saved_with_version(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        name = f"agent-{uuid.uuid4().hex[:8]}"
        _, resp = await create_agent_yaml(
            client, users["a"]["headers"], yaml_prompt_ref(name, prompt["id"], version=1), name
        )
        assert resp.status_code == 201
        got = await client.get(f"/api/v1/agents/{name}", headers=users["a"]["headers"])
        assert got.json()["prompt_ref"]["version"] == 1

    async def test_updating_to_inline_prompt_is_refused_and_stored_config_untouched(
        self, client, users
    ):
        """REQ-LEGACY-003: materialization applies to CREATION only — an update
        YAML carrying a NEW inline prompt definition is refused with an explicit
        error and the stored configuration stays untouched."""
        prompt = await _create_prompt(client, users["a"]["headers"])
        name = f"agent-{uuid.uuid4().hex[:8]}"
        _, created = await create_agent_yaml(client, users["a"]["headers"], yaml_prompt_ref(name, prompt["id"]), name)
        assert created.status_code == 201

        yaml_legacy = f'name: {name}\nmodel: test-model\nsystem_prompt: "inline now"\ntools: []\ndebug: false\n'
        resp = await client.put(
            f"/api/v1/agents/{name}",
            headers=users["a"]["headers"],
            files={"file": ("agent.yaml", yaml_legacy.encode(), "text/yaml")},
        )
        assert resp.status_code in VALIDATION_CODES, resp.text
        got = await client.get(f"/api/v1/agents/{name}", headers=users["a"]["headers"])
        assert got.status_code == 200
        # The rejected update must not be partially applied: the reference
        # stays the source of truth.
        assert got.json().get("prompt_ref") is not None
        assert got.json()["prompt_ref"]["id"] == prompt["id"]
        assert got.json().get("system_prompt") in (None, "")
        new_prompts = await client.get("/api/v1/prompts", headers=users["a"]["headers"])
        assert all(p["name"] != f"{name}-prompt" for p in new_prompts.json()), (
            "a rejected update must not materialize resources"
        )


class TestPromptRefExclusivity:
    async def test_prompt_ref_with_inline_prompt_rejected(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        name = f"agent-{uuid.uuid4().hex[:8]}"
        yaml_text = yaml_prompt_ref(name, prompt["id"], extra='system_prompt: "You are inline."\n')
        _, resp = await create_agent_yaml(client, users["a"]["headers"], yaml_text, name)
        assert resp.status_code in VALIDATION_CODES

    async def test_prompt_ref_with_file_prompt_rejected(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        name = f"agent-{uuid.uuid4().hex[:8]}"
        yaml_text = yaml_prompt_ref(name, prompt["id"], extra='system_prompt_file: "prompt.txt"\n')
        _, resp = await create_agent_yaml(client, users["a"]["headers"], yaml_text, name)
        assert resp.status_code in VALIDATION_CODES

    async def test_empty_inline_prompt_does_not_bypass_exclusivity(self, client, users):
        """AGT-002: validation uses field presence, not truthiness — an empty
        inline system_prompt must not quietly coexist with prompt_ref."""
        prompt = await _create_prompt(client, users["a"]["headers"])
        name = f"agent-{uuid.uuid4().hex[:8]}"
        yaml_text = yaml_prompt_ref(name, prompt["id"], extra='system_prompt: ""\n')
        _, resp = await create_agent_yaml(client, users["a"]["headers"], yaml_text, name)
        assert resp.status_code in VALIDATION_CODES

    async def test_inline_plus_file_still_rejected(self, client, users):
        name = f"agent-{uuid.uuid4().hex[:8]}"
        yaml_text = (
            f"name: {name}\nmodel: test-model\ntools: []\ndebug: false\n"
            'system_prompt: "a"\nsystem_prompt_file: "b.txt"\n'
        )
        _, resp = await create_agent_yaml(client, users["a"]["headers"], yaml_text, name)
        assert resp.status_code in VALIDATION_CODES


class TestPromptRefValidation:
    async def test_pinned_nonexistent_version_rejected_no_latest_fallback(self, client, users):
        """§9: pinned version missing → explicit failure; no Latest fallback."""
        prompt = await _create_prompt(client, users["a"]["headers"])
        name = f"agent-{uuid.uuid4().hex[:8]}"
        _, resp = await create_agent_yaml(
            client, users["a"]["headers"], yaml_prompt_ref(name, prompt["id"], version=99), name
        )
        assert resp.status_code in UNAVAILABLE_CODES, resp.text

    async def test_pinned_version_zero_rejected(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        name = f"agent-{uuid.uuid4().hex[:8]}"
        _, resp = await create_agent_yaml(
            client, users["a"]["headers"], yaml_prompt_ref(name, prompt["id"], version=0), name
        )
        assert resp.status_code in VALIDATION_CODES

    async def test_non_uuid_ref_rejected(self, client, users):
        name = f"agent-{uuid.uuid4().hex[:8]}"
        _, resp = await create_agent_yaml(client, users["a"]["headers"], yaml_prompt_ref(name, "not-a-uuid"), name)
        assert resp.status_code in VALIDATION_CODES

    async def test_missing_prompt_id_rejected(self, client, users):
        name = f"agent-{uuid.uuid4().hex[:8]}"
        yaml_text = f"name: {name}\nmodel: test-model\ntools: []\nprompt_ref:\n  id: {_uuid()}\n"
        _, resp = await create_agent_yaml(client, users["a"]["headers"], yaml_text, name)
        assert resp.status_code in UNAVAILABLE_CODES

    async def test_cross_user_prompt_ref_rejected_non_enumerating(self, client, users):
        a_prompt = await _create_prompt(client, users["a"]["headers"])
        name = f"agent-{uuid.uuid4().hex[:8]}"
        _, resp = await create_agent_yaml(client, users["b"]["headers"], yaml_prompt_ref(name, a_prompt["id"]), name)
        assert resp.status_code in UNAVAILABLE_CODES

    async def test_deleted_prompt_reference_rejected(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        deleted = await client.delete(f"/api/v1/prompts/{prompt['id']}", headers=users["a"]["headers"])
        assert deleted.status_code == 204
        name = f"agent-{uuid.uuid4().hex[:8]}"
        _, resp = await create_agent_yaml(client, users["a"]["headers"], yaml_prompt_ref(name, prompt["id"]), name)
        assert resp.status_code in UNAVAILABLE_CODES


class TestModelRef:
    async def test_model_ref_saved_and_preserved(self, client, users):
        model = await _create_model(client, users["a"]["headers"])
        name = f"agent-{uuid.uuid4().hex[:8]}"
        _, resp = await create_agent_yaml(client, users["a"]["headers"], yaml_model_ref(name, model["id"]), name)
        assert resp.status_code == 201, resp.text
        got = await client.get(f"/api/v1/agents/{name}", headers=users["a"]["headers"])
        assert got.json()["model_ref"] == model["id"]

    async def test_model_ref_takes_precedence_with_legacy_model_still_accepted(self, client, users):
        """AGT-003: model_ref precedence — the legacy model string may remain
        in the YAML (schema compatibility) when a valid model_ref exists."""
        model = await _create_model(client, users["a"]["headers"])
        name = f"agent-{uuid.uuid4().hex[:8]}"
        _, resp = await create_agent_yaml(
            client, users["a"]["headers"], yaml_model_ref(name, model["id"], legacy_model="legacy-string"), name
        )
        assert resp.status_code == 201
        got = await client.get(f"/api/v1/agents/{name}", headers=users["a"]["headers"])
        assert got.json()["model_ref"] == model["id"]

    async def test_invalid_model_ref_rejected_never_falls_back(self, client, users):
        """AGT-003: an invalid model_ref is an error, never a fallback to `model`."""
        name = f"agent-{uuid.uuid4().hex[:8]}"
        yaml_text = (
            f"name: {name}\ntools: []\ndebug: false\nmodel_ref: {_uuid()}\n"
            "model: legacy-fallback-string\n"
            'system_prompt: "s"\n'
        )
        _, resp = await create_agent_yaml(client, users["a"]["headers"], yaml_text, name)
        assert resp.status_code in UNAVAILABLE_CODES, resp.text

    async def test_cross_user_model_ref_rejected(self, client, users):
        a_model = await _create_model(client, users["a"]["headers"])
        name = f"agent-{uuid.uuid4().hex[:8]}"
        _, resp = await create_agent_yaml(client, users["b"]["headers"], yaml_model_ref(name, a_model["id"]), name)
        assert resp.status_code in UNAVAILABLE_CODES

    async def test_removing_model_ref_for_a_new_inline_model_is_refused(self, client, users):
        """REQ-LEGACY-003: switching BACK to a new inline model definition is
        refused on update (conversion to registered resources is the manual-
        edit conversion path); the stored configuration stays untouched."""
        model = await _create_model(client, users["a"]["headers"])
        name = f"agent-{uuid.uuid4().hex[:8]}"
        _, created = await create_agent_yaml(client, users["a"]["headers"], yaml_model_ref(name, model["id"]), name)
        assert created.status_code == 201

        yaml_legacy = f'name: {name}\nmodel: back-to-legacy\nsystem_prompt: "s"\ntools: []\ndebug: false\n'
        resp = await client.put(
            f"/api/v1/agents/{name}",
            headers=users["a"]["headers"],
            files={"file": ("agent.yaml", yaml_legacy.encode(), "text/yaml")},
        )
        assert resp.status_code in VALIDATION_CODES, resp.text
        got = await client.get(f"/api/v1/agents/{name}", headers=users["a"]["headers"])
        assert got.status_code == 200
        assert got.json()["model_ref"] == model["id"]
        new_models = await client.get("/api/v1/models", headers=users["a"]["headers"])
        assert all(m["name"] != f"{name}-model" for m in new_models.json()), (
            "a rejected update must not materialize resources"
        )


class TestLegacyCompatibility:
    async def test_inline_values_are_materialized_into_registered_resources(self, client, users):
        """Spec agent-resource-import: an inline-only creation is materialized —
        the persisted agent references the created catalog model and registry
        prompt instead of carrying the inline values (REQ-IMPORT-003)."""
        name = f"agent-{uuid.uuid4().hex[:8]}"
        yaml_text = f'name: {name}\nmodel: anthropic:my-legacy-model\nsystem_prompt: "Legacy prompt."\ntools: []\ndebug: false\n'
        _, resp = await create_agent_yaml(client, users["a"]["headers"], yaml_text, name)
        assert resp.status_code == 201

        listing_models = await client.get("/api/v1/models", headers=users["a"]["headers"])
        materialized = [m for m in listing_models.json() if m["name"] == f"{name}-model"]
        assert len(materialized) == 1
        listing_prompts = await client.get("/api/v1/prompts", headers=users["a"]["headers"])
        materialized_prompt = [p for p in listing_prompts.json() if p["name"] == f"{name}-prompt"]
        assert len(materialized_prompt) == 1

        got = await client.get(f"/api/v1/agents/{name}", headers=users["a"]["headers"])
        assert got.status_code == 200
        assert got.json()["model_ref"] == materialized[0]["id"]
        assert got.json()["prompt_ref"]["id"] == materialized_prompt[0]["id"]

    async def test_same_user_prompt_and_model_refs_saved(self, client, users):
        """A single agent may reference the user's own prompt AND model."""
        prompt = await _create_prompt(client, users["a"]["headers"])
        model = await _create_model(client, users["a"]["headers"])
        name = f"agent-{uuid.uuid4().hex[:8]}"
        yaml_text = (
            f"name: {name}\ntools: []\ndebug: false\nprompt_ref:\n  id: {prompt['id']}\nmodel_ref: {model['id']}\n"
        )
        _, resp = await create_agent_yaml(client, users["a"]["headers"], yaml_text, name)
        assert resp.status_code == 201
        got = await client.get(f"/api/v1/agents/{name}", headers=users["a"]["headers"])
        assert got.json()["prompt_ref"]["id"] == prompt["id"]
        assert got.json()["model_ref"] == model["id"]


async def _create_model(client, headers, name: str | None = None, provider_model="catalog-model"):
    """Create a Model through the REAL route, using the new identity-only
    contract (exactly ``name`` + ``provider_model`` — no ``provider``,
    no ``base_url``)."""
    resp = await client.post(
        "/api/v1/models",
        headers=headers,
        json={"name": name or f"m-{uuid.uuid4().hex[:8]}", "provider_model": provider_model},
    )
    assert resp.status_code == 201
    return resp.json()
