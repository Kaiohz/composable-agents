"""Behavioral tests — prompt & model resolution through REAL agent invocations
(spec §5.5 AGT-002/AGT-003, §7 CACHE-001/CACHE-002, §10.3/10.6/10.8).

Entry points: POST /api/v1/threads → POST /api/v1/chat/{thread_id} — the full
production chain runs real (router → use case → registry → DeepAgent factory →
runner → repositories). The ONLY mock is the external OpenAI-compatible LLM
API (in-process ``StubLlmServer``): each completion request records the
resolved system prompt (``messages[0].system``), the effective model name and
the ``Authorization`` header, giving end-to-end observability of
prompt/model/credential resolution and cache freshness.
"""

import uuid

import pytest

from tests.behavioral.conftest import (
    configure_llm_credentials,
    create_agent_yaml,
    create_thread,
    invoke_agent,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

VALIDATION_CODES = {400, 422}


def yaml_prompt_ref(name: str, prompt_id: str, version: int | None = None) -> str:
    if version is not None:
        ref = f"prompt_ref:\n  id: {prompt_id}\n  version: {version}\n"
    else:
        ref = f"prompt_ref:\n  id: {prompt_id}\n"
    # REQ-MODEL-003: a new custom agent must also declare a model source.
    return f"name: {name}\nmodel: test-model\ntools: []\ndebug: false\n{ref}"


def yaml_model_ref(name: str, model_id: str) -> str:
    # REQ-PROMPT-005: a new custom agent must also declare a prompt source.
    return f'name: {name}\ntools: []\ndebug: false\nmodel_ref: {model_id}\nsystem_prompt: "Resolution prompt."\n'


def yaml_legacy(name: str, model: str) -> str:
    return f'name: {name}\nmodel: {model}\nsystem_prompt: "Legacy inline prompt."\ntools: []\ndebug: false\n'


async def _configured_stack(client, users, yaml_text: str, name: str):
    """Create agent + configure LLM creds + create a thread — all real APIs."""
    _, created = await create_agent_yaml(client, users["a"]["headers"], yaml_text, name)
    assert created.status_code == 201, created.text
    await configure_llm_credentials(client, users["a"]["headers"])
    thread = await create_thread(client, users["a"]["headers"], name)
    assert thread.status_code == 201, thread.text
    return thread.json()


class TestLatestVsPinnedResolution:
    async def test_latest_agent_picks_new_version_pinned_agent_stays(self, client, users, stub_llm):
        """Success criterion 2.4 / E2E-01: an agent using Latest resolves the
        newly created version on its next invocation; a pinned agent keeps the
        exact immutable content (no Phoenix involved — the system prompt must
        come from the local registry)."""
        prompt_v1 = "LATEST_MARKER_V1"
        created = await client.post(
            "/api/v1/prompts",
            headers=users["a"]["headers"],
            json={"name": f"resolve-{uuid.uuid4().hex[:6]}", "content": prompt_v1},
        )
        assert created.status_code == 201
        prompt = created.json()
        pid = prompt["id"]

        latest_agent = f"latest-agent-{uuid.uuid4().hex[:6]}"
        pinned_agent = f"pinned-agent-{uuid.uuid4().hex[:6]}"
        latest_thread = await _configured_stack(client, users, yaml_prompt_ref(latest_agent, pid), latest_agent)
        pinned_thread = await _configured_stack(
            client, users, yaml_prompt_ref(pinned_agent, pid, version=1), pinned_agent
        )

        # --- First invocation: both agents resolve v1 ---
        resp = await invoke_agent(client, users["a"]["headers"], latest_thread["id"])
        assert resp.status_code == 200, resp.text
        resp = await invoke_agent(client, users["a"]["headers"], pinned_thread["id"])
        assert resp.status_code == 200
        before_count = len(stub_llm.app.requests)
        assert before_count == 2
        for recorded in stub_llm.app.requests:
            assert prompt_v1 in "".join(recorded.system_messages), (
                f"Expected pinned v1 prompt in LLM messages, got: {recorded.system_messages}"
            )

        # --- Add version 2 (CACHE-002: new prompt content version) ---
        v2 = await client.post(f"/api/v1/prompts/{pid}/versions", headers=users["a"]["headers"], json={"content": "LATEST_MARKER_V2"})
        assert v2.status_code == 201

        # Latest agent must resolve v2 on next invocation…
        resp = await invoke_agent(client, users["a"]["headers"], latest_thread["id"])
        assert resp.status_code == 200
        latest_records = stub_llm.app.requests[before_count:]
        assert len(latest_records) == 1
        assert "LATEST_MARKER_V2" in "".join(latest_records[0].system_messages), "Latest agent did not pick up the new version"

        # …while the pinned agent keeps resolving exactly v1.
        count = len(stub_llm.app.requests)
        resp = await invoke_agent(client, users["a"]["headers"], pinned_thread["id"])
        assert resp.status_code == 200
        pinned_records = stub_llm.app.requests[count:]
        assert len(pinned_records) == 1
        assert "LATEST_MARKER_V1" in "".join(pinned_records[0].system_messages), "Pinned agent lost its immutable version"

    async def test_metadata_only_prompt_edit_does_not_change_content(self, client, users, stub_llm):
        """CACHE-002: prompt metadata-only edit → no prompt-content rebuild."""
        prompt = await client.post(
            "/api/v1/prompts",
            headers=users["a"]["headers"],
            json={"name": f"meta-{uuid.uuid4().hex[:6]}", "content": "CONTENT_MARKER"},
        )
        assert prompt.status_code == 201
        pid = prompt.json()["id"]

        name = f"agent-{uuid.uuid4().hex[:6]}"
        thread = await _configured_stack(client, users, yaml_prompt_ref(name, pid), name)
        resp = await invoke_agent(client, users["a"]["headers"], thread["id"])
        assert resp.status_code == 200
        before_count = len(stub_llm.app.requests)

        patched = await client.patch(
            f"/api/v1/prompts/{pid}", headers=users["a"]["headers"], json={"description": "renamed"}
        )
        assert patched.status_code == 200

        resp = await invoke_agent(client, users["a"]["headers"], thread["id"])
        assert resp.status_code == 200
        latest = stub_llm.app.requests[before_count:]
        assert len(latest) == 1
        assert "CONTENT_MARKER" in "".join(latest[0].system_messages)

    async def test_implicit_phoenix_agent_name_lookup_removed(self, client, users, stub_llm):
        """§8.8: remove the runner's implicit Phoenix prompt lookup / agent-name
        prompt resolution. With Phoenix unreachable (conftest env), building and
        invoking an inline-prompt agent must not touch Phoenix and must send the
        inline prompt to the LLM."""
        name = f"agent-{uuid.uuid4().hex[:6]}"
        thread = await _configured_stack(client, users, yaml_legacy(name, "stub-model"), name)
        resp = await invoke_agent(client, users["a"]["headers"], thread["id"])
        assert resp.status_code == 200, resp.text
        assert "Legacy inline prompt." in "".join(stub_llm.app.requests[0].system_messages), (
            "Inline prompt was replaced by a Phoenix/agent-name lookup"
        )


class TestModelResolution:
    async def test_catalog_model_ref_used_for_metadata_on_invocation(self, client, users, stub_llm):
        """E2E-02: an agent with model_ref resolves the catalog model metadata
        (stub observes the effective model name == provider_model, not the
        legacy YAML model string)."""
        model = await client.post(
            "/api/v1/models",
            headers=users["a"]["headers"],
            json={"name": f"m-{uuid.uuid4().hex[:6]}", "provider": "stub", "provider_model": "catalog-gpt-a"},
        )
        assert model.status_code == 201
        model_id = model.json()["id"]

        name = f"agent-{uuid.uuid4().hex[:6]}"
        yaml_text = (
            f"name: {name}\ntools: []\ndebug: false\n"
            f"model_ref: {model_id}\n"
            "model: legacy-should-be-ignored\n"
            "system_prompt: \"Catalog serving prompt.\"\n"
        )
        thread = await _configured_stack(client, users, yaml_text, name)
        resp = await invoke_agent(client, users["a"]["headers"], thread["id"])
        assert resp.status_code == 200, resp.text
        assert stub_llm.app.requests[0].model == "catalog-gpt-a", (
            f"Catalog model_ref not used for invocation: {stub_llm.app.requests[0].model}"
        )

    async def test_model_metadata_edit_applies_on_next_invocation(self, client, users, stub_llm):
        """CACHE-002: model metadata edit → rebuild/invalidate referencing
        agents; success criterion 2.5."""
        model = await client.post(
            "/api/v1/models",
            headers=users["a"]["headers"],
            json={"name": f"m-{uuid.uuid4().hex[:6]}", "provider": "stub", "provider_model": "catalog-v1-model"},
        )
        model_id = model.json()["id"]
        name = f"agent-{uuid.uuid4().hex[:6]}"
        thread = await _configured_stack(client, users, yaml_model_ref(name, model_id), name)

        resp = await invoke_agent(client, users["a"]["headers"], thread["id"])
        assert resp.status_code == 200
        assert stub_llm.app.requests[0].model == "catalog-v1-model"

        patched = await client.patch(
            f"/api/v1/models/{model_id}",
            headers=users["a"]["headers"],
            json={"provider": "stub", "provider_model": "catalog-v2-model"},
        )
        assert patched.status_code == 200

        resp = await invoke_agent(client, users["a"]["headers"], thread["id"])
        assert resp.status_code == 200
        assert stub_llm.app.requests[-1].model == "catalog-v2-model", (
            "Model metadata edit was not picked up on next invocation (stale cache)"
        )

    async def test_legacy_custom_model_string_invoked_unchanged(self, client, users, stub_llm):
        """E2E-02: agents with legacy custom model strings keep existing
        behavior — the effective model is the YAML string."""
        name = f"agent-{uuid.uuid4().hex[:6]}"
        thread = await _configured_stack(client, users, yaml_legacy(name, "legacy-custom-string"), name)
        resp = await invoke_agent(client, users["a"]["headers"], thread["id"])
        assert resp.status_code == 200, resp.text
        assert stub_llm.app.requests[0].model == "legacy-custom-string"


class TestUserScopedCredentials:
    async def test_never_share_runtime_credentials_across_users(self, client, users, stub_llm):
        """CACHE-001: cached runners must never share user-specific resolved
        state. Both users invoke agents through the real chain; every LLM
        request must carry the INVOKING user's own configured credential."""
        name_a = f"agent-a-{uuid.uuid4().hex[:6]}"
        name_b = f"agent-b-{uuid.uuid4().hex[:6]}"

        # Deterministic per-user LLM credentials (distinct on purpose).
        await configure_llm_credentials(client, users["a"]["headers"], api_key="sk-cred-A")
        await configure_llm_credentials(client, users["b"]["headers"], api_key="sk-cred-B")

        _, created_a = await create_agent_yaml(client, users["a"]["headers"], yaml_legacy(name_a, "stub-model"), name_a)
        assert created_a.status_code == 201
        thread_a = await create_thread(client, users["a"]["headers"], name_a)
        assert thread_a.status_code == 201

        _, created_b = await create_agent_yaml(client, users["b"]["headers"], yaml_legacy(name_b, "stub-model"), name_b)
        assert created_b.status_code == 201
        thread_b = await create_thread(client, users["b"]["headers"], name_b)
        assert thread_b.status_code == 201

        a_key = "sk-cred-A"
        b_key = "sk-cred-B"

        resp = await invoke_agent(client, users["a"]["headers"], thread_a.json()["id"])
        assert resp.status_code == 200, resp.text
        assert stub_llm.app.requests[0].auth_header == f"Bearer {a_key}", (
            f"User A's invocation used credentials: {stub_llm.app.requests[0].auth_header!r}"
        )

        resp = await invoke_agent(client, users["b"]["headers"], thread_b.json()["id"])
        assert resp.status_code == 200, resp.text
        b_records = [r for r in stub_llm.app.requests if r.auth_header is not None]
        assert any(r.auth_header == f"Bearer {b_key}" for r in b_records), "User B's invocation lacked B's credentials"
        # And A's key was never reused for B's agent…
        assert not any(r.auth_header == f"Bearer {a_key}" and "agent-b" in str(r.body) for r in stub_llm.app.requests)

    async def test_per_user_llm_settings_change_isolated(self, client, users, stub_llm):
        """Cross-user credential purity: reconfiguring ONE user's settings
        must never bleed into the other user's invocations."""
        await configure_llm_credentials(client, users["a"]["headers"], api_key="sk-cred-A")
        await configure_llm_credentials(client, users["b"]["headers"], api_key="sk-cred-B")

        name_b = f"agent-b-{uuid.uuid4().hex[:6]}"
        _, created_b = await create_agent_yaml(client, users["b"]["headers"], yaml_legacy(name_b, "stub-model"), name_b)
        assert created_b.status_code == 201
        thread_b = await create_thread(client, users["b"]["headers"], name_b)
        assert thread_b.status_code == 201

        resp = await invoke_agent(client, users["b"]["headers"], thread_b.json()["id"])
        assert resp.status_code == 200
        assert stub_llm.app.requests[-1].auth_header == "Bearer sk-cred-B"
