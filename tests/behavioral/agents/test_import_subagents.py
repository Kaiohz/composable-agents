"""Behavioral tests — YAML import materializes inline sub-agents, skills and
memories (spec §4.G REQ-SUBAGENT-001..004, REQ-SKILL-001, REQ-MEMORY-001).

Entry point: the REAL ``POST /api/v1/agents`` multipart YAML create route,
plus the Store File API (``GET /api/v1/store/files/{path}``) for observing
skill/memory file creation, the agent registry routes for observing
sub-agent creation, and the real chat chain for observing the RUNTIME effect
of the imported files.

Target contract:
* a fully inline sub-agent is created as a real Agent resource, its own
  ``instructions``/``model`` materialized (prompt/model rules), and its
  inline definition in the parent is replaced by ``agent_ref``;
* MCP ownership is EXPLICIT and scoped: a server declared inside an inline
  sub-agent entry belongs to that sub-agent's ``mcp_refs`` selection and is
  never mounted by (or persisted on) the parent; a REFERENCE entry carries
  only ``agent_ref`` and never an inline MCP definition (REQ-MCP-003);
* an existing ``agent_ref`` is validated and reused without recreation;
  cyclic references are refused before any creation;
* ``path + content`` skill/memory definitions are written to the Store
  File API verbatim and only the textual path remains on the agent;
  existing store files are never overwritten; the imported files are copied
  into the agent's namespace at build time (the runtime contract);
* a textual skill/memory reference to a path missing from the user's Store
  fails the import instead of creating an empty resource.
"""

import uuid

import pytest
import yaml

from tests.behavioral.conftest import (
    configure_llm_credentials,
    create_agent_yaml,
    create_thread,
    invoke_agent,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

VALIDATION_CODES = {400, 422}
UNAVAILABLE_CODES = {400, 404, 422}


def _name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def import_yaml(name: str, *, model: str = "gpt-4o", system_prompt: str = "You are helpful.", **extra) -> str:
    data: dict = {"name": name, "model": model, "system_prompt": system_prompt, "tools": [], "debug": False}
    data.update(extra)
    return yaml.safe_dump(data, sort_keys=False)


async def import_agent(client, headers, yaml_text: str, name: str):
    _, resp = await create_agent_yaml(client, headers, yaml_text, name)
    return resp


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


import_yaml_ = import_yaml  # alias kept for readability in long test bodies


class TestSubagentMaterialization:
    async def test_inline_subagent_is_created_as_agent_and_parent_keeps_only_agent_ref(self, client, users):
        """REQ-SUBAGENT-001: a fully inline sub-agent is created as an Agent
        resource; the parent keeps the ``agent_ref`` and no inline definition."""
        agent = _name("imp-parent")
        helper = _name("imp-helper")
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml_(
                agent,
                subagents=[{"name": helper, "description": "Helper", "instructions": "Helper instructions.", "model": "gpt-helper", "tools": []}],
            ),
            agent,
        )
        assert resp.status_code == 201, resp.text

        got = await get_agent(client, users["a"]["headers"], helper)
        assert got.status_code == 200, f"the inline sub-agent must exist as a real agent: {got.text}"

        parent = await get_agent(client, users["a"]["headers"], agent)
        assert parent.status_code == 200
        subagents = parent.json()["subagents"]
        assert len(subagents) == 1
        assert subagents[0]["agent_ref"] == helper, "the parent must keep only the agent reference"
        assert not subagents[0].get("instructions"), "the inline instructions must not remain duplicated"
        assert subagents[0].get("model") in (None, ""), "the inline model must not remain duplicated"

    async def test_inline_subagent_resources_are_recursively_materialized(self, client, users):
        """REQ-SUBAGENT-001: the sub-agent's own `model` and `instructions`
        follow the Model/Prompt materialization rules named after the
        sub-agent."""
        agent = _name("imp-parentres")
        helper = _name("imp-helperres")
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml_(
                agent,
                subagents=[{"name": helper, "description": "Helper", "instructions": "Helper prompt text.", "model": "gpt-helper", "tools": []}],
            ),
            agent,
        )
        assert resp.status_code == 201, resp.text

        helper_model = await models_named(client, users["a"]["headers"], f"{helper}-model")
        assert len(helper_model) == 1, f"the sub-agent model must be materialized: {helper_model}"
        assert helper_model[0]["provider"] == "openai"
        assert helper_model[0]["provider_model"] == "gpt-helper"

        helper_prompt = await prompts_named(client, users["a"]["headers"], f"{helper}-prompt")
        assert len(helper_prompt) == 1, helper_prompt
        version = await client.get(
            f"/api/v1/prompts/{helper_prompt[0]['id']}/versions/1", headers=users["a"]["headers"]
        )
        assert version.json()["content"] == "Helper prompt text."

        helper_config = await get_agent(client, users["a"]["headers"], helper)
        assert helper_config.json()["model_ref"] == helper_model[0]["id"]
        assert helper_config.json()["prompt_ref"]["id"] == helper_prompt[0]["id"]

    async def test_agent_ref_to_existing_agent_is_reused_without_new_resources(self, client, users):
        """REQ-SUBAGENT-002: an existing agent reference is validated and
        reused — no duplicate sub-agent and no new resources."""
        helper = _name("imp-existinghelper")
        first = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml_(helper, model="shared-model", system_prompt="Shared prompt."),
            helper,
        )
        assert first.status_code == 201, first.text

        agent = _name("imp-refparent")
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml_(agent, subagents=[{"name": "helper", "description": "Helper", "agent_ref": helper}]),
            agent,
        )
        assert resp.status_code == 201, resp.text

        # No resource was created a second time for the referenced sub-agent.
        assert len(await models_named(client, users["a"]["headers"], f"{helper}-model")) == 1
        assert len(await prompts_named(client, users["a"]["headers"], f"{helper}-prompt")) == 1

    async def test_cyclic_agent_refs_are_refused_before_any_creation(self, client, users):
        """REQ-SUBAGENT-002: circular sub-agent references are refused by
        prevalidation — nothing is created."""
        agent = _name("imp-parent-loop")
        helper_a = _name("imp-loopa")
        helper_b = _name("imp-loopb")
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml_(
                agent,
                subagents=[
                    {"name": helper_a, "description": "Waiting on B", "agent_ref": helper_b},
                    {"name": helper_b, "description": "Waiting on A", "agent_ref": helper_a},
                ],
            ),
            agent,
        )
        assert resp.status_code in VALIDATION_CODES, resp.text
        assert "cycle" in resp.text.lower(), "the diagnostic must identify the cyclic reference"
        assert (await get_agent(client, users["a"]["headers"], agent)).status_code == 404
        assert (await get_agent(client, users["a"]["headers"], helper_a)).status_code == 404
        assert (await get_agent(client, users["a"]["headers"], helper_b)).status_code == 404
        assert await models_named(client, users["a"]["headers"], f"{helper_a}-model") == []
        assert await prompts_named(client, users["a"]["headers"], f"{helper_b}-prompt") == []


def http_server(name: str, url: str, *, token: str | None = None) -> dict:
    server: dict = {"name": name, "transport": "http", "url": url}
    if token:
        server["auth_token"] = token
    return server


class TestSubagentMcpOwnership:
    """C1/C2 (review): MCP selections are owned, scoped and never leaked."""

    async def test_subagent_owned_servers_stay_in_the_subagent_selection(
        self, client, users, stub_registry
    ):
        """REQ-SUBAGENT-001 / REQ-MCP-003: a server declared INSIDE an inline
        sub-agent entry belongs to THAT sub-agent's ``mcp_refs`` selection —
        the parent keeps (and mounts) only its own identities and never the
        sub-agent's connection parameters or secrets."""
        agent = _name("imp-parentmcp")
        helper = _name("imp-helpermcp")
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml_(
                agent,
                mcp_servers=[http_server("parent-srv", "http://127.0.0.1:9501/mcp")],
                subagents=[
                    {
                        "name": helper,
                        "description": "Child",
                        "instructions": "Child instructions.",
                        "model": "gpt-helper",
                        "tools": [],
                        "mcp_servers": [
                            http_server("child-srv", "http://127.0.0.1:9502/mcp", token="sekrit-child")
                        ],
                    }
                ],
            ),
            agent,
        )
        assert resp.status_code == 201, resp.text

        parent = await get_agent(client, users["a"]["headers"], agent)
        assert parent.status_code == 200, parent.text
        assert sorted(parent.json()["mcp_refs"]) == ["parent-srv"], (
            f"the parent must never inherit the sub-agent's selection: {parent.json()['mcp_refs']}"
        )
        assert parent.json()["mcp_servers"] == []
        assert "child-srv" not in parent.text, "the sub-agent's server must not leak onto the parent"
        assert "sekrit" not in parent.text, "no secret may persist in agent settings"

        child = await get_agent(client, users["a"]["headers"], helper)
        assert child.status_code == 200, child.text
        assert sorted(child.json()["mcp_refs"]) == ["child-srv"], child.json()["mcp_refs"]
        assert child.json()["mcp_servers"] == []
        assert "sekrit" not in child.text, "secrets stay in the registry"

    async def test_reference_entry_inline_mcp_definition_is_refused_before_any_write(
        self, client, users, stub_registry
    ):
        """REQ-MCP-003 (review C2): a REFERENCE sub-agent entry carries only
        its reference — an inline MCP definition (with its secrets) is refused
        before any write instead of being persisted on the parent as
        unregistered, unvalidated configuration."""
        helper = _name("imp-existingref")
        first = await import_agent(
            client, users["a"]["headers"], import_yaml_(helper, model="m-legacy", system_prompt="Shared prompt."), helper
        )
        assert first.status_code == 201, first.text
        stub_registry.reset(seed={})

        agent = _name("imp-refwithmcp")
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml_(
                agent,
                subagents=[
                    {
                        "name": "child",
                        "description": "Child",
                        "agent_ref": helper,
                        "mcp_servers": [
                            http_server("ghost-srv", "http://127.0.0.1:9599/mcp", token="sekrit-ref")
                        ],
                    }
                ],
            ),
            agent,
        )
        assert resp.status_code in VALIDATION_CODES, resp.text
        assert "sekrit-ref" not in resp.text, "diagnostics must not echo secret material"
        assert "ghost-srv" not in stub_registry.app.entries, "a refused entry must not be created in the registry"
        assert (await get_agent(client, users["a"]["headers"], agent)).status_code == 404, (
            "the refusing import must not create the agent"
        )

    async def test_reference_entry_is_normalized_to_the_reference_only(self, client, users):
        """C2: the persisted reference entry carries EXACTLY
        ``{name, description, agent_ref}`` — with no selection of its own it
        is represented empty (never a copy of the parent's or of an inline
        unregistered fallback)."""
        helper = _name("imp-normalizedref")
        first = await import_agent(
            client, users["a"]["headers"], import_yaml_(helper, model="m-legacy", system_prompt="Shared prompt."), helper
        )
        assert first.status_code == 201, first.text

        agent = _name("imp-refplain")
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml_(
                agent,
                subagents=[{"name": "child", "description": "Child", "agent_ref": helper}],
            ),
            agent,
        )
        assert resp.status_code == 201, resp.text

        parent = await get_agent(client, users["a"]["headers"], agent)
        assert parent.status_code == 200, parent.text
        entry = parent.json()["subagents"][0]
        assert entry["agent_ref"] == helper, (
            f"the reference entry must keep only the reference and empty defaults: {entry}"
        )
        # Nothing but the reference: the non-reference payload is empty.
        assert entry["mcp_servers"] == [], "the entry never carries an inline MCP fallback (C2)"
        assert entry["skills"] == []
        assert not entry.get("instructions")
        assert not entry.get("model")
        assert entry["tools"] == []
        assert parent.json()["mcp_refs"] == []


class TestInlineSkillsAndMemories:
    async def test_inline_skill_definition_writes_skill_md_and_keeps_only_the_path(self, client, users):
        """REQ-SKILL-001: a `{path, content}` skill definition is written
        verbatim to `<path>SKILL.md` via the Store File API and the agent
        keeps only the textual path reference."""
        agent = _name("imp-skill")
        skill_path = "skills/imp-skill-dir/"
        content = "# My skill\n\nInstructions verbatim.\n"
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml_(agent, skills=[{"path": skill_path, "content": content}]),
            agent,
        )
        assert resp.status_code == 201, resp.text

        stored = await client.get(
            "/api/v1/store/files/skills/imp-skill-dir/SKILL.md", headers=users["a"]["headers"]
        )
        assert stored.status_code == 200, f"the skill file must exist in the store: {stored.text}"
        assert stored.json()["content"] == content, "the content must be written without modification"

        got = (await get_agent(client, users["a"]["headers"], agent)).json()
        assert got["skills"] == [skill_path], "the agent must keep only the path reference"

    async def test_inline_memory_definition_writes_memory_file_and_keeps_only_the_path(self, client, users):
        """REQ-MEMORY-001: a `{path, content}` memory definition is written
        verbatim to the Store File API and the agent keeps only the path."""
        agent = _name("imp-memory")
        content = "# Project memory\n\nContexte immuable.\n"
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml_(agent, memory=[{"path": "memories/imp-project.md", "content": content}]),
            agent,
        )
        assert resp.status_code == 201, resp.text

        stored = await client.get(
            "/api/v1/store/files/memories/imp-project.md", headers=users["a"]["headers"]
        )
        assert stored.status_code == 200, stored.text
        assert stored.json()["content"] == content

        got = (await get_agent(client, users["a"]["headers"], agent)).json()
        assert got["memory"] == ["memories/imp-project.md"]

    async def test_existing_skill_store_file_is_never_overwritten(self, client, users):
        """REQ-SKILL-001: importing a `{path, content}` skill whose file already
        exists is refused — the existing content is preserved."""
        original = "# Original skill\n"
        put = await client.put(
            "/api/v1/store/files/skills/imp-dupe-skill/SKILL.md",
            headers=users["a"]["headers"],
            json={"content": original},
        )
        assert put.status_code == 200, put.text

        agent = _name("imp-skilldupe")
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml_(agent, skills=[{"path": "skills/imp-dupe-skill/", "content": "# New content\n"}]),
            agent,
        )
        assert resp.status_code in VALIDATION_CODES, resp.text
        stored = await client.get(
            "/api/v1/store/files/skills/imp-dupe-skill/SKILL.md", headers=users["a"]["headers"]
        )
        assert stored.json()["content"] == original, "the existing file must remain unchanged"

    async def test_textual_skill_reference_to_unstored_path_fails_import(self, client, users):
        """REQ-SUBAGENT-003: a textual skill entry is a reference to an existing
        Store resource — an unknown path fails the import instead of creating
        an empty resource."""
        agent = _name("imp-skillref")
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml_(agent, skills=["skills/imp-missing-skill/"]),
            agent,
        )
        assert resp.status_code in UNAVAILABLE_CODES, resp.text
        assert (await get_agent(client, users["a"]["headers"], agent)).status_code == 404

    async def test_textual_memory_reference_to_unstored_path_fails_import(self, client, users):
        """REQ-SUBAGENT-003: same rule for textual memory references."""
        agent = _name("imp-memoryref")
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml_(agent, memory=["memories/imp-missing.md"]),
            agent,
        )
        assert resp.status_code in UNAVAILABLE_CODES, resp.text
        assert (await get_agent(client, users["a"]["headers"], agent)).status_code == 404


class TestInlineFormatValidation:
    async def test_skill_inline_object_with_unexpected_field_is_rejected(self, client, users):
        """REQ-SKILL-001: the inline skill object accepts EXACTLY ``path`` and
        ``content`` — extra fields are refused."""
        agent = _name("imp-skill-extra")
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml_(
                agent,
                skills=[{"path": "skills/imp-extra/", "content": "# ok\n", "name": "unexpected"}],
            ),
            agent,
        )
        assert resp.status_code in VALIDATION_CODES, resp.text


class TestImportedFilesRuntimeEffect:
    """C5 (review): the imported skill/memory is not only WRITTEN to the
    Store — the runner looks it up where it was written and the imported
    resource becomes part of the agent's effective namespace at build time."""

    async def test_imported_skill_is_copied_into_the_agent_namespace(
        self, client, users, stub_llm
    ):
        agent = _name("imp-skilluse")
        skill_path = "skills/imp-use-skill/"
        content = "# Runtime skill\n\nUsed verbatim.\n"
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml_(agent, skills=[{"path": skill_path, "content": content}]),
            agent,
        )
        assert resp.status_code == 201, resp.text

        await configure_llm_credentials(client, users["a"]["headers"])
        thread = await create_thread(client, users["a"]["headers"], agent)
        assert thread.status_code == 201, thread.text
        response = await invoke_agent(client, users["a"]["headers"], thread.json()["id"])
        assert response.status_code == 200, response.text

        stored = await client.get(
            f"/api/v1/store/files/agents/{agent}/skills/imp-use-skill/SKILL.md",
            headers=users["a"]["headers"],
        )
        assert stored.status_code == 200, (
            f"the runner must resolve the imported skill under its canonical store key: {stored.text}"
        )
        assert stored.json()["content"] == content

    async def test_imported_memory_is_copied_into_the_agent_namespace(
        self, client, users, stub_llm
    ):
        agent = _name("imp-memoryuse")
        content = "# Runtime memory\n\nContexte.\n"
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml_(agent, memory=[{"path": "memories/imp-run.md", "content": content}]),
            agent,
        )
        assert resp.status_code == 201, resp.text

        await configure_llm_credentials(client, users["a"]["headers"])
        thread = await create_thread(client, users["a"]["headers"], agent)
        assert thread.status_code == 201, thread.text
        response = await invoke_agent(client, users["a"]["headers"], thread.json()["id"])
        assert response.status_code == 200, response.text

        stored = await client.get(
            f"/api/v1/store/files/agents/{agent}/memories/imp-run.md",
            headers=users["a"]["headers"],
        )
        assert stored.status_code == 200, (
            f"the runner must resolve the imported memory under its canonical store key: {stored.text}"
        )
        assert stored.json()["content"] == content
