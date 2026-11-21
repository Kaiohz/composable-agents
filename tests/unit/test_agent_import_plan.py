"""Unit tests for the agent import plan builder (pure prevalidation).

Covers the spec "agent-resource-import" prevalidation rules (§4.C..§4.G):
explicit-only declarations (no Pydantic default leaks), model/prompt
materialization descriptors, MCP fidelity + namespace uniqueness, sub-agent
cycles, ``path + content`` skill/memory rules, and the resource-only target
skeleton.
"""

import pytest

from src.domain.errors.config import ConfigError, ConfigValidationError
from src.domain.services.agent_import_plan import (
    build_agent_import_plan,
    derive_model_provider,
    derived_resource_names,
)


def plan_of(**overrides) -> dict:
    """A minimal VALID import mapping; tests override single fields."""
    data = {"name": "my-agent", "model": "anthropic:claude-x", "system_prompt": "Prompt body.", "tools": [], "debug": False}
    data.update(overrides)
    return {key: value for key, value in data.items() if value is not None}


class TestModelDeclaration:
    async def test_inline_model_materializes_with_derived_provider(self):
        plan = build_agent_import_plan(plan_of(), "my-agent")
        assert plan.model is not None
        assert plan.model.resource_name == "my-agent-model"
        assert plan.model.provider_model == "anthropic:claude-x"
        assert plan.model.provider == "anthropic"

    async def test_model_without_prefix_defaults_to_openai(self):
        plan = build_agent_import_plan(plan_of(model="gpt-4o-mini"), "my-agent")
        assert plan.model.provider == "openai"
        assert plan.model.provider_model == "gpt-4o-mini"

    async def test_derive_provider_edge_cases(self):
        assert derive_model_provider(":degenerate") == "openai"
        assert derive_model_provider("plain") == "openai"
        assert derive_model_provider("stub:model-x") == "stub"

    async def test_missing_model_and_model_ref_is_rejected(self):
        data = plan_of()
        del data["model"]
        with pytest.raises(ConfigError) as exc:
            build_agent_import_plan(data, "my-agent")
        assert "model" in str(exc.value)

    async def test_non_string_model_is_rejected(self):
        with pytest.raises(ConfigError):
            build_agent_import_plan(plan_of(model=42), "my-agent")

    async def test_model_ref_takes_precedence_and_is_kept_for_validation(self):
        plan = build_agent_import_plan(plan_of(model="legacy-string", model_ref="abc"), "my-agent")
        assert plan.model is None
        assert plan.model_ref == "abc"
        assert "model" not in plan.target


class TestPromptDeclaration:
    async def test_inline_prompt_materializes_verbatim(self):
        plan = build_agent_import_plan(plan_of(system_prompt="Line one.\nLine two.\n"), "my-agent")
        assert plan.prompt is not None
        assert plan.prompt.resource_name == "my-agent-prompt"
        assert plan.prompt.content == "Line one.\nLine two.\n"

    async def test_explicitly_empty_prompt_is_valid(self):
        plan = build_agent_import_plan(plan_of(system_prompt=""), "my-agent")
        assert plan.prompt.content == ""

    async def test_prompt_ref_short_circuits_materialization(self):
        data = {
            "name": "my-agent",
            "model": "gpt-4o",
            "tools": [],
            "prompt_ref": {"id": "3f4c66b4-1c6f-4c05-89f2-cdc6f3277c07"},
        }
        plan = build_agent_import_plan(data, "my-agent")
        assert plan.prompt is None
        assert plan.prompt_ref is not None
        assert str(plan.prompt_ref.id) == "3f4c66b4-1c6f-4c05-89f2-cdc6f3277c07"
        assert plan.prompt_ref.version is None

    async def test_non_uuid_prompt_ref_is_rejected(self):
        data = {"name": "my-agent", "model": "gpt-4o", "tools": [], "prompt_ref": {"id": "not-a-uuid"}}
        with pytest.raises(ConfigValidationError):
            build_agent_import_plan(data, "my-agent")

    async def test_missing_prompt_is_rejected(self):
        data = {"name": "my-agent", "model": "gpt-4o", "tools": [], "debug": False}
        with pytest.raises(ConfigError) as exc:
            build_agent_import_plan(data, "my-agent")
        assert "prompt" in str(exc.value)

    async def test_prompt_ref_and_inline_prompt_are_exclusive(self):
        data = plan_of(prompt_ref={"id": "3f4c66b4-1c6f-4c05-89f2-cdc6f3277c07"})
        with pytest.raises(ConfigError) as exc:
            build_agent_import_plan(data, "my-agent")
        assert "mutually exclusive" in str(exc.value)

    async def test_system_prompt_file_is_refused_on_import(self):
        with pytest.raises(ConfigError) as exc:
            build_agent_import_plan(plan_of(system_prompt="x", **{"system_prompt_file": "p.txt"}), "my-agent")
        assert "system_prompt_file" in str(exc.value)


class TestMcpDeclarations:
    async def test_inline_servers_fidelity(self):
        raw = plan_of(
            mcp_servers=[
                {
                    "name": "search",
                    "transport": "http",
                    "url": "http://127.0.0.1:8443/mcp",
                    "headers": {"X-Token": "sekrit"},
                    "env": {"KEY": "value"},
                    "auth_token": "tok",
                },
                {"name": "local", "transport": "stdio", "command": "uvx", "args": ["srv"]},
            ]
        )
        plan = build_agent_import_plan(raw, "my-agent")
        assert [m.writer.name for m in plan.mcp_creations] == ["search", "local"]
        search = plan.mcp_creations[0].writer
        assert search.url == "http://127.0.0.1:8443/mcp"
        assert search.headers == {"X-Token": "sekrit"}
        assert search.env == {"KEY": "value"}
        assert search.auth_token == "tok"
        assert plan.target.get("mcp_servers") is None

    async def test_duplicate_names_refused(self):
        raw = plan_of(mcp_servers=[{"name": "search", "transport": "http", "url": "http://a/mcp"}] * 1)
        raw["mcp_servers"].append({"name": "search", "transport": "http", "url": "http://b/mcp"})
        with pytest.raises(ConfigError) as exc:
            build_agent_import_plan(raw, "my-agent")
        assert "mcp_servers" in str(exc.value)
        assert "search" in str(exc.value)

    async def test_ref_and_inline_name_conflict_refused(self):
        raw = plan_of(
            mcp_servers=[{"name": "search", "transport": "http", "url": "http://a/mcp"}],
            mcp_refs=["search"],
        )
        with pytest.raises(ConfigError):
            build_agent_import_plan(raw, "my-agent")

    async def test_invalid_entry_shape_refused(self):
        with pytest.raises(ConfigError):
            build_agent_import_plan(plan_of(mcp_servers=["not-a-mapping"]), "my-agent")
        with pytest.raises(ConfigError):
            build_agent_import_plan(
                plan_of(mcp_servers=[{"name": "x", "transport": "http", "url": "http://a", "cwd": "/tmp"}]),
                "my-agent",
            )
        with pytest.raises(ConfigError):
            build_agent_import_plan(
                plan_of(mcp_servers=[{"name": "x", "transport": "carrier-pigeon"}]),
                "my-agent",
            )


class TestSubagentDeclarations:
    async def test_inline_subagent_materialization_descriptors(self):
        raw = plan_of(
            subagents=[
                {
                    "name": "helper",
                    "description": "Helper",
                    "instructions": "Helper prompt.",
                    "model": "gpt-helper",
                    "tools": [],
                }
            ]
        )
        plan = build_agent_import_plan(raw, "my-agent")
        assert len(plan.subagent_creations) == 1
        creation = plan.subagent_creations[0]
        assert creation.model.provider_model == "gpt-helper"
        assert creation.prompt.content == "Helper prompt."
        position = plan.subagent_entry_index["helper"]
        assert plan.target["subagents"][position] == {"name": "helper", "description": "Helper"}

    async def test_subagent_without_required_fields_refused(self):
        base = {
            "name": "helper",
            "description": "Helper",
            "instructions": "i",
            "model": "m",
            "tools": [],
        }
        for missing in ("description", "instructions", "model", "name"):
            incomplete = {key: value for key, value in base.items() if key != missing}
            with pytest.raises(ConfigError):
                build_agent_import_plan(plan_of(subagents=[incomplete]), "my-agent")

    async def test_subagent_self_reference_refused(self):
        with pytest.raises(ConfigError) as exc:
            build_agent_import_plan(
                plan_of(subagents=[{"name": "me", "description": "d", "agent_ref": "my-agent"}]),
                "my-agent",
            )
        assert "self-reference" in str(exc.value)

    async def test_cyclic_inline_refs_refused(self):
        with pytest.raises(ConfigError) as exc:
            build_agent_import_plan(
                plan_of(
                    subagents=[
                        {"name": "a", "description": "A", "agent_ref": "b"},
                        {"name": "b", "description": "B", "agent_ref": "a"},
                    ]
                ),
                "my-agent",
            )
        assert "cycle" in str(exc.value).lower()
        assert "a -> b -> a" in str(exc.value)

    async def test_self_loop_subagent_ref_refused(self):
        with pytest.raises(ConfigError) as exc:
            build_agent_import_plan(
                plan_of(subagents=[{"name": "a", "description": "A", "agent_ref": "a"}]),
                "my-agent",
            )
        assert "cycle" in str(exc.value).lower()

    async def test_duplicate_subagent_names_refused(self):
        entry = {"name": "helper", "description": "d", "agent_ref": "elsewhere"}
        with pytest.raises(ConfigError):
            build_agent_import_plan(plan_of(subagents=[entry, dict(entry)]), "my-agent")

    async def test_subagent_unknown_field_refused(self):
        with pytest.raises(ConfigError):
            build_agent_import_plan(
                plan_of(
                    subagents=[
                        {
                            "name": "helper",
                            "description": "d",
                            "instructions": "i",
                            "model": "m",
                            "nickname": "nope",
                        }
                    ]
                ),
                "my-agent",
            )

    async def test_subagent_inline_skill_object_refused(self):
        with pytest.raises(ConfigError):
            build_agent_import_plan(
                plan_of(
                    subagents=[
                        {
                            "name": "helper",
                            "description": "d",
                            "instructions": "i",
                            "model": "m",
                            "skills": [{"path": "skills/x/", "content": "c"}],
                        }
                    ]
                ),
                "my-agent",
            )

    async def test_subagent_mcp_duplicates_parent_name(self):
        raw = plan_of(
            mcp_servers=[{"name": "shared", "transport": "http", "url": "http://a/mcp"}],
            subagents=[
                {
                    "name": "helper",
                    "description": "d",
                    "instructions": "i",
                    "model": "m",
                    "mcp_servers": [{"name": "shared", "transport": "http", "url": "http://b/mcp"}],
                }
            ],
        )
        with pytest.raises(ConfigError):
            build_agent_import_plan(raw, "my-agent")


class TestSkillsAndMemories:
    async def test_textual_skill_reference(self):
        plan = build_agent_import_plan(plan_of(skills=["skills/docs/"]), "my-agent")
        assert plan.skill_refs == ["skills/docs/"]
        assert plan.files == []
        assert plan.target["skills"] == ["skills/docs/"]

    async def test_inline_skill_definition(self):
        plan = build_agent_import_plan(
            plan_of(skills=[{"path": "skills/my-dir/", "content": "# My skill\n"}]),
            "my-agent",
        )
        assert len(plan.files) == 1
        file_definition = plan.files[0]
        assert file_definition.file_path == "/skills/my-dir/SKILL.md"
        assert plan.target["skills"] == ["skills/my-dir/"]

    async def test_skill_object_with_unexpected_field_refused(self):
        with pytest.raises(ConfigError):
            build_agent_import_plan(
                plan_of(skills=[{"path": "skills/x/", "content": "c", "name": "bad"}]),
                "my-agent",
            )

    async def test_skill_path_shape_refused(self):
        with pytest.raises(ConfigError):
            build_agent_import_plan(plan_of(skills=[{"path": "wrong/x/", "content": "c"}]), "my-agent")
        with pytest.raises(ConfigError):
            build_agent_import_plan(plan_of(skills=[{"path": "skills/x", "content": "c"}]), "my-agent")
        with pytest.raises(ConfigError):
            build_agent_import_plan(plan_of(skills=["skills/no-trailing"]), "my-agent")

    async def test_textual_memory_reference(self):
        plan = build_agent_import_plan(plan_of(memory=["memories/p.md"]), "my-agent")
        assert plan.memory_refs == ["memories/p.md"]
        assert plan.target["memory"] == ["memories/p.md"]

    async def test_inline_memory_definition(self):
        plan = build_agent_import_plan(
            plan_of(memory=[{"path": "memories/p.md", "content": "# M\n"}]),
            "my-agent",
        )
        assert plan.files[0].file_path == "/memories/p.md"
        assert plan.target["memory"] == ["memories/p.md"]

    async def test_memory_path_shape_refused(self):
        with pytest.raises(ConfigError):
            build_agent_import_plan(plan_of(memory=[{"path": "memories/dir/", "content": "c"}]), "my-agent")
        with pytest.raises(ConfigError):
            build_agent_import_plan(plan_of(memory=["../escape"]), "my-agent")
        with pytest.raises(ConfigError):
            build_agent_import_plan(plan_of(memory=[{"path": "skills/x/", "content": "c"}]), "my-agent")

    async def test_memory_object_with_unexpected_field_refused(self):
        with pytest.raises(ConfigError):
            build_agent_import_plan(plan_of(memory=[{"path": "memories/p.md", "content": "c", "x": 1}]), "my-agent")


class TestTargetAndIdentity:
    async def test_inline_definitions_removed_from_target(self):
        plan = build_agent_import_plan(plan_of(), "my-agent")
        assert "model" not in plan.target
        assert "system_prompt" not in plan.target
        assert plan.target["name"] == "my-agent"
        assert plan.target["skills"] == []
        assert plan.target["memory"] == []

    async def test_deprecated_fields_stripped(self):
        plan = build_agent_import_plan(plan_of(middleware=[], is_builtin=True), "my-agent")
        assert "middleware" not in plan.target
        assert "is_builtin" not in plan.target

    async def test_yaml_name_mismatch_refused(self):
        with pytest.raises(ConfigError):
            build_agent_import_plan(plan_of(), "other-name")


class TestWritePolicy:
    """resource-only update boundary (domain/services/agent_resource_write_policy)."""

    async def test_new_inline_system_prompt_refused_on_reference_agent(self):
        from datetime import UTC, datetime

        from src.domain.entities.agent_config_metadata import AgentConfigMetadata
        from src.domain.services.agent_resource_write_policy import ensure_resource_only_update

        metadata = AgentConfigMetadata(
            name="a",
            model="m",
            minio_path="a.yaml",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            prompt_ref="p1",
        )
        with pytest.raises(ConfigError) as exc:
            ensure_resource_only_update({"system_prompt": "new"}, metadata)
        assert "system_prompt" in str(exc.value)

    async def test_new_inline_model_refused_on_reference_agent(self):
        from datetime import UTC, datetime

        from src.domain.entities.agent_config_metadata import AgentConfigMetadata
        from src.domain.services.agent_resource_write_policy import ensure_resource_only_update

        metadata = AgentConfigMetadata(
            name="a",
            model="m",
            minio_path="a.yaml",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            model_ref="m1",
        )
        with pytest.raises(ConfigError):
            ensure_resource_only_update({"model": "brand-new"}, metadata)
        # BUG-003: the inline value is refused EVEN next to an explicit
        # model_ref — the update persists the raw YAML verbatim, so the
        # "redundant" inline value would stay in double with the reference
        # (REQ-IMPORT-003) and override the display projection while the
        # runtime keeps the referenced model.
        with pytest.raises(ConfigError):
            ensure_resource_only_update({"model": "legacy", "model_ref": "other"}, metadata)

    async def test_nested_inline_mcp_servers_in_subagents_refused(self):
        """BUG-002: a nested `subagents[i].mcp_servers` (with its secrets) is
        refused at ANY depth — metadata-INDEPENDENT."""
        from datetime import UTC, datetime

        from src.domain.entities.agent_config_metadata import AgentConfigMetadata
        from src.domain.services.agent_resource_write_policy import ensure_resource_only_update

        inline = {"name": "srv", "transport": "http", "url": "http://x", "auth_token": "sekrit"}
        metadata = AgentConfigMetadata(
            name="a",
            model="m",
            minio_path="a.yaml",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            model_ref="m1",
            prompt_ref="p1",
        )
        with pytest.raises(ConfigError) as exc:
            ensure_resource_only_update(
                {
                    "subagents": [
                        {
                            "name": "c",
                            "description": "d",
                            "instructions": "i",
                            "model": "m",
                            "mcp_servers": [inline],
                        }
                    ]
                },
                metadata,
            )
        assert "subagents[0].mcp_servers" in str(exc.value)

        # An empty nested selection carries no definition: allowed.
        ensure_resource_only_update({"subagents": [{"name": "c", "description": "d", "mcp_servers": []}]}, metadata)

        # The nested rule is metadata-INDEPENDENT: secrets never pass the
        # boundary, even for a legacy agent without references.
        legacy = AgentConfigMetadata(
            name="a",
            model="m",
            minio_path="a.yaml",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        with pytest.raises(ConfigError):
            ensure_resource_only_update(
                {"subagents": [{"name": "c", "description": "d", "mcp_servers": [inline]}]},
                legacy,
            )

    async def test_inline_mcp_servers_refused_on_update(self):
        from datetime import UTC, datetime

        from src.domain.entities.agent_config_metadata import AgentConfigMetadata
        from src.domain.services.agent_resource_write_policy import ensure_resource_only_update

        metadata = AgentConfigMetadata(
            name="a",
            model="m",
            minio_path="a.yaml",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        server = {"name": "s", "transport": "http", "url": "http://x", "auth_token": "sekrit"}
        with pytest.raises(ConfigError):
            ensure_resource_only_update({"mcp_servers": [server]}, metadata)
        # Empty selection: no materializable definition.
        ensure_resource_only_update({"mcp_servers": []}, metadata)


class TestSubagentMcpOwnership:
    """C1/C2 (review): ownership of MCP selections in the plan + the target."""

    async def test_subagent_owned_server_is_scoped_to_the_subagent(self):
        """The plan carries the owner per declaration and the resource-only
        TARGET already scopes the sub-agent's server to its own entry — the
        root target must never carry a sub-agent-owned identity."""
        raw = plan_of(
            mcp_servers=[{"name": "root-srv", "transport": "http", "url": "http://a/mcp"}],
            subagents=[
                {
                    "name": "helper",
                    "description": "Child",
                    "instructions": "i",
                    "model": "m",
                    "tools": [],
                    "mcp_servers": [{"name": "child-srv", "transport": "http", "url": "http://b/mcp"}],
                }
            ],
        )
        plan = build_agent_import_plan(raw, "my-agent")
        assert [(m.writer.name, m.owner_agent) for m in plan.mcp_creations] == [
            ("root-srv", "my-agent"),
            ("child-srv", "helper"),
        ]
        # Root: no mcp_refs key until the ROOT-owned identities are injected.
        assert "mcp_refs" not in plan.target
        assert plan.target["subagents"] == [{"name": "helper", "description": "Child"}]

    async def test_derived_resource_names_cover_root_and_subagents(self):
        raw = plan_of(
            subagents=[
                {"name": "helper", "description": "Child", "instructions": "i", "model": "m", "tools": []}
            ]
        )
        plan = build_agent_import_plan(raw, "my-agent")
        assert set(derived_resource_names(plan)) == {
            ("my-agent-model", "model"),
            ("my-agent-prompt", "prompt"),
            ("helper-model", "model"),
            ("helper-prompt", "prompt"),
        }
        # A reference-only import creates nothing derived.
        refs_only = {
            "name": "my-agent",
            "model_ref": "0f4c66b4-1c6f-4c05-89f2-cdc6f3277c07",
            "prompt_ref": {"id": "3f4c66b4-1c6f-4c05-89f2-cdc6f3277c07"},
            "tools": [],
        }
        assert derived_resource_names(build_agent_import_plan(refs_only, "my-agent")) == []

    async def test_duplicate_subagent_name_names_the_right_resource(self):
        """I2: the diagnostic identifies the SUB-AGENT resource type."""
        entry = {"name": "helper", "description": "d", "agent_ref": "elsewhere"}
        with pytest.raises(ConfigError) as exc:
            build_agent_import_plan(plan_of(subagents=[dict(entry), dict(entry)]), "my-agent")
        assert "subagents" in str(exc.value)
        assert "mcp_servers" not in str(exc.value)

    async def test_reference_entry_inline_mcp_servers_are_refused(self):
        """C2: an inline MCP definition behind a REFERENCE entry is refused
        before any write (it would persist unregistered connection data with
        its secrets on an agent that does not own it)."""
        with pytest.raises(ConfigError) as exc:
            build_agent_import_plan(
                plan_of(
                    subagents=[
                        {
                            "name": "refd",
                            "description": "d",
                            "agent_ref": "other-agent",
                            "mcp_servers": [
                                {"name": "ref-srv", "transport": "http", "url": "http://c/mcp", "auth_token": "t"}
                            ],
                        }
                    ]
                ),
                "my-agent",
            )
        assert "mcp_servers" in str(exc.value)

    async def test_reference_entry_inline_body_is_refused(self):
        """C2: model/instructions/skills are never persisted on a reference
        entry — the referenced agent owns its own resources."""
        for forbidden in (
            {"model": "legacy-string"},
            {"instructions": "override"},
            {"skills": ["skills/docs/"]},
        ):
            entry = {"name": "refd", "description": "d", "agent_ref": "other-agent", **forbidden}
            with pytest.raises(ConfigError):
                build_agent_import_plan(plan_of(subagents=[entry]), "my-agent")

    async def test_reference_entry_target_is_the_reference_only(self):
        plan = build_agent_import_plan(
            plan_of(subagents=[{"name": "refd", "description": "d2", "agent_ref": "other-agent"}]),
            "my-agent",
        )
        assert plan.target["subagents"] == [{"name": "refd", "description": "d2", "agent_ref": "other-agent"}]
