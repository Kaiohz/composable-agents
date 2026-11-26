"""Unit tests for the ``mcp_refs`` resolution flow (REQ-MCP-003).

Covers the factory's trusted registry-ref tool loading (with/witout a wired
resolver) and the referenced-subagent merge of the referenced agent's MCP
selection, plus the cron unattended-safety placeholder scan (agent YAML and
registry raw reveal sources).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.cron_unattended import find_user_placeholder_sources
from src.domain.entities.agent_config import AgentConfig, SubAgentConfig
from src.domain.entities.mcp_server_config import McpServerConfig
from src.domain.errors.config import ConfigError
from src.infrastructure.deepagent.factory import (
    _resolve_agent_tools,
    _resolve_referenced_subagent,
)


def _loader():
    loader = AsyncMock()
    loader.load_tools.return_value = [MagicMock(name="tool")]
    return loader


def _resolver():
    resolver = AsyncMock()
    resolver.return_value = [McpServerConfig(name="selected", transport="http", url="http://x/mcp")]
    return resolver


class TestAgentToolsMcpRefs:
    async def test_mcp_refs_are_resolved_through_the_registry(self):
        loader = _loader()
        fake_tool = MagicMock(name="mcp_ref_tool")
        loader.load_tools.return_value = [fake_tool]
        resolver = _resolver()
        config = AgentConfig(name="r-agent", mcp_refs=["selected"])
        tools = await _resolve_agent_tools(config, loader, None, resolver)
        assert tools == [fake_tool]
        resolver.assert_awaited_once_with(["selected"])

    async def test_mcp_refs_without_resolver_is_an_explicit_error(self):
        config = AgentConfig(name="r-agent", mcp_refs=["selected"])
        with pytest.raises(ConfigError) as exc:
            await _resolve_agent_tools(config, _loader(), None, None)
        assert "mcp_refs" in str(exc.value)

    async def test_legacy_inline_servers_still_load(self):
        loader = _loader()
        fake_tool = MagicMock(name="inline_tool")
        loader.load_tools.return_value = [fake_tool]
        config = AgentConfig(
            name="legacy",
            mcp_servers=[McpServerConfig(name="inline", transport="http", url="http://x/mcp")],
        )
        tools = await _resolve_agent_tools(config, loader, None, None)
        assert tools == [fake_tool]

    async def test_every_source_down_fails_with_the_registry_error(self):
        from src.domain.errors.mcp import McpRegistryUnavailableError

        loader = AsyncMock()
        loader.load_tools.side_effect = McpRegistryUnavailableError("registry down")
        resolver = _resolver()
        config = AgentConfig(name="r-agent", mcp_refs=["selected"])
        with pytest.raises(McpRegistryUnavailableError):
            await _resolve_agent_tools(config, loader, None, resolver)


class TestReferencedSubagentMcpRefs:
    async def test_referenced_agent_mcp_refs_are_merged(self):
        loader = _loader()
        resolver = _resolver()
        ref_config = AgentConfig(name="child", mcp_refs=["selected"])

        async def config_resolver(_name: str) -> AgentConfig:
            return ref_config

        spec = await _resolve_referenced_subagent(
            SubAgentConfig(name="sub", description="child", agent_ref="child"),
            loader,
            config_resolver,
            resolver,
        )
        assert spec["name"] == "sub"
        assert spec["system_prompt"] is None
        # The tools list is built through the loader's resolved server.
        loader.load_tools.assert_awaited_once()
        assert loader.load_tools.await_args.args[0][0].name == "selected"

    async def test_legacy_inline_servers_of_entry_and_reference_merge(self):
        loader = _loader()
        loader.load_tools.side_effect = [[MagicMock(name="legacy_tool")], [MagicMock(name="ref_tool")]]
        ref_config = AgentConfig(
            name="child",
            mcp_servers=[McpServerConfig(name="ref-inline", transport="http", url="http://r/mcp")],
        )

        async def config_resolver(_name: str) -> AgentConfig:
            return ref_config

        spec = await _resolve_referenced_subagent(
            SubAgentConfig(
                name="sub",
                description="child",
                agent_ref="child",
                mcp_servers=[McpServerConfig(name="sa-inline", transport="http", url="http://s/mcp")],
            ),
            loader,
            config_resolver,
            None,
        )
        assert loader.load_tools.await_count == 2
        called_servers = [loader.load_tools.await_args_list[i].args[0][0].name for i in (0, 1)]
        assert called_servers == ["sa-inline", "ref-inline"]
        assert len(spec["tools"]) == 2

    async def test_referenced_agent_with_subagents_warns_only(self):
        loader = _loader()

        async def config_resolver(_name: str) -> AgentConfig:
            referenced = AgentConfig(name="child", system_prompt="Referenced prompt.")
            return referenced.model_copy(
                update={"subagents": [SubAgentConfig(name="nested", description="nested", agent_ref="elsewhere")]}
            )

        with patch("src.infrastructure.deepagent.factory.logger.warning") as warning:
            spec = await _resolve_referenced_subagent(
                SubAgentConfig(name="sub", description="child", agent_ref="child"),
                loader,
                config_resolver,
            )
        assert warning.called
        assert spec["description"] == "child"


class TestCronUnattendedScan:
    async def test_yaml_placeholder_source(self):
        yaml_text = (
            "name: a\nmcp_servers:\n"
            "  - name: s\n"
            "    transport: http\n"
            "    url: http://x\n"
            "    env:\n"
            '      CREDENTIAL: "${USER_JWT}"\n'
        )
        sources = await find_user_placeholder_sources(yaml_text, None, "a")
        assert sources == ["agent YAML"]

    async def test_refs_placeholder_source_via_raw_reveal(self):
        class _Registry:
            async def reveal_server(self, name: str) -> dict:
                assert name == "cred"
                return {"name": "cred", "env": {"CREDENTIAL": "${USER_API_KEY}"}}

        yaml_text = "name: a\nmcp_refs:\n  - cred\n"
        sources = await find_user_placeholder_sources(yaml_text, _Registry(), "a")
        assert sources == ["MCP server 'cred'"]

    async def test_safe_agent_has_no_sources(self):
        yaml_text = "name: a\nmodel: m\nsystem_prompt: s\nmcp_refs:\n  - clean\n"

        class _Registry:
            async def reveal_server(self, name: str) -> dict:
                assert name == "clean"
                return {"name": "clean", "url": "http://x", "env": {}}

        assert await find_user_placeholder_sources(yaml_text, _Registry(), "a") == []

    async def test_no_refs_or_no_registry_means_yaml_only(self):
        assert await find_user_placeholder_sources("name: a\nmodel: m\n", None, "a") == []
        assert await find_user_placeholder_sources("name: a\nmcp_refs:\n  - x\n", None, "a") == []

    async def test_broken_yaml_never_scans_refs(self):
        assert await find_user_placeholder_sources("not: [valid", None, "a") == []
