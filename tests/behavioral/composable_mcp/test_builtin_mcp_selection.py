"""Behavioral tests — MCP selection for the built-in assistant
(spec §4.B REQ-BUILTIN-001..003 / §4.F REQ-MCP-003).

Entry points: the REAL ``PUT /api/v1/agents/{builtin}`` customization route
and the real one-shot invocation chain (``composable_agents_run`` via the
mounted MCP endpoint) with the external LLM boundary scripted.

Target contract:
* the built-in customization accepts an explicit MCP selection carried by
  the registry identity list (``mcp_refs``) and PERSISTS it — the backend no
  longer forces an empty selection nor rejects the reference mechanism;
* the selection is per-user: only the owning user sees it;
* the runner mounts ONLY the explicitly selected user MCP servers, in
  addition to the platform tools and integrated platform MCP — mounting the
  entire user registry automatically is forbidden (REQ-BUILTIN-003);
* a selection referencing an unregistered server is refused without
  persisting the customization.
"""

import pytest

from tests.behavioral.composable_mcp.conftest import (
    StubMcpServer,
    assistant_text,
    assistant_tool_call,
    call_mcp_tool,
)
from tests.behavioral.conftest import configure_llm_credentials

pytestmark = pytest.mark.asyncio(loop_scope="session")

BUILTIN_NAME = "composable-ai-assistant"
UNAVAILABLE_CODES = {400, 404, 422}


def builtin_yaml_with_refs(model: str = "personalized-model", refs: list[str] | None = None) -> str:
    lines = [
        f"name: {BUILTIN_NAME}",
        f"model: {model}",
        'description: "Personalized built-in assistant."',
        'system_prompt: "Personalized built-in prompt."',
        "tools: []",
        "debug: false",
    ]
    if refs is not None:
        lines.append("mcp_refs:")
        for ref in refs:
            lines.append(f"  - {ref}")
    return "\n".join(lines) + "\n"


async def put_builtin(client, headers: dict[str, str], yaml_text: str):
    return await client.put(
        f"/api/v1/agents/{BUILTIN_NAME}",
        headers=headers,
        files={"file": ("builtin.yaml", yaml_text.encode(), "text/yaml")},
    )


async def run_assistant(mcp_base_url, users) -> dict:
    return await call_mcp_tool(
        mcp_base_url,
        users["a"]["headers"],
        "composable_agents_run",
        {"agent_name": BUILTIN_NAME, "message": "do work"},
    )


class TestBuiltinSelectionPersistence:
    async def test_selection_is_accepted_and_persisted_across_reads(self, client, users, stub_registry):
        """REQ-BUILTIN-001/002: the MCP selector mechanism (``mcp_refs``) is
        accepted by the customization and the selection is preserved on every
        subsequent read (page reloads included)."""
        seed = {
            "alpha": {
                "name": "alpha",
                "transport": "http",
                "url": "http://127.0.0.1:59997/mcp",
                "headers": {},
                "env": {},
                "auth_token": None,
                "tool_count": 1,
                "source_type": "external",
                "openapi_url": None,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        }
        stub_registry.reset(seed=seed)

        resp = await put_builtin(client, users["a"]["headers"], builtin_yaml_with_refs(refs=["alpha"]))
        assert resp.status_code == 200, resp.text

        got = await client.get(f"/api/v1/agents/{BUILTIN_NAME}", headers=users["a"]["headers"])
        assert got.status_code == 200, got.text
        assert got.json()["mcp_refs"] == ["alpha"], (
            "the built-in customization must persist and expose the selected MCP servers"
        )
        assert got.json()["is_customized"] is True

    async def test_selection_is_per_user_and_platform_default_is_untouched(self, client, users, stub_registry):
        """REQ-BUILTIN-002/005: the selection is scoped to the owning user; a
        user without customization keeps the untouched platform definition."""
        seed = {
            "alpha": {
                "name": "alpha",
                "transport": "http",
                "url": "http://127.0.0.1:59997/mcp",
                "headers": {},
                "env": {},
                "auth_token": None,
                "tool_count": 1,
                "source_type": "external",
                "openapi_url": None,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        }
        stub_registry.reset(seed=seed)

        resp = await put_builtin(client, users["a"]["headers"], builtin_yaml_with_refs(refs=["alpha"]))
        assert resp.status_code == 200, resp.text

        other = await client.get(f"/api/v1/agents/{BUILTIN_NAME}", headers=users["b"]["headers"])
        assert other.status_code == 200
        assert other.json()["is_customized"] is False, "another user must not inherit the selection"
        assert not (other.json().get("mcp_refs") or [])
        assert other.json()["system_prompt"].startswith("You are the Composable AI Assistant"), (
            "the untouched platform default must not be affected by the selection"
        )

    async def test_selection_referencing_an_unregistered_server_is_refused(self, client, users, stub_registry):
        """REQ-BUILTIN-001: the selector presents registered servers only — a
        customization referencing an unknown server is refused and not persisted."""
        stub_registry.reset(seed={})

        resp = await put_builtin(client, users["a"]["headers"], builtin_yaml_with_refs(refs=["ghost-server"]))
        assert resp.status_code in UNAVAILABLE_CODES, resp.text

        got = await client.get(f"/api/v1/agents/{BUILTIN_NAME}", headers=users["a"]["headers"])
        assert got.json()["is_customized"] is False, "a refused customization must not be persisted"


class TestSelectiveMounting:
    async def test_runner_mounts_only_the_selected_user_mcp(self, mcp_base_url, client, users, stub_llm, stub_registry):
        """REQ-BUILTIN-003: the runner mounts the platform tools plus ONLY the
        user MCP servers explicitly selected — not the entire user registry."""
        selected = StubMcpServer("selected", tool_name="selected-ping", tool_result="pong-selected").start()
        unselected = StubMcpServer("unselected", tool_name="other-ping", tool_result="pong-unselected").start()
        try:
            seed = {
                "selected": {
                    "name": "selected",
                    "transport": "http",
                    "url": selected.url,
                    "headers": {},
                    "env": {},
                    "auth_token": None,
                    "tool_count": 1,
                    "source_type": "external",
                    "openapi_url": None,
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                },
                "unselected": {
                    "name": "unselected",
                    "transport": "http",
                    "url": unselected.url,
                    "headers": {},
                    "env": {},
                    "auth_token": None,
                    "tool_count": 1,
                    "source_type": "external",
                    "openapi_url": None,
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                },
            }
            stub_registry.reset(seed=seed)

            resp = await put_builtin(client, users["a"]["headers"], builtin_yaml_with_refs(refs=["selected"]))
            assert resp.status_code == 200, resp.text

            await configure_llm_credentials(client, users["a"]["headers"])

            # The selected server's tools ARE reachable.
            stub_llm.script(assistant_tool_call("selected-ping", {}), assistant_text("used selected"))
            ok_run = await run_assistant(mcp_base_url, users)
            assert ok_run["ok"], f"the selected MCP server must be mounted: {ok_run}"
            assert "pong-selected" in str(ok_run["result"]), ok_run

            # A registered but UNSELECTED server is NOT mounted.
            stub_llm.script(assistant_tool_call("other-ping", {}), assistant_text("used unselected"))
            noshow = await run_assistant(mcp_base_url, users)
            assert not noshow["ok"] or "pong-unselected" not in str(noshow["result"]), (
                f"an unselected user MCP server must not be mounted automatically: {noshow}"
            )

            # Platform tools keep working with the selective mounting.
            stub_llm.script(assistant_tool_call("composable_skills_list", {}), assistant_text("listed skills"))
            platform = await run_assistant(mcp_base_url, users)
            assert platform["ok"], f"platform tools must remain mounted: {platform}"
        finally:
            selected.stop()
            unselected.stop()


class TestResolveOnceAndDegradation:
    """C3 (review): the built-in resolves and mounts each selected server
    EXACTLY ONCE per build, and a selection that can no longer be resolved
    DEGRADES the build per REQ-040 instead of failing the whole runner."""

    async def test_selected_server_is_resolved_exactly_once_per_build(
        self, mcp_base_url, client, users, stub_llm, stub_registry
    ):
        """NFR: one registry resolution (and one tool load) per selected MCP
        per construction — the namespaced dynamic resolution is THE build
        path; the graph must not resolve ``mcp_refs`` a second time."""
        seed = {
            "alpha": {
                "name": "alpha",
                "transport": "http",
                "url": "http://127.0.0.1:59997/mcp",
                "headers": {},
                "env": {},
                "auth_token": None,
                "tool_count": 1,
                "source_type": "external",
                "openapi_url": None,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        }
        stub_registry.reset(seed=seed)

        resp = await put_builtin(client, users["a"]["headers"], builtin_yaml_with_refs(refs=["alpha"]))
        assert resp.status_code == 200, resp.text
        await configure_llm_credentials(client, users["a"]["headers"])

        stub_llm.script(assistant_text("single resolution"))
        result = await run_assistant(mcp_base_url, users)
        assert result["ok"], f"the built-in must answer with its selection mounted: {result}"

        reveals = [r for r in stub_registry.app.requests if "reveal" in r["path"]]
        assert len(reveals) == 1, (
            f"the selected server must be resolved exactly ONCE per build: {[r['path'] for r in reveals]}"
        )

    async def test_registry_outage_after_selection_degrades_the_build(
        self, mcp_base_url, client, users, stub_llm, stub_registry
    ):
        """REQ-040: a selected server that can no longer resolve degrades to
        capability information — the built-in still answers (never a 5xx)."""
        seed = {
            "alpha": {
                "name": "alpha",
                "transport": "http",
                "url": "http://127.0.0.1:59997/mcp",
                "headers": {},
                "env": {},
                "auth_token": None,
                "tool_count": 1,
                "source_type": "external",
                "openapi_url": None,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        }
        stub_registry.reset(seed=seed)

        resp = await put_builtin(client, users["a"]["headers"], builtin_yaml_with_refs(refs=["alpha"]))
        assert resp.status_code == 200, resp.text

        await configure_llm_credentials(client, users["a"]["headers"])
        # The registry becomes unreachable AFTER the selection was persisted.
        stub_registry.reset(seed={}, fail=True)
        stub_llm.script(assistant_text("answered under degradation"))
        result = await run_assistant(mcp_base_url, users)
        assert result["ok"], f"the built-in must still answer under a registry outage: {result}"
        degraded = result["result"].get("degraded") if isinstance(result["result"], dict) else None
        assert degraded, f"the outage must be surfaced as degraded capability information: {result}"
        assert any("alpha" in str(entry.get("name")) for entry in degraded), result
