"""Behavioral tests — outbound MCP ``X-Composable-Agent`` header injection
(spec REQ-016 / REQ-017 / REQ-019 header absence).

Entry point: the REAL ``composable_agents_run`` one-shot execution (real
agent CRUD → runner build → MCP tool loader → REAL streamable-HTTP MCP stub
server capturing every request header). The external MCP server is the ONLY
stub here (same boundary class as the chat-attachments files stub); the LLM
provider API is scripted via ``StubLlmServer``.

What is proven behaviorally:

* a top-level agent's EVERY outbound MCP connection (tool-load phase:
  ``initialize`` requests; tool-execution phase: ``tools/call`` requests)
  carries ``X-Composable-Agent: <agent_name>`` — injected at tool-load time,
  never as an LLM-supplied parameter (REQ-016);
* a subagent's OWN ``mcp_servers`` entries carry the SUBAGENT name instead
  (per-subagent space), including the servers merged from a referenced agent
  (REQ-016);
* the optional ``${AGENT_NAME}`` placeholder in agent YAML MCP header values
  resolves to the running agent's name (REQ-017 env_utils mechanism);
* a YAML-configured colliding header (case-insensitive) is overridden by the
  automatically injected header (REQ-017 ``_merge_headers`` semantics);
* the built-in assistant sends NO ``X-Composable-Agent`` header (REQ-019:
  personal scope) and triggers no spaces provisioning.

RED phase: nothing injects the header yet → the positive assertions fail for
the right reason (the stub observes no/other agent header). The built-in
absence test guards GREEN against over-implementation (it is expected to pass
today too) while the placeholder/collision positive assertions fail today.
"""

import uuid

import pytest

from tests.behavioral.agent_spaces.conftest import (
    assistant_text,
    assistant_tool_call,
    call_mcp_tool,
    create_agent_ok,
    header_mcp_yaml,
    one_shot_run,
    scoped_agent_yaml,
)
from tests.behavioral.conftest import configure_llm_credentials, poll_condition

pytestmark = pytest.mark.asyncio(loop_scope="session")

BUILTIN_NAME = "composable-ai-assistant"


async def _run_with_stub(mcp_base_url, client, users, stub_llm, yaml_text: str, name: str, script=None):
    """Create the agent (header tests reuse the stub URL), run it one-shot."""
    await create_agent_ok(client, users["a"]["headers"], yaml_text, name)
    await configure_llm_credentials(client, users["a"]["headers"])
    if script is not None:
        stub_llm.script(*script)
    return await one_shot_run(mcp_base_url, users["a"]["headers"], name)


async def _register_stub_server(mcp_base_url, users, name: str, url: str) -> None:
    """Register the stub MCP server for a user through the REAL registry tool."""
    created = await call_mcp_tool(
        mcp_base_url,
        users["a"]["headers"],
        "composable_mcp_servers_create",
        {"name": name, "transport": "http", "url": url, "source_type": "external"},
    )
    assert created["ok"], f"registering the stub MCP server failed: {created}"


class TestTopLevelAgentHeaderInjection:
    async def test_tool_load_connection_carries_agent_header(
        self, mcp_base_url, client, users, stub_llm, hdr_mcp, raganything_stub
    ):
        """REQ-016: the tool-LOAD phase of a top-level agent's outbound HTTP
        MCP connection carries ``X-Composable-Agent: <agent_name>``.

        The load happens before ANY LLM exchange, proving the header is
        injected at tool-load time — never an LLM-supplied parameter.
        """
        agent = f"hdr-load-{uuid.uuid4().hex[:8]}"
        result = await _run_with_stub(mcp_base_url, client, users, stub_llm, header_mcp_yaml(agent, hdr_mcp.url), agent)
        assert result["ok"], f"the one-shot run must succeed against the stub server: {result}"

        initializes = hdr_mcp.recorder.requests_with(json_method="initialize")
        assert initializes, "the agent must have connected to the external MCP server at tool-load time"
        carrying = hdr_mcp.recorder.requests_with(
            json_method="initialize", header="X-Composable-Agent", header_value=agent
        )
        assert carrying, (
            f"the tool-load connection must inject X-Composable-Agent: {agent!r}; observed agent headers: "
            f"{hdr_mcp.recorder.agent_header_values()!r}"
        )
        # Exactly the agent's name — no other agent scope may leak.
        assert hdr_mcp.recorder.agent_header_values() == {agent}, (
            f"unexpected agent header values observed: {hdr_mcp.recorder.agent_header_values()!r}"
        )

    async def test_tool_execution_connection_carries_agent_header(
        self, mcp_base_url, client, users, stub_llm, hdr_mcp, raganything_stub
    ):
        """REQ-016: the tool-CALL phase (a real tool execution over a fresh
        streamable-HTTP session) carries the same injected header."""
        agent = f"hdr-call-{uuid.uuid4().hex[:8]}"
        script = [
            assistant_tool_call("ping", {}),
            assistant_text("invoked the external tool"),
        ]
        result = await _run_with_stub(
            mcp_base_url, client, users, stub_llm, header_mcp_yaml(agent, hdr_mcp.url), agent, script
        )
        assert result["ok"], f"the one-shot run with a tool call must succeed: {result}"
        assert "pong" in str(result["result"]), (
            f"the agent must have executed the stub server's ping tool: {result['result']}"
        )

        calls = hdr_mcp.recorder.requests_with(json_method="tools/call")
        assert calls, "the scripted LLM tool call must reach the external MCP server"
        carrying = hdr_mcp.recorder.requests_with(
            json_method="tools/call", header="X-Composable-Agent", header_value=agent
        )
        assert carrying, (
            "the tool-execution connection must inject X-Composable-Agent: "
            f"{agent!r}; observed agent headers: {hdr_mcp.recorder.agent_header_values()!r}"
        )

    async def test_platform_tool_agent_injects_header_too(
        self, mcp_base_url, client, users, stub_llm, hdr_mcp, raganything_stub
    ):
        """REQ-016: the injection applies to EVERY top-level agent build —
        including platform-tool agents (``tools: ["mcp"]``)."""
        agent = f"hdr-plat-{uuid.uuid4().hex[:8]}"
        yaml_text = header_mcp_yaml(agent, hdr_mcp.url, tools='["mcp"]')
        result = await _run_with_stub(mcp_base_url, client, users, stub_llm, yaml_text, agent)
        assert result["ok"], f"the one-shot run must succeed: {result}"
        carrying = hdr_mcp.recorder.requests_with(
            json_method="initialize", header="X-Composable-Agent", header_value=agent
        )
        assert carrying, (
            f"the platform-tool agent build must inject its own agent header; observed: "
            f"{hdr_mcp.recorder.agent_header_values()!r}"
        )


class TestScopeSlugHeaderMapping:
    async def test_nonslug_agent_name_maps_to_slug_header(
        self, mcp_base_url, client, users, stub_llm, hdr_mcp, raganything_stub
    ):
        """Review C-1: an agent whose legal name is NOT slug-valid
        (``MyAgent-…``) still boots and its outbound MCP connections carry
        the canonical mapped SLUG as ``X-Composable-Agent`` — never the raw
        name (the receiver's REQ-002 charset would 422 it) — and the spaces
        provisioning POSTs the very same slug (REQ-010/C-1 coherence)."""
        agent = f"MyAgent-{uuid.uuid4().hex[:8]}"
        slug = agent.lower()  # deterministic mapping of this ASCII name
        result = await _run_with_stub(mcp_base_url, client, users, stub_llm, header_mcp_yaml(agent, hdr_mcp.url), agent)
        assert result["ok"], f"the one-shot run must succeed: {result}"

        carrying = hdr_mcp.recorder.requests_with(
            json_method="initialize", header="X-Composable-Agent", header_value=slug
        )
        assert carrying, (
            f"the mapped slug {slug!r} must be injected as the agent scope; observed: "
            f"{hdr_mcp.recorder.agent_header_values()!r}"
        )
        assert not hdr_mcp.recorder.requests_with(
            json_method="initialize", header="X-Composable-Agent", header_value=agent
        ), "the RAW non-slug name must never be forwarded as the scope header"

        async def _slug_provisioned():
            posts = [call["agent_slug"] for call in raganything_stub.app.space_calls() if call["method"] == "POST"]
            return slug in posts, posts

        ok, posts = await poll_condition(_slug_provisioned, timeout=15.0)
        assert ok, f"the spaces API must be provisioned with the SAME mapped slug (last observed: {posts!r})"


class TestSubagentHeaderInjection:
    async def test_subagent_connection_carries_subagent_header(
        self, mcp_base_url, client, users, stub_llm, hdr_mcp, raganything_stub
    ):
        """REQ-016: top-level agent and its subagent inject DIFFERENT header
        values on their respective outbound MCP connections (per-subagent
        space): the agent's own servers see the agent name, the subagent's
        own servers see the subagent name."""
        agent = f"hdr-top-{uuid.uuid4().hex[:8]}"
        sub = f"sub-{uuid.uuid4().hex[:8]}"
        yaml_text = scoped_agent_yaml(agent, hdr_mcp.url, subagent_name=sub, subagent_url=hdr_mcp.url)
        result = await _run_with_stub(mcp_base_url, client, users, stub_llm, yaml_text, agent)
        assert result["ok"], f"the one-shot run must succeed: {result}"

        agent_carrying = hdr_mcp.recorder.requests_with(
            json_method="initialize", header="X-Composable-Agent", header_value=agent
        )
        sub_carrying = hdr_mcp.recorder.requests_with(
            json_method="initialize", header="X-Composable-Agent", header_value=sub
        )
        assert agent_carrying, (
            f"the top-level agent's MCP connection must carry {agent!r}; observed: "
            f"{hdr_mcp.recorder.agent_header_values()!r}"
        )
        assert sub_carrying, (
            f"the subagent's MCP connection must carry {sub!r} (per-subagent space); observed: "
            f"{hdr_mcp.recorder.agent_header_values()!r}"
        )

    async def test_agent_ref_subagent_binds_subagent_space(
        self, mcp_base_url, client, users, stub_llm, hdr_mcp, raganything_stub
    ):
        """REQ-016 (per-subagent space): a subagent referencing another agent
        (``agent_ref``) loads the REFERENCED agent's servers under the
        SUBAGENT's space — every connection of that subagent carries the
        subagent name, not the referenced agent's name."""
        ref_agent = f"hdr-ref-{uuid.uuid4().hex[:8]}"
        sub = f"sub-{uuid.uuid4().hex[:8]}"
        top = f"hdr-owner-{uuid.uuid4().hex[:8]}"
        ref_headers = users["a"]["headers"]

        # The referenced agent owns the stub MCP server…
        await create_agent_ok(client, ref_headers, header_mcp_yaml(ref_agent, hdr_mcp.url), ref_agent)

        # …the top agent's subagent references it with NO own servers. The
        # subagent pins a provider-prefixed model so the deepagents runtime
        # resolves a real BaseChatModel (production model resolution is NOT
        # under test here — only the per-subagent space binding is).
        yaml_text = (
            f"name: {top}\n"
            "model: stub-model\n"
            'system_prompt: "Scoped agent-ref test agent."\n'
            "tools: []\n"
            "debug: false\n"
            "subagents:\n"
            f"  - name: {sub}\n"
            f'    description: "Reuses {ref_agent} as a scoped sub-agent."\n'
            f"    agent_ref: {ref_agent}\n"
        )
        result = await _run_with_stub(mcp_base_url, client, users, stub_llm, yaml_text, top)
        assert result["ok"], f"the one-shot run must succeed: {result}"

        sub_carrying = hdr_mcp.recorder.requests_with(
            json_method="initialize", header="X-Composable-Agent", header_value=sub
        )
        assert sub_carrying, (
            f"the agent_ref subagent's merged MCP connections must carry {sub!r}; observed: "
            f"{hdr_mcp.recorder.agent_header_values()!r}"
        )
        assert not hdr_mcp.recorder.requests_with(
            json_method="initialize", header="X-Composable-Agent", header_value=ref_agent
        ), "the referenced agent's name must NOT be used as the subagent's scope header"


class TestAgentNamePlaceholder:
    async def test_agent_name_placeholder_resolves_to_agent_name(
        self, mcp_base_url, client, users, stub_llm, hdr_mcp, monkeypatch, raganything_stub
    ):
        """REQ-017: an agent YAML MCP header value containing
        ``${AGENT_NAME}`` resolves to the running agent's name (env_utils
        mechanism, pattern of ``${USER_JWT}``)."""
        monkeypatch.delenv("AGENT_NAME", raising=False)
        agent = f"hdr-repl-{uuid.uuid4().hex[:8]}"
        yaml_text = header_mcp_yaml(agent, hdr_mcp.url, headers_block=["X-Agent-Tag: '${AGENT_NAME}'"])
        result = await _run_with_stub(mcp_base_url, client, users, stub_llm, yaml_text, agent)
        assert result["ok"], f"the one-shot run must succeed: {result}"

        resolved = hdr_mcp.recorder.requests_with(json_method="initialize", header="X-Agent-Tag", header_value=agent)
        assert resolved, (
            "'${AGENT_NAME}' in the configured header must resolve to the agent's real name; observed "
            f"X-Agent-Tag values: "
            f"{sorted({r['headers'].get('x-agent-tag') for r in hdr_mcp.recorder.snapshot()})!r}"
        )

    async def test_agent_name_placeholder_wins_over_os_environ(
        self, mcp_base_url, client, users, stub_llm, hdr_mcp, monkeypatch, raganything_stub
    ):
        """Review I-6(c) / iter-5 I-9: ``${AGENT_NAME}`` resolves to the
        running agent's slug EVEN WHEN an ``AGENT_NAME`` variable is present
        in the environment — the extra-reserved precedence is provable, not
        assumed (a regression consulting ``os.environ`` first would silently
        change the agent's scope)."""
        monkeypatch.setenv("AGENT_NAME", "wrong-slug")
        agent = f"hdr-env-{uuid.uuid4().hex[:8]}"
        yaml_text = header_mcp_yaml(agent, hdr_mcp.url, headers_block=["X-Agent-Tag: '${AGENT_NAME}'"])
        result = await _run_with_stub(mcp_base_url, client, users, stub_llm, yaml_text, agent)
        assert result["ok"], f"the one-shot run must succeed: {result}"

        resolved = hdr_mcp.recorder.requests_with(json_method="initialize", header="X-Agent-Tag", header_value=agent)
        assert resolved, (
            "'${AGENT_NAME}' must resolve to the agent's own slug even with AGENT_NAME set in the "
            "environment; observed X-Agent-Tag values: "
            f"{sorted({r['headers'].get('x-agent-tag') for r in hdr_mcp.recorder.snapshot()})!r}"
        )
        wrong = hdr_mcp.recorder.requests_with(
            json_method="initialize", header="X-Agent-Tag", header_value="wrong-slug"
        )
        assert not wrong, "the environment value must never win the reserved-placeholder precedence"


class TestInjectedHeaderWinsCollision:
    async def test_yaml_configured_collision_is_overridden(
        self, mcp_base_url, client, users, stub_llm, hdr_mcp, raganything_stub
    ):
        """REQ-017: a YAML-configured ``x-composable-agent`` (lowercase name,
        same header case-insensitively) is overridden by the automatically
        injected header — the injected value WINS (``_merge_headers``
        semantics)."""
        agent = f"hdr-coll-{uuid.uuid4().hex[:8]}"
        yaml_text = header_mcp_yaml(agent, hdr_mcp.url, headers_block=["x-composable-agent: 'spoofed-by-yaml'"])
        result = await _run_with_stub(mcp_base_url, client, users, stub_llm, yaml_text, agent)
        assert result["ok"], f"the one-shot run must succeed: {result}"

        carrying = hdr_mcp.recorder.requests_with(
            json_method="initialize", header="X-Composable-Agent", header_value=agent
        )
        assert carrying, (
            "the automatically injected header must WIN over the YAML-configured collision; "
            f"observed agent headers: {hdr_mcp.recorder.agent_header_values()!r}"
        )
        spoofed = hdr_mcp.recorder.requests_with(
            json_method="initialize", header="x-composable-agent", header_value="spoofed-by-yaml"
        )
        assert not spoofed, "the spoofed YAML value must never reach the external server"


class TestBuiltinAssistantPersonalScope:
    async def test_builtin_assistant_sends_no_agent_header(
        self, mcp_base_url, client, users, stub_llm, raganything_stub, hdr_mcp
    ):
        """REQ-019: the built-in assistant runs under the PERSONAL scope —
        its outbound MCP connections carry NO ``X-Composable-Agent`` header
        and no spaces provisioning is ever triggered for it.

        Rebase semantics (main PR #46, editable built-in): user-registered
        MCP servers are mounted for the built-in ONLY when the per-user
        customization SELECTS them (``mcp_refs``) — the test selects the
        registered server in the customization before the one-shot."""
        await configure_llm_credentials(client, users["a"]["headers"])
        await _register_stub_server(mcp_base_url, users, "hdr-builtin-stub", hdr_mcp.url)

        # PR #46: SELECT the registered server in the built-in customization.
        customization = (
            "name: composable-ai-assistant\n"
            'description: "Personalized built-in assistant."\n'
            "tools: []\n"
            "debug: false\n"
            "mcp_refs:\n"
            '  - "hdr-builtin-stub"\n'
        )
        customized = await client.put(
            f"/api/v1/agents/{BUILTIN_NAME}",
            headers=users["a"]["headers"],
            files={"file": ("builtin.yaml", customization.encode(), "text/yaml")},
        )
        assert customized.status_code == 200, (
            f"the built-in customization with the mcp_refs selection must be accepted: {customized.text}"
        )

        stub_llm.script(
            assistant_tool_call("ping", {}),
            assistant_text("used the registered server"),
        )
        result = await one_shot_run(mcp_base_url, users["a"]["headers"], BUILTIN_NAME)
        assert result["ok"], f"the builtin assistant run must succeed: {result}"
        assert "pong" in str(result["result"]), (
            f"the builtin must resolve and execute the SELECTED user-registered server's tool: {result['result']}"
        )

        assert hdr_mcp.recorder.requests_with(header="x-composable-agent") == [], (
            "the built-in assistant must NOT inject a custom agent header (personal scope); observed: "
            f"{hdr_mcp.recorder.agent_header_values()!r}"
        )
        assert raganything_stub.app.space_calls() == [], (
            "the built-in assistant run must never trigger RAG space provisioning"
        )
