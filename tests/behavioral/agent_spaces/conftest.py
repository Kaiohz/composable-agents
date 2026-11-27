"""Feature fixtures — per-user × per-agent RAG/memories scoping suites.

Reuses the suite-wide ``stack`` / ``client`` / ``users`` / ``stub_llm`` /
``app`` fixtures from ``tests/behavioral/conftest.py`` (real app + testcontainers
PostgreSQL/MinIO + real auth chain) and adds the two EXTERNAL boundaries this
feature touches:

* ``hdr_mcp``   — a REAL streamable-HTTP MCP server stub recording every
  request header (``header_mcp_stub.py``): stands in for an external MCP
  server an agent connects to, so the injected ``X-Composable-Agent`` header
  (REQ-016/REQ-017) is observed where mcp-raganything would observe it.
* ``raganything_stub`` — a stub of the EXTERNAL mcp-raganything REST boundary
  (``raganything_stub.py``): the idempotent spaces API (REQ-010 contract,
  ``POST /api/v1/spaces`` / ``DELETE /api/v1/spaces/{agent_slug}``) plus the
  minimal masked MCP-server registry + reveal endpoints needed by the
  built-in assistant's dynamic tool resolution. Its base URL is published to
  the SAME configuration seam the composable-mcp suite uses
  (``MCP_REGISTRY_BASE_URL`` env var AND ``Settings`` attributes, best-effort)
  — a Green-phase provisioning adapter just needs to read one of them at CALL
  time (exactly like ``HttpMcpRegistryClient.base_url_provider``).
* ``mcp_base_url`` — the REAL app served over a real local socket so MCP
  tests speak the real streamable-HTTP transport (no ASGI-transport shortcut).
* MCP client helpers (``open_mcp_session`` / ``call_mcp_tool`` etc., copied
  verbatim from the composable-mcp suite) and YAML builders for scoped,
  subagent-carrying and platform-tool agents.
"""

import asyncio
import contextlib
import json
import os
import threading
import time
from collections.abc import AsyncGenerator
from typing import Any

import pytest
import pytest_asyncio

from tests.behavioral.agent_spaces.header_mcp_stub import HeaderRecordingMcpStubServer
from tests.behavioral.agent_spaces.raganything_stub import RagAnythingStubServer
from tests.behavioral.conftest import create_agent_yaml

# ---------------------------------------------------------------------------
# Header-recording MCP stub (external outbound MCP boundary)
# ---------------------------------------------------------------------------
_hdr_stub: HeaderRecordingMcpStubServer | None = None
_hdr_lock = threading.Lock()


def get_hdr_stub() -> HeaderRecordingMcpStubServer:
    """Process-wide recording MCP stub, started on first use (LLM-stub pattern)."""
    global _hdr_stub
    with _hdr_lock:
        if _hdr_stub is None:
            _hdr_stub = HeaderRecordingMcpStubServer().start()
    return _hdr_stub


@pytest.fixture
def hdr_mcp() -> HeaderRecordingMcpStubServer:
    """Per-test view over the recording MCP stub, with a clean recording state."""
    stub = get_hdr_stub()
    stub.reset()
    yield stub
    stub.reset()


# ---------------------------------------------------------------------------
# mcp-raganything REST boundary stub (spaces + minimal registry)
# ---------------------------------------------------------------------------
# Configuration seam — SAME contract as the composable-mcp suite: the base URL
# is published under this env var AND best-effort onto Settings attributes; a
# Green-phase implementation reads one of them at call time.
MCP_REGISTRY_BASE_URL_ENV = "MCP_REGISTRY_BASE_URL"

_RAG_SEAM_SETTINGS_ATTRS = (
    "mcp_registry_base_url",
    "mcp_registry_url",
    "mcp_registry_service_url",
    "raganything_base_url",
    "raganything_api_base_url",
    "rag_spaces_base_url",
)

_raganything_stub: RagAnythingStubServer | None = None
_raganything_lock = threading.Lock()


def raganything_stub_singleton() -> RagAnythingStubServer:
    """Process-wide mcp-raganything REST stub, started on first use."""
    global _raganything_stub
    with _raganything_lock:
        if _raganything_stub is None:
            stub = RagAnythingStubServer()
            stub.start()
            _raganything_stub = stub
    return _raganything_stub


def _publish_raganything_seam(base_url: str) -> list[tuple[str, str | None]]:
    """Publish the stub base URL to the config seam; return the prior values.

    Public seam helpers (``publish_raganything_seam`` /
    ``restore_raganything_seam``) are reused by the unreachable-boundary test,
    which republishes a DEAD URL under the SAME contract.
    """
    prior: list[tuple[str, str | None]] = [(MCP_REGISTRY_BASE_URL_ENV, os.environ.get(MCP_REGISTRY_BASE_URL_ENV))]
    os.environ[MCP_REGISTRY_BASE_URL_ENV] = base_url
    import src.dependencies as deps

    for attr in _RAG_SEAM_SETTINGS_ATTRS:
        prior.append((attr, getattr(deps.settings, attr, None)))
        try:
            setattr(deps.settings, attr, base_url)
        except Exception:  # noqa: BLE001 — unknown/frozen settings field
            continue
    return prior


def publish_raganything_seam(base_url: str) -> list[tuple[str, str | None]]:
    """Public alias of :func:`_publish_raganything_seam` (test seam helper)."""
    return _publish_raganything_seam(base_url)


def _restore_raganything_seam(prior: list[tuple[str, str | None]]) -> None:
    for attr, value in prior:
        if attr == MCP_REGISTRY_BASE_URL_ENV:
            if value is None:
                os.environ.pop(MCP_REGISTRY_BASE_URL_ENV, None)
            else:
                os.environ[MCP_REGISTRY_BASE_URL_ENV] = value
            continue
        if value is None:
            continue
        try:
            import src.dependencies as deps

            setattr(deps.settings, attr, value)
        except Exception:  # noqa: BLE001 — unknown/frozen settings field
            continue


def restore_raganything_seam(prior: list[tuple[str, str | None]]) -> None:
    """Public alias of :func:`_restore_raganything_seam` (test seam helper)."""
    _restore_raganything_seam(prior)


@pytest.fixture
def raganything_stub() -> RagAnythingStubServer:
    """The mcp-raganything REST boundary stub; publishes its URL to the seam.

    The prior seam values are restored at teardown so neighboring suites are
    never polluted (the composable-mcp suite sets the same seam for its own
    registry stub).
    """
    stub = raganything_stub_singleton()
    prior = _publish_raganything_seam(stub.base_url)
    stub.reset()
    yield stub
    _restore_raganything_seam(prior)
    stub.reset()


# ---------------------------------------------------------------------------
# Real app over a real socket (streamable-HTTP MCP transport)
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def mcp_base_url(real_app_and_db: dict) -> AsyncGenerator[str, None]:
    """Serve the REAL app on a real local socket for the whole session.

    The app lifespan runs, but ``init_persistence`` / ``close_persistence``,
    the cron scheduler loop and the MCP loader teardown are no-oped: the
    parent session fixture owns those resources (a second boot/shutdown would
    rip the shared engine out from under every other behavioral test).
    """
    import uvicorn

    import src.main as src_main

    orig_init = src_main.init_persistence
    orig_close = src_main.close_persistence
    orig_sched = src_main.start_scheduler_loop
    orig_loader = src_main.mcp_tool_loader

    class _NoopClosable:
        async def close(self) -> None:  # noqa: D102
            return None

    src_main.init_persistence = lambda: _noop_awaitable()  # type: ignore[assignment]
    src_main.close_persistence = lambda: _noop_awaitable()  # type: ignore[assignment]
    src_main.start_scheduler_loop = lambda _uc: None  # type: ignore[assignment]
    src_main.mcp_tool_loader = _NoopClosable()  # type: ignore[assignment]

    server = uvicorn.Server(uvicorn.Config(real_app_and_db["app"], host="127.0.0.1", port=0, log_level="warning"))
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    serve_task: Any = None
    try:
        serve_task = asyncio.create_task(server.serve())
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not server.started:
            await asyncio.sleep(0.05)
        if not server.started:
            raise RuntimeError("uvicorn test server failed to start")
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        if serve_task is not None:
            server.should_exit = True
            with contextlib.suppress(Exception):
                await asyncio.wait_for(serve_task, timeout=10)
        src_main.init_persistence = orig_init
        src_main.close_persistence = orig_close
        src_main.start_scheduler_loop = orig_sched
        src_main.mcp_tool_loader = orig_loader


def _noop_awaitable() -> Any:
    async def _noop() -> None:
        return None

    return _noop()


# ---------------------------------------------------------------------------
# MCP client helpers (copied verbatim from the composable-mcp suite)
# ---------------------------------------------------------------------------
@contextlib.asynccontextmanager
async def open_mcp_session(base_url: str, headers: dict[str, str] | None = None) -> AsyncGenerator[Any, None]:
    """Open a REAL MCP client session against the mounted streamable-HTTP app."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with (
        streamablehttp_client(f"{base_url.rstrip('/')}/mcp", headers=headers or {}) as (
            read,
            write,
            _get_session_id,
        ),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


async def call_mcp_tool(base_url: str, headers: dict[str, str], name: str, arguments: dict | None = None) -> dict:
    """Call an MCP tool and return ``{"ok": bool, "result": <json-safe dict>}``."""
    async with open_mcp_session(base_url, headers) as session:
        result = await session.call_tool(name, arguments or {})
        payload: Any = None
        if getattr(result, "structuredContent", None) is not None:
            payload = result.structuredContent
        else:
            texts = [c.text for c in result.content if getattr(c, "text", None)]
            if texts:
                try:
                    payload = json.loads(texts[0])
                except json.JSONDecodeError:
                    payload = {"text": "\n".join(texts)}
            else:
                payload = {}
        return {"ok": not getattr(result, "isError", False), "result": payload}


def assistant_tool_call(name: str, arguments: dict, text: str | None = None, call_id: str = "call_1") -> dict:
    """An OpenAI assistant message carrying exactly one tool call."""
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        ],
    }


def assistant_text(text: str) -> dict:
    """A plain final assistant message."""
    return {"role": "assistant", "content": text}


async def one_shot_run(
    mcp_base_url, headers: dict[str, str], agent_name: str, message: str = "please do your work"
) -> dict:
    """Run an agent one-shot through the REAL ``composable_agents_run`` tool."""
    return await call_mcp_tool(
        mcp_base_url, headers, "composable_agents_run", {"agent_name": agent_name, "message": message}
    )


# ---------------------------------------------------------------------------
# REST helpers (agent CRUD + user-namespaced store files)
# ---------------------------------------------------------------------------
async def create_agent_ok(client, headers: dict[str, str], yaml_text: str, name: str) -> str:
    """Create an agent through the REAL multipart endpoint; assert 201."""
    _, resp = await create_agent_yaml(client, headers, yaml_text, name)
    assert resp.status_code == 201, resp.text
    return name


async def update_agent_ok(client, headers: dict[str, str], yaml_text: str, name: str) -> None:
    """Update an agent through the REAL multipart endpoint; assert 200."""
    resp = await client.put(
        f"/api/v1/agents/{name}",
        headers=headers,
        files={"file": ("agent.yaml", yaml_text.encode(), "text/yaml")},
    )
    assert resp.status_code == 200, resp.text


async def delete_agent(client, headers: dict[str, str], name: str) -> Any:
    """Delete an agent through the REAL endpoint (returns the response)."""
    return await client.delete(f"/api/v1/agents/{name}", headers=headers)


async def store_put(client, headers: dict[str, str], path: str, content: str) -> Any:
    """Put a (possibly nested) store file through the REAL store API."""
    return await client.put(f"/api/v1/store/files/{path.lstrip('/')}", headers=headers, json={"content": content})


async def store_list(client, headers: dict[str, str], prefix: str) -> list[str]:
    """List store file paths under a prefix through the REAL store API."""
    resp = await client.get("/api/v1/store/files", params={"prefix": prefix}, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


async def store_get(client, headers: dict[str, str], path: str) -> Any:
    """Get one store file through the REAL store API (returns the response)."""
    return await client.get(f"/api/v1/store/files/{path.lstrip('/')}", headers=headers)


# ---------------------------------------------------------------------------
# YAML builders
# ---------------------------------------------------------------------------
def header_mcp_yaml(name: str, url: str, *, headers_block: list[str] | None = None, tools: str = "[]") -> str:
    """Agent YAML whose only MCP server points at the recording stub.

    ``headers_block`` appends additional configured header lines (e.g. a
    ``${AGENT_NAME}`` placeholder or a colliding ``x-composable-agent`` value)
    under the MCP server entry's ``headers:`` mapping. ``tools`` is emitted as
    the agent's ``tools:`` YAML scalar (``"[]"`` or ``'["mcp"]'``).
    """
    lines = [
        f"name: {name}",
        "model: stub-model",
        'system_prompt: "Scoped header-injection test agent."',
        f"tools: {tools}",
        "debug: false",
        "mcp_servers:",
        "  - name: header-stub",
        "    transport: http",
        f"    url: {url}",
    ]
    if headers_block:
        lines.append("    headers:")
        lines.extend(f"      {line}" for line in headers_block)
    return "\n".join(lines) + "\n"


SUBAGENT_MODEL_YAML = "openai:stub-model"


def scoped_agent_yaml(
    name: str,
    url: str,
    *,
    subagent_name: str | None = None,
    subagent_url: str | None = None,
) -> str:
    """Agent YAML referencing the stub server; optionally with a subagent.

    The TOP-LEVEL agent declares its own ``url`` MCP server (its space,
    REQ-016) and — when ``subagent_name``/``subagent_url`` are given — the
    subagent declares its OWN ``mcp_servers`` entry (its per-subagent space,
    REQ-016), so both connections are observable at the same recording stub.
    The subagent pins a provider-prefixed ``model`` so the deepagents runtime
    resolves a real ``BaseChatModel`` for its middleware stack (the installed
    deepagents release rejects a bare/None subagent model — the tests
    exercise the per-subagent SPACE binding, not model resolution).
    """
    lines = [
        f"name: {name}",
        "model: stub-model",
        'system_prompt: "Scoped lifecycle test agent."',
        "tools: []",
        "debug: false",
        "mcp_servers:",
        "  - name: header-stub",
        "    transport: http",
        f"    url: {url}",
    ]
    if subagent_name is not None and subagent_url is not None:
        lines += [
            "subagents:",
            f"  - name: {subagent_name}",
            '    description: "Scoped sub-agent."',
            f"    model: {SUBAGENT_MODEL_YAML}",
            f'    instructions: "Scoped sub-agent instructions for {subagent_name}."',
            "    mcp_servers:",
            "      - name: sub-header-stub",
            "        transport: http",
            f"        url: {subagent_url}",
        ]
    return "\n".join(lines) + "\n"


def lifecycle_update_yaml(name: str, subagents: list[str]) -> str:
    """Agent YAML for the UPDATE of an imported topology.

    The resource-only write policy (main PR #47) refuses inline top-level
    ``model``/``system_prompt``/``mcp_servers`` on every update once
    ``model_ref``/``prompt_ref`` are materialized — the update declares its
    sub-agents inline (name+description) and nothing else.
    """
    lines = [f"name: {name}", "debug: false"]
    if subagents:
        lines.append("subagents:")
        for sub in subagents:
            lines += [
                f"  - name: {sub}",
                f"    description: \"Lifecycle sub-agent '{sub}'.\"",
            ]
    return "\n".join(lines) + "\n"


def lifecycle_yaml_refs(name: str, subagent_refs: list[str]) -> str:
    """Agent YAML declaring sub-agents as ``agent_ref`` references.

    Inline sub-agents are MATERIALIZED into real agents named after
    themselves at creation (main PR #47) — a topology re-declaring its
    sub-agents references those created agents instead of re-inlining them
    (a re-inlined body collides with the materialized row's name).
    """
    lines = [
        f"name: {name}",
        "model: stub-model",
        'system_prompt: "Scoped lifecycle test agent."',
        "debug: false",
    ]
    if subagent_refs:
        lines.append("subagents:")
        for sub in subagent_refs:
            lines += [
                f"  - name: {sub}",
                f"    description: \"Reference sub-agent '{sub}'.\"",
                f"    agent_ref: {sub}",
            ]
    return "\n".join(lines) + "\n"


def platform_tools_yaml(name: str) -> str:
    """Agent YAML opting into the agent-facing platform tools (``tools: ["mcp"]``)."""
    return (
        f"name: {name}\n"
        "model: stub-model\n"
        'system_prompt: "Scoped memories platform-tools agent."\n'
        'tools: ["mcp"]\n'
        "debug: false\n"
    )


def lifecycle_yaml(name: str, subagents: list[str]) -> str:
    """Agent YAML for lifecycle (spaces provisioning) tests: plain subagents."""
    lines = [
        f"name: {name}",
        "model: stub-model",
        'system_prompt: "Spaces lifecycle test agent."',
        "tools: []",
        "debug: false",
    ]
    if subagents:
        lines.append("subagents:")
        for sub in subagents:
            # main PR #47: inline sub-agents are MATERIALIZED into real
            # agents — they must declare a model source and instructions.
            lines += [
                f"  - name: {sub}",
                f"    description: \"Lifecycle sub-agent '{sub}'.\"",
                f"    model: {SUBAGENT_MODEL_YAML}",
                f'    instructions: "Scoped lifecycle instructions for {sub}."',
            ]

    return "\n".join(lines) + "\n"
