"""Feature fixtures — chat PDF attachments behavioral suites.

Reuses the suite-wide ``stack`` / ``client`` / ``users`` / ``stub_llm``
fixtures from ``tests/behavioral/conftest.py`` (real app + testcontainers
PostgreSQL/MinIO + real auth chain) and adds:

* ``mcp_files`` — the in-process REAL FastMCP stub server standing in for the
  external ``/files/mcp`` RAGAnything server (see ``mcp_files_stub.py``).
* a session registry REST boundary (``StubRegistryServer``, same convention as
  the composable_mcp suite): since the resource-only import materializes
  declared inline MCP servers into the registry (spec agent-resource-import),
  an agent import now REQUIRES a reachable registry; the external registry is
  the mocked boundary here.
* helpers to build attachable agents, threads and attachment payloads.
"""

import base64
import os
import threading
from typing import Any

import pytest
import pytest_asyncio

from tests.behavioral.chat_attachments.mcp_files_stub import McpFilesStubServer
from tests.behavioral.composable_mcp.conftest import MCP_REGISTRY_BASE_URL_ENV, StubRegistryServer
from tests.behavioral.conftest import configure_llm_credentials, create_agent_yaml, create_thread

PNG_1PX_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def mcp_files_server_session() -> Any:
    """Per-session view over the lazily-started Files-MCP stub singleton."""
    server = mcp_files_stub()
    try:
        yield server
    finally:
        server.stop()


def mcp_files_stub() -> McpFilesStubServer:
    """Return the session-wide Files-MCP stub, starting it on first use.

    The stub backs THREE usage paths: the ``mcp_files`` fixture (per-test
    recording reset), and the DEFAULT ``agent_with_thread`` YAML — whose
    contract (see ``files_mcp_yaml`` docstring, REQ-009) is an agent carrying a
    Files MCP server ending with ``/files/mcp``. Starting lazily keeps
    attachment-free suites cheap while guaranteeing the default agents can
    actually receive temporary document uploads.
    """
    global _STUB_SERVER
    if _STUB_SERVER is None:
        _STUB_SERVER = McpFilesStubServer()
        _STUB_SERVER.start()
    return _STUB_SERVER


_STUB_SERVER: McpFilesStubServer | None = None


# ---------------------------------------------------------------------------
# Import registry boundary mock (spec agent-resource-import): declared inline
# MCP servers are materialized in the user's registry at agent creation, so
# every attachment-suite import needs a reachable registry REST boundary.
# ---------------------------------------------------------------------------
_registry_server: StubRegistryServer | None = None
_registry_lock = threading.Lock()


def import_registry_stub() -> StubRegistryServer:
    """Session-wide registry REST boundary mock, started on first use."""
    global _registry_server
    with _registry_lock:
        if _registry_server is None:
            _registry_server = StubRegistryServer()
            _registry_server.start()
        return _registry_server


@pytest.fixture(autouse=True)
def stub_import_registry():
    """Point the production registry seam at a fresh (empty) mock per test."""
    stub = import_registry_stub()
    stub.reset(seed={})

    os.environ[MCP_REGISTRY_BASE_URL_ENV] = stub.base_url
    import src.dependencies as deps

    deps.settings.mcp_registry_base_url = stub.base_url
    yield stub
    stub.reset(seed={})


@pytest_asyncio.fixture(loop_scope="session")
async def mcp_files(mcp_files_server_session) -> Any:
    """Per-test view over the MCP files stub, with a clean recording state."""
    mcp_files_server_session.reset()
    return mcp_files_server_session


# ---------------------------------------------------------------------------
# Payload helpers (REQ-001 attachment shape)
# ---------------------------------------------------------------------------
def image_attachment(**overrides: Any) -> dict[str, Any]:
    """A minimal valid PNG attachment payload."""
    payload = {
        "filename": "capture.png",
        "content_type": "image/png",
        "content_base64": PNG_1PX_BASE64,
    }
    payload.update(overrides)
    return payload


def document_attachment(**overrides: Any) -> dict[str, Any]:
    """A minimal valid PDF document attachment payload."""
    payload = {
        "filename": "rapport.pdf",
        "content_type": "application/pdf",
        "content_base64": base64.b64encode(b"%PDF-1.7 stub document").decode(),
    }
    payload.update(overrides)
    return payload


def byte_attachment(filename: str, content_type: str, size_bytes: int) -> dict[str, str]:
    """Attachment whose decoded payload is exactly ``size_bytes`` bytes."""
    return {
        "filename": filename,
        "content_type": content_type,
        "content_base64": base64.b64encode(b"\x00" * size_bytes).decode(),
    }


def files_mcp_yaml(name: str, *, mcp_url: str | None = None, extra_servers: list[str] | None = None) -> str:
    """Agent YAML with a Files MCP server ending in ``/files/mcp`` (REQ-009).

    ``extra_servers`` appends additional MCP server URLs (e.g. a
    ``/classical/mcp`` server) to verify EDG-008 / EDG-009 selection rules.

    The ``mcp_servers`` key is emitted as an EXPLICIT EMPTY LIST when no
    server is requested — a bare ``mcp_servers:`` key parses to YAML null and
    the agent-config validation rejects ``null`` with 422
    ("Input should be a valid list").
    """
    lines = [
        f"name: {name}",
        "model: stub-model",
        'system_prompt: "Attachments test agent."',
        "tools: []",
        "debug: false",
    ]
    if mcp_url is not None or extra_servers:
        lines.append("mcp_servers:")
        if mcp_url is not None:
            lines += [
                "  - name: files-server",
                "    transport: http",
                f"    url: {mcp_url}",
            ]
        for url in extra_servers or []:
            lines += [
                "  - name: classical-server",
                "    transport: http",
                f"    url: {url}",
            ]
    else:
        lines.append("mcp_servers: []")
    return "\n".join(lines) + "\n"


async def create_agent(client, headers: dict[str, str], name: str, yaml_text: str) -> str:
    """Create an agent via the REAL endpoint and return its name."""
    _, resp = await create_agent_yaml(client, headers, yaml_text, name)
    assert resp.status_code == 201, resp.text
    return name


async def agent_with_thread(
    client,
    headers: dict[str, str],
    yaml_text: str | None = None,
    agent_name: str | None = None,
) -> str:
    """Create an agent + a thread for that agent; return the thread id.

    Without an explicit ``yaml_text``, the agent carries a Files MCP server
    pointing at the session stub (the ``files_mcp_yaml`` documented contract,
    REQ-009) so image+document turns and temporary uploads work out of the box.

    When a caller supplies ``yaml_text`` WITHOUT ``agent_name``, the agent
    name is derived from the YAML's ``name:`` key — creating the agent under a
    DIFFERENT name than the one declared inside its YAML is rejected by the
    endpoint ("Agent name in YAML ... does not match provided name").
    """
    import uuid

    import yaml as yaml_parser  # type: ignore[import-untyped]

    if yaml_text is not None and agent_name is None:
        derived = (yaml_parser.safe_load(yaml_text) or {}).get("name")
        if isinstance(derived, str) and derived.strip():
            agent_name = derived.strip()

    resolved_name = agent_name or f"att-{uuid.uuid4().hex[:8]}"
    default_yaml = (
        yaml_text if yaml_text is not None else files_mcp_yaml(resolved_name, mcp_url=mcp_files_stub().url)
    )
    await create_agent(client, headers, resolved_name, default_yaml)
    await configure_llm_credentials(client, headers)
    thread_resp = await create_thread(client, headers, resolved_name)
    assert thread_resp.status_code == 201, thread_resp.text
    return thread_resp.json()["id"]
