"""Behavioral fixtures for the agent-import materialization suite.

The ``mcp-raganything`` MCP registry is an OUTBOUND external service
(spec §9): it is the ONLY mocked boundary in this directory. The
``StubRegistryServer`` convention of ``tests/behavioral/composable_mcp/
conftest.py`` is reused (imported, not duplicated). Everything else —
PostgreSQL (testcontainers), MinIO store, prompt/model registries, routes,
use cases, RLS — runs real, booted by ``tests/behavioral/conftest.py``.
"""

import os
import threading

import pytest

from tests.behavioral.composable_mcp.conftest import (
    MCP_REGISTRY_BASE_URL_ENV,
    StubRegistryServer,
)

_registry_server: StubRegistryServer | None = None
_registry_lock = threading.Lock()


@pytest.fixture
def stub_registry() -> StubRegistryServer:
    """The mcp-raganything registry REST boundary mock with its config seam."""
    global _registry_server
    with _registry_lock:
        if _registry_server is None:
            _registry_server = StubRegistryServer()
            _registry_server.start()
    stub = _registry_server
    assert stub is not None
    stub.reset(seed={})

    os.environ[MCP_REGISTRY_BASE_URL_ENV] = stub.base_url
    # Best-effort Settings seam (implementation will settle the attribute name).
    try:
        import src.dependencies as deps

        for attr in (
            "mcp_registry_base_url",
            "mcp_registry_url",
            "mcp_registry_service_url",
            "raganything_base_url",
        ):
            try:
                setattr(deps.settings, attr, stub.base_url)
            except Exception:  # noqa: BLE001 — unknown/frozen settings field
                continue
    except Exception:  # noqa: BLE001 — deps not imported yet
        pass

    yield stub
    stub.reset()
