"""Behavioral fixtures for the cron suites.

Since the resource-only import materializes declared inline MCP servers into
the user's registry (spec agent-resource-import), agent creation inside the
cron scenarios needs a reachable registry REST boundary. The
``StubRegistryServer`` convention of ``tests/behavioral/composable_mcp/
conftest.py`` is reused (imported, not duplicated); PostgreSQL (testcontainers)
and the rest of the stack stay real, booted by ``tests/behavioral/conftest.py``.
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
    """The registry REST boundary mock with its configuration seam."""
    global _registry_server
    with _registry_lock:
        if _registry_server is None:
            _registry_server = StubRegistryServer()
            _registry_server.start()
    stub = _registry_server
    assert stub is not None
    stub.reset(seed={})

    os.environ[MCP_REGISTRY_BASE_URL_ENV] = stub.base_url
    import src.dependencies as deps

    deps.settings.mcp_registry_base_url = stub.base_url
    yield stub
    stub.reset(seed={})
