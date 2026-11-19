"""Unit tests for the composition-root shutdown lifecycle (I2 remediation).

Pins the hygiene contract of ``close_persistence``: every owned resource is
closed exactly once (persistent registry, engine, JWT adapter, MCP-registry
HTTP client), the singletons are nulled so a later boot starts clean, the MCP
registry client's ``aclose`` failure never blocks the remaining shutdown
steps, and an unwired shutdown is a safe no-op.
"""

from unittest.mock import AsyncMock

import pytest

import src.dependencies as deps
from src.infrastructure.persistent_registry.adapter import PersistentAgentRegistry


def _real_like_registry() -> PersistentAgentRegistry:
    """A real registry instance (isinstance-guarded branch) with a stub close."""
    registry = PersistentAgentRegistry(
        config_loader=AsyncMock(),  # type: ignore[arg-type] — unit stub
        config_store=AsyncMock(),  # type: ignore[arg-type] — unit stub
        config_repository=AsyncMock(),  # type: ignore[arg-type] — unit stub
        mcp_tool_loader=AsyncMock(),  # type: ignore[arg-type] — unit stub
    )
    registry.close = AsyncMock()
    return registry


@pytest.fixture
def _saved_state():
    """Snapshot + restore the composition-root singletons around each test."""
    root = deps._root
    saved_fields = {name: getattr(root, name) for name in ("agent_registry", "async_engine", "jwt_adapter")}
    saved_client = deps._mcp_registry_client
    yield root
    root.agent_registry = saved_fields["agent_registry"]
    root.async_engine = saved_fields["async_engine"]
    root.jwt_adapter = saved_fields["jwt_adapter"]
    deps._mcp_registry_client = saved_client


@pytest.mark.usefixtures("_saved_state")
class TestClosePersistence:
    async def test_closes_every_owned_resource_once(self):
        root = deps._root
        root.agent_registry = _real_like_registry()
        root.async_engine = AsyncMock()
        root.jwt_adapter = AsyncMock()
        jwt = root.jwt_adapter
        client = deps._mcp_registry_client = AsyncMock()

        await deps.close_persistence()

        root.agent_registry.close.assert_awaited_once()
        root.async_engine.dispose.assert_awaited_once()
        jwt.close.assert_awaited_once()
        client.aclose.assert_awaited_once()
        assert deps._mcp_registry_client is None
        assert root.jwt_adapter is None

    async def test_registry_client_failure_does_not_stop_shutdown(self):
        root = deps._root
        root.agent_registry = _real_like_registry()
        root.async_engine = AsyncMock()
        root.jwt_adapter = AsyncMock()
        client = deps._mcp_registry_client = AsyncMock()
        client.aclose.side_effect = RuntimeError("client already closed")

        await deps.close_persistence()

        # The failing aclose is contained: the remaining shutdown steps ran.
        root.agent_registry.close.assert_awaited_once()
        root.async_engine.dispose.assert_awaited_once()
        client.aclose.assert_awaited_once()

    async def test_noop_shutdown_when_nothing_is_wired(self):
        root = deps._root
        root.agent_registry = None
        root.async_engine = None
        root.jwt_adapter = None
        deps._mcp_registry_client = None

        await deps.close_persistence()  # no raise, nothing closed
        assert deps._mcp_registry_client is None
