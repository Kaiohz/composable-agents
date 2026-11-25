"""Tests for PersistentAgentRegistry.

Uses real YamlAgentConfigLoader and real NoopTracingProvider (internal).
Mocks external boundaries: mock_agent_config_store (MinIO),
mock_agent_config_repository (PostgreSQL), mock_mcp_tool_loader (MCP).
Patches create_agent_from_config to return a fake graph (LangGraph boundary).
The DeepAgentRunner is built for real.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.entities.agent_config_metadata import AgentConfigMetadata
from src.infrastructure.database.rls_context import current_credential, current_user_id
from src.infrastructure.persistent_registry.adapter import PersistentAgentRegistry

VALID_YAML = (
    "name: test-agent\n"
    "model: claude-sonnet-4-5-20250929\n"
    'system_prompt: "You are a test agent."\n'
    "tools: []\n"
    "debug: false\n"
)


class TestPersistentAgentRegistry:
    @pytest.fixture
    def registry(self, yaml_loader, mock_agent_config_store, mock_agent_config_repository, mock_mcp_tool_loader):
        return PersistentAgentRegistry(
            config_loader=yaml_loader,
            config_store=mock_agent_config_store,
            config_repository=mock_agent_config_repository,
            mcp_tool_loader=mock_mcp_tool_loader,
        )

    @patch("src.infrastructure.persistent_registry.adapter.create_agent_from_config", new_callable=AsyncMock)
    async def test_get_runner_returns_real_runner(self, mock_create, registry, mock_agent_config_store):
        # Arrange
        mock_agent_config_store.get.return_value = VALID_YAML
        mock_create.return_value = (MagicMock(), None)

        # Act
        runner = await registry.get_runner("test-agent")

        # Assert
        mock_agent_config_store.get.assert_awaited_once_with("test-agent")
        mock_create.assert_awaited_once()
        assert runner is not None

    @patch("src.infrastructure.persistent_registry.adapter.create_agent_from_config", new_callable=AsyncMock)
    async def test_get_runner_cache_hit_returns_same_instance(self, mock_create, registry, mock_agent_config_store):
        # Arrange
        mock_agent_config_store.get.return_value = VALID_YAML
        mock_create.return_value = (MagicMock(), None)

        # Act
        first = await registry.get_runner("test-agent")
        second = await registry.get_runner("test-agent")

        # Assert
        assert first is second
        assert mock_agent_config_store.get.await_count == 1

    async def test_list_agents_returns_names_from_repository(self, registry, mock_agent_config_repository):
        # Arrange
        now = datetime.now(UTC)
        mock_agent_config_repository.list_all.return_value = [
            AgentConfigMetadata(
                name="agent-a",
                model="gpt-4o",
                minio_path="agent-configs/agent-a.yaml",
                created_at=now,
                updated_at=now,
            ),
            AgentConfigMetadata(
                name="agent-b",
                model="claude-sonnet-4-5-20250929",
                minio_path="agent-configs/agent-b.yaml",
                created_at=now,
                updated_at=now,
            ),
        ]

        # Act
        result = await registry.list_agents()

        # Assert
        # feat-composable-mcp REQ-031: the virtual built-in assistant is
        # overlaid FIRST in every user's catalog (never a stored row).
        assert result == ["composable-ai-assistant", "agent-a", "agent-b"]
        mock_agent_config_repository.list_all.assert_awaited_once()

    @patch("src.infrastructure.persistent_registry.adapter.create_agent_from_config", new_callable=AsyncMock)
    async def test_invalidate_forces_rebuild_on_next_get(self, mock_create, registry, mock_agent_config_store):
        # Arrange
        mock_agent_config_store.get.return_value = VALID_YAML
        mock_create.side_effect = [(MagicMock(), None), (MagicMock(), None)]

        # Act
        first = await registry.get_runner("test-agent")
        await registry.invalidate("test-agent")
        second = await registry.get_runner("test-agent")

        # Assert
        assert first is not second
        assert mock_agent_config_store.get.await_count == 2

    @patch("src.infrastructure.persistent_registry.adapter.create_agent_from_config", new_callable=AsyncMock)
    async def test_close_forces_rebuild_on_next_get(self, mock_create, registry, mock_agent_config_store):
        # Arrange
        mock_agent_config_store.get.return_value = VALID_YAML
        mock_create.side_effect = [(MagicMock(), None), (MagicMock(), None)]

        # Act
        first = await registry.get_runner("test-agent")
        await registry.close()
        second = await registry.get_runner("test-agent")

        # Assert
        assert first is not second
        assert mock_agent_config_store.get.await_count == 2


class TestCredentialScopedFingerprintInvalidation:
    """feat-mcp-oauth2 review M-3: the caller's raw credential is part of the
    runner cache identity. Custom agents whose YAML declares
    ``mcp_servers: [.., auth_mode: user_token]`` bind the credential INTO their
    MCP connections at tool-load time (langchain-mcp-adapters pins connection
    headers when the session is created), so a ROTATED or REVOKED credential
    must evict the cached runner instead of re-transmitting the stale token.
    The stored fingerprint carries only a SHA-256 prefix of the credential —
    never the credential itself.
    """

    @pytest.fixture
    def fresh_registry(
        self, yaml_loader, mock_agent_config_store, mock_agent_config_repository, mock_mcp_tool_loader
    ) -> tuple[PersistentAgentRegistry, MagicMock]:
        """A registry wired with a MOCK freshness port (stable fingerprint)."""
        freshness = MagicMock()
        freshness.fingerprint = AsyncMock(return_value="freshness-1")
        registry = PersistentAgentRegistry(
            config_loader=yaml_loader,
            config_store=mock_agent_config_store,
            config_repository=mock_agent_config_repository,
            mcp_tool_loader=mock_mcp_tool_loader,
            runner_freshness=freshness,
        )
        return registry, freshness

    async def _get(self, registry: PersistentAgentRegistry, credential: str | None):
        """get_runner under an explicit (user, credential) RLS context."""
        user_token = current_user_id.set("user-a")
        credential_token = current_credential.set(credential)
        try:
            return await registry.get_runner("test-agent")
        finally:
            current_user_id.reset(user_token)
            current_credential.reset(credential_token)

    @patch("src.infrastructure.persistent_registry.adapter.create_agent_from_config", new_callable=AsyncMock)
    async def test_same_credential_is_a_cache_hit(self, mock_create, fresh_registry, mock_agent_config_store):
        # Arrange
        registry, _freshness = fresh_registry
        mock_agent_config_store.get.return_value = VALID_YAML
        mock_create.return_value = (MagicMock(), None)

        # Act
        first = await self._get(registry, "credential-token-1")
        second = await self._get(registry, "credential-token-1")

        # Assert — identical (agent, user, credential) state → cached runner
        assert first is second
        assert mock_create.await_count == 1

    @patch("src.infrastructure.persistent_registry.adapter.create_agent_from_config", new_callable=AsyncMock)
    async def test_rotated_credential_invalidates_cached_runner(self, mock_create, fresh_registry, mock_agent_config_store):
        # Arrange — the rotated token is the REVIEW M-3 scenario: a stale
        # cached runner would keep transmitting the old credential.
        registry, _freshness = fresh_registry
        mock_agent_config_store.get.return_value = VALID_YAML
        mock_create.side_effect = [(MagicMock(), None), (MagicMock(), None)]

        # Act
        stale = await self._get(registry, "credential-token-1")
        rotated = await self._get(registry, "credential-token-2")

        # Assert — the credential change evicted the cached runner
        assert stale is not rotated
        assert mock_create.await_count == 2
        assert mock_agent_config_store.get.await_count == 2

    @patch("src.infrastructure.persistent_registry.adapter.create_agent_from_config", new_callable=AsyncMock)
    async def test_unattended_run_without_credential_is_stable(self, mock_create, fresh_registry, mock_agent_config_store):
        # Arrange
        registry, _freshness = fresh_registry
        mock_agent_config_store.get.return_value = VALID_YAML
        mock_create.return_value = (MagicMock(), None)

        # Act — no credential (cron/unattended context) in either call
        first = await self._get(registry, None)
        second = await self._get(registry, None)

        # Assert — stable empty credential fingerprint never thrashes the cache
        assert first is second
        assert mock_create.await_count == 1

    @patch("src.infrastructure.persistent_registry.adapter.create_agent_from_config", new_callable=AsyncMock)
    async def test_fingerprint_carries_only_a_hash_of_the_credential(
        self, mock_create, fresh_registry, mock_agent_config_store
    ):
        # Arrange
        registry, _freshness = fresh_registry
        mock_agent_config_store.get.return_value = VALID_YAML
        mock_create.return_value = (MagicMock(), None)

        # Act
        await self._get(registry, "credential-token-secret")

        # Assert — the stored cache fingerprint never contains the credential
        entry = registry._runners[("user-a", "test-agent")]
        assert "credential-token-secret" not in entry.fingerprint
        assert entry.fingerprint.startswith("freshness-1\x1f")
