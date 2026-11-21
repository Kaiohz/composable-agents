"""Behavioral tests — legacy (inline) agents stay readable and executable
(spec §4.I REQ-LEGACY-001 / §5 Compatibilité).

A legacy agent created BEFORE the resource-only rule (inline `model`,
inline `system_prompt`, inline `mcp_servers`, no `model_ref`/`prompt_ref`)
must keep working unchanged: no destructive migration at read time, no
fallback rewrite, execution still resolves the legacy values. The legacy
row is seeded through the REAL composition-root store + repository
(exactly what the use cases persist) because the import route now
materializes new agents into resource-only configurations.

Entry points: ``GET /api/v1/agents/{name}`` and the real chat chain
(thread creation + invocation) with the external LLM boundary stubbed.
"""

import uuid
from datetime import UTC, datetime

import pytest

from src.domain.entities.agent_config_metadata import AgentConfigMetadata
from src.infrastructure.database.rls_context import user_rls_context
from tests.behavioral.conftest import configure_llm_credentials, create_thread, invoke_agent

pytestmark = pytest.mark.asyncio(loop_scope="session")


def legacy_yaml(name: str) -> str:
    return (
        f"name: {name}\n"
        "model: legacy-runner-model\n"
        'system_prompt: "Legacy legacy prompt."\n'
        "tools: []\n"
        "debug: false\n"
    )


async def seed_legacy_agent(_stack: dict, user_id: str, name: str, yaml_text: str) -> None:
    """Persist a legacy agent through the REAL store + repository (user-scoped)."""
    import src.dependencies as deps

    now = datetime.now(UTC)
    metadata = AgentConfigMetadata(name=name, model="legacy-runner-model", minio_path=f"{name}.yaml", created_at=now, updated_at=now)
    async with user_rls_context(user_id):
        await deps._root.pg_repository.save(metadata)
        await deps._root.rustfs_store.put(name, yaml_text)


class TestLegacyAgentRead:
    async def test_read_preserves_inline_definitions_unchanged(self, client, users, stack):
        """REQ-LEGACY-001: a legacy inline agent is read back unchanged —
        inline model/prompt/MCP values are served, no auto-conversion."""
        name = f"legacy-{uuid.uuid4().hex[:8]}"
        await seed_legacy_agent(stack, users["a"]["id"], name, legacy_yaml(name))

        got = await client.get(f"/api/v1/agents/{name}", headers=users["a"]["headers"])
        assert got.status_code == 200, got.text
        body = got.json()
        assert body["model"] == "legacy-runner-model"
        assert body["system_prompt"] == "Legacy legacy prompt."
        assert body.get("model_ref") is None
        assert body.get("prompt_ref") is None


class TestLegacyAgentExecution:
    async def test_legacy_agent_remains_executable_with_inline_model_and_prompt(
        self, client, users, stack, stub_llm
    ):
        """REQ-LEGACY-001: the runner still resolves the legacy model string and
        the inline prompt for an agent created before the resource-only rule."""
        name = f"legacy-{uuid.uuid4().hex[:8]}"
        await seed_legacy_agent(stack, users["a"]["id"], name, legacy_yaml(name))

        await configure_llm_credentials(client, users["a"]["headers"])
        thread = await create_thread(client, users["a"]["headers"], name)
        assert thread.status_code == 201, thread.text

        response = await invoke_agent(client, users["a"]["headers"], thread.json()["id"])
        assert response.status_code == 200, response.text

        record = stub_llm.app.requests[-1]
        assert record.model == "legacy-runner-model", (
            f"the legacy agent must keep executing with its inline model: {record.model}"
        )
        assert "Legacy legacy prompt." in "".join(record.system_messages), (
            f"the legacy agent must keep executing with its inline prompt: {record.system_messages}"
        )
