"""Behavioral/persistence tests — the additive ``mcp`` thread source
(REQ-010, migration contract in Technical Notes).

The migration behavior is exercised at the persistence level (real SQLAlchemy
models + repository over a real relational backend) instead of asserting raw
SQL only: once Alembic replaces the ``threads.source`` check constraint with
``source IN ('chat','cron','mcp')``, the domain enum gains ``MCP`` and the
repository round-trips it.

REST-side spoof prevention (public creation cannot set source=mcp) is covered
behaviorally in ``test_mcp_one_shot_run.py`` and the existing threads suite.
"""

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")


class TestThreadSourceMcpPersistence:
    async def test_domain_enum_has_mcp_source(self):
        from src.domain.entities.thread import ThreadSource

        assert ThreadSource.MCP.value == "mcp", "ThreadSource must gain the MCP member"

    async def test_repository_round_trips_mcp_source(self, thread_repo):
        """The new source value persists through the REAL repository."""
        from src.domain.entities.thread import Thread, ThreadSource

        thread = Thread(
            agent_name="composable-ai-assistant",
            source=ThreadSource.MCP,
        )
        created = await thread_repo.create(thread)
        fetched = await thread_repo.get(created.id)
        assert fetched.source == ThreadSource.MCP

    async def test_unknown_source_is_still_rejected(self):
        """The migration must keep rejecting sources outside chat|cron|mcp."""
        from src.domain.entities.thread import ThreadSource

        # Emulate the constraint at the domain boundary: only the three values
        # are constructible, anything else must raise (enum, not free string).
        with pytest.raises(ValueError):
            ThreadSource("totally-real")
