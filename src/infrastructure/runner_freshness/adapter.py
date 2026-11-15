"""PostgreSQL adapter for cross-worker runner freshness (CACHE-002).

Computes a per-agent dependency fingerprint from committed database state with
a single LEFT-JOIN query over ``agent_configs``, ``prompts`` and ``models``:

* the agent row's ``updated_at`` — changes on every agent YAML save, so each
  worker rebuilds after an agent create/update (cross-worker, not just the
  mutating process);
* the referenced prompt's existence — a deleted prompt changes the
  fingerprint so stale references fail explicitly instead of replaying cached
  content (SEC-002), plus its ``latest_version`` for ``Latest`` refs so a new
  immutable version forces a rebuild while pinned agents (whose resolution is
  immutable) and metadata-only edits do not;
* the referenced model's existence and ``updated_at`` — edits apply on the
  referencing agents' next invocation without touching credential resolution.

Agents without registry references fingerprint only their own revision,
preserving legacy inline/file/custom-model behavior with user-safe
invalidation.
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from src.domain.errors.agent import AgentNotFoundError
from src.domain.errors.messages import ErrorMessage
from src.domain.ports.runner_freshness import RunnerFreshness
from src.infrastructure.database.models.agent_config import AgentConfigModel
from src.infrastructure.database.models.model_config import ModelConfigModel
from src.infrastructure.database.models.prompt import PromptModel

logger = logging.getLogger(__name__)


class PostgresRunnerFreshness(RunnerFreshness):
    """Fingerprint adapter over the agent's dependency rows."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def fingerprint(self, agent_name: str) -> str:
        stmt = (
            select(
                AgentConfigModel.updated_at,
                AgentConfigModel.prompt_ref,
                AgentConfigModel.model_ref,
                PromptModel.latest_version,
                ModelConfigModel.updated_at.label("model_updated_at"),
            )
            .outerjoin(PromptModel, PromptModel.id == AgentConfigModel.prompt_ref)
            .outerjoin(ModelConfigModel, ModelConfigModel.id == AgentConfigModel.model_ref)
            .where(AgentConfigModel.name == agent_name)
        )
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).first()

        if row is None:
            # The agent metadata row is gone (deleted or foreign under RLS) —
            # explicit not-found instead of replaying a cached runner.
            raise AgentNotFoundError(ErrorMessage.AGENT_CONFIG_NOT_FOUND.format(name=agent_name))

        agent_updated_at, prompt_ref, model_ref, prompt_latest_version, model_updated_at = row
        fingerprint = "|".join(
            (
                agent_updated_at.isoformat() if agent_updated_at else "",
                str(prompt_ref) if prompt_ref else "",
                str(prompt_latest_version) if prompt_ref and prompt_latest_version is not None else "",
                str(model_ref) if model_ref else "",
                model_updated_at.isoformat() if model_updated_at else "",
            )
        )
        return fingerprint
