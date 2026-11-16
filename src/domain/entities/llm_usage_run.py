"""One costly run of the top-N usage list (spec §5).

A single entry of the ``GET /api/v1/traces/usage/top`` payload: the run
identity (thread + turn), the serving agent/model, token totals, estimated USD
cost, wall-time duration (HUMAN_MESSAGE → AI_MESSAGE) and the run's start
timestamp (the turn's HUMAN_MESSAGE timestamp). Error runs never appear here.
"""

from datetime import datetime

from pydantic import BaseModel


class LlmUsageRun(BaseModel):
    """One ranked run of the most-expensive-usage top list."""

    thread_id: str
    turn_id: str
    agent_name: str
    model_name: str
    total_tokens: int
    cost_usd: float
    duration_ms: float
    started_at: datetime
