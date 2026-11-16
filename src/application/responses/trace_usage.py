"""Response DTOs for the /traces/usage endpoints (dashboard spec §4-§5).

The payloads carry the domain aggregation entities verbatim; the wrapper only
adds the exact response shapes:

* ``TraceUsageResponse`` — totals + ``series``; the ``series`` wrapper exposes
  ``by_agent`` and/or ``by_model`` keys only (``group_by`` narrowing is
  rendered by omitting the other key — serialized with
  ``response_model_exclude_none``).
* ``TopRunsResponse`` — ``{"items": [...]}`` ranked runs.
"""

from pydantic import BaseModel, Field

from src.domain.entities.llm_usage_agent_breakdown import LlmUsageAgentBreakdown
from src.domain.entities.llm_usage_model_breakdown import LlmUsageModelBreakdown
from src.domain.entities.llm_usage_run import LlmUsageRun


class UsageSeries(BaseModel):
    """Token breakdown series; the unselected breakdown stays ``None``.

    ``None`` members are dropped by ``response_model_exclude_none`` so the
    serialized ``series`` object carries only the requested ``by_agent`` /
    ``by_model`` keys (exact group_by contract).
    """

    by_agent: list[LlmUsageAgentBreakdown] | None = None
    by_model: list[LlmUsageModelBreakdown] | None = None


class TraceUsageResponse(BaseModel):
    """Aggregate LLM usage response for the dashboard KPIs."""

    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    run_count: int
    error_count: int
    duration_ms_p50: float
    duration_ms_p95: float
    series: UsageSeries


class TopRunsResponse(BaseModel):
    """Top-N most expensive runs."""

    items: list[LlmUsageRun] = Field(default_factory=list)
