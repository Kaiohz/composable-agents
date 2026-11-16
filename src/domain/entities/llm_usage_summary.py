"""Global LLM usage summary for the authenticated user (spec §4).

Aggregated over one window of the user's runs (one run = one turn): totals,
cost, run/error counts and the two token breakdowns (by agent, by model).
``run_count`` counts COMPLETED runs (turns with a terminal AI_MESSAGE);
``error_count`` counts error runs (turns missing the terminal AI_MESSAGE) —
the two sets are disjoint. Duration percentiles are computed over COMPLETED
runs only: error runs carry no terminal timestamp and never contribute.
"""

from pydantic import BaseModel, Field

from src.domain.entities.llm_usage_agent_breakdown import LlmUsageAgentBreakdown
from src.domain.entities.llm_usage_model_breakdown import LlmUsageModelBreakdown


class LlmUsageSummary(BaseModel):
    """Aggregated token / cost / latency usage for a window."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    run_count: int = 0
    error_count: int = 0
    duration_ms_p50: float = 0.0
    duration_ms_p95: float = 0.0
    series_by_agent: list[LlmUsageAgentBreakdown] = Field(default_factory=list)
    series_by_model: list[LlmUsageModelBreakdown] = Field(default_factory=list)
