"""Per-agent token/cost breakdown of the LLM usage aggregation (spec §4).

One series entry of ``series.by_agent``: token totals and the priced cost of
all completed runs of that agent within the requested window.
"""

from pydantic import BaseModel


class LlmUsageAgentBreakdown(BaseModel):
    """Token / cost totals for one agent name."""

    agent_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    priced: bool = False
