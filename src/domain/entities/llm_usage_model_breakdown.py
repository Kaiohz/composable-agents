"""Per-model token/cost breakdown of the LLM usage aggregation (spec §2, §4).

One series entry of ``series.by_model``: token totals and the priced cost of
all completed runs served by that model. ``priced`` is ``False`` (and the cost
``0``) when the model has no row in the global ``llm_prices`` table — unpriced
models still appear so the dashboard can show their usage.
"""

from pydantic import BaseModel


class LlmUsageModelBreakdown(BaseModel):
    """Token / cost totals for one served model name."""

    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    priced: bool = False
