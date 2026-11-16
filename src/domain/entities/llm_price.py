"""Price reference for one served LLM model (LLM usage dashboard, spec §3).

A single row of the global ``llm_prices`` table: the price of one million
input/output tokens in the row's currency (USD in the seeded catalog). Models
without a matching row are reported by the usage pipeline as ``cost=0`` with
``priced=false`` — unknown models never break the aggregation.
"""

from pydantic import BaseModel, ConfigDict


class LlmPrice(BaseModel):
    """Per-model token price (USD per 1M tokens by default)."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    provider: str
    input_price_per_million: float
    output_price_per_million: float
    currency: str | None = None
