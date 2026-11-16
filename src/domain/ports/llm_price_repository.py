"""Outbound port: LlmPriceRepository (LLM usage dashboard, spec §3).

Persists the per-model token price rows used by the usage pipeline to estimate
run cost. Writes are system-side only (the startup seeder keeps the seeded
catalog fresh; RLS restricts writes to system contexts at the database level).
"""

from abc import ABC, abstractmethod

from src.domain.entities.llm_price import LlmPrice


class LlmPriceRepository(ABC):
    """Outbound port for the global LLM price reference table."""

    @abstractmethod
    async def upsert(self, price: LlmPrice) -> None:
        """Insert or update the price row of a single model.

        Idempotent: keyed on ``model_name`` (single row per model). The caller
        must run inside a system RLS scope — writes are restricted to system
        contexts by the table's RLS policy.

        Args:
            price: The price row to persist.
        """
        ...

    @abstractmethod
    async def seed_known_defaults(self) -> int:
        """Seed the known Anthropic/OpenAI/Google/Mistral/OpenRouter prices.

        Idempotent upsert of the built-in price catalog (single row per model).
        Runs inside its own system RLS scope so it can be called from the
        plain startup path. Unknown/unpriced models are unaffected — they are
        reported as ``cost=0`` + ``priced=false`` by the usage pipeline.

        Returns:
            The number of catalog rows kept fresh.
        """
        ...
