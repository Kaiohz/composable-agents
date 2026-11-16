"""PostgreSQL adapter for the LlmPriceRepository port (dashboard spec §3).

System-side persistence of the global ``llm_prices`` reference table:

* ``upsert`` — single-row-per-model idempotent INSERT ... ON CONFLICT upsert,
  keyed on ``model_name`` (the primary key). Writes require the
  ``app.bypass_rls`` GUC: RLS restricts writes to system contexts, so callers
  run inside ``system_rls_context`` (the seeding path wraps its own scope).
* ``seed_known_defaults`` — the built-in Anthropic / OpenAI / Google / Mistral
  / OpenRouter catalog (approximate USD per 1M tokens), upserted with one
  executemany statement at startup. The usage aggregation resolves the price
  by LONGEST-PREFIX match (spec §3 "model names and common prefixes", BUG-002):
  any seeded row also acts as the prefix of its dated variants
  (``gpt-4o-2024-08-06`` → ``gpt-4o``), and distinct families (``o3-mini``
  vs ``o3``) are seeded separately so the shorter row never grabs their runs.
  Models without a matching prefix stay unpriced: the usage pipeline reports
  them as ``cost=0`` + ``priced=false``.

No ORM model: the table is accessed with raw parameterized SQL (matches the
bypass-aware RLS write path used by the test fixture and the seeder alike).
"""

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from src.domain.entities.llm_price import LlmPrice
from src.domain.errors.messages import ErrorMessage
from src.domain.errors.storage import StorageError
from src.domain.ports.llm_price_repository import LlmPriceRepository
from src.infrastructure.database.rls_context import system_rls_context

# Built-in price catalog (approximate ~USD per 1M input/output tokens, spec §3
# "known Anthropic / OpenAI / Google / Mistral / OpenRouter model names and
# common prefixes"). OpenRouter entries use provider-qualified model ids.
DEFAULT_LLM_PRICES: tuple[LlmPrice, ...] = (
    # Anthropic
    LlmPrice(
        model_name="claude-sonnet-4-5", provider="anthropic", input_price_per_million=3.0, output_price_per_million=15.0
    ),
    LlmPrice(
        model_name="claude-haiku-4", provider="anthropic", input_price_per_million=0.8, output_price_per_million=4.0
    ),
    LlmPrice(
        model_name="claude-opus-4-5", provider="anthropic", input_price_per_million=5.0, output_price_per_million=25.0
    ),
    # OpenAI
    LlmPrice(model_name="gpt-4o", provider="openai", input_price_per_million=2.5, output_price_per_million=10.0),
    LlmPrice(model_name="gpt-4o-mini", provider="openai", input_price_per_million=0.15, output_price_per_million=0.6),
    LlmPrice(model_name="gpt-4.1", provider="openai", input_price_per_million=2.0, output_price_per_million=8.0),
    LlmPrice(model_name="gpt-4.1-mini", provider="openai", input_price_per_million=0.4, output_price_per_million=1.6),
    LlmPrice(model_name="gpt-5", provider="openai", input_price_per_million=1.25, output_price_per_million=10.0),
    LlmPrice(model_name="o3", provider="openai", input_price_per_million=2.0, output_price_per_million=8.0),
    # Distinct OpenAI family: with longest-prefix matching a bare `o3` row is a
    # boundary-prefix of `o3-mini-...` — seeding the longer id keeps the cheaper
    # family resolving its OWN prices (BUG-002 prefix disambiguation).
    LlmPrice(model_name="o3-mini", provider="openai", input_price_per_million=1.1, output_price_per_million=4.4),
    # Google
    LlmPrice(
        model_name="gemini-2.0-flash", provider="google", input_price_per_million=0.1, output_price_per_million=0.4
    ),
    LlmPrice(
        model_name="gemini-2.5-flash", provider="google", input_price_per_million=0.3, output_price_per_million=2.5
    ),
    LlmPrice(
        model_name="gemini-2.5-pro", provider="google", input_price_per_million=1.25, output_price_per_million=10.0
    ),
    # Mistral
    LlmPrice(model_name="mistral-large", provider="mistral", input_price_per_million=2.0, output_price_per_million=6.0),
    LlmPrice(
        model_name="mistral-medium", provider="mistral", input_price_per_million=0.4, output_price_per_million=2.0
    ),
    LlmPrice(model_name="mistral-small", provider="mistral", input_price_per_million=0.1, output_price_per_million=0.3),
    # OpenRouter (provider-qualified paths: provider='openrouter' + slash model ids)
    LlmPrice(
        model_name="anthropic/claude-sonnet-4.5",
        provider="openrouter",
        input_price_per_million=3.0,
        output_price_per_million=15.0,
    ),
    LlmPrice(
        model_name="openai/gpt-4o", provider="openrouter", input_price_per_million=2.5, output_price_per_million=10.0
    ),
    LlmPrice(
        model_name="google/gemini-2.0-flash-001",
        provider="openrouter",
        input_price_per_million=0.1,
        output_price_per_million=0.4,
    ),
    LlmPrice(
        model_name="mistralai/mistral-large",
        provider="openrouter",
        input_price_per_million=2.0,
        output_price_per_million=6.0,
    ),
)

_UPSERT_SQL = text(
    """
    INSERT INTO llm_prices
        (model_name, provider, input_price_per_million, output_price_per_million, currency)
    VALUES
        (:model_name, :provider, :input_price_per_million, :output_price_per_million, :currency)
    ON CONFLICT (model_name) DO UPDATE SET
        provider = EXCLUDED.provider,
        input_price_per_million = EXCLUDED.input_price_per_million,
        output_price_per_million = EXCLUDED.output_price_per_million,
        currency = EXCLUDED.currency
    """
)


class PostgresLlmPriceRepository(LlmPriceRepository):
    """Adapter persisting per-model LLM prices in PostgreSQL."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @staticmethod
    def _params(price: LlmPrice) -> dict:
        """Build the bound-parameter dict for one price row."""
        return price.model_dump()

    async def upsert(self, price: LlmPrice) -> None:
        """Insert or update one price row (single row per model).

        Args:
            price: The price row to persist.

        Raises:
            StorageError: On infrastructure failure.
        """
        try:
            async with self._engine.begin() as conn:
                await conn.execute(_UPSERT_SQL, self._params(price))
        except SQLAlchemyError as e:
            raise StorageError(ErrorMessage.LLM_PRICES_SEED_FAILED.format(error=e)) from e

    async def seed_known_defaults(self) -> int:
        """Upsert the built-in catalog inside its own system RLS scope.

        Returns:
            The number of catalog rows kept fresh.

        Raises:
            StorageError: On infrastructure failure.
        """
        try:
            async with system_rls_context(), self._engine.begin() as conn:
                await conn.execute(_UPSERT_SQL, [self._params(price) for price in DEFAULT_LLM_PRICES])
        except SQLAlchemyError as e:
            raise StorageError(ErrorMessage.LLM_PRICES_SEED_FAILED.format(error=e)) from e
        # The caller (init_persistence) logs the seeded count once — no
        # duplicate log line here.
        return len(DEFAULT_LLM_PRICES)
