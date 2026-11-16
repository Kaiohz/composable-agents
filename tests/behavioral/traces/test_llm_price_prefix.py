"""Behavioral tests — longest-prefix price resolution (dashboard §3, BUG-002).

Route under test: ``GET /api/v1/traces/usage`` (and ``/usage/top``) with turns
whose ``llm_usage.model_name`` is a DATED variant of a seeded model.

Contract fixed here (BUG-002 — pricing by "known model names AND common
prefixes"):

* a dated id must resolve the price of the seeded base model row it extends
  (``gpt-4o-2024-08-06`` → ``gpt-4o``: 2.5 USD/1M in, 10 USD/1M out) and be
  reported ``priced=true`` with the exact USD math;
* an EXACT id keeps its full price (the exact match is trivially the longest
  prefix);
* when several seeded rows are boundary-prefixes of the observed id
  (``gpt-4o-minl-…``: ``gpt-4o`` AND ``gpt-4o-mini``), the LONGEST prefix
  wins (``gpt-4o-mini``'s price, 0.15/0.6 — never the shorter family's);
* the anchor is a version boundary: a prefix followed by an alphanumeric
  character does NOT match (``gpt-4ox9`` is unrelated to ``gpt-4o``);
* models with NO prefix match keep the safe unpriced fallback
  (``cost=0`` + ``priced=false``, AC7).

Data: raw-seeded turns mirroring the capture contract (real HTTP endpoints,
real PostgreSQL — the QA-stack evidence ids are replayed deterministically).
No ``llm_prices`` row is written by the fixtures: everything resolves against
the production startup seed (19 built-in rows + ``o3-mini``).
"""

from datetime import UTC, datetime

import pytest

from tests.behavioral.traces.usage_fixture import seed_usage_turn

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def get_usage(client, headers, **params):
    return await client.get("/api/v1/traces/usage", headers=headers, params=params)


async def get_top(client, headers, **params):
    return await client.get("/api/v1/traces/usage/top", headers=headers, params=params)


def _model_entry(data: dict, name: str) -> dict:
    return next(e for e in data["series"]["by_model"] if e["model_name"] == name)


class TestLongestPrefixPricing:
    async def test_dated_model_id_resolves_base_model_price(self, stack, client, users):
        """BUG-002 core: `gpt-4o-2024-08-06` and `claude-sonnet-4-5-20250929`
        must resolve the seeded base rows' prices — `priced=true`, exact
        USD math, on both the totals and the by-model series (exactly the
        QA-stack evidence ids)."""
        base = datetime.now(UTC).replace(microsecond=0)
        openai_run = await seed_usage_turn(
            stack,
            user_id=users["a"]["id"],
            agent="PrefixDatedOpenAI",
            model="gpt-4o-2024-08-06",
            input_tokens=2_000_000,
            output_tokens=1_000_000,
            base=base,
        )
        anthropic_run = await seed_usage_turn(
            stack,
            user_id=users["a"]["id"],
            agent="PrefixDatedAnthropic",
            model="claude-sonnet-4-5-20250929",
            input_tokens=2_000_000,
            output_tokens=1_000_000,
            base=base,
        )
        window = dict(openai_run["window"])

        data = (await get_usage(client, users["a"]["headers"], **window)).json()
        # gpt-4o (2.5/10): 2M in / 1M out → 5.0 + 10.0 = 15.0
        # claude-sonnet-4-5 (3/15): 2M in / 1M out → 6.0 + 15.0 = 21.0
        assert openai_run["model"] == "gpt-4o-2024-08-06"
        assert anthropic_run["model"] == "claude-sonnet-4-5-20250929"
        assert data["run_count"] == 2 and data["error_count"] == 0
        assert data["total_input_tokens"] == 4_000_000
        assert data["total_output_tokens"] == 2_000_000
        assert data["total_cost_usd"] == pytest.approx(36.0)

        openai_entry = _model_entry(data, "gpt-4o-2024-08-06")
        assert openai_entry["priced"] is True
        assert openai_entry["total_tokens"] == 3_000_000
        assert openai_entry["cost_usd"] == pytest.approx(15.0)

        anthropic_entry = _model_entry(data, "claude-sonnet-4-5-20250929")
        assert anthropic_entry["priced"] is True
        assert anthropic_entry["total_tokens"] == 3_000_000
        assert anthropic_entry["cost_usd"] == pytest.approx(21.0)

        top = (await get_top(client, users["a"]["headers"], **window, limit=5)).json()
        assert {i["turn_id"] for i in top["items"]} == {openai_run["turn_id"], anthropic_run["turn_id"]}
        assert all(i["model_name"] in {"gpt-4o-2024-08-06", "claude-sonnet-4-5-20250929"} for i in top["items"])
        assert [i["cost_usd"] for i in top["items"]] == pytest.approx([21.0, 15.0])

    async def test_exact_id_keeps_its_full_price(self, stack, client, users):
        """An EXACT seeded id is the longest possible prefix — its own price
        (2.5/10 for gpt-4o), not any other row's."""
        fixture = await seed_usage_turn(
            stack,
            user_id=users["a"]["id"],
            agent="PrefixExact",
            model="gpt-4o",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )

        data = (await get_usage(client, users["a"]["headers"], **fixture["window"])).json()
        assert data["total_cost_usd"] == pytest.approx(12.5)
        assert _model_entry(data, "gpt-4o")["priced"] is True
        assert _model_entry(data, "gpt-4o")["cost_usd"] == pytest.approx(12.5)

    async def test_longest_prefix_wins_over_shorter_family_member(self, stack, client, users):
        """`gpt-4o-mini-2024-07-18` is a boundary-prefix of BOTH `gpt-4o`
        (2.5/10) and `gpt-4o-mini` (0.15/0.6) — the LONGEST prefix
        (gpt-4o-mini) must supply the price: 0.75, never 12.5."""
        fixture = await seed_usage_turn(
            stack,
            user_id=users["a"]["id"],
            agent="PrefixLongest",
            model="gpt-4o-mini-2024-07-18",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )

        data = (await get_usage(client, users["a"]["headers"], **fixture["window"])).json()
        assert data["total_cost_usd"] == pytest.approx(0.75), data["series"]["by_model"]
        entry = _model_entry(data, "gpt-4o-mini-2024-07-18")
        assert entry["priced"] is True and entry["cost_usd"] == pytest.approx(0.75)

    async def test_boundary_anchor_never_matches_alphanumeric_overlap(self, stack, client, users):
        """The prefix must be followed by a non-alphanumeric char (a '-' of a
        dated suffix) — `gpt-4ox9` overlaps `gpt-4o` only INSIDE a token, so
        it is NOT priced (no family collision)."""
        fixture = await seed_usage_turn(
            stack,
            user_id=users["a"]["id"],
            agent="PrefixBoundary",
            model="gpt-4ox9",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )

        data = (await get_usage(client, users["a"]["headers"], **fixture["window"])).json()
        assert data["total_cost_usd"] == pytest.approx(0.0)
        assert _model_entry(data, "gpt-4ox9")["priced"] is False

    async def test_genuinely_unknown_model_stays_unpriced(self, stack, client, users):
        """AC7 stays intact: a model with NO prefix match reports cost=0 +
        priced=false — the pipeline never breaks (QA evidence id)."""
        fixture = await seed_usage_turn(
            stack,
            user_id=users["a"]["id"],
            agent="PrefixUnknown",
            model="totally-unknown-model-xyz",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )

        data = (await get_usage(client, users["a"]["headers"], **fixture["window"])).json()
        assert data["total_cost_usd"] == pytest.approx(0.0)
        entry = _model_entry(data, "totally-unknown-model-xyz")
        assert entry["priced"] is False and entry["total_tokens"] == 2_000_000
