"""Unit tests for the canonical agent scope-slug mapping (review C-1).

``src.domain.services.agent_scope`` is the ONE owner of the name→slug rule
shared by the header injection, the spaces API and the memories paths.
"""

from src.domain.services.agent_scope import (
    AGENT_SCOPE_PATTERN,
    MEMORIES_PREFIX,
    agent_memory_prefix,
    agent_scope_slug,
    is_reserved_scope_slug,
    is_valid_scope_slug,
)


class TestAgentScopeSlug:
    def test_slug_valid_name_passes_through_unchanged(self):
        """REQ-016 acceptance: a slug-valid name maps to itself."""
        assert agent_scope_slug("sub-1") == "sub-1"
        assert agent_scope_slug("agent-abc123") == "agent-abc123"

    def test_legal_agent_names_map_deterministically(self):
        """Legal agent names (uppercase, dots, underscores) map onto
        slug-valid values instead of being forwarded raw (C-1)."""
        assert agent_scope_slug("MyAgent") == "myagent"
        assert agent_scope_slug("data_analyst") == "data-analyst"
        assert agent_scope_slug("agent.v2") == "agent-v2"

    def test_control_characters_never_reach_the_header_value(self):
        """A hostile subagent name (CR/LF, control bytes) maps to a safe
        slug — no illegal header byte can ever be emitted (C-1)."""
        slug = agent_scope_slug("x\r\nX-Injected: 1")
        assert slug == "x-x-injected-1"
        assert "\r" not in slug and "\n" not in slug and ":" not in slug

    def test_output_always_matches_the_req_002_charset(self):
        """EVERY mapping result satisfies the receiver's charset rule."""
        for raw in ("MyAgent", "data_analyst", "x\r\nX-Injected: 1", "my_agent", "sub-1", "a" * 250, "", "  "):
            slug = agent_scope_slug(raw)
            assert AGENT_SCOPE_PATTERN.match(slug), (raw, slug)

    def test_mapping_is_idempotent(self):
        """Mapping a slug returns the slug (the consumers may map twice)."""
        once = agent_scope_slug("My.Agent-1")
        assert agent_scope_slug(once) == once

    def test_long_names_truncate_to_the_charset_limit(self):
        slug = agent_scope_slug("a" * 250 + "-tail")
        assert len(slug) == 100
        assert AGENT_SCOPE_PATTERN.match(slug)

    def test_degenerate_name_falls_back_deterministically(self):
        """An all-invalid name deterministically maps to the fallback slug."""
        assert agent_scope_slug("---") == "agent"
        assert agent_scope_slug("___") == "agent"

    def test_empty_name_falls_back_deterministically(self):
        assert agent_scope_slug("") == "agent"


class TestScopeHelpers:
    def test_is_valid_scope_slug(self):
        assert is_valid_scope_slug("agent-1")
        assert not is_valid_scope_slug("MyAgent")
        assert not is_valid_scope_slug("-agent")
        assert not is_valid_scope_slug("a" * 101)

    def test_memory_prefix_personal_is_flat(self):
        assert agent_memory_prefix(None) == MEMORIES_PREFIX == "/memories/"

    def test_memory_prefix_maps_the_bound_scope(self):
        """REQ-020 + C-1: the bound scope name maps onto its slug."""
        assert agent_memory_prefix("MyAgent") == "/memories/myagent/"
        assert agent_memory_prefix("sub-1") == "/memories/sub-1/"


class TestReservedPersonalScope:
    """Review N-1: ``personal`` is the RESERVED default scope — the mapping
    must never send a name INTO it."""

    def test_reserved_spellings_never_map_onto_personal(self):
        """The reviewer's 8-spelling sweep: none maps to the reserved slug."""
        for raw in (
            "personal",
            "Personal",
            "PERSONAL",
            "personal.",
            "_personal",
            "-personal",
            "Personal.",
            " personal ",
        ):
            slug = agent_scope_slug(raw)
            assert slug != "personal", raw
            assert not is_reserved_scope_slug(slug), raw
            assert AGENT_SCOPE_PATTERN.match(slug), raw

    def test_reserved_fallback_is_deterministic(self):
        assert agent_scope_slug("personal") == agent_scope_slug("personal")

    def test_reserved_fallback_is_idempotent(self):
        """The constraint from iteration 5: mapping a slug returns the slug —
        including the fallback output."""
        once = agent_scope_slug("Personal")
        assert agent_scope_slug(once) == once

    def test_reserved_fallback_distinct_per_spelling(self):
        slugs = {agent_scope_slug(raw) for raw in ("personal", "Personal", "PERSONAL", "personal ")}
        assert len(slugs) == 4

    def test_reserved_fallback_matches_the_charset(self):
        import hashlib

        slug = agent_scope_slug("personal")
        assert slug == f"personal-{hashlib.sha256(b'personal').hexdigest()[:8]}"

    def test_memory_prefix_cannot_bind_the_reserved_scope(self):
        """Layer (d): a bound agent scope named 'personal' resolves INSIDE a
        distinct agent namespace — never the flat personal namespace."""
        assert agent_memory_prefix("personal") != "/memories/personal/"
        assert agent_memory_prefix("Personal") != "/memories/personal/"
        assert agent_memory_prefix("Personal").startswith("/memories/personal-")

    def test_is_reserved_scope_slug(self):
        assert is_reserved_scope_slug("personal")
        assert not is_reserved_scope_slug("personal-x")
        assert not is_reserved_scope_slug("myagent")
