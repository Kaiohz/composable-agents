"""Canonical agent scope-slug rule (REQ-002 / REQ-015 / REQ-016 — single owner).

mcp-raganything resolves every request's space from ``X-Composable-Agent``
and validates the value against ``^[a-z0-9][a-z0-9-]{0,99}$`` (REQ-002 —
invalid values are rejected with 422). composable-agents' own agent-name
rules are looser (``^[a-zA-Z0-9][a-zA-Z0-9._-]{0,98}[a-zA-Z0-9]$`` for
top-level agents, and NO pattern at all for subagent names), so a raw agent
name can never be forwarded as a scope value.

This module is the ONE owner of the mapping between an agent/subagent NAME
and its scope SLUG. Every scope consumer uses the same deterministic mapping
so the three surfaces can never disagree (review C-1/I-3):

* the ``X-Composable-Agent`` header value injected at tool-load time;
* the ``agent_slug`` sent to the spaces API (REQ-010) and used by the
  lifecycle cascade (deprovision + memories purge);
* the ``/memories/{slug}/…`` segment bound into the agent-facing memories
  tools (REQ-020).

Mapping (documented contract — ``agent_scope_slug``):

* lowercase; every character outside ``[a-z0-9-]`` (spaces, dots, dashes
  runs, CR/LF, control characters…) becomes a single ``-``;
* leading/trailing dashes are stripped, dash runs collapsed, and the result
  truncated to 100 characters — the output ALWAYS matches
  :data:`AGENT_SCOPE_PATTERN` (no illegal header bytes, no charset 422);
* a name that is already a valid slug passes through UNCHANGED (REQ-016
  acceptance: subagent ``sub-1`` reports header ``sub-1``);
* **reserved default scope** (review iteration-7 N-1): a name whose mapping
  would produce the reserved ``personal`` slug NEVER maps onto it — the
  mapping derives the deterministic, name-specific slug
  ``personal-{sha256(name)[:8]}`` instead. ``personal`` is mcp-raganything's
  default scope for requests WITHOUT the agent header: an agent owning it
  would share the user's personal documents and its deletion would cascade
  ``DELETE /api/v1/spaces/personal`` — wiping the whole personal RAG corpus.
  The fallback keeps such agents working WITHOUT ever touching the personal
  namespace (chosen over an outright rejection so pre-existing agents keep
  working; the lifecycle validation additionally rejects any topology whose
  resolved slug is reserved — defence against mapping drift);
* a degenerate name that maps to nothing (only dashes/underscores/spaces)
  deterministically falls back to ``agent``.

Known, documented consequence of the deterministic mapping: two distinct
names may map onto the same slug (``my_agent`` / ``my-agent`` →
``my-agent``). The lifecycle therefore treats the slug — not the name — as
the space key and rejects (create/update validation) or skips (cascade
ownership guard) any mapping collision with ANOTHER agent, see
:class:`~src.domain.services.agent_rag_spaces.AgentRagSpaces`. The mapping is
idempotent: ``agent_scope_slug(slug) == slug`` for every valid slug
(including the reserved-scope fallback, whose output is itself slug-valid).
"""

import hashlib
import re

#: The scope charset shared with mcp-raganything (REQ-002).
AGENT_SCOPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,99}$")

#: Scope slugs the platform RESERVES (never agent-owned). ``personal`` is
#: mcp-raganything's default scope for requests WITHOUT ``X-Composable-Agent``
#: (review iteration-7 N-1): an agent slug equal to it would share — and, on
#: agent deletion, cascade-delete — the user's whole personal RAG space.
RESERVED_SCOPE_SLUGS = frozenset({"personal"})

#: The memories store prefix of the per-user store namespace (single owner —
#: the application route-validation and the domain lifecycle service both
#: import this constant instead of duplicating the literal).
MEMORIES_PREFIX = "/memories/"

#: Maximum slug length accepted by REQ-002 (the pattern caps at 100 chars).
_MAX_SLUG_LENGTH = 100

#: Deterministic fallback for names that map to nothing (all-invalid names).
_DEGENERATE_SLUG_FALLBACK = "agent"

_NON_SLUG_CHARS = re.compile(r"[^a-z0-9-]+")
_DASH_RUNS = re.compile(r"-{2,}")


def is_reserved_scope_slug(slug: str) -> bool:
    """Whether ``slug`` is reserved by the platform (never agent-owned)."""
    return slug in RESERVED_SCOPE_SLUGS


def is_valid_scope_slug(slug: str) -> bool:
    """Return whether ``slug`` already matches the REQ-002 slug charset."""
    return bool(AGENT_SCOPE_PATTERN.match(slug))


def _reserved_scope_fallback(original_name: str) -> str:
    """Deterministic, name-specific slug for names mapping onto a reserved one.

    Args:
        original_name: The RAW (pre-mapping) name — different spellings map
            onto different fallback slugs, never onto the reserved scope.

    Returns:
        ``{reserved_slug}-{sha256(original_name)[:8]}`` (charset-valid).
    """
    digest = hashlib.sha256((original_name or "").encode("utf-8")).hexdigest()[:8]
    return f"personal-{digest}"


def agent_scope_slug(name: str) -> str:
    """Map an agent/subagent NAME onto its deterministic scope SLUG.

    See the module docstring for the documented contract. Deterministic,
    idempotent, and always slug-valid; NEVER maps onto a reserved scope
    (review N-1) — names that would land on ``personal`` deterministically
    map onto their own ``personal-{hash}`` slug instead.

    Args:
        name: The raw agent or subagent name (coerced — the helper is total).

    Returns:
        The scope slug (pass-through for already-valid slugs).
    """
    raw = str(name or "")
    collapsed = _DASH_RUNS.sub("-", _NON_SLUG_CHARS.sub("-", raw.strip().lower()))
    slug = collapsed.strip("-")[:_MAX_SLUG_LENGTH]
    if not slug:
        return _DEGENERATE_SLUG_FALLBACK
    if is_reserved_scope_slug(slug):
        return _reserved_scope_fallback(raw)
    return slug


def agent_memory_prefix(agent_scope: str | None) -> str:
    """Return the memories store prefix for a bound agent scope.

    Args:
        agent_scope: The bound agent scope NAME (mapped through
            :func:`agent_scope_slug` — bound scopes can therefore never
            resolve to the reserved ``personal`` namespace, review N-1);
            ``None`` returns the flat personal prefix ``/memories/``
            (REQ-019/REQ-020).
    """
    if agent_scope is None:
        return MEMORIES_PREFIX
    return f"{MEMORIES_PREFIX}{agent_scope_slug(agent_scope)}/"
