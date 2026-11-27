"""Behavioral tests — agent lifecycle RAG-space provisioning (spec REQ-018,
plus the built-in assistant's non-provisioning guard REQ-019).

Entry point: the REAL agent CRUD REST routes (``POST/PUT/DELETE
/api/v1/agents``). The external mcp-raganything REST boundary (its idempotent
spaces API, REQ-010 contract) is the ONLY stub (``raganything_stub``): every
provisioning/deprovisioning call the lifecycle use cases make is recorded
(method, slug, forwarded credential, body) at the receiving server.

What is proven behaviorally:

* creating an agent provisions ITS space and EVERY subagent's space through
 ``POST /api/v1/spaces`` with the CREATING USER's forwarded credential
 (REQ-018);
* creation is BEST-EFFORT: an unreachable mcp-raganything never fails agent
 creation and the failure is warned (spec acceptance: creation completes +
 warning logged; the space is created lazily on first use);
* provisioning tolerates an ALREADY-EXISTING space record (the idempotent
 200 answer must be treated as a success) and a per-slug registry rejection;
* deleting an agent deprovisions the spaces (agent + subagents, cascade) and
 PURGES ``/memories/{agent}/`` (+ the subagents') in the running user's
 store — without touching the user's personal memories or another user's
 same-named namespace (REQ-018);
* deleting succeeds even when the spaces API is unavailable (best-effort
 deprovisioning);
* deleting then re-creating the same name starts a FRESH space (documented
 no-rename semantics);
* updating an agent DIFFS the subagent spaces: removed subagents are
 deprovisioned, added ones provisioned, and the agent's own space stays.

RED phase: no provisioning exists yet → every positive provisioning
observation fails for the right reason (the stub records no spaces calls).
The creation-still-succeeds guards are expected to pass today too (they
protect GREEN against hard-fail implementations).
"""

import logging
import socket
import uuid

import pytest

from tests.behavioral.agent_spaces.conftest import (
    create_agent_ok,
    delete_agent,
    lifecycle_yaml,
    store_get,
    store_list,
    store_put,
    update_agent_ok,
)
from tests.behavioral.conftest import poll_condition

pytestmark = pytest.mark.asyncio(loop_scope="session")

BUILTIN_NAME = "composable-ai-assistant"


async def _poll_for(fn, description: str, timeout: float = 15.0):
    """Await ``poll_condition`` and fail with an explicit message."""
    ok, last = await poll_condition(fn, timeout=timeout)
    assert ok, f"{description} (last observed: {last!r})"
    return last


def _dead_base_url() -> str:
    """A port nothing can reach (bound then released) — deterministic refusal."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        probe_address = probe.getsockname()[1]
    return f"http://127.0.0.1:{probe_address}"


class TestCreateProvisionsSpaces:
    async def test_create_provisions_agent_and_subagent_spaces(self, client, users, raganything_stub):
        """REQ-018: ``CreateAgentConfigUseCase`` provisions the agent's space
        AND each subagent's space via idempotent ``POST /api/v1/spaces``,
        forwarding the creating user's credential (McpRegistryClient
        pattern)."""
        unique = uuid.uuid4().hex[:8]
        agent = f"agent-{unique}"
        sub = f"sub-{unique}"

        await create_agent_ok(client, users["a"]["headers"], lifecycle_yaml(agent, [sub]), agent)

        async def _two_posts():
            posts = [call for call in raganything_stub.app.space_calls() if call["method"] == "POST"]
            return len(posts) >= 2, posts

        posts = await _poll_for(_two_posts, "the provisioning POSTs for agent + subagent must reach mcp-raganything")
        slugs = {call["agent_slug"] for call in posts}
        assert slugs == {agent, sub}, f"unexpected provisioned slugs: {posts}"
        credential = users["a"]["key"]
        for call in posts:
            assert call["api_key"] == credential, (
                "the creating user's credential (X-API-Key) must be forwarded to the spaces API"
            )

    async def test_create_is_best_effort_when_raganything_is_unreachable(self, client, users, raganything_stub, caplog):
        """Spec acceptance: with mcp-raganything UNREACHABLE the agent
        creation still SUCCEEDS and a warning is logged (the space is
        created lazily on first use). The seam (env var AND Settings
        attributes) is republished at a dead URL under the SAME contract.

        Review C-3: provisioning runs in a fire-and-forget background task,
        so the warning is polled for AFTER the (instant) 201 — the contract
        (warning logged, agent created) is unchanged, only its timing moved
        to the task boundary."""
        from tests.behavioral.agent_spaces.conftest import publish_raganything_seam, restore_raganything_seam

        dead = _dead_base_url()
        prior = publish_raganything_seam(dead)
        try:
            with caplog.at_level(logging.WARNING):
                unique = uuid.uuid4().hex[:8]
                agent = f"agent-{unique}"
                response = await client.post(
                    "/api/v1/agents",
                    headers=users["a"]["headers"],
                    data={"agent_name": agent},
                    files={"file": ("agent.yaml", lifecycle_yaml(agent, []).encode(), "text/yaml")},
                )
                assert response.status_code == 201, response.text

                async def _warned():
                    warnings = [
                        record
                        for record in caplog.records
                        if record.levelno >= logging.WARNING and agent in record.getMessage()
                    ]
                    return bool(warnings), warnings

                warnings = await _poll_for(
                    _warned,
                    "an unreachable mcp-raganything must be warned (best-effort provisioning) for agent "
                    f"{agent!r}; captured warnings: "
                    f"{[record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]!r}",
                )
                assert warnings, f"an unreachable mcp-raganything must be warned for agent {agent!r}: {warnings!r}"
        finally:
            restore_raganything_seam(prior)

    async def test_create_tolerates_already_provisioned_space(self, client, users, raganything_stub):
        """REQ-010/REQ-018 idempotency: provisioning into a space record that
        ALREADY exists answers 200; the agent creation must treat it as a
        success."""
        unique = uuid.uuid4().hex[:8]
        agent = f"agent-{unique}"
        raganything_stub.reset(seed_spaces=[agent])

        response = await client.post(
            "/api/v1/agents",
            headers=users["a"]["headers"],
            data={"agent_name": agent},
            files={"file": ("agent.yaml", lifecycle_yaml(agent, []).encode(), "text/yaml")},
        )
        assert response.status_code == 201, response.text

        async def _provisioned():
            posts = [call for call in raganything_stub.app.space_calls() if call["method"] == "POST"]
            return agent in {call["agent_slug"] for call in posts}, posts

        await _poll_for(_provisioned, "the provisioning POST for the already-provisioned agent must be attempted")
        statuses = [
            reply["status"] for reply in raganything_stub.app.response_statuses() if reply["agent_slug"] == agent
        ]
        assert statuses[-1] == 200, f"the pre-existing space must answer idempotently (200), got: {statuses}"
        fetched = await client.get(f"/api/v1/agents/{agent}", headers=users["a"]["headers"])
        assert fetched.status_code == 200, "the agent must exist after the idempotent provisioning"

    async def test_create_succeeds_when_space_api_rejects_the_slug(self, client, users, raganything_stub, caplog):
        """Spec edge: agent creation is best-effort even when the spaces API
        REJECTS the slug (422, REQ-010/REQ-002 contract) — the agent must
        still be created. Review iteration-7 fold-in 3: the reject path is
        SCRIPTED on the stub for the agent's MAPPED slug (the C-1 mapping
        makes every provisioned slug charset-valid, so a real 422 is driven
        explicitly instead of relying on an invalid raw name)."""
        from src.domain.services.agent_scope import agent_scope_slug

        unique = uuid.uuid4().hex[:8]
        agent = f"Weird.Name-{unique}"  # valid REST agent name, non-slug charset
        mapped = agent_scope_slug(agent)  # the slug the platform actually sends
        raganything_stub.reset(reject_slugs=[mapped])
        with caplog.at_level(logging.WARNING):
            response = await client.post(
                "/api/v1/agents",
                headers=users["a"]["headers"],
                data={"agent_name": agent},
                files={"file": ("agent.yaml", lifecycle_yaml(agent, []).encode(), "text/yaml")},
            )
            assert response.status_code == 201, response.text

            async def _rejected():
                statuses = [
                    reply for reply in raganything_stub.app.response_statuses() if reply["agent_slug"] == mapped
                ]
                return bool(statuses) and statuses[-1]["status"] == 422, statuses

            statuses = await _poll_for(_rejected, "the spaces API must have rejected the provisioned slug (422)")
            assert statuses[-1]["status"] == 422, statuses

            async def _warned():
                warnings = [
                    record
                    for record in caplog.records
                    if record.levelno >= logging.WARNING and mapped in record.getMessage()
                ]
                return bool(warnings), warnings

            warnings = await _poll_for(
                _warned,
                "a rejected slug must be warned (best-effort provisioning); captured warnings: "
                f"{[record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]!r}",
            )
            assert warnings, "the spaces-API rejection must be warned (best-effort, REQ-018)"

        assert agent not in raganything_stub.app.spaces, "a rejected slug must not be provisioned"
        assert mapped not in raganything_stub.app.spaces, "the 422'd slug must not be provisioned"
        fetched = await client.get(f"/api/v1/agents/{agent}", headers=users["a"]["headers"])
        assert fetched.status_code == 200, "the agent must exist even though provisioning was rejected"


class TestDeleteCascadesSpacesAndMemories:
    async def test_delete_deprovisions_spaces_and_purges_agent_memories(self, client, users, raganything_stub):
        """REQ-018: deleting an agent cascades — the agent's and every
        subagent's space record is deprovisioned (``DELETE
        /api/v1/spaces/{agent_slug}``) and ``/memories/{agent}/`` is purged
        from the RUNNING USER's store, leaving the user's personal memories
        and OTHER users' same-named namespaces untouched."""
        unique = uuid.uuid4().hex[:8]
        agent = f"agent-{unique}"
        sub = f"sub-{unique}"

        await create_agent_ok(client, users["a"]["headers"], lifecycle_yaml(agent, [sub]), agent)

        async def _provisioned():
            posts = [call for call in raganything_stub.app.space_calls() if call["method"] == "POST"]
            return {agent, sub} <= {call["agent_slug"] for call in posts}, posts

        await _poll_for(_provisioned, "the spaces must be provisioned before deletion")

        # Seed memories: the agent's space (two files), the user's personal
        # space, and ANOTHER user's same-named namespace (cross-user isolation).
        headers_a = users["a"]["headers"]
        headers_b = users["b"]["headers"]
        for path, content in (
            (f"/memories/{agent}/notes.md", "AGENT NOTES"),
            (f"/memories/{agent}/prefs.md", "AGENT PREFS"),
        ):
            seeded = await store_put(client, headers_a, path, content)
            assert seeded.status_code == 200, seeded.text
        personal_path = f"/memories/personal-keep-{unique}.md"
        seeded_personal = await store_put(client, headers_a, personal_path, "PERSONAL KEEP")
        assert seeded_personal.status_code == 200, seeded_personal.text
        foreign = await store_put(client, headers_b, f"/memories/{agent}/b-notes.md", "B KEEP")
        assert foreign.status_code == 200, foreign.text

        deleted = await delete_agent(client, headers_a, agent)
        assert deleted.status_code == 204, deleted.text

        async def _deprovisioned():
            deletes = [call for call in raganything_stub.app.space_calls() if call["method"] == "DELETE"]
            return {agent, sub} <= set(raganything_stub.app.call_slugs("DELETE")), deletes

        await _poll_for(_deprovisioned, "the agent and subagent spaces must be deprovisioned on agent deletion")

        async def _purged():
            remaining = await store_list(client, headers_a, f"/memories/{agent}/")
            return not remaining, remaining

        await _poll_for(_purged, "the agent's /memories/{agent}/ namespace must be purged on agent deletion")
        kept = await store_get(client, headers_a, personal_path)
        assert kept.status_code == 200, "the user's personal memories must survive the agent purge"
        foreign_kept = await store_get(client, headers_b, f"/memories/{agent}/b-notes.md")
        assert foreign_kept.status_code == 200, (
            "another user's same-named /memories/{agent}/ namespace must never be purged"
        )

    async def test_delete_succeeds_even_when_space_api_is_failing(self, client, users, raganything_stub):
        """Spec acceptance/edge: deprovisioning is best-effort — an unavailable
        spaces API must never prevent the agent from being deleted."""
        unique = uuid.uuid4().hex[:8]
        agent = f"agent-{unique}"
        await create_agent_ok(client, users["a"]["headers"], lifecycle_yaml(agent, []), agent)

        async def _provisioned():
            posts = [call for call in raganything_stub.app.space_calls() if call["method"] == "POST"]
            return agent in {call["agent_slug"] for call in posts}, posts

        await _poll_for(_provisioned, "the space must be provisioned before deletion")

        raganything_stub.reset(fail=True)  # the spaces API is now unavailable (503 every call)
        deleted = await delete_agent(client, users["a"]["headers"], agent)
        assert deleted.status_code == 204, f"the agent deletion must succeed despite the spaces outage: {deleted.text}"
        fetched = await client.get(f"/api/v1/agents/{agent}", headers=users["a"]["headers"])
        assert fetched.status_code == 404, "the agent must be gone after the successful deletion"

    async def test_delete_then_recreate_starts_a_fresh_space(self, client, users, raganything_stub):
        """Spec edge (documented no-rename semantics): deleting and re-creating
        the same agent name deprovisions, purges, then re-provisions — the
        recreated agent starts with an EMPTY space."""
        unique = uuid.uuid4().hex[:8]
        agent = f"agent-{unique}"
        sub = f"sub-{unique}"
        await create_agent_ok(client, users["a"]["headers"], lifecycle_yaml(agent, [sub]), agent)

        async def _provisioned():
            posts = [call for call in raganything_stub.app.space_calls() if call["method"] == "POST"]
            return {agent, sub} <= {call["agent_slug"] for call in posts}, posts

        await _poll_for(_provisioned, "first creation must provision the agent + subagent spaces")
        seeded = await store_put(client, users["a"]["headers"], f"/memories/{agent}/notes.md", "BEFORE")
        assert seeded.status_code == 200, seeded.text

        deleted = await delete_agent(client, users["a"]["headers"], agent)
        assert deleted.status_code == 204, deleted.text

        async def _deprovisioned():
            deletes = [call for call in raganything_stub.app.space_calls() if call["method"] == "DELETE"]
            return agent in set(raganything_stub.app.call_slugs("DELETE")), deletes

        await _poll_for(_deprovisioned, "the space must be deprovisioned on deletion")

        # Re-create the SAME name: the space starts fresh.
        await create_agent_ok(client, users["a"]["headers"], lifecycle_yaml(agent, [sub]), agent)

        async def _reprovisioned():
            posts = [call for call in raganything_stub.app.space_calls() if call["method"] == "POST"]
            return len([call for call in posts if call["agent_slug"] == agent]) >= 2, posts

        await _poll_for(_reprovisioned, "the recreated agent must provision its space again (fresh, idempotent POST)")

        async def _fresh():
            remaining = await store_list(client, users["a"]["headers"], f"/memories/{agent}/")
            return not remaining, remaining

        await _poll_for(_fresh, "the recreated agent's space must start EMPTY (delete + create = fresh space)")


class TestUpdateDiffsSubagentSpaces:
    async def test_update_removes_and_provisions_subagent_spaces(self, client, users, raganything_stub):
        """REQ-018: ``UpdateAgentConfigUseCase`` diffs the subagent spaces —
        a removed subagent's space is DEPROVISIONED, a newly added one is
        PROVISIONED, and the agent's OWN space is left intact (no
        self-deprovision)."""
        unique = uuid.uuid4().hex[:8]
        agent = f"agent-{unique}"
        kept_sub = f"sub-kept-{unique}"
        removed_sub = f"sub-removed-{uuid.uuid4().hex[:8]}"
        added_sub = f"sub-added-{uuid.uuid4().hex[:8]}"

        await create_agent_ok(client, users["a"]["headers"], lifecycle_yaml(agent, [kept_sub, removed_sub]), agent)

        async def _provisioned():
            posts = [call for call in raganything_stub.app.space_calls() if call["method"] == "POST"]
            return {agent, kept_sub, removed_sub} <= {call["agent_slug"] for call in posts}, posts

        await _poll_for(_provisioned, "the initial subagent spaces must be provisioned before the update")

        await update_agent_ok(client, users["a"]["headers"], lifecycle_yaml(agent, [kept_sub, added_sub]), agent)

        async def _diffed():
            deletes = [call for call in raganything_stub.app.space_calls() if call["method"] == "DELETE"]
            posts = [call for call in raganything_stub.app.space_calls() if call["method"] == "POST"]
            deprovisioned = removed_sub in raganything_stub.app.call_slugs("DELETE")
            provisioned = added_sub in {call["agent_slug"] for call in posts}
            return deprovisioned and provisioned, {"deletes": deletes, "posts": posts}

        observed = await _poll_for(
            _diffed, "the update must deprovision the removed subagent space and provision the added one"
        )

        # The removed subagent's space record is gone; the agent's own space STAYS.
        assert removed_sub not in raganything_stub.app.spaces, (
            f"the removed subagent's space must be deprovisioned: {observed!r}"
        )
        assert agent in raganything_stub.app.spaces, "the agent's own space must NEVER be deprovisioned by an update"
        assert kept_sub in raganything_stub.app.spaces, "a kept subagent's space must be left untouched"
        assert added_sub in raganything_stub.app.spaces, "the added subagent's space must be provisioned"


class TestSpaceOwnershipGuard:
    """Review C-2: the slug namespace is shared between top-level agents and
    subagents — a cascade must never destroy a slug ANOTHER top-level agent
    owns, and the create boundary rejects colliding topologies."""

    async def test_create_rejects_subagent_colliding_with_existing_agent(self, client, users, raganything_stub):
        """C-2 option (a), creation layer: an agent whose subagent name maps
        onto an EXISTING top-level agent's slug is rejected (400) BEFORE any
        persisted write, and NO space is provisioned for the rejected
        topology."""
        existing = f"agent-{uuid.uuid4().hex[:8]}"
        await create_agent_ok(client, users["a"]["headers"], lifecycle_yaml(existing, []), existing)

        colliding = f"agent-{uuid.uuid4().hex[:8]}"
        rejected = await client.post(
            "/api/v1/agents",
            headers=users["a"]["headers"],
            data={"agent_name": colliding},
            files={"file": ("agent.yaml", lifecycle_yaml(colliding, [existing]).encode(), "text/yaml")},
        )
        assert rejected.status_code == 400, f"the colliding topology must be rejected: {rejected.text}"
        fetched = await client.get(f"/api/v1/agents/{colliding}", headers=users["a"]["headers"])
        assert fetched.status_code == 404, "the rejected agent must not have been persisted"
        assert existing not in {call["agent_slug"] for call in raganything_stub.app.call_slugs("DELETE")}, (
            "a rejected creation must never deprovision the existing agent's space"
        )

    async def test_delete_guard_spares_toplevel_owned_slug(self, client, users, raganything_stub):
        """C-2 option (a), cascade layer (legacy-data case): agent B declares
        subagent S, and a top-level agent S is created AFTERWARDS (only the
        subagent-vs-existing-agent direction is validate-able at creation
        time). Deleting B must deprovision/purge B's OWN space but SKIP the
        S slug — the top-level agent S keeps its space and its memories."""
        agent = f"agent-{uuid.uuid4().hex[:8]}"
        shared = f"shared-{uuid.uuid4().hex[:8]}"

        await create_agent_ok(client, users["a"]["headers"], lifecycle_yaml(agent, [shared]), agent)
        await create_agent_ok(client, users["a"]["headers"], lifecycle_yaml(shared, []), shared)

        marker = f"SHARED-KEEP-{uuid.uuid4().hex[:6]}"
        seeded = await store_put(client, users["a"]["headers"], f"/memories/{shared}/keep.md", marker)
        assert seeded.status_code == 200, seeded.text

        deleted = await delete_agent(client, users["a"]["headers"], agent)
        assert deleted.status_code == 204, deleted.text

        async def _own_space_gone():
            owned = raganything_stub.app.call_slugs("DELETE")
            return agent in owned, owned

        await _poll_for(_own_space_gone, "the deleted agent's OWN space must be deprovisioned")
        assert shared not in raganything_stub.app.call_slugs("DELETE"), (
            "a slug owned by the top-level agent 'shared' must never be deprovisioned by another agent's deletion"
        )
        assert shared in raganything_stub.app.spaces, "the top-level agent's space record must survive"
        kept = await store_get(client, users["a"]["headers"], f"/memories/{shared}/keep.md")
        assert kept.status_code == 200 and kept.json()["content"] == marker, (
            "the top-level agent's memories must survive another agent's deletion (C-2)"
        )

    async def test_delete_spares_a_subagent_slug_shared_with_another_agent(self, client, users, raganything_stub):
        """Review C-2b (residual): two agents may each declare their own
        subagent named ``shared`` (creation-time validation deliberately
        accepts it — documented residual). Deleting ONE of them must not
        deprovision ``DELETE /api/v1/spaces/{shared}`` nor purge
        ``/memories/{shared}/`` — the peer agent claims the same slug
        through its own declared subagent, and the ownership guard covers
        declared-subagent slugs as well as top-level names."""
        writer = f"writer-{uuid.uuid4().hex[:8]}"
        editor = f"editor-{uuid.uuid4().hex[:8]}"
        shared_sub = f"researcher-{uuid.uuid4().hex[:8]}"

        await create_agent_ok(client, users["a"]["headers"], lifecycle_yaml(writer, [shared_sub]), writer)
        await create_agent_ok(client, users["a"]["headers"], lifecycle_yaml(editor, [shared_sub]), editor)

        marker = f"PEER-KEEP-{uuid.uuid4().hex[:6]}"
        seeded = await store_put(client, users["a"]["headers"], f"/memories/{shared_sub}/keep.md", marker)
        assert seeded.status_code == 200, seeded.text

        deleted = await delete_agent(client, users["a"]["headers"], writer)
        assert deleted.status_code == 204, deleted.text

        async def _own_space_gone():
            return writer in raganything_stub.app.call_slugs("DELETE"), raganything_stub.app.space_calls()

        await _poll_for(_own_space_gone, "the deleted agent's OWN space must be deprovisioned")

        # The peer-claimed subagent slug is NEVER deprovisioned (skip + warn).
        assert shared_sub not in raganything_stub.app.call_slugs("DELETE"), (
            "a subagent slug claimed by another persisted agent must never be deprovisioned"
        )
        assert shared_sub in raganything_stub.app.spaces, "the peer-claimed subagent space must survive"
        kept = await store_get(client, users["a"]["headers"], f"/memories/{shared_sub}/keep.md")
        assert kept.status_code == 200 and kept.json()["content"] == marker, (
            "the peer agent's memories under the shared subagent slug must survive the deletion"
        )


class TestReservedPersonalScope:
    """Review N-1: ``personal`` is the RESERVED default scope — an agent can
    never own it and a cascade can never destroy it."""

    async def test_agent_named_personal_never_takes_the_personal_scope(self, client, users, raganything_stub):
        """An agent literally named ``personal`` maps onto its own distinct
        slug (never the reserved default): its provisioning POST targets the
        fallback slug, and its deletion cascades the SAME fallback slug —
        the reserved ``personal`` space is never created nor deleted."""
        from src.domain.services.agent_scope import agent_scope_slug

        mapped = agent_scope_slug("personal")  # deterministic fallback slug
        assert mapped != "personal"

        created = await client.post(
            "/api/v1/agents",
            headers=users["a"]["headers"],
            data={"agent_name": "personal"},
            files={"file": ("agent.yaml", lifecycle_yaml("personal", []).encode(), "text/yaml")},
        )
        assert created.status_code == 201, created.text

        async def _fallback_provisioned():
            posts = [call["agent_slug"] for call in raganything_stub.app.space_calls() if call["method"] == "POST"]
            return mapped in posts, posts

        posts = await _poll_for(_fallback_provisioned, "the fallback slug must be provisioned")
        assert "personal" not in posts, "the reserved default scope must never be created by an agent"
        assert "personal" not in raganything_stub.app.spaces, "the reserved space must stay un-owned"

        deleted = await delete_agent(client, users["a"]["headers"], "personal")
        assert deleted.status_code == 204, deleted.text

        async def _fallback_deprovisioned():
            deletes = raganything_stub.app.call_slugs("DELETE")
            return mapped in deletes, deletes

        deletes = await _poll_for(_fallback_deprovisioned, "the fallback slug must be deprovisioned on deletion")
        assert "personal" not in deletes, "the user's personal RAG space must never be cascade-deleted"


class TestBuiltinAssistantIsNeverProvisioned:
    async def test_creating_the_reserved_builtin_name_conflicts_without_provisioning(
        self, client, users, raganything_stub
    ):
        """REQ-019: the built-in assistant's name is reserved — a custom
        creation conflicts (409) and must NEVER trigger space provisioning."""
        response = await client.post(
            "/api/v1/agents",
            headers=users["a"]["headers"],
            data={"agent_name": BUILTIN_NAME},
            files={"file": ("agent.yaml", lifecycle_yaml(BUILTIN_NAME, []).encode(), "text/yaml")},
        )
        assert response.status_code == 409, response.text
        assert raganything_stub.app.space_calls() == [], "the reserved built-in name must never be provisioned"

    async def test_deleting_the_builtin_is_refused_without_provisioning(self, client, users, raganything_stub):
        """REQ-019: the built-in assistant is read-only — deleting it is
        refused (403) and must NEVER trigger a space deprovisioning."""
        deleted = await delete_agent(client, users["a"]["headers"], BUILTIN_NAME)
        assert deleted.status_code == 403, deleted.text
        assert not raganything_stub.app.call_slugs("DELETE"), "the built-in assistant must never be deprovisioned"
