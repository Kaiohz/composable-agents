"""Unit tests for the agent RAG-space lifecycle domain service (REQ-018 + C-2/C-2b + C-3 + N-1).

Covers the branch surfaces the behavioral suite reaches only partially:

* ``validate_space_ownership`` — the C-2 create/update boundary (topology
  slug collisions rejected BEFORE any persisted write; best-effort on an
  unreadable ownership set) and the N-1 reserved-scope drift guard;
* the cascade ownership guard (``_cascadable_slugs`` through the awaited
  deprovision/purge entry points) — a slug owned by another agent (top-level
  or DECLARED subagent, review C-2b) is never destroyed, the reserved
  ``personal`` scope is never cascaded (N-1), and the guard fails safe when
  the ownership set cannot be resolved;
* the fire-and-forget scheduling (C-3) and the shutdown drain.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.domain.entities.agent_config_metadata import AgentConfigMetadata
from src.domain.errors.config import ConfigError
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_config_repository import AgentConfigRepository
from src.domain.ports.agent_config_store import AgentConfigStore
from src.domain.ports.rag_space_provisioner import RagSpaceProvisioner
from src.domain.ports.store_file_repository import StoreFileRepository
from src.domain.services.agent_rag_spaces import (
    AgentRagSpaces,
    drain_pending_lifecycle_tasks,
)
from src.infrastructure.yaml_config.adapter import YamlAgentConfigLoader


def _metadata(name: str) -> AgentConfigMetadata:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return AgentConfigMetadata(name=name, model="test-model", minio_path=f"{name}.yaml", created_at=now, updated_at=now)


class RecordingProvisioner(RagSpaceProvisioner):
    """Records provision/deprovision slugs; optionally fails on demand."""

    def __init__(self) -> None:
        self.provisioned: list[str] = []
        self.deprovisioned: list[str] = []
        self.provision_error: Exception | None = None
        self.deprovision_error: Exception | None = None

    async def provision(self, agent_slug: str) -> None:
        self.provisioned.append(agent_slug)
        if self.provision_error is not None:
            raise self.provision_error

    async def deprovision(self, agent_slug: str) -> None:
        self.deprovisioned.append(agent_slug)
        if self.deprovision_error is not None:
            raise self.deprovision_error


class RecordingStoreFiles(StoreFileRepository):
    """Records list/delete; serves a FIXED, prefix-respecting listing."""

    def __init__(self, listing: list[str] | None = None) -> None:
        self.listing = listing if listing is not None else []
        self.deleted: list[str] = []

    async def list_files(self, prefix: str) -> list[str]:
        return [path for path in self.listing if path.startswith(prefix)]

    async def list_files_with_preview(self, prefix: str, preview_chars: int):  # noqa: ANN201 — not under test
        raise NotImplementedError

    async def get_file(self, path: str) -> str | None:
        raise NotImplementedError

    async def put_file(self, path: str, content: str) -> None:
        raise NotImplementedError

    async def delete_file(self, path: str) -> None:
        self.deleted.append(path)


def _logged(caplog, member: LogMessage) -> bool:
    """Whether a record carrying ``member``'s static message prefix was logged."""
    prefix = member.value.split("%s", 1)[0]
    return any(prefix in record.getMessage() for record in caplog.records)


def _repo(names: list[str], *, fail: bool = False) -> AgentConfigRepository:
    repo = AsyncMock(spec=AgentConfigRepository)
    if fail:
        repo.list_all.side_effect = RuntimeError("connection lost")
    else:
        repo.list_all.return_value = [_metadata(name) for name in names]
    return repo


class TestValidateSpaceOwnership:
    async def test_without_repository_nothing_is_rejected(self):
        spaces = AgentRagSpaces(RecordingProvisioner())
        await spaces.validate_space_ownership("agent-a", ["agent-b", "agent-c"])  # no raise

    async def test_two_subagents_mapping_to_one_slug_are_rejected(self):
        spaces = AgentRagSpaces(RecordingProvisioner(), config_repository=_repo([]))
        with pytest.raises(ConfigError, match=r"already claimed by 'a_b'"):
            await spaces.validate_space_ownership("top", ["a_b", "a-b"])

    async def test_subagent_sharing_the_parent_slug_is_allowed(self):
        spaces = AgentRagSpaces(RecordingProvisioner(), config_repository=_repo([]))
        await spaces.validate_space_ownership("my.agent", ["my-agent"])  # same topology slug

    async def test_unresolvable_ownership_set_fails_open_with_warning(self, caplog):
        # Validation is best-effort: an unreadable list defers to the cascade
        # guard (fail-safe there) instead of bricking agent creation.
        spaces = AgentRagSpaces(RecordingProvisioner(), config_repository=_repo([], fail=True))
        await spaces.validate_space_ownership("agent-a", ["agent-b"])  # no raise
        assert _logged(caplog, LogMessage.RAG_SPACE_OWNERSHIP_UNRESOLVED)

    async def test_legacy_own_slug_collision_downgrades_to_a_warning(self, caplog):
        """Review I-6(a) / iter-5 I-3: an agent's own name cannot change on
        update, so an own-slug mapping collision ALWAYS predates the update
        (a legacy pair) — it downgrades to a WARNING so both agents stay
        editable; the cascade ownership guard still protects both spaces."""
        # 'my_agent' maps onto the slug of the EXISTING agent 'my-agent'.
        spaces = AgentRagSpaces(RecordingProvisioner(), config_repository=_repo(["my-agent"]))
        await spaces.validate_space_ownership("My_Agent", [])  # no raise
        assert _logged(caplog, LogMessage.RAG_SPACE_SLUG_COLLISION_LEGACY)

    async def test_subagent_slug_colliding_with_existing_agent_is_rejected(self):
        spaces = AgentRagSpaces(RecordingProvisioner(), config_repository=_repo(["Coll.X"]))
        with pytest.raises(ConfigError, match="already claimed by 'Coll.X'"):
            await spaces.validate_space_ownership("top-agent", ["coll-x"])

    async def test_the_updated_agent_itself_is_ignored(self):
        # The updating agent's own row must not collide against itself.
        spaces = AgentRagSpaces(RecordingProvisioner(), config_repository=_repo(["top-agent"]))
        await spaces.validate_space_ownership("top-agent", ["fresh-sub"])  # no raise


class TestCascadeOwnershipGuard:
    async def test_without_repository_only_the_own_slug_is_cascaded(self):
        """Fail-safe (C-2): no ownership data ⇒ never destroy a non-own slug."""
        provisioner = RecordingProvisioner()
        spaces = AgentRagSpaces(provisioner)
        await spaces.deprovision_for_agent("agent-a", ["agent-b", "agent-c"])
        assert provisioner.deprovisioned == ["agent-a"]

    async def test_unresolvable_ownership_set_fails_safe(self, caplog):
        provisioner = RecordingProvisioner()
        spaces = AgentRagSpaces(provisioner, config_repository=_repo([], fail=True))
        await spaces.deprovision_for_agent("agent-a", ["agent-b"])
        assert provisioner.deprovisioned == ["agent-a"]
        assert _logged(caplog, LogMessage.RAG_SPACE_OWNERSHIP_UNRESOLVED)

    async def test_owned_by_another_agent_slugs_are_skipped_with_warning(self, caplog):
        """Deleting 'agent-a' whose subagent 'agent-b' collides with the
        top-level agent 'agent-b' cascades ONLY 'agent-a' (C-2)."""
        provisioner = RecordingProvisioner()
        spaces = AgentRagSpaces(provisioner, config_repository=_repo(["agent-a", "agent-b"]))
        await spaces.deprovision_for_agent("agent-a", ["agent-b"])
        assert provisioner.deprovisioned == ["agent-a"]
        assert _logged(caplog, LogMessage.RAG_SPACE_OWNERSHIP_SKIPPED)

    async def test_purge_purges_only_the_cascadable_prefixes(self):
        store_files = RecordingStoreFiles(
            ["/memories/agent-a/f1.md", "/memories/agent-b/keep.md", "/memories/agent-a/f2.md"]
        )
        spaces = AgentRagSpaces(RecordingProvisioner(), store_files, config_repository=_repo(["agent-b"]))
        await spaces.purge_agent_memories("agent-a", ["agent-b"])
        assert {"f1.md", "f2.md"} == {path.rsplit("/", 1)[-1] for path in store_files.deleted}
        assert all("/memories/agent-b/" not in path for path in store_files.deleted)

    async def test_purge_listing_failure_warns_and_deletes_nothing(self, caplog):
        store_files = RecordingStoreFiles(["/memories/agent-a/f1.md"])
        store_files.list_files = AsyncMock(side_effect=RuntimeError("store down"))
        spaces = AgentRagSpaces(RecordingProvisioner(), store_files)
        await spaces.purge_agent_memories("agent-a", [])
        assert store_files.deleted == []
        assert _logged(caplog, LogMessage.AGENT_MEMORIES_PURGE_FAILED)

    async def test_purge_delete_failures_warn_per_path(self, caplog):
        store_files = RecordingStoreFiles()
        store_files.list_files = AsyncMock(return_value=["/memories/agent-a/boom.md"])
        store_files.delete_file = AsyncMock(side_effect=RuntimeError("delete refused"))
        spaces = AgentRagSpaces(RecordingProvisioner(), store_files)
        await spaces.purge_agent_memories("agent-a", [])
        assert _logged(caplog, LogMessage.AGENT_MEMORIES_PURGE_FAILED)

    async def test_diff_skips_owned_removed_subagent_and_provisions_added(self):
        """Removing a subagent whose slug belongs to the updating agent's OWN
        slug must not deprovision that slug either."""
        provisioner = RecordingProvisioner()
        spaces = AgentRagSpaces(provisioner, config_repository=_repo(["my.agent"]))
        await spaces.diff_subagent_spaces("my.agent", ["my-agent", "gone-sub"], ["fresh-sub"])
        assert provisioner.deprovisioned == ["gone-sub"]  # 'my-agent' == own slug: kept
        assert provisioner.provisioned == ["fresh-sub"]

    async def test_per_slug_provision_failures_are_warned_not_raised(self, caplog):
        provisioner = RecordingProvisioner()
        provisioner.provision_error = RuntimeError("spaces api down")
        spaces = AgentRagSpaces(provisioner)
        await spaces.provision_for_agent("agent-a", ["agent-b"])  # no raise
        assert _logged(caplog, LogMessage.RAG_SPACE_PROVISION_FAILED)


class TestFireAndForgetScheduling:
    async def test_schedule_provision_runs_to_completion(self):
        provisioner = RecordingProvisioner()
        spaces = AgentRagSpaces(provisioner)
        spaces.schedule_provision_for_agent("agent-a", ["agent-b"])
        drained = await drain_pending_lifecycle_tasks()
        assert drained >= 1
        assert provisioner.provisioned == ["agent-a", "agent-b"]

    async def test_schedule_deletion_cascade_runs_guarded_steps(self):
        """The delete cascade resolves ownership ONCE, then deprovisions and
        purges the cascadable slugs in the same background task."""
        provisioner = RecordingProvisioner()
        store_files = RecordingStoreFiles(["/memories/agent-a/f1.md"])
        repo = _repo(["agent-a", "agent-b"])
        spaces = AgentRagSpaces(provisioner, store_files, repo)
        spaces.schedule_deletion_cascade("agent-a", ["agent-b"])
        await drain_pending_lifecycle_tasks()
        assert provisioner.deprovisioned == ["agent-a"]
        assert [path.rsplit("/", 1)[-1] for path in store_files.deleted] == ["f1.md"]
        assert repo.list_all.await_count == 1

    async def test_provision_proceeds_when_agent_still_exists(self):
        """Normal path: the agent row exists → the provisioning POSTs happen."""
        provisioner = RecordingProvisioner()
        repo = _repo(["agent-a"])
        repo.exists.return_value = True
        spaces = AgentRagSpaces(provisioner, config_repository=repo)
        spaces.schedule_provision_for_agent("agent-a", ["agent-b"])
        await drain_pending_lifecycle_tasks()
        assert provisioner.provisioned == ["agent-a", "agent-b"]

    async def test_provision_skipped_when_agent_deleted_before_the_task_runs(self, caplog):
        """Review I-6(b) / iter-5 I-5 (create→delete race): a fire-and-forget
        provisioning task whose agent no longer exists (an immediate delete
        won the race) must NOT leave an orphaned space — the task verifies
        the agent's existence in the caller-captured context and skips."""
        provisioner = RecordingProvisioner()
        repo = _repo([])
        repo.exists.return_value = False
        spaces = AgentRagSpaces(provisioner, config_repository=repo)
        spaces.schedule_provision_for_agent("agent-a", ["agent-b"])
        await drain_pending_lifecycle_tasks()
        assert provisioner.provisioned == [], "no orphan space from a late provisioning POST"
        assert _logged(caplog, LogMessage.RAG_SPACE_PROVISION_SKIPPED_GONE)

    async def test_drain_cancels_tasks_that_outlive_the_timeout(self, monkeypatch):
        """Shutdown does not hang on a stuck task: it is cancelled (C-3)."""
        stuck = asyncio.Event()
        provisioner = RecordingProvisioner()
        spaces = AgentRagSpaces(provisioner)
        spaces.schedule_provision_for_agent("agent-a", [])

        # Sonar S7483: the drain bound is the module constant — the spec
        # patches it to keep the cancellation path fast (behavioral
        # semantics unchanged: tasks outliving the bound are cancelled).
        monkeypatch.setattr("src.domain.services.agent_rag_spaces._DRAIN_TIMEOUT_SECONDS", 0.05)
        blocker = asyncio.create_task(stuck.wait())
        from src.domain.services.agent_rag_spaces import _PENDING_LIFECYCLE_TASKS

        _PENDING_LIFECYCLE_TASKS.add(blocker)
        try:
            drained = await drain_pending_lifecycle_tasks()
            assert drained <= 1  # the provisioning task may finish; the blocker must not block the drain
            assert blocker.cancelled() or stuck.is_set()
        finally:
            if not blocker.done():
                blocker.cancel()
        assert provisioner.provisioned == ["agent-a"]


def _agent_yaml(name: str, subagents: list[str]) -> str:
    """A minimal valid agent YAML declaring the given subagent names."""
    lines = [f"name: {name}", "model: test-model", 'system_prompt: "Topology fixture."']
    if subagents:
        lines.append("subagents:")
        for sub in subagents:
            lines += [f"  - name: {sub}", '    description: "Fixture sub-agent."']
    return "\n".join(lines) + "\n"


def _yaml_store(yamls: dict[str, str]) -> AgentConfigStore:
    store = AsyncMock(spec=AgentConfigStore)
    store.get.side_effect = lambda name: yamls[name]
    return store


class TestPeerDeclaredSubagentOwnership:
    """Review C-2b: the owned set extends to the subagent slugs each peer's
    stored YAML declares."""

    async def test_peer_declared_subagent_slug_is_protected(self):
        writer_yaml = _agent_yaml("writer-a1", ["researcher-x"])
        editor_yaml = _agent_yaml("editor-b2", ["researcher-x"])
        provisioner = RecordingProvisioner()
        from src.infrastructure.agent_topology.adapter import StoreAgentTopologyResolver

        spaces = AgentRagSpaces(
            provisioner,
            config_repository=_repo(["writer-a1", "editor-b2"]),
            topologies=StoreAgentTopologyResolver(
                _yaml_store({"writer-a1": writer_yaml, "editor-b2": editor_yaml}),
                YamlAgentConfigLoader(),
            ),
        )
        await spaces.deprovision_for_agent("writer-a1", ["researcher-x"])
        assert provisioner.deprovisioned == ["writer-a1"], (
            "the subagent slug claimed by the PEER editor's declared subagent must be skipped"
        )

    async def test_without_topology_resolver_the_peer_subagents_are_not_protected(self):
        """Documented degradation: a ``None`` resolver keeps the (weaker)
        pre-C-2b behavior — the composition root always wires the resolver."""
        provisioner = RecordingProvisioner()
        spaces = AgentRagSpaces(provisioner, config_repository=_repo(["editor-b2"]))
        await spaces.deprovision_for_agent("writer-a1", ["researcher-x"])
        assert provisioner.deprovisioned == ["writer-a1", "researcher-x"]

    async def test_unreadable_peer_yaml_fails_safe_to_own_slug(self, caplog):
        """Security review F-1: an individually unreadable peer YAML makes
        the WHOLE owned set unresolvable — the cascade narrows to the agent's
        OWN slug only (never the peer's, never a shared subagent slug)."""
        yamls = {
            "writer-a1": _agent_yaml("writer-a1", []),
            "editor-b2": _agent_yaml("editor-b2", ["researcher-x"]),
            "broken-c3": "name: [uncloseable",
        }
        provisioner = RecordingProvisioner()
        from src.infrastructure.agent_topology.adapter import StoreAgentTopologyResolver

        spaces = AgentRagSpaces(
            provisioner,
            config_repository=_repo(["writer-a1", "editor-b2", "broken-c3"]),
            topologies=StoreAgentTopologyResolver(_yaml_store(yamls), YamlAgentConfigLoader()),
        )
        await spaces.deprovision_for_agent("writer-a1", ["researcher-x", "broken-c3", "other-sub"])
        assert provisioner.deprovisioned == ["writer-a1"], (
            "the unreadable peer must invalidate the whole ownership set: ONLY the agent's own slug "
            "is cascaded — the peer's top-level slug, its declared subagent slugs and any other slug "
            "are never destroyed while the owned set cannot be proven"
        )
        assert _logged(caplog, LogMessage.AGENT_TOPOLOGY_UNRESOLVED)

    async def test_oversized_peer_yaml_fails_safe_to_own_slug(self, caplog):
        """Security review NV-1: a stored definition exceeding the 1 MiB
        write cap never reaches the parser — the peer topology is treated as
        unresolvable and the cascade fails safe to the own slug."""
        oversized = "x: " + "a" * (1024 * 1024 + 100)  # > MAX_YAML_SIZE bytes
        yamls = {
            "writer-a1": _agent_yaml("writer-a1", []),
            "editor-b2": _agent_yaml("editor-b2", ["researcher-x"]),
            "bloated-c4": oversized,
        }
        provisioner = RecordingProvisioner()
        from src.infrastructure.agent_topology.adapter import StoreAgentTopologyResolver

        spaces = AgentRagSpaces(
            provisioner,
            config_repository=_repo(["writer-a1", "editor-b2", "bloated-c4"]),
            topologies=StoreAgentTopologyResolver(_yaml_store(yamls), YamlAgentConfigLoader()),
        )
        await spaces.deprovision_for_agent("writer-a1", ["researcher-x", "huge-sub"])
        assert provisioner.deprovisioned == ["writer-a1"], (
            "an oversized stored definition must make the owned set unresolvable (read-side bound, NV-1)"
        )
        assert _logged(caplog, LogMessage.AGENT_TOPOLOGY_YAML_OVERSIZED)

    async def test_foreign_definition_mismatch_fails_safe_to_own_slug(self, caplog):
        """Security review F-2: the fetched stored definition must
        self-identify as the requested agent (name-keyed store) — a
        mismatching/foreign definition never influences the owned set and
        makes the whole cascade fail safe to the own slug."""
        yamls = {
            "writer-a1": _agent_yaml("writer-a1", []),
            "editor-b2": _agent_yaml("someone-else", ["researcher-x"]),  # root name ≠ requested
        }
        provisioner = RecordingProvisioner()
        from src.infrastructure.agent_topology.adapter import StoreAgentTopologyResolver

        spaces = AgentRagSpaces(
            provisioner,
            config_repository=_repo(["writer-a1", "editor-b2"]),
            topologies=StoreAgentTopologyResolver(_yaml_store(yamls), YamlAgentConfigLoader()),
        )
        await spaces.deprovision_for_agent("writer-a1", ["researcher-x"])
        assert provisioner.deprovisioned == ["writer-a1"], (
            "a foreign definition (root name mismatch) must make the owned set unresolvable, "
            "never silently narrow the peer's protection"
        )
        assert _logged(caplog, LogMessage.AGENT_TOPOLOGY_FOREIGN_DEFINITION)

    async def test_unexpected_peer_resolution_failure_fails_safe_to_own_slug(self, caplog):
        """Review I-3: the fanned-out outcomes are INSPECTED, never swallowed
        into a success — an unexpected (non-sentinel) peer failure also makes
        the whole owned set unresolvable (own slug only)."""
        from src.domain.ports.agent_topology_resolver import AgentTopologyResolver

        class _ExplodingResolver(AgentTopologyResolver):
            async def subagent_names(self, agent_name: str) -> list[str]:
                if agent_name == "editor-b2":
                    raise RuntimeError("boom")
                return []

        provisioner = RecordingProvisioner()
        spaces = AgentRagSpaces(
            provisioner,
            config_repository=_repo(["writer-a1", "editor-b2"]),
            topologies=_ExplodingResolver(),
        )
        await spaces.deprovision_for_agent("writer-a1", ["researcher-x", "editor-b2"])
        assert provisioner.deprovisioned == ["writer-a1"], (
            "an unexpected peer failure must invalidate the whole owned set (fan-out inspection)"
        )
        assert _logged(caplog, LogMessage.RAG_SPACE_OWNERSHIP_UNRESOLVED)


class TestReservedPersonalScopeGuard:
    """Review N-1 — layers (b) and (c) of the reserved default scope."""

    async def test_cascade_never_destroys_the_reserved_scope(self, caplog):
        """Layer (c): the guard refuses to deprovision/purge ``personal`` —
        white-box via the slug list (the branch is a mapping-drift guard)."""
        provisioner = RecordingProvisioner()
        spaces = AgentRagSpaces(provisioner, config_repository=_repo(["agent-a"]))
        cascadable = await spaces._cascadable_slugs("agent-a", ["agent-a", "personal", "agent-b"])
        assert cascadable == ["agent-a", "agent-b"]
        assert _logged(caplog, LogMessage.RAG_SPACE_RESERVED_SLUG_SKIPPED)

    async def test_validation_rejects_a_topology_resolving_to_the_reserved_scope(self):
        """Layer (b): the create/update boundary rejects a topology whose
        resolved slug is the reserved scope. Driven through a patched
        mapping — with the documented mapping, ``agent_scope_slug`` derives a
        distinct slug for names that would map onto ``personal``, so this
        branch exists to catch mapping drift (review N-1)."""
        spaces = AgentRagSpaces(RecordingProvisioner(), config_repository=_repo([]))
        with patch("src.domain.services.agent_rag_spaces.agent_scope_slug", return_value="personal"):
            with pytest.raises(ConfigError, match="reserved"):
                await spaces.validate_space_ownership("agent-a", [])
            with pytest.raises(ConfigError, match="reserved"):
                await spaces.validate_space_ownership("agent-a", ["sub-a"])

    async def test_mapped_personal_named_agent_is_never_rejected(self):
        """Layer (a) coherence: an agent literally named ``personal`` maps
        onto its own distinct fallback slug and passes validation — its
        space can never be the reserved scope."""
        spaces = AgentRagSpaces(RecordingProvisioner(), config_repository=_repo([]))
        await spaces.validate_space_ownership("personal", [])  # no raise
