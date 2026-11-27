"""Agent RAG-space + agent-memories lifecycle policy (REQ-018 + C-2 + C-3).

The agent-lifecycle use cases (create / update / delete) share the SAME
business rules about WHICH spaces belong to an agent and HOW failures are
absorbed:

* an agent's space set = the canonical scope slugs (see
  :mod:`src.domain.services.agent_scope` — ONE name→slug owner shared with
  the header injection and the memories closure) of the agent itself and of
  every subagent (an ``agent_ref`` subagent owns a space under ITS own
  name's slug, not the referenced agent's — REQ-016 per-subagent space);
* provisioning/deprovisioning is IDEMPOTENT and BEST-EFFORT and NON-BLOCKING
  (REQ-018 + review C-3): the lifecycle is executed in fire-and-forget
  background tasks (the request's contextvars are captured at task creation,
  so the forwarded credential and the store namespace stay the caller's),
  slugs are fanned out concurrently, and ANY failure — unreachable or
  rejecting spaces API, hung connection (short dedicated timeout) — only
  warns and never fails the agent operation;
* C-2 (intra-user data-loss guard, option (a)): the slug namespace is shared
  between top-level agents AND subagents, so a cascade must NEVER destroy a
  slug that another agent of the same user owns — through its TOP-LEVEL name
  or through its DECLARED subagent names (review iteration-7 C-2b). Two
  layers:
  - validation at create/update time (fail-fast): a topology whose mapped
    slugs collide with ANOTHER existing agent's TOP-LEVEL slug is rejected
    (400). Creation-time validation deliberately does NOT reject subagent↔
    subagent collisions (documented residual): two agents may each declare
    their own subagent ``researcher`` and share that space by the flat
    namespace design — the cascade guard below protects it instead;
  - an ownership guard at delete/update-cascade time: the owned set covers
    every persisted agent's top-level slug AND the subagent slugs declared
    in its stored YAML (name-keyed store reads via the
    :class:`~src.domain.ports.agent_topology_resolver.AgentTopologyResolver`
    port — see the port contract for the boundary: the store is global by
    agent NAME, not per user); a slug claimed by a PEER is SKIPPED
    (deprovision + memories purge) with a warning. When the ownership set
    cannot be resolved the guard FAILS SAFE: only the agent's OWN slug may
    ever be destroyed — and per review F-1 a peer definition that cannot be
    read, parsed, size-checked or self-identified raises
    :class:`AgentTopologyUnresolvedError`, which makes the WHOLE set
    unresolvable (a degraded peer can never silently drop its own claimed
    slugs from the protection);
* N-1 (reserved default scope): ``personal`` is mcp-raganything's default
  scope for requests WITHOUT the agent header — an agent can never own it
  and a cascade can never destroy it (see
  :mod:`src.domain.services.agent_scope` for the mapping rule that derives a
  distinct slug for names that would map onto the reserved one);
* agent deletion ALSO purges the agent's ``/memories/{slug}/`` files (plus
  each cascaded subagent's) from the RUNNING user's store — best-effort too
  (the purge is bounded by the store listing cap, same as the pre-existing
  thread cascade).

Pure domain class over the outbound ports (no transport knowledge); the use
cases stay thin orchestrators.
"""

import asyncio
import logging
from collections.abc import Iterable
from typing import cast

from src.domain.errors.base import DomainError
from src.domain.errors.config import ConfigError
from src.domain.errors.messages import ErrorMessage
from src.domain.logging.messages import LogMessage
from src.domain.ports.agent_config_repository import AgentConfigRepository
from src.domain.ports.agent_topology_resolver import AgentTopologyResolver, AgentTopologyUnresolvedError
from src.domain.ports.rag_space_provisioner import RagSpaceProvisioner
from src.domain.ports.store_file_repository import StoreFileRepository
from src.domain.services.agent_scope import MEMORIES_PREFIX, agent_scope_slug, is_reserved_scope_slug

logger = logging.getLogger(__name__)

#: Strong references to the pending fire-and-forget lifecycle tasks (C-3):
#: ``asyncio.create_task`` results are otherwise garbage-collectable
#: mid-flight; the set also gives the shutdown path a drain handle.
_PENDING_LIFECYCLE_TASKS: set[asyncio.Task] = set()

#: How long the shutdown drain lets pending lifecycle tasks finish before
#: cancelling them (:func:`drain_pending_lifecycle_tasks` — the timeout
#: policy is owned by the module constant, per Sonar S7483: the caller never
#: passes a timeout to an async function).
_DRAIN_TIMEOUT_SECONDS: float = 10.0


def agent_space_slugs(agent_name: str, subagent_names: Iterable[str]) -> list[str]:
    """Return the deduplicated, mapped space slugs of an agent topology.

    Args:
        agent_name: The agent's own name (mapped through
            :func:`agent_scope_slug` — the ONE canonical slug owner).
        subagent_names: Every subagent's own name (mapped the same way,
            order preserved).

    Returns:
        The ordered, deduplicated slugs: the agent first, then the subagents.
    """
    return list(dict.fromkeys([agent_scope_slug(agent_name), *(agent_scope_slug(name) for name in subagent_names)]))


def safe_reason(exc: Exception) -> str:
    """Return the sanitized reason of a provisioning failure for warning logs.

    The adapter's domain errors already carry status-only details (the
    upstream response body is never echoed); unexpected exceptions are
    reduced to their exception-type name so no credential/URL detail leaks.

    Args:
        exc: The caught exception.

    Returns:
        The safe log reason.
    """
    if isinstance(exc, DomainError):
        return str(exc)
    return type(exc).__name__


def _mapped_topology_slugs(agent_name: str, subagent_names: Iterable[str]) -> dict[str, str]:
    """Map a topology onto slugs, rejecting the immediate violations.

    Args:
        agent_name: The agent's own name.
        subagent_names: Every subagent's own name.

    Returns:
        The ordered mapping slug → first subagent name claiming it (the
        agent's own slug may appear when a subagent shares it — same
        topology, one space).

    Raises:
        ConfigError: N-1 (b) — a topology element (own name or subagent)
            resolves onto the RESERVED ``personal`` scope (a mapping-drift
            guard: :func:`agent_scope_slug` derives distinct slugs for names
            that would map onto the reserved one); C-2 — two DIFFERENT
            subagent names of this topology map onto the same slug.
    """
    own = agent_scope_slug(agent_name)
    if is_reserved_scope_slug(own):
        raise ConfigError(ErrorMessage.AGENT_SPACE_SLUG_RESERVED.format(slug=own, name=agent_name))
    mapped: dict[str, str] = {}
    for sub_name in subagent_names:
        slug = agent_scope_slug(sub_name)
        if is_reserved_scope_slug(slug):
            raise ConfigError(ErrorMessage.AGENT_SPACE_SLUG_RESERVED.format(slug=slug, name=sub_name))
        if slug != own and slug in mapped:
            raise ConfigError(
                ErrorMessage.AGENT_SPACE_SLUG_COLLISION.format(slug=slug, name=sub_name, other=mapped[slug])
            )
        mapped.setdefault(slug, sub_name)
    return mapped


class AgentRagSpaces:
    """Best-effort, non-blocking RAG-space and agent-memories lifecycle (REQ-018)."""

    def __init__(
        self,
        provisioner: RagSpaceProvisioner | None,
        store_files: StoreFileRepository | None = None,
        config_repository: AgentConfigRepository | None = None,
        topologies: AgentTopologyResolver | None = None,
    ) -> None:
        """Wire the outbound ports.

        Args:
            provisioner: The spaces-API client; ``None`` disables RAG-space
                provisioning rather than failing the lifecycle (the space is
                created lazily on first use inside mcp-raganything).
            store_files: The caller's store file repository used for the
                deletion-time memories purge; ``None`` skips the purge.
            config_repository: The owner-scoped agent metadata repository,
                powering the C-2 ownership validation (create/update) and the
                cascade ownership guard (delete/update diff). ``None``
                degrades fail-safe (see :meth:`_cascadable_slugs`).
            topologies: The peer-topology resolver (C-2b) — extends the
                owned set with the subagent slugs each persisted agent's
                stored YAML declares; ``None`` degrades to top-level slugs
                only (the subagent↔subagent direction is then unprotected —
                the composition root always wires it).
        """
        self._provisioner = provisioner
        self._store_files = store_files
        self._config_repository = config_repository
        self._topologies = topologies

    # ------------------------------------------------------------------
    # validation (fail-fast, awaited)
    # ------------------------------------------------------------------
    async def validate_space_ownership(self, agent_name: str, subagent_names: Iterable[str]) -> None:
        """Reject topologies whose mapped slugs collide with another agent's.

        C-2 option (a), creation/update-time layer: two distinct names may
        map onto the same scope slug (documented mapping), and sharing the
        slug means sharing the RAG space and the memories namespace — the
        create/update boundary rejects it instead of silently sharing (the
        cascade ownership guard remains the safety net for legacy data).

        Residual (documented): the comparison covers the peers' TOP-LEVEL
        slugs only — two agents may each declare their own subagent
        ``researcher`` (the flat namespace shares that space by design); the
        peer-claimed subagent slugs are protected by the cascade guard
        (:meth:`_cascadable_slugs`, review C-2b). The comparison also
        deliberately ignores the agent's own row, so an update never blocks
        on itself.

        Args:
            agent_name: The agent's own name (a subagent may share the
                parent's own slug — same topology, one space).
            subagent_names: Every subagent's own name.

        Raises:
            ConfigError: N-1 (b) when a topology element resolves onto the
                reserved ``personal`` scope (mapping-drift guard); C-2 when a
                subagent slug collides with a DIFFERENT existing agent's
                slug, or when two different subagent names of this topology
                map onto the same slug. NOTE (review iter-5 I-3, still
                deliberate): the agent's OWN slug colliding with a peer's
                legacy slug only WARNS (the name cannot change on update, and
                blocking the edit would lock a pre-existing pair out of
                every change) — the destructive side stays protected by the
                cascade ownership guard.
        """
        if self._config_repository is None:
            return
        own = agent_scope_slug(agent_name)
        mapped_subs = _mapped_topology_slugs(agent_name, subagent_names)

        own_subagent_slugs = {agent_scope_slug(name) for name in subagent_names}
        try:
            metadata = await self._config_repository.list_all()
        except Exception as exc:  # noqa: BLE001
            # Validation is best-effort: the cascade ownership guard stays fail-safe (the owned-set resolution below returns None).
            logger.warning(LogMessage.RAG_SPACE_OWNERSHIP_UNRESOLVED, agent_name, safe_reason(exc))
            return
        existing: dict[str, str] = {}
        for entry in metadata:
            if entry.name == agent_name:
                continue
            # Rebase semantics: this topology's OWN materialized sub-agents
            # (real top-level rows named after our sub-agents) are not peers.
            if agent_scope_slug(entry.name) in own_subagent_slugs:
                continue
            existing.setdefault(agent_scope_slug(entry.name), entry.name)

        if own in existing:
            # Review iter-5 I-3 / I-6(a): the agent's own name cannot change
            # on update (rename = delete + re-create), so an own-slug mapping
            # collision ALWAYS predates the update (a legacy pair created
            # before the create-time validation existed). Warn instead of
            # permanently locking the pair out of edits; NEW collisions
            # created by the update itself (below) are still rejected.
            logger.warning(LogMessage.RAG_SPACE_SLUG_COLLISION_LEGACY, agent_name, own, existing[own])
        for slug, sub_name in mapped_subs.items():
            if slug != own and slug in existing:
                raise ConfigError(
                    ErrorMessage.AGENT_SPACE_SLUG_COLLISION.format(slug=slug, name=sub_name, other=existing[slug])
                )

    # ------------------------------------------------------------------
    # fire-and-forget scheduling (C-3 — agent CRUD latency is independent of
    # mcp-raganything health); tasks capture the caller's contextvars, so the
    # forwarded credential and store namespace remain the originating user's
    # ------------------------------------------------------------------
    def schedule_provision_for_agent(self, agent_name: str, subagent_names: Iterable[str]) -> None:
        """Schedule the create-time provisioning cascade in the background."""
        _spawn(self.provision_for_agent(agent_name, subagent_names))

    def schedule_subagent_space_diff(self, agent_name: str, previous: Iterable[str], current: Iterable[str]) -> None:
        """Schedule the update-time subagent-space diff in the background."""
        _spawn(self.diff_subagent_spaces(agent_name, previous, current))

    def schedule_deletion_cascade(self, agent_name: str, subagent_names: Iterable[str]) -> None:
        """Schedule the delete-time deprovision + memories purge in the background."""
        _spawn(self._deletion_cascade(agent_name, subagent_names))

    async def _deletion_cascade(self, agent_name: str, subagent_names: Iterable[str]) -> None:
        """Run the guarded deprovision and purge with ONE ownership resolution."""
        slugs = await self._cascadable_slugs(agent_name, agent_space_slugs(agent_name, subagent_names))
        await self._deprovision_slugs(slugs)
        await self._purge_slugs(slugs)

    # ------------------------------------------------------------------
    # awaited lifecycle implementations (guarded, fanned out)
    # ------------------------------------------------------------------
    async def provision_for_agent(self, agent_name: str, subagent_names: Iterable[str]) -> None:
        """Ensure every space of the agent topology exists (create flow).

        Idempotent POSTs (the 200 already-exists answer is a success); ANY
        per-slug failure is warned and absorbed. Non-destructive: no
        ownership guard needed.

        Review iter-5 I-5 (create→delete race, chosen option): the
        fire-and-forget provisioning task first re-checks that the agent
        STILL exists in the owner-scoped repository (the task captured the
        creating user's context) — a create immediately followed by a delete
        must not leave an orphaned space behind from a late provisioning
        POST. Residual micro-window: a POST that already passed the check can
        still land against a concurrent cascade — accepted (the space is
        empty and its owner-gone state is the documented delete+re-create
        semantics); the reviewer's alternative (a lock per slug) adds
        serialization machinery for that window only.

        Args:
            agent_name: The agent's own name.
            subagent_names: Every subagent's own name.
        """
        if self._config_repository is not None and not await self._config_repository.exists(agent_name):
            logger.warning(LogMessage.RAG_SPACE_PROVISION_SKIPPED_GONE, agent_name)
            return
        await self._provision_slugs(agent_space_slugs(agent_name, subagent_names))

    async def deprovision_for_agent(self, agent_name: str, subagent_names: Iterable[str]) -> None:
        """Cascade-delete every OWNED space of the agent topology (delete flow).

        Strict-REST 404 answers are the target state ("no space") and are
        absorbed by :meth:`RagSpaceProvisioner.deprovision` itself. Slugs
        owned by ANOTHER persisted agent (top-level name or DECLARED
        subagent — review C-2b) are skipped.

        Args:
            agent_name: The agent's own name.
            subagent_names: Every subagent's own name.
        """
        cascadable = await self._cascadable_slugs(agent_name, agent_space_slugs(agent_name, subagent_names))
        await self._deprovision_slugs(cascadable)

    async def diff_subagent_spaces(self, agent_name: str, previous: Iterable[str], current: Iterable[str]) -> None:
        """Apply the update diff: deprovision removed, provision added.

        The agent's OWN space is never part of the diff (kept intact), and
        removed subagent slugs owned by ANOTHER top-level agent are skipped
        (C-2 guard) instead of being deprovisioned.

        Args:
            agent_name: The agent's own name (protects its own slug when a
                removed subagent mapped onto it).
            previous: The subagent names BEFORE the update.
            current: The subagent names AFTER the update.
        """
        before = {agent_scope_slug(name) for name in previous}
        after = {agent_scope_slug(name) for name in current}
        # The agent's OWN slug is never part of the diff, even when a removed
        # subagent mapped onto it (the agent keeps its space on any update).
        own = agent_scope_slug(agent_name)
        cascadable = await self._cascadable_slugs(agent_name, sorted((before - after) - {own}))
        await self._deprovision_slugs(cascadable)
        await self._provision_slugs(sorted(after - before))

    async def purge_agent_memories(self, agent_name: str, subagent_names: Iterable[str]) -> None:
        """Purge ``/memories/{slug}/`` of the agent's OWNED topology (delete flow).

        Only the RUNNING user's namespace is ever touched (the repository
        resolves the caller's namespace), and only the exact prefix of each
        CASCADED slug — slugs owned by another persisted agent (top-level or
        declared subagent, C-2b) and the reserved ``personal`` scope (N-1)
        are skipped, so another agent's memories — and the user's personal
        memories — are never destroyed.

        Args:
            agent_name: The agent's own name.
            subagent_names: Every subagent's own name.
        """
        cascadable = await self._cascadable_slugs(agent_name, agent_space_slugs(agent_name, subagent_names))
        await self._purge_slugs(cascadable)

    # ------------------------------------------------------------------
    async def _owned_space_slugs(
        self, agent_name: str, exclude_own_subagent_slugs: set[str] | None = None
    ) -> set[str] | None:
        """Resolve the scope slugs owned by OTHER persisted agents (C-2b).

        The set covers every persisted agent's TOP-LEVEL slug AND the slugs
        of the subagent names its stored YAML declares (name-keyed store
        reads via the :class:`AgentTopologyResolver` port). A slug claimed by
        a peer through EITHER channel is never cascaded by this agent's
        lifecycle.

        Review F-1 + I-3: the peer resolutions fan out CONCURRENTLY and
        their outcomes are INSPECTED, never swallowed into a success — an
        individually unresolvable peer topology (the adapter's sentinel) or
        an UNEXPECTED peer failure invalidates the WHOLE owned set, so the
        cascade fails safe to the agent's own slug.

        Args:
            agent_name: The agent whose cascade is being prepared (its own
                row/topology is excluded).

        Returns:
            The owned-by-peers slug set, or ``None`` when the ownership set
            cannot be resolved (repository outage or any unresolvable peer
            topology → caller fails safe to the own slug only).
        """
        if self._config_repository is None:
            return None
        try:
            metadata = await self._config_repository.list_all()
        except Exception as exc:  # noqa: BLE001
            # The caller fails safe below: an unresolvable owned set narrows the cascade to the own slug.
            logger.warning(LogMessage.RAG_SPACE_OWNERSHIP_UNRESOLVED, agent_name, safe_reason(exc))
            return None
        excluded = exclude_own_subagent_slugs or set()
        peers = [entry.name for entry in metadata if entry.name != agent_name]
        # Rebase semantics: inline sub-agents are MATERIALIZED into real
        # agents named after themselves (origin/main PR #47) — a topology
        # re-declaring its OWN subagents (create/update revalidation) must
        # not be refused by the top-level row its own earlier import
        # created. The CASCade guard calls this WITHOUT the exclusion so
        # every real agent's space stays protected (F-1/C-2b parity).
        owned = {agent_scope_slug(name) for name in peers} - excluded
        if self._topologies is None or not peers:
            return owned
        if (declared := await self._collect_declared_subagent_slugs(self._topologies, peers)) is None:
            return None
        owned.update(declared)
        return owned

    async def _collect_declared_subagent_slugs(
        self, topologies: AgentTopologyResolver, peers: list[str]
    ) -> set[str] | None:
        """Fan the peer topology resolutions out CONCURRENTLY (review I-3).

        One gather over every peer instead of N sequential store reads. The
        outcomes are INSPECTED, never swallowed into a success: the sentinel
        (an individually unresolvable peer — review F-1) or ANY unexpected
        peer failure invalidates the WHOLE declared-slug set.

        Args:
            topologies: The (non-``None``) resolver, validated by the caller.
            peers: The peer agent names to resolve.

        Returns:
            The union of the peers' declared subagent slugs, or ``None`` when
            any peer topology is unresolvable.
        """
        resolved = await asyncio.gather(*(topologies.subagent_names(name) for name in peers), return_exceptions=True)
        declared: set[str] = set()
        for peer, outcome in zip(peers, resolved, strict=True):
            if isinstance(outcome, AgentTopologyUnresolvedError):
                # Review F-1: the adapter already logged the specific cause —
                # ONE unresolvable peer invalidates the whole owned set.
                return None
            if isinstance(outcome, Exception):
                logger.warning(LogMessage.RAG_SPACE_OWNERSHIP_UNRESOLVED, peer, safe_reason(outcome))
                return None
            # ``gather(..., return_exceptions=True)`` types outcomes as
            # ``... | BaseException``, but only Exceptions are captured per
            # child here (a child-internal cancellation is an Exception
            # result) — the remainder IS the declared names list.
            declared.update(agent_scope_slug(name) for name in cast("list[str]", outcome))
        return declared

    async def _cascadable_slugs(self, agent_name: str, slugs: list[str]) -> list[str]:
        """Filter the topology slugs down to the ones this agent MAY destroy.

        C-2 ownership guard (extended per review iteration-7 C-2b): a slug
        mapped onto ANOTHER persisted agent — top-level name or DECLARED
        subagent name — is skipped with a warning (the space and
        ``/memories/{slug}/`` belong to that agent too). When the ownership
        set cannot be resolved the guard FAILS SAFE — an unresolvable
        ownership set must never widen the blast radius, so only the
        agent's own slug may ever be destroyed.

        N-1 (c): the reserved ``personal`` default scope is NEVER cascaded
        (defense in depth — :func:`agent_scope_slug` derives distinct slugs
        for names that would map onto it, so the branch guards mapping drift
        and any hand-built slug list).

        Args:
            agent_name: The agent whose cascade is being prepared (its own
                slug is always cascadable — it owns that space).
            slugs: The mapped topology slugs.

        Returns:
            The slugs that may be deprovisioned and purged.
        """
        own = agent_scope_slug(agent_name)
        others: set[str] | None = None
        if self._config_repository is not None:
            owned = await self._owned_space_slugs(agent_name, None)
            if owned is not None:
                others = owned - {own}
        cascadable: list[str] = []
        for slug in slugs:
            if is_reserved_scope_slug(slug):
                logger.warning(LogMessage.RAG_SPACE_RESERVED_SLUG_SKIPPED, slug, agent_name)
            elif others is None:
                # Fail safe (review C-2): an unresolvable ownership set must
                # never widen the blast radius — only the own slug.
                if slug == own:
                    cascadable.append(slug)
            elif slug == own or slug not in others:
                cascadable.append(slug)
            else:
                logger.warning(LogMessage.RAG_SPACE_OWNERSHIP_SKIPPED, slug, agent_name)
        return cascadable

    async def _provision_slugs(self, slugs: list[str]) -> None:
        """Provision each slug CONCURRENTLY, warning per-slug failures (C-3)."""
        provisioner = self._provisioner
        if provisioner is None:
            return
        results = await asyncio.gather(*(self._provision_one(slug) for slug in slugs), return_exceptions=True)
        for slug, result in zip(slugs, results, strict=True):
            if isinstance(result, Exception):
                logger.warning(LogMessage.RAG_SPACE_PROVISION_FAILED, slug, safe_reason(result))

    async def _deprovision_slugs(self, slugs: list[str]) -> None:
        """Deprovision each slug CONCURRENTLY, warning per-slug failures (C-3)."""
        provisioner = self._provisioner
        if provisioner is None:
            return
        results = await asyncio.gather(*(self._deprovision_one(slug) for slug in slugs), return_exceptions=True)
        for slug, result in zip(slugs, results, strict=True):
            if isinstance(result, Exception):
                logger.warning(LogMessage.RAG_SPACE_DEPROVISION_FAILED, slug, safe_reason(result))

    async def _purge_slugs(self, slugs: list[str]) -> None:
        """Purge ``/memories/{slug}/`` per slug (guarded by the caller)."""
        if self._store_files is None:
            return
        for slug in slugs:
            await self._purge_one(slug)

    async def _provision_one(self, slug: str) -> None:
        """Provision ONE slug (no error escapes — gathered by the caller)."""
        provisioner = self._provisioner
        if provisioner is None:
            return
        await provisioner.provision(slug)

    async def _deprovision_one(self, slug: str) -> None:
        """Deprovision ONE slug (no error escapes — gathered by the caller)."""
        provisioner = self._provisioner
        if provisioner is None:
            return
        await provisioner.deprovision(slug)

    async def _purge_one(self, slug: str) -> None:
        """Purge ONE slug's memory prefix, warning per failed path.

        The listing is sequential (single bounded store query), the deletes
        are fanned out concurrently (C-3).
        """
        store_files = self._store_files
        if store_files is None:
            return
        prefix = f"{MEMORIES_PREFIX}{slug}/"
        try:
            paths = await store_files.list_files(prefix)
        except Exception as exc:
            logger.warning(LogMessage.AGENT_MEMORIES_PURGE_FAILED, prefix, safe_reason(exc))
            return
        results = await asyncio.gather(*(store_files.delete_file(path) for path in paths), return_exceptions=True)
        for path, result in zip(paths, results, strict=True):
            if isinstance(result, Exception):
                logger.warning(LogMessage.AGENT_MEMORIES_PURGE_FAILED, path, safe_reason(result))


def _spawn(coroutine) -> None:
    """Schedule one fire-and-forget lifecycle task (C-3) with a strong ref."""
    task = asyncio.create_task(coroutine)
    _PENDING_LIFECYCLE_TASKS.add(task)
    task.add_done_callback(_log_task_result)


def _log_task_result(task: asyncio.Task) -> None:
    """Discard the finished task and surface any unexpected failure once."""
    _PENDING_LIFECYCLE_TASKS.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if isinstance(exc, Exception):
        logger.warning(LogMessage.RAG_SPACE_TASK_FAILED, safe_reason(exc))


async def drain_pending_lifecycle_tasks() -> int:
    """Await (then cancel) pending lifecycle tasks on the CURRENT loop.

    Shutdown hook (review C-3): no orphan fire-and-forget task may outlive
    the process. Tasks pinned to ANOTHER event loop are left to that loop
    (best-effort); the coroutine internals already absorb every failure.

    Sonar S7483: the timeout policy is NOT a parameter of the awaited
    function — the bound is the module constant
    :data:`_DRAIN_TIMEOUT_SECONDS` (callers that need a different bound
    patch the constant, as the shutdown-drain unit spec does).

    Returns:
        The number of tasks that completed within the drain bound.
    """
    loop = asyncio.get_running_loop()
    pending = [task for task in _PENDING_LIFECYCLE_TASKS if not task.done() and task.get_loop() is loop]
    if not pending:
        return 0
    done: set[asyncio.Task]
    still_pending: set[asyncio.Task]
    try:
        async with asyncio.timeout(_DRAIN_TIMEOUT_SECONDS):
            done, still_pending = await asyncio.wait(pending)
    except TimeoutError:
        done = set()
        still_pending = set(pending)
    for task in still_pending:
        task.cancel()
    if still_pending:
        await asyncio.gather(*still_pending, return_exceptions=True)
    return len(done)
