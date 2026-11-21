"""Pure planning rules for the create-time agent YAML import.

Spec "Configuration des agents par ressources et matérialisation des imports
YAML" (§4.C..§4.G). The plan builder is the PREVALIDATION step of the import:
it inspects only what the user EXPLICITLY declared in the raw YAML mapping
(REQ-IMPORT-002 — no Pydantic default may be interpreted as a declaration),
raises a controlled error for every violation it can detect without I/O
(REQ-IMPORT-006), and produces the declarative materialization plan plus the
resource-only TARGET configuration skeleton the import materializer completes
with the created resource identities (REQ-IMPORT-003).

No framework and no I/O here: the module operates on plain dicts and domain
entities only. Raw YAML text handling stays in the application layer.
"""

import copy
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from src.domain.entities.agent_config import PromptRef, SubAgentConfig
from src.domain.entities.registered_mcp_server import McpServerWrite
from src.domain.errors.config import ConfigError, ConfigValidationError
from src.domain.errors.messages import ErrorMessage

#: Legacy top-level fields stripped before import planning (single source,
#: reused by the YAML adapter's stored-config compatibility stripping).
DEPRECATED_AGENT_FIELDS = frozenset({"middleware", "is_builtin", "is_customized"})

_SKILLS_PREFIX = "skills/"
_MEMORIES_PREFIX = "memories/"
_SKILL_FILE_NAME = "SKILL.md"

#: Fields the import planning removes from the resource-only target: inline
#: definitions are replaced by materialized references (REQ-IMPORT-003).
_INLINE_DEFINITION_KEYS = ("model", "system_prompt", "prompt_ref", "mcp_servers", "system_prompt_file")

#: Provider applied to legacy model strings without a ``provider:model``
#: separator: the OpenAI-compatible serving chain (REQ-MODEL-001).
DEFAULT_SERVING_PROVIDER = "openai"


@dataclass(frozen=True)
class ModelMaterialization:
    """An inline ``model`` value to materialize as a Model catalog (REQ-MODEL-001)."""

    agent_name: str
    provider_model: str
    provider: str

    @property
    def resource_name(self) -> str:
        """The deterministic catalog name ``<agent-name>-model``."""
        return f"{self.agent_name}-model"


@dataclass(frozen=True)
class PromptMaterialization:
    """An inline ``system_prompt`` value to materialize as a Prompt (REQ-PROMPT-001)."""

    agent_name: str
    content: str

    @property
    def resource_name(self) -> str:
        """The deterministic registry name ``<agent-name>-prompt``."""
        return f"{self.agent_name}-prompt"


@dataclass(frozen=True)
class McpMaterialization:
    """One inline MCP server definition to create in the registry (REQ-MCP-001).

    ``owner_agent`` names the agent (root or inline sub-agent) whose YAML will
    carry the created server's identity in its ``mcp_refs`` selection.
    """

    writer: McpServerWrite
    owner_agent: str


@dataclass(frozen=True)
class SubagentMaterialization:
    """A fully inline sub-agent entry to create as a real Agent (REQ-SUBAGENT-001)."""

    entry_name: str
    entry: dict
    model: ModelMaterialization
    prompt: PromptMaterialization


@dataclass(frozen=True)
class ResourceFileDefinition:
    """An inline ``{path, content}`` skill/memory file to write to the Store.

    One dataclass serves both kinds: ``is_skill`` distinguishes the Store key
    derivation (``<path>SKILL.md`` for skills, the file itself for memories)
    and the path policy (skill paths end with ``/``, memory paths are files).
    """

    is_skill: bool
    path: str
    content: str

    @property
    def file_path(self) -> str:
        """The Store key written via the Store File API (canonical, leading slash)."""
        base = self.path if self.path.startswith("/") else f"/{self.path}"
        return f"{base}{_SKILL_FILE_NAME}" if self.is_skill else base


@dataclass(frozen=True)
class AgentImportPlan:
    """The declarative materialization plan for one agent YAML import.

    ``target`` is the RESOURCE-ONLY configuration skeleton: inline
    model/prompt/MCP definitions and sub-agent inline bodies are removed; the
    import materializer injects ``model_ref``/``prompt_ref``/``mcp_refs`` and
    the created sub-agents' ``agent_ref`` values before persisting.
    """

    agent_name: str
    model: ModelMaterialization | None
    model_ref: str | None
    prompt: PromptMaterialization | None
    prompt_ref: PromptRef | None
    mcp_creations: list[McpMaterialization]
    mcp_refs: list[str]
    subagent_creations: list[SubagentMaterialization]
    subagent_entry_index: dict[str, int]
    files: list[ResourceFileDefinition]
    skill_refs: list[str]
    memory_refs: list[str]
    target: dict


def derive_model_provider(model_value: str) -> str:
    """The serving provider inferred from a legacy model string (REQ-MODEL-001).

    The prefix preceding the first ``:`` wins; without a separator (or with a
    degenerate empty prefix) the OpenAI-compatible serving chain applies.
    """
    prefix = model_value.split(":", 1)[0] if ":" in model_value else ""
    return prefix if prefix else DEFAULT_SERVING_PROVIDER


def derived_resource_names(plan: AgentImportPlan) -> list[tuple[str, str]]:
    """The PREDICTABLE resource names this import will create (REQ-IMPORT-006).

    Returns ``(resource_name, kind)`` pairs for the root and every created
    sub-agent's derived model/prompt names — the name-conflict prevalidation
    checks them against the owner's catalogs BEFORE any write (REQ-TX-002).
    """
    declared: list[tuple[str, str]] = []
    if plan.model is not None:
        declared.append((plan.model.resource_name, "model"))
    if plan.prompt is not None:
        declared.append((plan.prompt.resource_name, "prompt"))
    declared.extend((creation.model.resource_name, "model") for creation in plan.subagent_creations)
    declared.extend((creation.prompt.resource_name, "prompt") for creation in plan.subagent_creations)
    return declared


def build_agent_import_plan(raw: dict, agent_name: str) -> AgentImportPlan:
    """Prevalidate an import YAML mapping and build its materialization plan.

    Pure: everything this function can detect is detected BEFORE any external
    write or remote call. The remaining checks (existence of referenced
    resources, registry availability, name availability) belong to the import
    service's I/O prevalidation.

    Args:
        raw: The parsed (mapping) YAML document.
        agent_name: The requested agent name (URL / form field).

    Returns:
        The validated :class:`AgentImportPlan`.

    Raises:
        ConfigError: On any schema-level or policy violation.
        ConfigValidationError: On malformed declarative structures.
    """
    raw = {key: value for key, value in raw.items() if key not in DEPRECATED_AGENT_FIELDS}
    _check_agent_identity(raw, agent_name)

    model, model_ref = _plan_model(raw, agent_name)
    prompt, prompt_ref = _plan_prompt(raw, agent_name)
    subagent_creations, subagent_entry_index, subagent_mcp, subagent_skill_refs = _plan_subagents(raw, agent_name)
    mcp_creations, mcp_refs = _plan_mcp(raw, agent_name, subagent_mcp)
    files, skill_refs, memory_refs, skill_targets, memory_targets = _plan_files(raw)
    skill_refs = [*skill_refs, *subagent_skill_refs]

    return AgentImportPlan(
        agent_name=agent_name,
        model=model,
        model_ref=model_ref,
        prompt=prompt,
        prompt_ref=prompt_ref,
        mcp_creations=mcp_creations,
        mcp_refs=mcp_refs,
        subagent_creations=subagent_creations,
        subagent_entry_index=subagent_entry_index,
        files=files,
        skill_refs=skill_refs,
        memory_refs=memory_refs,
        target=_build_target(raw, subagent_creations, subagent_entry_index, skill_targets, memory_targets),
    )


def _check_agent_identity(raw: dict, agent_name: str) -> None:
    """The YAML name must match the requested agent name (reused message)."""
    yaml_name = raw.get("name")
    if yaml_name != agent_name:
        raise ConfigError(ErrorMessage.AGENT_NAME_MISMATCH.format(yaml_name=yaml_name, name=agent_name))


def _plan_model(raw: dict, agent_name: str) -> tuple[ModelMaterialization | None, str | None]:
    """Resolve the root's explicit model declaration (REQ-MODEL-001/002)."""
    inline = raw.get("model")
    ref = raw.get("model_ref")
    if ref is not None:
        # REQ-MODEL-002: an existing reference takes precedence; the legacy
        # string stays inert and is dropped from the resource-only target.
        if not isinstance(ref, str):
            raise ConfigValidationError(
                [{"loc": ["model_ref"], "msg": "model_ref must be a catalog model UUID", "type": "value_error"}]
            )
        return None, ref
    if inline is not None:
        return _model_materialization(agent_name, inline), None
    # REQ-MODEL-003: the Pydantic legacy default must never materialize nor
    # mask an incomplete configuration.
    raise ConfigError(ErrorMessage.AGENT_IMPORT_MISSING_MODEL.format(name=agent_name))


def _model_materialization(agent_name: str, model_value: Any) -> ModelMaterialization:
    """Build the Model materialization descriptor for an inline model value."""
    if not isinstance(model_value, str) or not model_value:
        raise ConfigError(ErrorMessage.AGENT_IMPORT_MODEL_DECLARATION_INVALID.format(name=agent_name))
    return ModelMaterialization(
        agent_name=agent_name,
        provider_model=model_value,
        provider=derive_model_provider(model_value),
    )


def _plan_prompt(raw: dict, agent_name: str) -> tuple[PromptMaterialization | None, PromptRef | None]:
    """Resolve the root's explicit prompt declaration (REQ-PROMPT-001..004)."""
    if raw.get("system_prompt_file"):
        # REQ-PROMPT-004: system_prompt_file stays forbidden in a YAML
        # transmitted as a string or uploaded (no archive mechanism yet).
        raise ConfigError(ErrorMessage.YAML_SYSTEM_PROMPT_FILE_DISALLOWED.format(source=agent_name))
    inline = raw.get("system_prompt")
    ref = raw.get("prompt_ref")
    if ref is not None:
        if inline is not None:
            # REQ-PROMPT-004: the exclusivity stays invalid.
            raise ConfigError(
                ErrorMessage.AGENT_IMPORT_EXCLUSIVE_SOURCES.format(
                    name=agent_name, first="prompt_ref", second="system_prompt"
                )
            )
        # Non-UUID refs / non-positive versions fail here, before any write.
        try:
            return None, PromptRef.model_validate(ref)
        except ValidationError as e:
            raise ConfigValidationError([dict(item) for item in e.errors()]) from e
    if inline is not None:
        if not isinstance(inline, str):
            raise ConfigValidationError(
                [{"loc": ["system_prompt"], "msg": "system_prompt must be a string", "type": "value_error"}]
            )
        # REQ-PROMPT-002: an explicitly empty content stays valid.
        return PromptMaterialization(agent_name=agent_name, content=inline), None
    # REQ-PROMPT-005: a new custom agent must declare a prompt.
    raise ConfigError(ErrorMessage.AGENT_IMPORT_MISSING_PROMPT.format(name=agent_name))


def _plan_mcp(raw: dict, agent_name: str, subagent_mcp: list[tuple[dict, str]]) -> tuple[list[McpMaterialization], list[str]]:
    """Collect inline MCP definitions and declared references (REQ-MCP-001..005).

    Root- and sub-agent-declared servers share one user-scoped name namespace:
    duplicates anywhere in the import are refused (REQ-MCP-005), and a name
    that is both referenced AND defined inline is refused because a reference
    is never materialized again.
    """
    creations: list[McpMaterialization] = []
    names: set[str] = set()
    for entry in raw.get("mcp_servers") or []:
        writer = _mcp_entry_writer(entry)
        if writer.name in names:
            raise ConfigError(ErrorMessage.AGENT_IMPORT_MCP_DUPLICATE.format(name=writer.name))
        names.add(writer.name)
        creations.append(McpMaterialization(writer=writer, owner_agent=agent_name))
    for entry, owner in subagent_mcp:
        writer = _mcp_entry_writer(entry)
        if writer.name in names:
            raise ConfigError(ErrorMessage.AGENT_IMPORT_MCP_DUPLICATE.format(name=writer.name))
        names.add(writer.name)
        creations.append(McpMaterialization(writer=writer, owner_agent=owner))
    refs = _string_list(raw.get("mcp_refs"), "mcp_refs")
    for name in refs:
        if name in names:
            raise ConfigError(ErrorMessage.AGENT_IMPORT_MCP_REF_CONFLICT.format(name=name))
        names.add(name)
    return creations, refs


def _mcp_entry_writer(entry: Any) -> McpServerWrite:
    """Validate one inline MCP entry into its registry write payload."""
    if not isinstance(entry, dict):
        raise ConfigError(ErrorMessage.AGENT_IMPORT_MCP_ENTRY_INVALID.format(reason="each server must be a mapping"))
    name = str(entry.get("name") or "<unnamed>")
    unexpected = sorted(set(entry.keys()) - set(McpServerWrite.model_fields.keys()))
    if unexpected:
        # REQ-MCP-002: no functional field may be silently abandoned.
        raise ConfigError(
            ErrorMessage.AGENT_IMPORT_MCP_ENTRY_INVALID.format(
                reason=f"server '{name}' declares unsupported field(s) {unexpected}"
            )
        )
    try:
        return McpServerWrite.model_validate(entry)
    except ValidationError as e:
        raise ConfigError(ErrorMessage.AGENT_IMPORT_MCP_ENTRY_INVALID.format(reason=_summarize_errors(e))) from e


def _plan_subagents(
    raw: dict, agent_name: str
) -> tuple[list[SubagentMaterialization], dict[str, int], list[tuple[dict, str]], list[str]]:
    """Collect fully inline sub-agent entries (REQ-SUBAGENT-001/002).

    Returns the creation descriptors, the mapping ``entry name → position in
    the raw ``subagents`` list``, the sub-agents' inline MCP server entries
    awaiting the shared namespace validation, and their textual skill
    reference paths (existence-checked with the root ones).
    """
    entries = raw.get("subagents") or []
    if not entries:
        return [], {}, [], []
    entry_index: dict[str, int] = {}
    creations: list[SubagentMaterialization] = []
    subagent_mcp: list[tuple[dict, str]] = []
    subagent_skill_refs: list[str] = []
    for position, entry in enumerate(entries):
        _validate_subagent_shape(entry, agent_name)
        name = entry["name"]
        if name in entry_index:
            raise ConfigError(ErrorMessage.AGENT_IMPORT_SUBAGENT_NAME_DUPLICATE.format(name=name))
        entry_index[name] = position
        if entry.get("agent_ref") is not None:
            # Reference entry: validated for existence/cycles; never created
            # and never carrying an inline definition (its resources belong to
            # the referenced agent, REQ-MCP-003).
            continue
        subagent_skill_refs.extend(entry.get("skills") or [])
        for mcp_entry in entry.get("mcp_servers") or []:
            subagent_mcp.append((mcp_entry, name))
        model = _model_materialization(name, entry["model"])
        prompt = PromptMaterialization(agent_name=name, content=entry["instructions"])
        creations.append(SubagentMaterialization(entry_name=name, entry=entry, model=model, prompt=prompt))

    _detect_reference_cycles(entries, agent_name, set(entry_index))
    return creations, entry_index, subagent_mcp, subagent_skill_refs


def _validate_subagent_shape(entry: Any, agent_name: str) -> None:
    """Structural validation applied to EVERY sub-agent entry."""
    if not isinstance(entry, dict):
        raise ConfigError(ErrorMessage.AGENT_IMPORT_SUBAGENT_MUST_DECLARE.format(name="<unnamed>", field="mapping"))
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError(ErrorMessage.AGENT_IMPORT_SUBAGENT_MUST_DECLARE.format(name=name, field="name"))
    unexpected = sorted(set(entry.keys()) - set(SubAgentConfig.model_fields.keys()))
    if unexpected:
        raise ConfigError(
            ErrorMessage.AGENT_IMPORT_SUBAGENT_MUST_DECLARE.format(
                name=name, field=f"no unsupported field(s) {unexpected}"
            )
        )
    if not isinstance(entry.get("description"), str) or not entry["description"]:
        raise ConfigError(ErrorMessage.AGENT_IMPORT_SUBAGENT_MUST_DECLARE.format(name=name, field="description"))
    ref = entry.get("agent_ref")
    if ref is None:
        _validate_inline_subagent_body(entry, name)
    elif not isinstance(ref, str) or not ref:
        raise ConfigError(ErrorMessage.AGENT_IMPORT_SUBAGENT_MUST_DECLARE.format(name=name, field="agent_ref"))
    elif ref == agent_name:
        raise ConfigError(ErrorMessage.AGENT_IMPORT_SUBAGENT_SELF_REFERENCE.format(name=name, agent=agent_name))
    else:
        _validate_reference_subagent_entry(entry, name)


def _validate_inline_subagent_body(entry: dict, name: str) -> None:
    """A fully inline entry must carry every field needed to create it."""
    for required in ("instructions", "model"):
        if not isinstance(entry.get(required), str) or not entry[required]:
            raise ConfigError(ErrorMessage.AGENT_IMPORT_SUBAGENT_MUST_DECLARE.format(name=name, field=required))
    skills = entry.get("skills") or []
    if skills and not all(isinstance(item, str) for item in skills):
        raise ConfigError(
            ErrorMessage.AGENT_IMPORT_SKILL_PATH_INVALID.format(
                path=name,
                reason="inline {path, content} skill definitions are only supported on the top-level import",
            )
        )


def _validate_reference_subagent_entry(entry: dict, name: str) -> None:
    """A REFERENCE entry carries ONLY the reference (REQ-MCP-003 / C2).

    Inline definitions (connection parameters with their secrets, or body
    overrides) are never persisted on the parent: the referenced agent owns
    its own model, prompt and MCP selection.
    """
    for forbidden in ("mcp_servers", "model", "instructions", "skills"):
        if entry.get(forbidden) is not None and entry.get(forbidden) != []:
            raise ConfigError(
                ErrorMessage.AGENT_IMPORT_SUBAGENT_REF_FIELD_FORBIDDEN.format(name=name, field=forbidden)
            )


def _detect_reference_cycles(entries: list, agent_name: str, entry_names: set[str]) -> None:
    """Refuse circular ``agent_ref`` graphs before any creation (REQ-SUBAGENT-002)."""
    edges = _collect_reference_edges(entries, agent_name, entry_names)
    for start, targets in edges.items():
        for target in targets:
            rendered = _trace_cycle(start, target, edges)
            if rendered is not None:
                raise ConfigError(ErrorMessage.AGENT_IMPORT_SUBAGENT_CYCLE.format(names=rendered))


def _collect_reference_edges(entries: list, agent_name: str, entry_names: set[str]) -> dict[str, list[str]]:
    """Collect valid sub-agent-to-agent edges; refuse direct self references."""
    edges: dict[str, list[str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        origin, ref = entry.get("name"), entry.get("agent_ref")
        if not isinstance(origin, str) or not isinstance(ref, str):
            continue
        if ref == agent_name:
            raise ConfigError(ErrorMessage.AGENT_IMPORT_SUBAGENT_SELF_REFERENCE.format(name=origin, agent=agent_name))
        if ref in entry_names:
            edges.setdefault(origin, []).append(ref)
    return edges


def _trace_cycle(start: str, target: str, edges: dict[str, list[str]]) -> str | None:
    """Return ``a -> b -> ... -> a`` when ``start`` reaches itself from ``target``."""
    path = [start, target]
    node = target
    while node in edges:
        node = edges[node][0]
        if node in path:
            path.append(node)
            return " -> ".join(path)
        path.append(node)
    return None


def _plan_files(raw: dict) -> tuple[list[ResourceFileDefinition], list[str], list[str], list[str], list[str]]:
    """Resolve ``skills``/``memory`` entries: textual refs or inline ``path+content``.

    Returns ``(files, skill_refs, memory_refs, skill_targets, memory_targets)``
    where ``files`` holds the inline definitions to materialize, the ``*_refs``
    are the textual Store references the import must validate for existence,
    and the ``*_targets`` are the textual values that replace the YAML entries
    in the resource-only target configuration.
    """
    files: list[ResourceFileDefinition] = []
    skill_refs: list[str] = []
    memory_refs: list[str] = []
    skill_targets: list[str] = []
    memory_targets: list[str] = []
    for entry in raw.get("skills") or []:
        textual, path, content = _file_entry(entry, is_skill=True)
        if textual:
            skill_refs.append(textual)
            skill_targets.append(textual)
        else:
            files.append(ResourceFileDefinition(is_skill=True, path=path, content=content))
            skill_targets.append(path)
    for entry in raw.get("memory") or []:
        textual, path, content = _file_entry(entry, is_skill=False)
        if textual:
            memory_refs.append(textual)
            memory_targets.append(textual)
        else:
            files.append(ResourceFileDefinition(is_skill=False, path=path, content=content))
            memory_targets.append(path)
    return files, skill_refs, memory_refs, skill_targets, memory_targets


def _file_entry(entry: Any, *, is_skill: bool) -> tuple[str, str, str]:
    """Validate one ``skills``/``memory`` entry.

    Returns ``(textual_ref, "", "")`` for a textual reference entry and
    ``("", path, content)`` for an inline definition. The path shape policy
    (prefix + trailing ``/`` for skills, plain file under ``memories/`` for
    memories) applies to BOTH forms; a controlled error is raised for any
    malformed or policy-violating entry (REQ-SKILL-001 / REQ-MEMORY-001: the
    object form accepts EXACTLY ``path`` and ``content``).
    """
    if isinstance(entry, str):
        textual = entry
        path, content = entry, ""
    elif isinstance(entry, dict) and set(entry.keys()) == {"path", "content"}:
        textual = ""
        path, content = entry["path"], entry["content"]
        if not isinstance(path, str) or not isinstance(content, str) or not path:
            raise ConfigError(ErrorMessage.AGENT_IMPORT_FILE_SHAPE_INVALID.format(kind=_kind(is_skill)))
    else:
        if isinstance(entry, dict) and {"path", "content"}.issubset(entry.keys()):
            unexpected = sorted(set(entry.keys()) - {"path", "content"})
            template = (
                ErrorMessage.AGENT_IMPORT_SKILL_FIELD_UNEXPECTED
                if is_skill
                else ErrorMessage.AGENT_IMPORT_MEMORY_FIELD_UNEXPECTED
            )
            raise ConfigError(template.format(fields=", ".join(unexpected)))
        raise ConfigError(ErrorMessage.AGENT_IMPORT_FILE_SHAPE_INVALID.format(kind=_kind(is_skill)))
    _reject_traversal(path, is_skill)
    _validate_path_shape(path, is_skill)
    return textual, path, content


def _validate_path_shape(path: str, is_skill: bool) -> None:
    """The path policy: skills are directories under ``skills/`` ending with
    ``/``; memories are file paths under ``memories/``."""
    if is_skill:
        if not path.startswith(_SKILLS_PREFIX) or not path.endswith("/"):
            raise ConfigError(
                ErrorMessage.AGENT_IMPORT_SKILL_PATH_INVALID.format(
                    path=path, reason="a skill path must be under skills/ and end with '/'"
                )
            )
        return
    if not path.startswith(_MEMORIES_PREFIX) or path.endswith("/"):
        raise ConfigError(
            ErrorMessage.AGENT_IMPORT_MEMORY_PATH_INVALID.format(
                path=path, reason="a memory path must designate a file under memories/"
            )
        )


def _reject_traversal(path: str, is_skill: bool) -> None:
    """Reject any path escaping its required prefix."""
    if ".." in path:
        invalid = (
            ErrorMessage.AGENT_IMPORT_SKILL_PATH_INVALID
            if is_skill
            else ErrorMessage.AGENT_IMPORT_MEMORY_PATH_INVALID
        )
        raise ConfigError(invalid.format(path=path, reason="path traversal is forbidden"))


def _string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ConfigValidationError(
            [
                {
                    "loc": [field_name],
                    "msg": f"{field_name} must be a list of non-empty strings",
                    "type": "value_error",
                }
            ]
        )
    return value


def _build_target(
    raw: dict,
    subagent_creations: list[SubagentMaterialization],
    subagent_entry_index: dict[str, int],
    skill_targets: list[str],
    memory_targets: list[str],
) -> dict:
    """Build the resource-only TARGET skeleton.

    Inline model/prompt/MCP declarations and inline sub-agent bodies are
    removed; EVERY sub-agent entry is normalized to its resource-only shape —
    a created entry keeps its display identity (the materializer injects its
    ``agent_ref``), a reference entry keeps ONLY ``agent_ref`` so a referenced
    agent's own model/prompt/MCP selection can never leak connection
    parameters or secrets onto the parent (REQ-MCP-003).
    """
    target = copy.deepcopy({k: v for k, v in raw.items() if k not in DEPRECATED_AGENT_FIELDS})
    for source_key in _INLINE_DEFINITION_KEYS:
        target.pop(source_key, None)
    target["skills"] = list(skill_targets)
    target["memory"] = list(memory_targets)
    created_by_position = {
        subagent_entry_index[creation.entry_name]: creation.entry_name for creation in subagent_creations
    }
    subagents = target.get("subagents") or []
    for position, entry in enumerate(subagents):
        subagents[position] = _normalized_subagent_entry(entry, created_by_position.get(position))
    return target


def _normalized_subagent_entry(entry: Any, created_name: str | None) -> dict:
    """The resource-only shape of one sub-agent entry on the parent."""
    if created_name is not None:
        return {"name": created_name, "description": entry["description"]}
    return {
        "name": entry["name"],
        "description": entry["description"],
        "agent_ref": entry["agent_ref"],
    }


def _summarize_errors(e: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(loc) for loc in err.get('loc', []))}: {err.get('msg', '')}" for err in e.errors()
    )


def _kind(is_skill: bool) -> str:
    return "skill" if is_skill else "memory"
