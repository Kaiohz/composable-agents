"""Resource-only write boundary for agent configuration updates.

Materialization applies to CREATION only (REQ-LEGACY-003). The policy
enforces what the STORED METADATA projection can verify authoritatively:

* an agent whose prompt/model is reference-based (``prompt_ref``/``
  ``model_ref`` in its metadata — every agent imported since the
  resource-only rule) refuses ANY inline ``system_prompt`` / ``model``
  definition, because that would be a NEW inline definition (req
  agent-resource-import REQ-IMPORT-003: an inline value must never persist
  in double with the reference it is superseded by);
* inline MCP server definitions (connection parameters, tokens) are refused
  on EVERY update, at ANY depth — the registry owns them, they are never
  resurrected through the update boundary, and a NESTED ``subagents[]``
  entry must not smuggle them back (req agent-resource-import REQ-MCP-003,
  §5 Sécurité);
* a LEGACY agent (no registered reference in its metadata) keeps its existing
  edit path (REQ-LEGACY-001 read/execute compatibility): the conversion of
  its inline values to registered resources happens through the
  UI-driven manual-edit flow of REQ-LEGACY-002, which presents registered
  resources instead of re-submitting inline values.
"""

from collections.abc import Mapping

from src.domain.entities.agent_config_metadata import AgentConfigMetadata
from src.domain.errors.config import ConfigError
from src.domain.errors.messages import ErrorMessage


def ensure_resource_only_update(raw: Mapping, metadata: AgentConfigMetadata) -> None:
    """Refuse update YAMLs introducing NEW inline materializable definitions.

    Args:
        raw: The parsed (mapping) update YAML document. Pydantic defaults are
            never interpreted as declarations: only explicitly present keys
            are examined. ``subagents[]`` entries are inspected as well — an
            inline MCP definition is refused at ANY depth.
        metadata: The STORED metadata of the agent being updated — the
            authoritative projection of its current reference-based resources.

    Raises:
        ConfigError: When a new inline model, prompt or MCP server definition
            would enter the configuration (first violation wins).
    """
    if metadata.prompt_ref is not None and raw.get("system_prompt") is not None:
        # The stored configuration resolves its prompt through the registry:
        # an inline prompt would be a NEW definition (update import-with-
        # conversion is not implemented, REQ-LEGACY-003).
        raise ConfigError(
            ErrorMessage.AGENT_IMPORT_UPDATE_INLINE_FORBIDDEN.format(
                name=metadata.name, field="system_prompt", alternatives="prompt_ref"
            )
        )
    if raw.get("model") is not None and metadata.model_ref is not None:
        # The stored configuration resolves its model through the catalog.
        # An inline `model` next to its `model_ref` must be refused exactly
        # like the prompt-side rule: the update persists the raw YAML
        # verbatim, so the "redundant" inline value would stay in double with
        # the reference (REQ-IMPORT-003) and override the display projection
        # while the runtime keeps the referenced model (BUG-003).
        raise ConfigError(
            ErrorMessage.AGENT_IMPORT_UPDATE_INLINE_FORBIDDEN.format(
                name=metadata.name, field="model", alternatives="model_ref"
            )
        )
    # The registry owns MCP connection parameters: an inline definition
    # (with its URLs, headers, env entries or tokens) is never accepted
    # through the update boundary — at the top level NOR inside any
    # sub-agent entry (BUG-002: one write surface deeper).
    nested_mcp_servers = _nested_mcp_servers_field(raw)
    if raw.get("mcp_servers") or nested_mcp_servers is not None:
        field = "mcp_servers" if raw.get("mcp_servers") else nested_mcp_servers
        raise ConfigError(
            ErrorMessage.AGENT_IMPORT_UPDATE_INLINE_FORBIDDEN.format(
                name=metadata.name, field=field, alternatives="mcp_refs"
            )
        )


def _nested_mcp_servers_field(raw: Mapping) -> str | None:
    """The field path of the FIRST nested inline MCP definition, if any.

    Only the connection-data key (``mcp_servers``) is walked: nested
    ``model``/``instructions`` sub-agent overrides are legacy body fields,
    not registry-owned resources, and nested MCP data is the only shape that
    smuggles connection parameters or secrets through the boundary.
    """
    subagents = raw.get("subagents")
    if not isinstance(subagents, list):
        return None
    for position, entry in enumerate(subagents):
        if isinstance(entry, Mapping) and entry.get("mcp_servers"):
            return f"subagents[{position}].mcp_servers"
    return None
