"""Unattended-run credential-placeholder scan (spec cron §9 / CRN-002).

An agent requiring ``${USER_JWT}``/``${USER_API_KEY}`` credentials can never
run unattended: scheduled executions must not borrow an interactive user's
credential. Under the resource-only agents regime the placeholders live in
the MCP REGISTRY (the selection is carried by ``mcp_refs``), so the scan
extends the legacy stored-YAML scan with the trusted execution-config
resolution of the agent's MCP selection. The scan only PATTERN-MATCHES the
locally resolved values and labels the source — the credential material is
never logged nor forwarded.
"""

import logging
import re

import yaml  # type: ignore[import-untyped]

from src.domain.logging.messages import LogMessage
from src.domain.ports.mcp_registry import McpRegistryClient

logger = logging.getLogger(__name__)

_USER_PLACEHOLDER_PATTERN = re.compile(r"\$\{(?:USER_JWT|USER_API_KEY)\}")


async def find_user_placeholder_sources(
    yaml_text: str,
    mcp_registry: McpRegistryClient | None,
    agent_name: str,
) -> list[str]:
    """Return the labels of every user-credential placeholder source.

    Two sources are scanned: (1) the stored agent YAML itself (legacy inline
    MCP definitions, env entries or tokens) and (2) the agent's ``mcp_refs``
    selections resolved through the trusted registry RAW reveal path —
    a resource-only agent carries its MCP selection as registry identities,
    so a user-credentialed MCP server is only visible after resolution, and
    the RAW payload is used because user placeholders are resolved (and
    dropped when empty) inside the execution-config projection.

    Args:
        yaml_text: The stored agent YAML text.
        mcp_registry: The trusted registry client (``None`` skips resolution;
            the stored-YAML scan still applies).
        agent_name: Log context.

    Returns:
        The list of source labels; empty when the agent is unattended-safe.

    Raises:
        McpRegistryError: When the selection cannot be resolved because the
            registry is unavailable — a controlled failure, never a false
            "unattended-safe" answer (REQ-TX-003).
    """
    if _USER_PLACEHOLDER_PATTERN.search(yaml_text):
        logger.info(LogMessage.CRON_UNATTENDED_SCAN, agent_name, 1)
        return ["agent YAML"]
    refs = _mcp_ref_identities(yaml_text)
    if not refs or mcp_registry is None:
        return []
    sources: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        entry = await mcp_registry.reveal_server(ref)
        if _carries_user_placeholder(entry):
            sources.append(f"MCP server '{ref}'")
    logger.info(LogMessage.CRON_UNATTENDED_SCAN, agent_name, len(sources))
    return sources


def _mcp_ref_identities(yaml_text: str) -> list[str]:
    """Extract the agent's ``mcp_refs`` from its stored YAML (light parse)."""
    try:
        raw = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        return []
    if not isinstance(raw, dict):
        return []
    refs = raw.get("mcp_refs")
    if not isinstance(refs, list) or not all(isinstance(ref, str) and ref for ref in refs):
        return []
    return refs


def _carries_user_placeholder(entry: dict) -> bool:
    """Whether one RAW registry entry carries a user-credential placeholder."""
    values = [*(entry.get("env") or {}).values(), *(entry.get("headers") or {}).values()]
    if entry.get("auth_token"):
        values.append(entry["auth_token"])
    return any(_USER_PLACEHOLDER_PATTERN.search(str(value) if value is not None else "") for value in values)
