"""Behavioral tests — YAML import materializes inline MCP servers and persists
only resource identities (spec §4.F REQ-MCP-001..005, §4.H REQ-TX-001..004).

Entry point: the REAL ``POST /api/v1/agents`` multipart YAML create route.
The ``mcp-raganything`` registry is an external service boundary and is
mocked (``stub_registry``, same convention as the composable_mcp suite);
everything else runs real.

Target contract:
* every server defined inline is created in the user's registry, with the
  YAML name as the resource key and full functional fidelity;
* the persisted agent keeps ONLY the server identities (``mcp_refs``), never
  URLs, headers, env entries or auth tokens;
* an empty selection creates nothing; duplicate names fail prevalidation;
  a name conflict is an explicit error without overwriting;
* a failing import compensates the resources it created: no agent, no orphans;
  the error diagnostics name the resource without leaking secrets.
"""

import uuid

import pytest
import yaml

from tests.behavioral.conftest import create_agent_yaml

pytestmark = pytest.mark.asyncio(loop_scope="session")

VALIDATION_CODES = {400, 422}
# Non-enumerating not-found / validation codes for references (legacy seam).
UNAVAILABLE_CODES = {404, 422}
# Name conflicts are explicit errors; the registry contract decides 409 vs
# validation-level wording ({400,409,422} accepted).
CONFLICT_CODES = {400, 409, 422}
# Registry unavailability is a controlled retryable failure (503 semantics).
UNAVAILABLE_SERVICE_CODES = {400, 422, 503}


def _name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def http_server(name: str, url: str, *, headers: dict | None = None, env: dict | None = None, token: str | None = None) -> dict:
    server: dict = {"name": name, "transport": "http", "url": url}
    if headers:
        server["headers"] = headers
    if env:
        server["env"] = env
    if token:
        server["auth_token"] = token
    return server


def stdio_server(name: str, command: str, args: list[str]) -> dict:
    return {"name": name, "transport": "stdio", "command": command, "args": args}


def import_yaml(name: str, *, mcp_servers: list[dict], model: str = "gpt-4o", system_prompt: str = "You are helpful.") -> str:
    return yaml.safe_dump(
        {"name": name, "model": model, "system_prompt": system_prompt, "tools": [], "debug": False, "mcp_servers": mcp_servers},
        sort_keys=False,
    )


async def import_agent(client, headers, yaml_text: str, name: str):
    _, resp = await create_agent_yaml(client, headers, yaml_text, name)
    return resp


async def models_named(client, headers: dict[str, str], name: str) -> list[dict]:
    listing = await client.get("/api/v1/models", headers=headers)
    assert listing.status_code == 200, listing.text
    return [m for m in listing.json() if m.get("name") == name]


async def prompts_named(client, headers: dict[str, str], name: str) -> list[dict]:
    listing = await client.get("/api/v1/prompts", headers=headers)
    assert listing.status_code == 200, listing.text
    return [p for p in listing.json() if p.get("name") == name]


class TestMcpMaterialization:
    async def test_definitions_are_created_in_registry_and_agent_keeps_only_identities(
        self, client, users, stub_registry
    ):
        """REQ-MCP-001/002/003: inline servers (http + stdio, with connection
        data AND secrets) are created in the registry; the agent persists
        ``mcp_refs`` identities and no connection parameters or secrets."""
        agent = _name("imp-mcp")
        search_url = "http://127.0.0.1:8443/mcp"
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml(
                agent,
                mcp_servers=[
                    http_server(
                        "search",
                        search_url,
                        headers={"X-Token": "sekrit-header"},
                        env={"SEARCH_KEY": "sekrit-env"},
                        token="sekrit-token",
                    ),
                    stdio_server("local-scan", "uvx", ["some-mcp-server"]),
                ],
            ),
            agent,
        )
        assert resp.status_code == 201, resp.text

        # Registry owns the functional definitions, secrets included.
        assert stub_registry.app.entries["search"]["url"] == search_url
        assert stub_registry.app.entries["search"]["headers"] == {"X-Token": "sekrit-header"}
        assert stub_registry.app.entries["search"]["env"] == {"SEARCH_KEY": "sekrit-env"}
        assert stub_registry.app.entries["search"]["auth_token"] == "sekrit-token"
        assert stub_registry.app.entries["local-scan"]["transport"] == "stdio"
        assert stub_registry.app.entries["local-scan"]["command"] == "uvx"
        assert stub_registry.app.entries["local-scan"]["args"] == ["some-mcp-server"]

        got = await client.get(f"/api/v1/agents/{agent}", headers=users["a"]["headers"])
        assert got.status_code == 200, got.text
        body = got.json()
        assert sorted(body["mcp_refs"]) == ["local-scan", "search"], (
            "the agent must persist the SELECTED registry identities (mcp_refs)"
        )
        inline = got.json().get("mcp_servers") or []
        assert inline == [], "inline definitions must not remain in the target configuration"
        # No connection parameter and no secret may survive in the agent config.
        assert search_url not in got.text
        assert "sekrit" not in got.text

    async def test_empty_selection_creates_no_registry_entries(self, client, users, stub_registry):
        """REQ-MCP-004: an empty MCP selection is valid and creates nothing."""
        agent = _name("imp-mcp-empty")
        resp = await import_agent(
            client, users["a"]["headers"], import_yaml(agent, mcp_servers=[]), agent
        )
        assert resp.status_code == 201, resp.text

        creates = [r for r in stub_registry.app.requests if r["method"] == "POST" and r["path"] == "/api/v1/mcp/servers"]
        assert creates == [], "an empty MCP selection must not create registry resources"
        body = (await client.get(f"/api/v1/agents/{agent}", headers=users["a"]["headers"])).json()
        assert not (body.get("mcp_refs") or [])

    async def test_duplicate_names_fail_prevalidation_without_registry_writes(self, client, users, stub_registry):
        """REQ-MCP-005: two inline definitions with the same name are rejected
        before any registry write; the diagnostic names the resource type/path."""
        agent = _name("imp-mcp-dupe")
        token = "sekrit-diagnostic"
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml(
                agent,
                mcp_servers=[
                    http_server("search", "http://127.0.0.1:8444/mcp"),
                    http_server("search", "http://127.0.0.1:8445/mcp", token=token),
                ],
            ),
            agent,
        )
        assert resp.status_code in VALIDATION_CODES, resp.text
        assert stub_registry.app.entries == {}, "prevalidation must happen before any registry write"
        assert "mcp_servers" in resp.text, "the error must identify the YAML resource path"
        assert "search" in resp.text
        assert token not in resp.text, "error diagnostics must never carry secrets"

    async def test_name_conflict_rejected_and_existing_entry_untouched(self, client, users, stub_registry):
        """REQ-TX-002: a conflict with an existing registry entry is an explicit
        error — never an implicit overwrite, rename or silent reuse."""
        existing = http_server("conflict", "http://127.0.0.1:9100/original", token="original-token")
        stub_registry.reset(seed={"conflict": existing})

        agent = _name("imp-mcp-conflict")
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml(agent, mcp_servers=[http_server("conflict", "http://127.0.0.1:9101/other")]),
            agent,
        )
        assert resp.status_code in CONFLICT_CODES, resp.text
        assert stub_registry.app.entries["conflict"]["url"] == "http://127.0.0.1:9100/original", (
            "a name conflict must not overwrite the existing resource"
        )


class TestTransactionalImport:
    async def test_registry_unavailable_fails_without_partial_agent_or_resources(self, client, users, stub_registry):
        """REQ-TX-003: a registry outage is a controlled failure — no agent is
        created, no partial resource survives, no inline fallback is used."""
        agent = _name("imp-outage")
        stub_registry.reset(fail=True)
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml(
                agent,
                mcp_servers=[
                    http_server("search", "http://127.0.0.1:8443/mcp", token="sekrit-token"),
                ],
            ),
            agent,
        )
        assert resp.status_code in UNAVAILABLE_SERVICE_CODES, resp.text
        assert "sekrit" not in resp.text

        listing = await client.get("/api/v1/agents", headers=users["a"]["headers"])
        assert listing.status_code == 200
        assert agent not in [entry["name"] for entry in listing.json()], "no partially imported agent"
        assert await models_named(client, users["a"]["headers"], f"{agent}-model") == []
        assert await prompts_named(client, users["a"]["headers"], f"{agent}-prompt") == []

    async def test_failure_compensates_created_resources_and_leaves_no_orphans(self, client, users, stub_registry):
        """REQ-TX-001: when an import step fails, the resources created earlier
        in the same import are compensated; preexisting resources are unchanged.
        The error body must not leak secrets (user story / acceptance)."""
        conflict = http_server("taken", "http://127.0.0.1:9100/original", token="original-token")
        stub_registry.reset(seed={"taken": conflict})

        agent = _name("imp-compensate")
        token = "sekrit-compensation"
        resp = await import_agent(
            client,
            users["a"]["headers"],
            import_yaml(
                agent,
                mcp_servers=[
                    http_server("fresh-one", "http://127.0.0.1:9101/fresh", token=token),
                    http_server("taken", "http://127.0.0.1:9102/dup"),
                ],
            ),
            agent,
        )
        assert resp.status_code in CONFLICT_CODES, resp.text
        assert token not in resp.text

        listing = await client.get("/api/v1/agents", headers=users["a"]["headers"])
        assert agent not in [entry["name"] for entry in listing.json()], "the agent must not be created"
        assert "fresh-one" not in stub_registry.app.entries, "the created resource must be compensated"
        assert stub_registry.app.entries["taken"]["url"] == "http://127.0.0.1:9100/original", (
            "a preexisting resource is never modified or deleted by compensation"
        )
        assert await models_named(client, users["a"]["headers"], f"{agent}-model") == [], (
            "no orphan Model resource may survive a failed import"
        )
        assert await prompts_named(client, users["a"]["headers"], f"{agent}-prompt") == [], (
            "no orphan Prompt resource may survive a failed import"
        )
