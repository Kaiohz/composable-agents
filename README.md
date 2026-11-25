# composable-agents

Configure Deep Agent LangGraph agents in YAML and expose them via FastAPI.

**composable-agents** is a Python framework that lets you declare AI agents as simple YAML files and instantly expose them as a full-featured HTTP API. It is built on [deepagents](https://pypi.org/project/deepagents/) (LangGraph-based Deep Agent) with a strict hexagonal architecture, making every component testable and replaceable.

It also provides a local, per-user **prompt registry with immutable versions**, a metadata-only **model catalog**, validated **crons** that produce cron-sourced threads, and a global **trace summary** API. Phoenix may still be used as an optional tracing provider, but prompts are completely independent from Phoenix.

The server supports **multi-agent mode**: multiple agents are defined as separate YAML files in an `agents/` directory, each thread is bound to a specific agent at creation time, and agents are lazily instantiated on first use. Sub-agents declared in the `subagents` config are fully traced: every event emitted by a sub-agent carries its name in the `source` field of the corresponding `TraceEvent`, so the client can group events per sub-agent (see [TraceEvent](#traceevent-format)).

The server enforces **per-user isolation** via **dual authentication** (JWT via Logto OIDC **or** per-user API keys), **PostgreSQL Row-Level Security** (RLS) on every per-user table, **per-user LLM credentials** (each user brings their own provider/key), and **per-user LangGraph Store namespaces** for skills and memories. See [Authentication](#authentication), [Row-Level Security (RLS)](#row-level-security-rls), [Per-User LLM Settings](#per-user-llm-settings), and [Store Namespaces per User](#store-namespaces-per-user).

---

## Authentication

composable-agents uses **dual authentication**: every protected endpoint accepts **either** a JWT bearer token (validated against the Logto OIDC JWKS) **or** a per-user API key. The chosen credential determines the authenticated `user_id` that is propagated to PostgreSQL Row-Level Security and to the per-user LLM credential resolver.

### Methods

| Method | Header | `user_id` source | Validated by |
|---|---|---|---|
| **JWT** | `Authorization: Bearer <jwt>` | the JWT `sub` claim | Logto OIDC JWKS (`LOGTO_URL` + `JWT_AUDIENCES` / legacy `JWT_AUDIENCE`) |
| **API key** | `X-API-Key: cpk_...` | the `user_id` column on the matching `api_keys` row | SHA-256 hash lookup in the `api_keys` table |

The two methods are mutually exclusive on a given request — send **one** of the two headers. If neither validates, the server returns `401 {"detail": "Invalid or missing credentials"}`.

> **Deprecated for auth:** The old single master `OPENAI_API_KEY` / `API_KEY` settings are no longer used to authenticate requests. `OPENAI_API_KEY` is also no longer used to call the LLM — each user configures their own provider via [Per-User LLM Settings](#per-user-llm-settings).

### Obtaining a JWT (production)

In production, the user authenticates against **Logto** (via an oauth2-proxy in front of composable-agents) and receives a JWT. The `Authorization: Bearer <jwt>` header is then forwarded to composable-agents, which validates the signature against the Logto JWKS and the `aud` claim against the accepted audience list: `JWT_AUDIENCES` (comma-separated; legacy `JWT_AUDIENCE` is still honored as an extra audience). This lets several Logto applications (the oauth2-proxy UI client AND code-agent clients such as the Logto "OpenCode" native app) obtain accepted tokens. The JWT `sub` becomes the `user_id` for RLS and per-user LLM resolution. An invalid JWT never falls back to an API key — an `Authorization` header always means JWT.

### Obtaining a per-user API key

A user first authenticates with a JWT, then creates an API key they can reuse for script/automation use:

```bash
# Requires a JWT first (the user must be logged in via Logto).
curl -X POST http://localhost:8000/api/v1/api-keys \
  -H "Authorization: Bearer <jwt>" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-ci-key"}'
# → 201 Created
# {
#   "id": "8f3c...",
#   "name": "my-ci-key",
#   "key_prefix": "cpk_abcde",
#   "plaintext": "cpk_abcdefghijklmnopqrstuvwxyz0123456789...",  # shown ONCE
#   "created_at": "2026-07-27T10:00:00Z"
# }
```

The plaintext is returned **exactly once**; only its SHA-256 hash is persisted in `api_keys` (column `key_hash`). Store the plaintext securely — it cannot be recovered. The `cpk_...` prefix identifies composable-agents keys.

### QA / local mode (no Logto)

In QA and local dev, there is no Logto instance. The QA stack seeds two API keys directly in the `api_keys` table via the `composable-agents-qa-init` one-shot service (see [Testing](#testing)). The two seed keys are:

| `user_id` | plaintext `X-API-Key` |
|---|---|
| `qa-user-1` | `cpk_qa_test_key_12345` |
| `qa-user-2` | `cpk_qa_test_key_67890` |

When `LOGTO_URL` is empty, the JWT path is disabled and only the per-user API key path is active.

### Authenticated request examples

```bash
# JWT
curl http://localhost:8000/api/v1/agents \
  -H "Authorization: Bearer <jwt>"

# Per-user API key
curl http://localhost:8000/api/v1/agents \
  -H "X-API-Key: cpk_..."
```

---

## Composable MCP Control Plane

The same app exposes an authenticated, streamable-HTTP MCP endpoint at:

```text
/mcp
```

It is not an agent-only wrapper. It is the platform control plane and exposes typed tools for

- agent CRUD and one-shot execution (`composable_agents_*`);
- the per-user MCP-server registry (`composable_mcp_servers_*`);
- global skills and memories (`composable_skills_*`, `composable_memories_*`);
- prompts and immutable versions (`composable_prompts_*`, `composable_prompt_versions_*`);
- metadata-only model definitions (`composable_models_*`);
- crons and cron-run history (`composable_crons_*`, `composable_cron_runs_list`);
- thread history/traces (`composable_threads_*`, `composable_traces_*`);
- and the current-user profile (`composable_users_me`).

LLM settings, API keys, credential revelation, and cross-user administration are intentionally not exposed. Registry server list/get/create/update responses mask `auth_token`, `headers`, and `env`; secrets are write-only.

Each `composable_agents_run` call is a one-shot task: it creates a trusted thread with `source="mcp"`, persists the normal agent trace, and returns the final message plus the thread ID. There is no caller-selected thread ID, implicit conversation continuity, or MCP HITL resume in this version. If the agent stops in HITL, the response reports `awaiting_hitl` and the thread can be resumed in the UI.

### Connect an MCP client

Any MCP-over-streamable-HTTP client can connect with the same credential used by REST:

```json
{
  "mcpServers": {
    "composable-platform": {
      "url": "http://localhost:8000/mcp",
      "headers": { "X-API-Key": "cpk_..." }
    }
  }
}
```

Use the same `Authorization: Bearer <jwt>` header for JWT-based clients. An invalid presented JWT does not silently fall back to an API key.

### Native OAuth2 (MCP authorization spec, 2025-06-18 discovery)

The MCP endpoint speaks the MCP OAuth2 discovery chain for code agents (`OpenCode`, `Claude Code`, …):

1. an unauthenticated request to `/mcp` is answered `401` with
   `WWW-Authenticate: Bearer resource_metadata="<origin>/.well-known/oauth-protected-resource"`
   (plus `Cache-Control: no-store`, generic non-enumerating body);
2. `GET /.well-known/oauth-protected-resource` is public (CORS preflighted, no auth) and returns
   `{"resource": "<origin>/mcp", "authorization_servers": ["https://logto.soludev.tech/oidc"]}`;
3. a pre-registered **public Logto native app** (PKCE S256, no secret) issues resource-bound JWTs
   whose `aud` = the MCP resource indicator — validated offline against the cached JWKS.

Recommended `opencode.jsonc` for the composable-agents server:

```jsonc
{
  "mcp": {
    "servers": {
      "composables": {
        "type": "remote",
        "url": "https://composables-agent.soludev.tech/mcp",
        "oauth": {
          "client_id": "<logto-opencode-app-id>",
          "callback_port": 19876,
          "redirect_uri": "http://127.0.0.1:19876/callback"
        }
      }
    }
  }
}
```

Machine-to-machine access keeps using `oauth: false` + an `X-API-Key` header. The discovery origin is
`MCP_PUBLIC_BASE_URL` when set, else the request `Host` only when it is allow-listed in `MCP_ALLOWED_HOSTS`
(default-port suffixes `:443`/`:80` are stripped before the match; any other host fails closed).

### Built-in assistant

The virtual, centrally managed agent `composable-ai-assistant` is overlaid into every authenticated user's catalog. It is listed first and marked with `is_builtin=true`. Its name is reserved, creation is refused, and deletion is unavailable. Each user can store a personal customization through `PUT /api/v1/agents/composable-ai-assistant`; the response and all effective reads carry `is_customized=true` while the override exists. `DELETE /api/v1/agents/composable-ai-assistant/customization` removes only that user's override and restores the default. The assistant is manually selectable in the UI, uses that user's LLM credentials, and dynamically aggregates:

- the platform management/execution tools;
- the built-in Files, Classical RAG, Web, and Memories MCP servers;
- every MCP server registered by the current user;
- `create_pdf`; and
- the user's other composable agents through one-shot execution (excluding self-invocation).

A built-in customization is a full snapshot, but server invariants are re-applied at write time and read time: `tools`, `subagents`, and `mcp_servers` are rejected/forced empty because the dynamic toolset is authoritative, and the destructive-delete HITL floor must stay enabled. Its destructive delete tools are configured with HITL rules. Direct MCP management tool calls are explicit client commands and execute normally, subject to authorization.

---

## Quickstart (5 minutes)

### Prerequisites

- Python 3.11+
- [UV](https://docs.astral.sh/uv/) package manager
- PostgreSQL 15+ (required for thread, agent config, API key, and LLM-settings persistence, with Row-Level Security enabled)
- A **Logto** instance (OIDC issuer) for JWT validation in production — optional in QA/local (see [Authentication](#authentication))
- Each end-user configures their own LLM provider credentials via [Per-User LLM Settings](#per-user-llm-settings); there is no longer a server-wide LLM key

### Installation

```bash
git clone https://github.com/your-org/composable-agents.git
cd composable-agents
uv sync
cp .env.example .env
```

Edit `.env` and add your database and dual-auth credentials:

```dotenv
# PostgreSQL (required)
DATABASE_URL=postgresql://raganything:raganything@localhost:5433/raganything

# Dual auth (JWT via Logto OIDC + per-user API keys)
# Leave LOGTO_URL empty in local QA (only X-API-Key auth is used).
# In prod, set both to enable JWT validation.
LOGTO_URL=
JWT_AUDIENCE=

# Fernet key used to encrypt per-user LLM API keys at rest.
# Generate one with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
SECRET_ENCRYPTION_KEY=<your-fernet-key>
```

> **⚠️ Breaking change:** The `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DATABASE` environment variables have been replaced by a single `DATABASE_URL` variable. If you are upgrading from a previous version, construct your `DATABASE_URL` as `postgresql://<user>:<password>@<host>:<port>/<database>` and remove the old `POSTGRES_*` variables from your `.env`.

> **⚠️ Breaking change (auth):** The single master `X-API-Key` / `API_KEY` model has been replaced by **dual JWT + per-user API keys**. The `OPENAI_API_KEY` env var is no longer used to call the LLM — each user configures their own provider via `PUT /api/v1/settings/llm`. See [Authentication](#authentication) and [Per-User LLM Settings](#per-user-llm-settings).

> **⚠️ Breaking change (trace events):** The legacy `StreamEvent` SSE format and the `messages` table have been removed. The `/stream` and WebSocket endpoints now emit `TraceEvent` JSON objects. See [Breaking Changes](#breaking-changes) for migration details.

### Configure your agents

Each agent is a standalone YAML file inside the `agents/` directory. A minimal agent only needs a name. Create `agents/my-agent.yaml`:

```yaml
name: my-agent
```

Or use one of the provided examples in the `agents/` directory (see [Examples](#examples)).

### Validate the configuration

```bash
uv run python -m src validate agents/my-agent.yaml
```

### Launch the server

```bash
uv run python -m src.main serve
```

The API starts on `http://localhost:8000`. On startup, the server:

1. **Runs Alembic migrations** automatically to bring the database schema up to date.
2. **Initializes persistence** (PostgreSQL engine, MinIO store, agent seeding).
3. Reads the `AGENTS_DIR` environment variable (default: `./agents`) to discover available agents.

Agents are not loaded into memory until a thread references them for the first time.

### Test with curl

> Replace `<API_KEY>` with a per-user API key (e.g. `cpk_qa_test_key_12345` in QA) or use `-H "Authorization: Bearer <jwt>"`. See [Authentication](#authentication).

```bash
# Health check (public, no auth)
curl http://localhost:8000/health

# Create a thread bound to an agent (agent_name must match a YAML filename in agents/)
curl -X POST http://localhost:8000/api/v1/threads \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"agent_name": "my-agent"}'

# Send a message (replace <thread_id> with the id from the previous response)
curl -X POST http://localhost:8000/api/v1/chat/<thread_id> \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"message": "Hello, what can you do?"}'

# Stream a message (yields TraceEvent JSON objects, one per SSE line, ends with [DONE])
curl -N -X POST http://localhost:8000/api/v1/chat/<thread_id>/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"message": "Hello, what can you do?"}'

# Send a message with attachments (images → vision content blocks; documents
# require an agent whose MCP config declares an HTTP server ending with
# /files/mcp — they are uploaded to a temporary, user-isolated chat_tmp/
# RustFS prefix via the backend and deleted after the terminal response)
curl -X POST http://localhost:8000/api/v1/chat/<thread_id> \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"message": "Analyse cette image et ce document", "attachments": [
         {"filename": "capture.png", "content_type": "image/png", "content_base64": "<base64>"},
         {"filename": "rapport.pdf", "content_type": "application/pdf", "content_base64": "<base64>"}
       ]}'

# Get the full thread history (thread + turns grouped by turn_id)
curl http://localhost:8000/api/v1/threads/<thread_id>/history \
  -H "X-API-Key: <API_KEY>"

# Get the flat trace of events for a thread
curl http://localhost:8000/api/v1/threads/<thread_id>/trace \
  -H "X-API-Key: <API_KEY>"
```

### Attachments limits (environment variables)

| Variable | Default | Description |
|---|---|---|
| `CHAT_ATTACHMENT_MAX_COUNT` | `10` | Max attachments per chat request |
| `CHAT_ATTACHMENT_MAX_TOTAL_BYTES` | `52428800` | Max total decoded size (50 MiB) |
| `CHAT_ATTACHMENT_MAX_IMAGE_BYTES` | `10485760` | Max size per image (10 MiB) |
| `CHAT_ATTACHMENT_MAX_DOCUMENT_BYTES` | `52428800` | Max size per document (50 MiB) |
| `CREATE_PDF_MAX_MARKDOWN_CHARS` | `1000000` | Max Markdown length accepted by the `create_pdf` tool |

Present-but-invalid values stop startup; unset/empty values fall back silently to the defaults (hierarchy `image ≤ document ≤ total` enforced). These are declared in the `flux` GitOps repo as `prd/composables/{composable-agents,raganything}` ConfigMaps.

### Builtin `create_pdf` tool

`src.infrastructure.deepagent.pdf_tools:create_pdf` renders A4 PDFs from Markdown (headings, bold/italic/strikethrough, lists, inline/fenced code, tables, links, blockquotes, horizontal rules — no remote images) via `fpdf2` + `mistletoe` and stores them in the user's LangGraph Store at `/documents/<thread_id>/<uuid>/<filename>.pdf` (collision-free by construction). Tool result: `{"path": ..., "filename": ..., "mime_type": "application/pdf"}` — no binary in traces. Referencing in YAML:

```yaml
tools:
  - "src.infrastructure.deepagent.pdf_tools:create_pdf"
```

---

## Multi-Agent Architecture

composable-agents now supports running **multiple agents simultaneously**. Each agent is defined by a separate YAML file in the `agents/` directory. Sub-agents declared via `subagents` are traced individually: each `TraceEvent` they emit includes the sub-agent name in its `source` field, so the timeline can be grouped per sub-agent on the client side.

### How it works

1. **Discovery** -- On startup, the server scans `AGENTS_DIR` (default: `./agents`) for `.yaml` files. The filename (without extension) becomes the agent name.
2. **Thread creation** -- When creating a thread via `POST /api/v1/threads`, you specify an `agent_name`. If no matching YAML file exists, the API returns `404`.
3. **Lazy loading** -- The agent (LangGraph graph + runner) is created only when a thread first sends a message to it. Subsequent requests reuse the cached runner.
4. **Per-thread binding** -- Each thread is permanently bound to its agent. Different threads can use different agents.

### Key components

| Component | Location | Role |
|---|---|---|
| `AgentRegistry` (port) | `src/domain/ports/agent_registry.py` | Abstract interface for retrieving agent runners by name. |
| `DeepAgentRegistry` (adapter) | `src/infrastructure/deepagent/registry.py` | Scans `agents/` directory, creates and caches runners on demand. |
| `AgentNotFoundError` | `src/domain/errors/agent.py` | Raised when a requested agent name has no corresponding YAML file. |

### Example: two agents, two threads

> All requests require `X-API-Key` or `Authorization: Bearer` (see [Authentication](#authentication)).

```bash
# Create a thread using the research assistant agent
curl -X POST http://localhost:8000/api/v1/threads \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"agent_name": "research-assistant"}'
# Returns: {"id": "thread-1-uuid", "agent_name": "research-assistant", ...}

# Create another thread using the code reviewer agent
curl -X POST http://localhost:8000/api/v1/threads \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"agent_name": "code-reviewer"}'
# Returns: {"id": "thread-2-uuid", "agent_name": "code-reviewer", ...}

# Each thread talks to its own agent
curl -X POST http://localhost:8000/api/v1/chat/<thread-1-uuid> \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"message": "Summarize the latest research on transformers."}'

curl -X POST http://localhost:8000/api/v1/chat/<thread-2-uuid> \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"message": "Review this Python function for security issues."}'
```

---

## Configuration YAML Reference

Every agent is defined by a single YAML file validated against the `AgentConfig` Pydantic schema.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `string` (required) | -- | Unique agent name (1-100 characters). |
| `description` | `string` | `null` | Optional human-readable description of the agent (max 500 characters). Stored in the `agent_configs` table and exposed via `AgentConfigMetadata`. |
| `model` | `string` | `"claude-sonnet-4-5-20250929"` | LLM model identifier. See [Supported Models](#supported-models). |
| `system_prompt` | `string` | `null` | Inline system prompt. Mutually exclusive with `system_prompt_file`. |
| `system_prompt_file` | `string` | `null` | Path to a text file containing the system prompt (resolved relative to the YAML file). Mutually exclusive with `system_prompt`. |
| `tools` | `list[string]` | `[]` | Python tool references in `module.path:attribute` format. |
| `backend` | `BackendConfig` | `{"type": "state", "store_backend": "memory", "checkpoint_backend": "postgres"}` | Persistence backend. See [Backends](#backends). |
| `hitl` | `HITLConfig` | `{"rules": {}}` | Human-in-the-loop interrupt rules. |
| `memory` | `list[string]` | `[]` | Paths to memory files (e.g. `"./AGENTS.md"`). |
| `skills` | `list[string]` | `[]` | Paths to skill directories (e.g. `"./skills/"`). |
| `subagents` | `list[SubAgentConfig]` | `[]` | Sub-agent definitions for delegation. |
| `mcp_servers` | `list[McpServerConfig]` | `[]` | MCP server connections. See [MCP Servers](#mcp-servers). |
| `debug` | `bool` | `false` | Enable debug mode. |
| `response_format` | `dict` | `null` | Inline JSON Schema dict for structured output. See [Structured Output (response_format)](#structured-output-response_format). |

### SubAgentConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `string` (required) | -- | Sub-agent name. |
| `description` | `string` (required) | -- | Description of the sub-agent's role. |
| `instructions` | `string` | `null` | System prompt / instructions for the sub-agent. |
| `model` | `string` | `null` | Override model for this sub-agent. |
| `tools` | `list[string]` | `[]` | Tool references specific to this sub-agent. |
| `skills` | `list[string]` | `[]` | Skill paths for this sub-agent. |
| `mcp_servers` | `list[McpServerConfig]` | `[]` | MCP servers for this sub-agent. |
| `agent_ref` | `string` | `null` | Name of an existing agent to reference. When set, the backend resolves the referenced agent's `model`, `system_prompt`, `mcp_servers`, `tools`, and `response_format` at runner-build time. Explicit values on the `SubAgentConfig` override the referenced agent's values. See [Agent references (`agent_ref`)](#agent-references-agent_ref). |

### Agent references (`agent_ref`)

A `SubAgentConfig` can reference another existing agent by name via the `agent_ref` field. This lets you compose agents without duplicating their configuration.

**Resolution rules:**

- At runner-build time, the backend looks up the referenced agent's `AgentConfig` and copies its `model`, `system_prompt`, `mcp_servers`, `tools`, and `response_format` into the sub-agent.
- Any field set explicitly on the `SubAgentConfig` (e.g. `model`, `instructions`) **overrides** the value inherited from the referenced agent.
- **One level only** — the referenced agent's own `subagents` are ignored. References are not resolved recursively.

**Validation (enforced at create/update time):**

- **Self-reference** (`agent_ref` equal to the parent agent's `name`) is rejected with a `ConfigError`.
- **Non-existent reference** (`agent_ref` pointing to an agent that does not exist) is rejected with a `ConfigError`.

**Cache invalidation:**

When an agent is updated or deleted, the registry also invalidates any agents that reference it via `agent_ref` in their `subagents`. This ensures stale runners are rebuilt on next use.

Example:

```yaml
name: orchestrator
subagents:
  - name: researcher
    agent_ref: research-assistant   # inherits model, system_prompt, mcp_servers, tools, response_format
    instructions: "Focus on recent papers."  # overrides system_prompt
```

### McpServerConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `string` (required) | -- | Server identifier. |
| `transport` | `"stdio"` or `"http"` (required) | -- | Transport type. |
| `command` | `string` | `null` | Command to run (required for `stdio` transport). |
| `args` | `list[string]` | `[]` | Command arguments (for `stdio` transport). |
| `url` | `string` | `null` | Server URL (required for `http` transport). |
| `headers` | `dict[string, string]` | `{}` | HTTP headers (for `http` transport). |
| `env` | `dict[string, string]` | `{}` | Environment variables for the server process. Supports `${VAR_NAME}` syntax for resolving env vars. |

### HITL Rules

HITL rules map tool names to either a boolean or a detailed rule:

```yaml
hitl:
  rules:
    write_file: true                  # Simple: interrupt on any call
    execute:                          # Detailed: restrict allowed decisions
      allowed_decisions:
        - approve
        - reject
```

Allowed decisions: `approve`, `edit`, `reject`.

---

## Structured Output (`response_format`)

The `response_format` field in agent YAML configures structured output — forcing the LLM to reply with JSON conforming to a JSON Schema. Define it inline as a dict (a valid JSON Schema) in the agent's YAML:

```yaml
name: invoice-extractor
model: claude-sonnet-4-5-20250929
response_format:
  type: object
  properties:
    invoice_number: { type: string }
    total_cents: { type: integer }
    currency: { type: string, enum: ["USD", "EUR", "GBP"] }
    paid: { anyOf: [{ type: boolean }, { type: "null" }] }
  required: [invoice_number, total_cents, currency]
  additionalProperties: false
```

### Native passthrough to deepagents/langchain

The dict is passed **natively** to `create_deep_agent(response_format=dict)`. No custom tool injection or prompt instruction concatenation is performed — the previous "tool leurre" hack (`_create_response_tool`, `_JSON_TYPE_MAP`, `STRUCTURED_OUTPUT_INSTRUCTION`) has been deleted.

langchain uses an **`AutoStrategy`** internally to pick the right delivery mechanism based on the model name:

- **`ProviderStrategy`** — the schema is passed as a native provider parameter (Anthropic `response_format`/tool_use strict mode, OpenAI `response_format` with `json_schema`, etc.).
- **`ToolStrategy`** — when the provider does not support native structured output, langchain injects a real tool whose schema is the JSON Schema, and the LLM is asked to call it.

You do not need to choose the strategy yourself — `AutoStrategy` selects based on the configured `model`.

### `structured_response` delivery

When the LLM produces a structured response, it is attached to the `Message` as `structured_response` (a dict validated against the schema). The `AI_MESSAGE` trace event carries the structured payload **inside `content`** as a JSON-serialized `Message` — it is **not** placed in `metadata`. The `metadata` of an `AI_MESSAGE` now only contains `{"status": ...}`.

Clients consuming `AI_MESSAGE` events must JSON-parse `content` and read `structured_response` from the resulting `Message` object.

### Missing structured response

If the LLM fails to produce a structured response despite a `response_format` being configured:

- A warning is logged (`STRUCTURED_RESPONSE_MISSING`).
- `Message.structured_response` is set to `None`.
- **No error is raised** — the client decides how to handle the absence.

### Supported JSON Schema constructs

`schema_utils.py` converts the JSON Schema dict to a Pydantic model at agent build time. The converter supports:

- `type` (string, integer, number, boolean, object, array, string)
- `type: ["string", "null"]` — array form for nullable scalars
- `anyOf` — nullable fields (use `anyOf: [{type: <T>}, {type: "null"}]`)
- `enum` — maps to `Literal` on the Pydantic side
- `properties` / `required` — nested objects
- `items` — arrays of objects or scalars

Not supported (will raise at build time or be ignored):

- `$ref`, `$defs`
- `oneOf`, `allOf`
- `if` / `then` / `else`
- `dependentSchemas`
- `patternProperties`

### Provider strict-mode constraints

When `AutoStrategy` selects `ProviderStrategy` against a provider that enforces strict mode (e.g. Anthropic's strict tool-use), the schema must satisfy the provider's constraints or the request will be rejected:

- Set `additionalProperties: false` on every object (recommended default).
- Use `anyOf` for nullable fields — do not use `type: ["string", "null"]` for Anthropic strict mode; use `anyOf: [{type: "string"}, {type: "null"}]` instead.
- Do **not** put `default` on `required` fields (Anthropic strict mode forbids it).
- All properties listed in `required` must appear in `properties`.

These constraints only apply when the provider enforces strict mode; `ToolStrategy` is more permissive. Since `AutoStrategy` picks automatically, authoring schemas that satisfy the strict constraints up front is the safest approach.

---


| Provider | Format | Example |
|---|---|---|
| Anthropic | `claude-<variant>` | `claude-sonnet-4-5-20250929` |
| OpenAI | `openai:<model>` | `openai:gpt-4o` |
| Google | `google_genai:<model>` | `google_genai:gemini-2.0-flash` |

The default model is `claude-sonnet-4-5-20250929`.

For OpenAI-compatible endpoints (OpenRouter, LiteLLM, vLLM, etc.), set the `OPENAI_BASE_URL` environment variable to point to your endpoint. The OpenAI SDK reads this variable automatically.

---

## MCP Servers

Agents can connect to [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers for tool access. MCP servers are defined in the agent's YAML config:

> **Registry migration:** The MCP server **registry** (CRUD API, OpenAPI→MCP generation, Swagger 2.0 conversion, Fernet encryption, mounter, startup rehydration) **and the `mcp_servers` table schema** used to live in this brick. Both have been **moved to [mcp-raganything](https://github.com/soludev/mcp-raganything)**, which now owns the `/api/v1/mcp/servers` REST surface and the table's Alembic migrations (`001_create_mcp_servers_table`, tracked in the `raganything_alembic_version` table). composable-agents no longer ships the registry routes, use cases, repository, OpenAPI factory, Swagger 2.0 converter, Fernet cipher, mounter, or any `mcp_servers` migration.
>
> What remains in this brick is:
>
> - `McpServerConfig` — the entity used by agent YAML to declare an MCP server connection.
> - `McpToolLoader` (port) / `LangchainMcpToolLoader` (adapter) — loads tools from an MCP server **by URL** at agent build time. The URLs come from the agent YAML directly, or from entries registered in mcp-raganything's registry (which composable-agents reads via its UI/backend when needed).
> - `McpConnectionError` / `McpToolLoadError` — domain errors raised when an MCP server cannot be reached or its tools fail to load.
>
> `SECRET_ENCRYPTION_KEY` is now **optional** in composable-agents (it is no longer used here); it must still be set on the mcp-raganything service that owns the registry.

```yaml
name: mcp-agent
model: claude-sonnet-4-5-20250929
system_prompt: "You are an agent with MCP tool access."
mcp_servers:
  - name: filesystem
    transport: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

Environment variables in `env` fields support the `${VAR_NAME}` resolution syntax.

### Authenticating to MCP servers with API keys

When an MCP server requires authentication via a custom HTTP header (e.g. `X-API-Key`), use the `headers` field. The `${VAR_NAME}` syntax is supported for resolving environment variables at runtime:

```yaml
mcp_servers:
  - name: bricks
    transport: http
    url: https://raganything.soludev.tech/bricks/mcp
    headers:
      X-API-Key: "${MCP_RAGANYTHING_API_KEY}"
```

Set the corresponding environment variable in `.env`:

```dotenv
MCP_RAGANYTHING_API_KEY=your-shared-secret-key
```

The value must match the `API_KEY` configured on the [mcp-raganything](https://github.com/soludev/mcp-raganything) server.

### Per-user credential propagation (`${USER_JWT}` / `${USER_API_KEY}`)

In addition to env-var placeholders, MCP server `headers` support two **per-user** placeholders that are resolved from the authenticated caller's credential at agent-build time:

| Placeholder | Resolved to | When |
|---|---|---|
| `${USER_JWT}` | the caller's raw JWT (without `Bearer ` prefix) | the request was authenticated with a JWT |
| `${USER_API_KEY}` | the caller's raw per-user API key (`cpk_...`) | the request was authenticated with an API key |

This lets an agent forward the caller's identity to a downstream MCP server (e.g. raganything) without storing a shared secret, so the downstream server can apply its own per-user RLS:

```yaml
mcp_servers:
  - name: raganything
    transport: http
    url: https://raganything.soludev.tech/bricks/mcp
    headers:
      Authorization: "Bearer ${USER_JWT}"
      X-API-Key: "${USER_API_KEY}"
```

Resolution rules:

- The placeholder is filled with `current_credential` only when `current_auth_method` matches (`"jwt"` for `${USER_JWT}`, `"api_key"` for `${USER_API_KEY}`).
- If the method does not match (e.g. the caller used an API key but the header uses `${USER_JWT}`), or no credential is set, the resolved value is **empty**.
- **Empty resolved headers are dropped** from the outgoing MCP request, so the downstream server receives no spurious empty auth header.

---

## Middlewares

The Todo List, Filesystem, and Sub-Agent middlewares are **always installed** by `create_deep_agent` defaults and cannot be toggled via the YAML configuration. Sub-agent delegation is enabled automatically when `subagents` is non-empty.

---

## Backends

The `BackendConfig` schema controls where agent state and checkpoints are persisted.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `BackendType` (`state` \| `store`) | `state` | Backend kind. `state` = in-memory LangGraph state, `store` = LangGraph store-backed. |
| `store_backend` | `Literal["memory", "postgres"]` | `memory"` | Where the LangGraph store lives. `memory` = in-process, `postgres` = PostgreSQL-backed (singleton reused across all agent builds). |
| `checkpoint_backend` | `Literal["memory", "postgres"]` | `"postgres"` | Where the LangGraph checkpointer lives. `memory` = in-process, `postgres` = PostgreSQL-backed (singleton reused across all agent builds). Defaults to `postgres` for durability; falls back to `memory` with a warning if Postgres is unreachable at agent-build time. |

### Supported `type` values

| Name | Enum Value | Description |
|---|---|---|
| State | `state` | Default in-memory state backend (no extra config). |
| Store | `store` | LangGraph store-based backend. The store itself is backed by `store_backend`. |

### Postgres-backed store and checkpointer

When `store_backend` or `checkpoint_backend` is set to `"postgres"`, the framework instantiates a single `PostgresStore` / `PostgresSaver` (from `langgraph-checkpoint-postgres`) and **reuses the same instance across every agent build**. This avoids opening a new connection pool per agent.

> **Note:** The `langgraph-checkpoint-postgres` package is required and is included in the project dependencies.

Example YAML enabling Postgres for both store and checkpointer:

```yaml
name: persistent-agent
backend:
  type: store
  store_backend: postgres
  checkpoint_backend: postgres
```

The `StoreBackend` is wired with a per-run namespace via `StoreBackend(store=store, namespace=lambda r: ("filesystem",))` (the deprecated `StoreBackend(runtime)` pattern has been removed).

---

## API Reference

All endpoints are prefixed appropriately. The server runs on `http://localhost:8000` by default.

> **Auth:** Every endpoint **except** `GET /health` requires either `Authorization: Bearer <jwt>` or `X-API-Key: cpk_...` (see [Authentication](#authentication)). The `X-API-Key` shown in the curl examples below is a placeholder — replace it with your per-user key. Each authenticated user only sees rows they own (see [Row-Level Security (RLS)](#row-level-security-rls)).

| Method | Path | Description | Success Status |
|---|---|---|---|
| `GET` | `/health` | Health check (public, no auth) | `200` |
| `POST` | `/api/v1/api-keys` | Create a new per-user API key (returns plaintext once) | `201` |
| `GET` | `/api/v1/api-keys` | List the authenticated user's API keys (no plaintext) | `200` |
| `DELETE` | `/api/v1/api-keys/{key_id}` | Revoke a per-user API key (idempotent) | `204` |
| `GET` | `/api/v1/settings/llm` | Get the authenticated user's LLM provider settings (masked key) | `200` |
| `PUT` | `/api/v1/settings/llm` | Insert or update the authenticated user's LLM provider settings | `200` |
| `DELETE` | `/api/v1/settings/llm` | Delete the authenticated user's LLM provider settings (idempotent) | `204` |
| `POST` | `/api/v1/threads` | Create a new conversation thread (bound to an agent) | `201` |
| `GET` | `/api/v1/threads` | List all threads owned by the authenticated user | `200` |
| `GET` | `/api/v1/threads/{thread_id}` | Get a specific thread | `200` |
| `DELETE` | `/api/v1/threads/{thread_id}` | Delete a thread | `204` |
| `GET` | `/api/v1/threads/{thread_id}/history` | Get thread history grouped by turn (`ThreadHistory`) | `200` |
| `GET` | `/api/v1/threads/{thread_id}/trace` | Get the flat list of `TraceEvent`s for a thread | `200` |
| `GET` | `/api/v1/threads/{thread_id}/messages` | List messages in a thread (projection from `trace_events`: `HUMAN_MESSAGE` + `AI_MESSAGE` only, backward-compat) | `200` |
| `POST` | `/api/v1/chat/{thread_id}` | Send a message and get the full response | `200` |
| `POST` | `/api/v1/chat/{thread_id}/stream` | Send a message and stream the response (SSE) | `200` |
| `POST` | `/api/v1/chat/{thread_id}` | Submit a human-in-the-loop decision (approve/reject/edit, single or multi `decisions`) | `200` |
| `GET` | `/api/v1/agents` | List all agent configs from `agents/` directory | `200` |
| `GET` | `/api/v1/agents/{agent_name}` | Get a specific agent configuration | `200` |
| `POST` | `/api/v1/agents` | Create a custom agent from multipart YAML (`agent_name` + `file`) | `201` |
| `PUT` | `/api/v1/agents/{agent_name}` | Replace a custom agent, or store the user's built-in customization | `200` |
| `DELETE` | `/api/v1/agents/{agent_name}` | Delete a custom agent (built-in returns `403`) | `204` |
| `DELETE` | `/api/v1/agents/{agent_name}/customization` | Reset the current user's built-in customization (other names return `404`) | `204` |
| `GET` | `/api/v1/store/files` | List file paths in the store (optional `prefix` query param) | `200` |
| `GET` | `/api/v1/store/files/{path}` | Get a single file's content by path | `200` |
| `PUT` | `/api/v1/store/files/{path}` | Create or replace a file in the store | `200` |
| `DELETE` | `/api/v1/store/files/{path}` | Delete a file from the store | `204` |
| `WS` | `/api/v1/ws/{thread_id}` | WebSocket endpoint for streaming chat | -- |
| `GET` | `/api/v1/models` | List the authenticated user's identity-only model catalog entries | `200` |
| `POST` | `/api/v1/models` | Create an identity-only model entry (`name` + `provider_model`) | `201` |
| `GET` | `/api/v1/models/{id}` | Get a reusable model entry | `200` |
| `PATCH` | `/api/v1/models/{id}` | Edit `name` and/or `provider_model` | `200` |
| `DELETE` | `/api/v1/models/{id}` | Delete an unreferenced model entry (in-use returns `409`; malformed IDs return `404`) | `204` |
| `GET` | `/api/v1/prompts` | List the authenticated user's prompts | `200` |
| `POST` | `/api/v1/prompts` | Create a prompt (creates immutable version 1) | `201` |
| `GET` | `/api/v1/prompts/{id}` | Get prompt metadata | `200` |
| `PATCH` | `/api/v1/prompts/{id}` | Edit name/description only (`description: null` clears it) | `200` |
| `DELETE` | `/api/v1/prompts/{id}` | Delete an unreferenced prompt (in-use returns `409`) | `204` |
| `GET` | `/api/v1/prompts/{id}/versions` | List immutable version summaries | `200` |
| `POST` | `/api/v1/prompts/{id}/versions` | Create the next immutable prompt version | `201` |
| `GET` | `/api/v1/prompts/{id}/versions/{version}` | Get one prompt version's verbatim content | `200` |
| `GET` | `/api/v1/crons` | List the authenticated user's cron jobs | `200` |
| `POST` | `/api/v1/crons` | Create a five-field cron job (validated agent/message/timezone) | `201` |
| `GET` | `/api/v1/crons/{id}` | Get a cron job and its scheduling state | `200` |
| `PATCH` | `/api/v1/crons/{id}` | Edit a cron job, including enable/disable | `200` |
| `DELETE` | `/api/v1/crons/{id}` | Delete a cron job and stop future scheduling | `204` |
| `POST` | `/api/v1/crons/{id}/run` | Trigger **Run now** (`202`); concurrent call returns `409` | `202` |
| `GET` | `/api/v1/crons/{id}/runs` | List paginated run history (`limit` 1-200) | `200` |
| `GET` | `/api/v1/traces` | List global trace summaries; optional exact `agent_name`, cursor pagination | `200` |
| `GET` | `/api/v1/traces/usage` | LLM usage aggregates (tokens/cost/duration/errors) for a window, optional exact `agent_name`, breakdown `series` | `200` |
| `GET` | `/api/v1/traces/usage/top` | Top-N most expensive LLM runs for a window (ranked by cost) | `200` |

### Error Responses

| Status | Condition |
|---|---|
| `400` | General configuration error |
| `401` | Missing or invalid credentials (no JWT, no API key, or unknown/revoked key) |
| `404` | Thread not found, agent not found, or config file not found |
| `422` | Validation error (bad request body, invalid config schema, **or no LLM provider configured for the user** — `LlmNotConfiguredError`) |
| `502` | Agent execution error (LLM failure) |
| `500` | Unexpected domain error |

---

## curl Examples

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Response:

```json
{"status": "ok"}
```

### 2. List Available Agents

```bash
curl http://localhost:8000/api/v1/agents \
  -H "X-API-Key: <API_KEY>"
```

Response (`200`):

```json
[
  {
    "name": "code-reviewer",
    "model": "claude-sonnet-4-5-20250929",
    "system_prompt": "You are an expert code reviewer...",
    "tools": [],
    "backend": {"type": "state", "store_backend": "memory", "checkpoint_backend": "memory"},
    "hitl": {"rules": {"write_file": true, "execute": {"allowed_decisions": ["approve", "reject"]}}},
    "subagents": [...]
  },
  {
    "name": "example-agent",
    "model": "openai:anthropic/claude-haiku-4.5:nitro",
    "system_prompt": "You are a helpful assistant.",
    "tools": [],
    "backend": {"type": "state", "store_backend": "memory", "checkpoint_backend": "memory"},
    "hitl": {"rules": {}},
    "subagents": []
  }
]
```

### 3. Get a Specific Agent Configuration

```bash
curl http://localhost:8000/api/v1/agents/example-agent \
  -H "X-API-Key: <API_KEY>"
```

Response (`200`):

```json
{
  "name": "example-agent",
  "model": "openai:anthropic/claude-haiku-4.5:nitro",
  "system_prompt": "You are a helpful assistant.",
  "system_prompt_file": null,
  "tools": [],
  "backend": {"type": "state", "store_backend": "memory", "checkpoint_backend": "memory"},
  "hitl": {"rules": {}},
  "memory": [],
  "skills": [],
  "subagents": [],
  "mcp_servers": [],
  "debug": false
}
```

If the agent does not exist:

```bash
curl http://localhost:8000/api/v1/agents/nonexistent \
  -H "X-API-Key: <API_KEY>"
```

Response (`404`):

```json
{"detail": "Fichier de configuration introuvable: agents/nonexistent.yaml"}
```

### 3a. Personalize and reset the built-in assistant

Store (or replace) the authenticated user's built-in customization:

```bash
cat > composable-ai-assistant.yaml <<'EOF'
name: composable-ai-assistant
model: openai:gpt-4o
description: "My personal platform assistant"
system_prompt: "You are my personal Composable assistant."
tools: []
debug: false
EOF

curl -X PUT http://localhost:8000/api/v1/agents/composable-ai-assistant \
  -H "X-API-Key: <API_KEY>" \
  -F "file=@composable-ai-assistant.yaml;type=text/yaml"
```

Response (`200`):

```json
{
  "name": "composable-ai-assistant",
  "model": "openai:gpt-4o",
  "description": "My personal platform assistant",
  "system_prompt": "You are my personal Composable assistant.",
  "tools": [],
  "is_builtin": true,
  "is_customized": true
}
```

Reset only the current user's customization (idempotent):

```bash
curl -X DELETE http://localhost:8000/api/v1/agents/composable-ai-assistant/customization \
  -H "X-API-Key: <API_KEY>"
```

Response: `204 No Content`.

Server-managed built-in fields are rejected or forced: non-empty `tools`, `subagents`, or `mcp_servers` return `422`, and disabling a destructive-delete HITL rule returns `422`. Deleting the assistant itself returns `403`.

### 4. Create a Thread

The `agent_name` must match an existing YAML filename (without the `.yaml` extension) in the `agents/` directory.

```bash
curl -X POST http://localhost:8000/api/v1/threads \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"agent_name": "example-agent"}'
```

Response (`201`):

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "agent_name": "example-agent",
  "messages": [],
  "created_at": "2025-01-15T10:30:00.000000",
  "updated_at": "2025-01-15T10:30:00.000000"
}
```

If the agent name does not match any YAML file:

```bash
curl -X POST http://localhost:8000/api/v1/threads \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"agent_name": "nonexistent-agent"}'
```

Response (`404`):

```json
{"detail": "Agent introuvable: nonexistent-agent"}
```

### 5. Send a Message

```bash
curl -X POST http://localhost:8000/api/v1/chat/a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"message": "Explain the hexagonal architecture pattern in 3 sentences."}'
```

Response (`200`):

```json
{
  "role": "ai",
  "content": "Hexagonal architecture separates core business logic from external concerns...",
  "timestamp": "2025-01-15T10:30:05.000000",
  "tool_calls": null,
  "tool_call_id": null
}
```

### 6. Stream a Message (SSE)

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/a1b2c3d4-e5f6-7890-abcd-ef1234567890/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"message": "Write a haiku about programming."}'
```

Response (Server-Sent Events, one `TraceEvent` JSON object per line):

```
data: {"id":"...","thread_id":"...","turn_id":"...","type":"HUMAN_MESSAGE","source":null,"name":null,"content":"Write a haiku about programming.","metadata":{},"timestamp":"2025-04-24T10:30:00.000000Z","sequence":0}

data: {"id":"...","thread_id":"...","turn_id":"...","type":"THINKING","source":null,"name":null,"content":"Hmm, a haiku needs 5-7-5 syllables...","metadata":{},"timestamp":"...","sequence":1}

data: {"id":"...","thread_id":"...","turn_id":"...","type":"CONTENT","source":null,"name":null,"content":"Lines","metadata":{},"timestamp":"...","sequence":2}

data: {"id":"...","thread_id":"...","turn_id":"...","type":"CONTENT","source":null,"name":null,"content":" of","metadata":{},"timestamp":"...","sequence":3}

data: {"id":"...","thread_id":"...","turn_id":"...","type":"CONTENT","source":null,"name":null,"content":" code","metadata":{},"timestamp":"...","sequence":4}

data: {"id":"...","thread_id":"...","turn_id":"...","type":"AI_MESSAGE","source":null,"name":null,"content":"Lines of code align...","metadata":{},"timestamp":"...","sequence":5}

data: [DONE]
```

#### `TraceEvent` format

Each `data:` line (except the final `[DONE]`) is a JSON-serialized `TraceEvent` with the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Unique event ID. |
| `thread_id` | `string` | Owning thread ID. |
| `turn_id` | `string` | Turn ID grouping all events from one user message to the next AI reply. |
| `type` | `enum` | One of `HUMAN_MESSAGE`, `AI_MESSAGE`, `THINKING`, `CONTENT`, `TOOL_CALL`, `TOOL_RESULT`. |
| `source` | `string \| null` | Sub-agent name when the event was emitted by a sub-agent, `null` for the parent agent. |
| `name` | `string \| null` | Tool name (only for `TOOL_CALL` / `TOOL_RESULT`). |
| `content` | `string` | Text payload (message text, thinking text, content chunk, tool arguments/result). |
| `metadata` | `object` | Additional structured data (e.g. tool call ID, `{"status": ...}` for `AI_MESSAGE`). The structured response for an `AI_MESSAGE` is **not** in `metadata` — it lives in `content` as part of the JSON-serialized `Message`. See [Structured Output](#structured-output-response_format). |
| `timestamp` | `string` | ISO 8601 timestamp. |
| `sequence` | `int` | Monotonic sequence number within the thread (ordering). |

#### Error payload

On error the stream emits a single JSON object (NOT a valid `TraceEvent`) followed by `[DONE]`:

```
data: {"type":"error","data":"Agent execution failed: ..."}

data: [DONE]
```

Clients should check for `type === "error"` before parsing as `TraceEvent`.

#### Rendering guidance

- Render `THINKING` events in a collapsible reasoning panel.
- Append `CONTENT` events directly to the chat bubble (or the relevant sub-agent panel when `source` is set).
- Render `TOOL_CALL` as a badge and `TOOL_RESULT` as a terminal-style block.
- Group events by `source` to display sub-agent panels separately.
- Wait for the `AI_MESSAGE` event to finalize the turn.

This design prevents Cloudflare timeout issues (~100s on idle connections) because chunks and SSE pings (every 15s) keep the connection active.

### 6b. Get Thread History

```bash
curl http://localhost:8000/api/v1/threads/a1b2c3d4-e5f6-7890-abcd-ef1234567890/history \
  -H "X-API-Key: <API_KEY>"
```

Response (`200`) — `ThreadHistory`:

```json
{
  "thread": {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "agent_name": "example-agent",
    "created_at": "2025-01-15T10:30:00.000000",
    "updated_at": "2025-01-15T10:30:05.000000"
  },
  "turns": [
    {
      "turn_id": "turn-uuid-1",
      "human_message": { "id": "...", "type": "HUMAN_MESSAGE", "content": "Hello", ... },
      "ai_message": { "id": "...", "type": "AI_MESSAGE", "content": "Hi!", ... },
      "events": [
        { "id": "...", "type": "THINKING", "source": null, "content": "...", ... },
        { "id": "...", "type": "TOOL_CALL", "source": "researcher", "name": "search", ... },
        { "id": "...", "type": "TOOL_RESULT", "source": "researcher", "name": "search", ... }
      ]
    }
  ]
}
```

`events` contains the intermediate events (`THINKING`, `CONTENT`, `TOOL_CALL`, `TOOL_RESULT`) for the turn, in `sequence` order. `human_message` and `ai_message` are the terminal `HUMAN_MESSAGE` / `AI_MESSAGE` events.

### 6c. Get Flat Trace

```bash
curl http://localhost:8000/api/v1/threads/a1b2c3d4-e5f6-7890-abcd-ef1234567890/trace \
  -H "X-API-Key: <API_KEY>"
```

Response (`200`):

```json
{
  "events": [
    { "id": "...", "type": "HUMAN_MESSAGE", "content": "Hello", ... },
    { "id": "...", "type": "THINKING", "content": "...", ... },
    { "id": "...", "type": "AI_MESSAGE", "content": "Hi!", ... }
  ]
}
```

Returns the full flat list of `TraceEvent`s for the thread, ordered by `sequence`.

### 7. List All Threads

```bash
curl http://localhost:8000/api/v1/threads \
  -H "X-API-Key: <API_KEY>"
```

Response (`200`):

```json
[
  {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "agent_name": "example-agent",
    "messages": [],
    "created_at": "2025-01-15T10:30:00.000000",
    "updated_at": "2025-01-15T10:30:00.000000"
  },
  {
    "id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
    "agent_name": "research-assistant",
    "messages": [],
    "created_at": "2025-01-15T10:31:00.000000",
    "updated_at": "2025-01-15T10:31:00.000000"
  }
]
```

### 8. Get a Specific Thread

```bash
curl http://localhost:8000/api/v1/threads/a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "X-API-Key: <API_KEY>"
```

### 9. List Messages in a Thread

```bash
curl http://localhost:8000/api/v1/threads/a1b2c3d4-e5f6-7890-abcd-ef1234567890/messages \
  -H "X-API-Key: <API_KEY>"
```

Response (`200`):

```json
[
  {
    "role": "human",
    "content": "Explain the hexagonal architecture pattern in 3 sentences.",
    "timestamp": "2025-01-15T10:30:00.000000",
    "tool_calls": null,
    "tool_call_id": null
  },
  {
    "role": "ai",
    "content": "Hexagonal architecture separates core business logic from external concerns...",
    "timestamp": "2025-01-15T10:30:05.000000",
    "tool_calls": null,
    "tool_call_id": null
  }
]
```

### 10. HITL -- Approve a Pending Tool Call

When the agent is configured with HITL rules and one or more tool calls are
interrupted, submit decisions via `POST /api/v1/chat/{thread_id}`. The preferred
payload is a `decisions` list (one entry per interrupted tool call):

```bash
curl -X POST http://localhost:8000/api/v1/chat/a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{
    "decisions": [
      {"tool_call_id": "call_abc123", "action": "approve"}
    ]
  }'
```

Response (`200`):

```json
{
  "role": "ai",
  "content": "Action approved. Proceeding with file write.",
  "timestamp": "2025-01-15T10:31:00.000000",
  "tool_calls": null,
  "status": "completed"
}
```

A legacy single-decision shape (`tool_call_id` + `action`) is still accepted and
internally converted to a one-element `decisions` list.

### 11. HITL -- Reject a Pending Tool Call

```bash
curl -X POST http://localhost:8000/api/v1/chat/a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{
    "decisions": [
      {"tool_call_id": "call_abc123", "action": "reject", "reason": "This operation is too risky for production."}
    ]
  }'
```

### 12. HITL -- Edit and Approve a Pending Tool Call

```bash
curl -X POST http://localhost:8000/api/v1/chat/a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{
    "decisions": [
      {"tool_call_id": "call_abc123", "action": "edit", "edits": {"filename": "safe_output.txt", "content": "sanitized content"}}
    ]
  }'
```

### 12b. HITL -- Multiple Tool Calls in One Resume

When several tool calls are interrupted in the same turn, provide one decision
per tool call (positional order matches the interrupted actions):

```bash
curl -X POST http://localhost:8000/api/v1/chat/a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{
    "decisions": [
      {"tool_call_id": "call_001", "action": "approve"},
      {"tool_call_id": "call_002", "action": "reject", "reason": "Not safe"}
    ]
  }'
```

### 13. Delete a Thread

```bash
curl -X DELETE http://localhost:8000/api/v1/threads/a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "X-API-Key: <API_KEY>"
```

Response: `204 No Content`

### 14. Prompt Registry (Phoenix-independent)

Prompts use a local, per-user PostgreSQL registry. Phoenix is never used for prompt storage, lookup, versioning, or agent fallback.

#### 14.1 Create a Prompt

```bash
curl -X POST http://localhost:8000/api/v1/prompts \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{
    "name": "customer-support",
    "description": "General customer support system prompt",
    "content": "You are a helpful customer support agent. Be polite and professional."
  }'
```

Response (`201`):

```json
{
  "id": "7e6a56fd-83e7-46a1-a2f2-f5c5f8d9a52e",
  "name": "customer-support",
  "description": "General customer support system prompt",
  "latest_version": 1,
  "created_at": "2026-11-15T10:30:00.000000Z",
  "updated_at": "2026-11-15T10:30:00.000000Z"
}
```

Creating a prompt also creates immutable version 1. Version content is retrievable separately.

#### 14.2 Get Prompt Metadata

```bash
curl http://localhost:8000/api/v1/prompts/7e6a56fd-83e7-46a1-a2f2-f5c5f8d9a52e \
  -H "X-API-Key: <API_KEY>"
```

#### 14.3 Edit Metadata (no new version)

```bash
curl -X PATCH http://localhost:8000/api/v1/prompts/7e6a56fd-83e7-46a1-a2f2-f5c5f8d9a52e \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"description": "More detailed support prompt"}'
```

`{"description": null}` clears the description. `name: null` is rejected with `422`.

#### 14.4 Create Version 2

```bash
curl -X POST http://localhost:8000/api/v1/prompts/7e6a56fd-83e7-46a1-a2f2-f5c5f8d9a52e/versions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"content": "You are a knowledgeable, thorough, and polite support agent."}'
```

Response (`201`):

```json
{
  "prompt_id": "7e6a56fd-83e7-46a1-a2f2-f5c5f8d9a52e",
  "version": 2,
  "created_at": "2026-11-15T10:31:00.000000Z"
}
```

#### 14.5 List Summary History

```bash
curl http://localhost:8000/api/v1/prompts/7e6a56fd-83e7-46a1-a2f2-f5c5f8d9a52e/versions \
  -H "X-API-Key: <API_KEY>"
```

#### 14.6 Read Immutable Version Content

```bash
curl http://localhost:8000/api/v1/prompts/7e6a56fd-83e7-46a1-a2f2-f5c5f8d9a52e/versions/1 \
  -H "X-API-Key: <API_KEY>"
```

#### 14.7 Referencing a Prompt from Agent YAML

```yaml
name: research-agent
prompt_ref:
  id: 7e6a56fd-83e7-46a1-a2f2-f5c5f8d9a52e
  # version: 1  # Omit for Latest; specify to pin.
```

Rules:

- `prompt_ref` is mutually exclusive with `system_prompt` and `system_prompt_file`.
- Omitted `version` resolves the prompt's current `latest_version`.
- A supplied version is immutable.
- Missing/inaccessible references fail explicitly; there is no Phoenix fallback or old agent-name lookup.
- Deleting a referenced prompt returns `409`.

#### 14.8 Delete a Prompt

```bash
curl -X DELETE http://localhost:8000/api/v1/prompts/7e6a56fd-83e7-46a1-a2f2-f5c5f8d9a52e \
  -H "X-API-Key: <API_KEY>"
```

Success: `204`. A referenced prompt returns `409`.

### 15. Model Catalog (identity-only)

Models are reusable, per-user identity records: a display name plus the `provider_model` identifier to execute. The provider connection (`provider`, `base_url`, and `api_key`) is owned exclusively by [Per-User LLM Settings](#per-user-llm-settings). Model records contain **no provider endpoints, API keys, or other credentials**.

#### 15.1 Create a Model

```bash
curl -X POST http://localhost:8000/api/v1/models \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{
    "name": "GLM Flash via OpenRouter",
    "provider_model": "z-ai/glm-5.3-flash:nitro"
  }'
```

Unknown or removed fields such as `provider` and `base_url` are rejected with `422`. The response exposes only `id`, `name`, `provider_model`, `created_at`, and `updated_at`.

#### 15.2 Edit a Model

```bash
curl -X PATCH http://localhost:8000/api/v1/models/<model_id> \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"provider_model": "z-ai/glm-5.3-flash"}'
```

Send at least one editable field; omitted fields keep their current value. Editing `provider_model` affects referencing agents on their next invocation.

#### 15.3 Reference a Model from Agent YAML

```yaml
name: research-agent
model_ref: 1a5bca0d-0f14-4ed6-98db-b07c39e0d4a2
```

`model_ref` takes precedence over legacy `model` for catalog selection; an invalid reference is an error, not a fallback.

#### 15.4 Delete a Model

```bash
curl -X DELETE http://localhost:8000/api/v1/models/<model_id> \
  -H "X-API-Key: <API_KEY>"
```

Success: `204`. A referenced model returns `409`.

### 16. Crons

Crons are per-user scheduled agent invocations. Standard five-field expressions and IANA timezones are validated. Every accepted run creates a new thread with immutable `source="cron"`.

#### 16.1 Create a Cron

```bash
curl -X POST http://localhost:8000/api/v1/crons \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{
    "name": "daily-briefing",
    "agent_name": "research-agent",
    "message": "Summarize the latest papers.",
    "cron_expression": "0 9 * * *",
    "timezone": "America/New_York",
    "enabled": true
  }'
```

#### 16.2 Edit or Enable/Disable

```bash
curl -X PATCH http://localhost:8000/api/v1/crons/<cron_id> \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"enabled": false}'
```

#### 16.3 Run Now

```bash
curl -X POST http://localhost:8000/api/v1/crons/<cron_id>/run \
  -H "X-API-Key: <API_KEY>"
```

Response: `202`. A concurrent manual run returns `409`; disabled jobs may still be run manually.

#### 16.4 Run History

```bash
curl 'http://localhost:8000/api/v1/crons/<cron_id>/runs?limit=50' \
  -H "X-API-Key: <API_KEY>"
```

Run records expose trigger, terminal status (`succeeded`/`failed`/`skipped`/etc.), `occurrence_at`, `thread_id`, timestamps, and a safe error summary. The linked `thread_id` can be opened in traces.

Policies:

- Scheduled and manual runs use one execution path.
- Database atomic claiming prevents duplicate execution across workers.
- Overlapping occurrences are skipped; downtime occurrences coalesce into at most one run.
- Failures are terminal; there is no automatic retry.
- Agents requiring `${USER_JWT}` or `${USER_API_KEY}` cannot run unattended and fail explicitly.

### 17. Global Traces

`GET /api/v1/traces` returns a bounded, paginated summary of the caller's threads. It does not fetch full thread histories.

```bash
curl 'http://localhost:8000/api/v1/traces?agent_name=research-agent&limit=50' \
  -H "X-API-Key: <API_KEY>"
```

Response shape:

```json
{
  "items": [
    {
      "thread_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "agent_name": "research-agent",
      "source": "chat",
      "started_at": "2026-11-15T10:30:00Z",
      "last_event_at": "2026-11-15T10:32:00Z",
      "event_count": 42,
      "turn_count": 2
    }
  ],
  "next_cursor": null
}
```

`agent_name` is optional but exact and case-sensitive. Detail reuses `GET /api/v1/threads/{thread_id}/history`; trace detail is read-only.

### 18. LLM Usage Dashboard (tokens / cost / latency)

Two aggregation endpoints power the `/traces` dashboard. Both read the same per-user RLS-scoped `trace_events` and join the server-side `llm_prices` price table (USD per 1M tokens). Aggregation semantics:

- One **run** = one turn (`thread_id` + `turn_id`). A run is **COMPLETED** when the turn has a terminal `AI_MESSAGE`, otherwise it is an **ERROR** run — `run_count` and `error_count` are disjoint sets (total = `run_count + error_count`).
- **Tokens** come from the terminal `AI_MESSAGE`'s `metadata.llm_usage` (`input_tokens`, `output_tokens`, `total_tokens`, `model_name`, `provider`, `duration_ms`), captured automatically by the chat path (including stream and HITL-resume turns) from LangGraph's `usage_metadata`. Runs without usage metadata count as 0 tokens.
- **Cost** resolves each run's effective model against `llm_prices` by **longest-prefix match** (exact id wins; `gpt-4o-2024-08-06` is priced as `gpt-4o`; `o3-mini-2025-04-16` is priced as `o3-mini`, never as `o3` thanks to a version-boundary anchor). A model with **no** price match contributes `cost=0` and is reported `priced=false` so usage still shows.
- **Duration** is derived per run from the turn opener (`HUMAN_MESSAGE`, or the leading `HITL_DECISION` for an HITL-resume turn) minus the terminal `AI_MESSAGE` timestamp; `duration_ms_p50`/`p95` are computed over COMPLETED runs only.
- The window bounds apply to the turn OPENING event; `from`/`to` are ISO datetimes (naive values are UTC-normalized).

```bash
# KPI aggregation (defaults: last 7 days, both breakdowns)
curl 'http://localhost:8000/api/v1/traces/usage?agent_name=research-agent&from=2026-11-01T00:00:00Z&to=2026-11-30T23:59:59Z' \
  -H "X-API-Key: <API_KEY>"
```

Response (`200`) — `TraceUsageResponse`:

```json
{
  "total_input_tokens": 15420,
  "total_output_tokens": 8300,
  "total_tokens": 23720,
  "total_cost_usd": 0.0953,
  "run_count": 12,
  "error_count": 1,
  "duration_ms_p50": 3150.5,
  "duration_ms_p95": 9800.0,
  "series": {
    "by_agent": [
      {
        "agent_name": "research-agent",
        "input_tokens": 15420,
        "output_tokens": 8300,
        "total_tokens": 23720,
        "cost_usd": 0.0953,
        "priced": true
      }
    ],
    "by_model": [
      {
        "model_name": "claude-sonnet-4-5-20250929",
        "input_tokens": 14000,
        "output_tokens": 7600,
        "total_tokens": 21600,
        "cost_usd": 0.0881,
        "priced": true
      },
      {
        "model_name": "gpt-4o-2024-08-06",
        "input_tokens": 1420,
        "output_tokens": 700,
        "total_tokens": 2120,
        "cost_usd": 0.0,
        "priced": false
      }
    ]
  }
}
```

Pass `group_by=agent` or `group_by=model` to narrow `series` to a single breakdown (the other key is then omitted from the payload).

```bash
# Top-N most expensive runs (limit 1-100, default 10), ranked by cost
curl 'http://localhost:8000/api/v1/traces/usage/top?limit=5&from=2026-11-01T00:00:00Z&to=2026-11-30T23:59:59Z' \
  -H "X-API-Key: <API_KEY>"
```

Response (`200`) — `TopRunsResponse` (envelope):

```json
{
  "items": [
    {
      "thread_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "turn_id": "turn-uuid-7",
      "agent_name": "research-agent",
      "model_name": "claude-sonnet-4-5-20250929",
      "total_tokens": 6420,
      "cost_usd": 0.0311,
      "duration_ms": 9150.5,
      "started_at": "2026-11-15T10:30:00.000000Z"
    }
  ]
}
```

Pricing table: the `llm_prices` table (Alembic `020_create_llm_prices_table`, forced RLS, server-visible, one row per model with `input_price_per_million` / `output_price_per_million` USD) is seeded at startup for common Anthropic / OpenAI / Google / Mistral / OpenRouter model names and families (prefix rows, e.g. `gpt-4o`, `claude-sonnet-4-5`, `mistral-large`, `openrouter/anthropic/claude-…`). Seeding is idempotent and never fatal. Editing/deleting price rows from the UI is out of scope for this iteration.

### Thread Source

Every thread has immutable `source: "chat" | "cron"`. Public thread creation always produces `chat`; scheduled and manual cron execution produces `cron`. Existing rows were backfilled to `chat`. Clients cannot override it.

### WebSocket

Connect to the WebSocket endpoint and send JSON messages. The WebSocket handshake accepts either `Authorization: Bearer <jwt>` or `X-API-Key: cpk_...` as headers; if neither validates, the server rejects the upgrade with HTTP 401.

```javascript
const ws = new WebSocket("ws://localhost:8000/api/v1/ws/<thread_id>", {
  headers: { "X-API-Key": "<API_KEY>" }   // or "Authorization": "Bearer <jwt>"
});
ws.onopen = () => ws.send(JSON.stringify({ message: "Hello" }));
ws.onmessage = (event) => {
  if (event.data === "[END]") {
    console.log("Response complete");
    return;
  }
  const data = JSON.parse(event.data);
  if (data.type === "error") { console.error("Error:", data.data); return; }
  switch (data.type) {
    case "THINKING":   console.log("[Thinking]", data.content); break;
    case "CONTENT":    process.stdout.write(data.content); break;
    case "AI_MESSAGE": console.log("Final message:", data.content); break;
    case "TOOL_CALL":  console.log("Tool call:", data.name, data.content); break;
    case "TOOL_RESULT":console.log("Tool result:", data.name, data.content); break;
  }
};
```

The WebSocket stream emits `TraceEvent` JSON objects (same shape as the `/stream` SSE endpoint), then `[END]`. On error, emits `{"type":"error","data":"..."}` before `[END]`.

---

## Store File API

composable-agents exposes a small REST surface for managing **files in the LangGraph store**. Files are stored as UTF-8 text blobs keyed by a path string (e.g. `/skills/my-skill/SKILL.md`). The store is shared across all agents and backed by the same `BaseStore` instance configured per agent (`store_backend: memory` or `postgres`).

### Endpoints

| Method | Path | Description | Success Status |
|---|---|---|---|
| `GET` | `/api/v1/store/files?prefix=<prefix>` | List file paths matching `prefix` (default `/` = all) | `200` |
| `GET` | `/api/v1/store/files/{path}` | Retrieve a single file's content | `200` |
| `PUT` | `/api/v1/store/files/{path}` | Create or replace a file (body: `{"content": "..."}`) | `200` |
| `DELETE` | `/api/v1/store/files/{path}` | Delete a file (idempotent) | `204` |

The `{path}` segment uses FastAPI's `:path` converter, so it can contain slashes (e.g. `skills/my-skill/SKILL.md`). Do not include a leading slash in the URL.

### Listing files

```bash
curl 'http://localhost:8000/api/v1/store/files?prefix=/skills/' \
  -H "X-API-Key: <API_KEY>"
```

Response (`200`) — a JSON array of path strings:

```json
["skills/code-review/SKILL.md", "skills/debugging/SKILL.md"]
```

### Getting a file

```bash
curl http://localhost:8000/api/v1/store/files/skills/code-review/SKILL.md \
  -H "X-API-Key: <API_KEY>"
```

Response (`200`):

```json
{"path": "skills/code-review/SKILL.md", "content": "# Code Review Skill\n\n..."}
```

If the file does not exist, the API returns `404` with `{"detail": "File not found: skills/code-review/SKILL.md"}`.

### Creating or replacing a file

```bash
curl -X PUT http://localhost:8000/api/v1/store/files/skills/code-review/SKILL.md \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"content": "# Code Review Skill\n\nReview code for correctness and security."}'
```

Response (`200`):

```json
{"path": "skills/code-review/SKILL.md", "content": "# Code Review Skill\n\nReview code for correctness and security."}
```

### Deleting a file

```bash
curl -X DELETE http://localhost:8000/api/v1/store/files/skills/code-review/SKILL.md \
  -H "X-API-Key: <API_KEY>"
```

Response: `204 No Content`. The operation is idempotent — deleting a non-existent path does not raise.

### Agent Namespace

When an agent is created (or updated) with `skills` or `memory` configured, the selected files are **copied into a dedicated namespace** in the store, scoped to that agent:

- **Skills:** `/agents/{agent_name}/skills/{skill_name}/SKILL.md`
- **Memories:** `/agents/{agent_name}/memories/{filename}`

This ensures each agent only loads the skills and memories explicitly selected in its configuration, not every file in the global `/skills/` and `/memories/` directories.

When an agent is updated and a skill or memory is **removed** from the selection, the corresponding copy in the agent namespace is **deleted** automatically.

> **Per-user isolation:** As of the `feat/dual-auth-rls-llm-peruser` branch, both the Store File API (`/api/v1/store/files`) and the agent namespace copies (`/agents/{name}/skills/`, `/agents/{name}/memories/`) are scoped **per authenticated user** via a namespace prefix `(user_id, "filesystem")`. A user only ever sees the skills/memories they own. See [Store Namespaces per User](#store-namespaces-per-user).

To discover which agents reference a given skill, use the usage-tracking endpoint:

```bash
curl http://localhost:8000/api/v1/store/skills/my-skill/usage \
  -H "X-API-Key: <API_KEY>"
```

Response (`200`) — the list of agent names that have `my-skill` in their namespace:

```json
["code-reviewer", "research-assistant"]
```

---

## Skills Management

Skills are `SKILL.md` files stored in the LangGraph store under the `/skills/` prefix. They describe reusable capabilities that an agent can load via its `skills` config field. Skills can now be created, edited, and deleted **via the Store File API** or the **frontend UI** (dedicated "Skills" page in the sidebar).

### Creating a skill via curl

```bash
curl -X PUT http://localhost:8000/api/v1/store/files/skills/my-skill/SKILL.md \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"content": "# my-skill\n\n## Description\nA skill for ..."}'
```

### Referencing a skill in an agent

In the agent YAML, point the `skills` field at the skill path (without the leading slash):

```yaml
name: my-agent
skills:
  - "skills/my-skill/"
```

In the frontend agent form, the Skills field is a **multi-select dropdown** (`PillMultiSelect`) that lists all available skills discovered in the store (paths matching `/skills/`), replacing the previous free-text input.

---

## Memories Management

Memories are Markdown files (e.g. `AGENTS.md`) stored in the LangGraph store, typically under the `/memories/` prefix. They provide persistent context that an agent loads via its `memory` config field. Memories can now be created, edited, and deleted **via the Store File API** or the **frontend UI** (dedicated "Memories" page in the sidebar).

### Creating a memory via curl

```bash
curl -X PUT http://localhost:8000/api/v1/store/files/memories/AGENTS.md \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"content": "# Project Guidelines\n\n- Always write tests.\n- Follow the hexagonal architecture."}'
```

### Referencing a memory in an agent

In the agent YAML, point the `memory` field at the memory path:

```yaml
name: my-agent
memory:
  - "memories/AGENTS.md"
```

In the frontend agent form, the Memory field is a **multi-select dropdown** (`PillMultiSelect`) that lists all available memories from the store, replacing the previous free-text input.

---

## Row-Level Security (RLS)

composable-agents enforces per-user isolation at the **database level** using PostgreSQL Row-Level Security. RLS is enabled **and forced** (so even the table owner is subject to the policies) on every per-user table.

### RLS-protected tables

| Table | `user_id` column added by |
|---|---|
| `agent_configs` | migration `012_add_user_id_to_rls_tables` |
| `threads` | migration `012_add_user_id_to_rls_tables` |
| `trace_events` | migration `012_add_user_id_to_rls_tables` |
| `api_keys` | migration `011_create_api_keys_table` (column present at creation) |
| `user_llm_settings` | migration `014_create_user_llm_settings_table` (column is the PK) |

### How the `app.user_id` GUC is set

For each authenticated request, the `ComposableAgentsSecurity.verify_credentials` dependency resolves the `user_id` (JWT `sub` or the API key's `user_id`) and stores it in a `contextvars.ContextVar` (`current_user_id`). A SQLAlchemy `before_cursor_execute` event listener on the engine then emits:

```sql
SET LOCAL app.user_id = '<user_id>';
```

inside the current transaction. The RLS policies compare each row's `user_id` against this GUC:

```sql
CREATE POLICY <table>_user_isolation ON <table>
  USING       (user_id = current_setting('app.user_id', true))
  WITH CHECK  (user_id = current_setting('app.user_id', true));
```

When the GUC is unset (unauthenticated session), `current_setting(..., true)` returns `NULL` and the policy filters out **every** row — the defensive default.

### Bypassing RLS (migrations / background jobs)

System operations that must read across all users (Alembic migrations, cron jobs) wrap their work in the `system_rls_context()` async context manager, which sets a `bypass_rls` contextvar that makes the listener emit `SET LOCAL row_security = off` for that transaction.

### Tables NOT protected by RLS

The LangGraph **`store`** table is **not** RLS-protected. It is accessed via its own asyncpg connection pool (the `PostgresStore` from `langgraph-checkpoint-postgres`), not via the SQLAlchemy engine that sets the `app.user_id` GUC. Per-user isolation for skills and memories is therefore enforced at the **application layer** via namespace prefixes — see [Store Namespaces per User](#store-namespaces-per-user).

---

## Per-User LLM Settings

Each authenticated user configures their **own** OpenAI-compatible LLM provider (provider label, base URL, and API key). The agent factory resolves the credentials per request from the authenticated `user_id` and builds a per-user `ChatOpenAI` instance. The server-wide `OPENAI_API_KEY` env var is **no longer used** to call the LLM.

### Endpoints

| Method | Path | Description | Success Status |
|---|---|---|---|
| `GET` | `/api/v1/settings/llm` | Get the authenticated user's LLM settings (masked key) | `200` (or `null` body if not configured) |
| `PUT` | `/api/v1/settings/llm` | Insert or update the user's LLM settings | `200` |
| `DELETE` | `/api/v1/settings/llm` | Delete the user's LLM settings (idempotent) | `204` |

The `api_key` is plaintext on the wire (HTTPS) and stored **encrypted at rest** with Fernet using `SECRET_ENCRYPTION_KEY`. GET responses return only a masked preview (`api_key_masked`), never the plaintext.

### Get current settings

```bash
curl http://localhost:8000/api/v1/settings/llm \
  -H "X-API-Key: <API_KEY>"
```

Response (`200`) — `null` when nothing is configured yet:

```json
{
  "user_id": "qa-user-1",
  "provider": "openrouter",
  "base_url": "https://openrouter.ai/api/v1",
  "api_key_masked": "sk-or-v1-8e1a…b147",
  "created_at": "2026-07-27T10:00:00Z",
  "updated_at": "2026-07-27T10:00:00Z"
}
```

### Configure (insert or update)

```bash
curl -X PUT http://localhost:8000/api/v1/settings/llm \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{
    "provider": "openrouter",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "sk-or-v1-..."
  }'
```

Response (`200`):

```json
{
  "user_id": "qa-user-1",
  "provider": "openrouter",
  "base_url": "https://openrouter.ai/api/v1",
  "api_key_masked": "sk-or-v1-8e1a…b147",
  "created_at": "2026-07-27T10:00:00Z",
  "updated_at": "2026-07-27T10:05:00Z"
}
```

### Delete settings

```bash
curl -X DELETE http://localhost:8000/api/v1/settings/llm \
  -H "X-API-Key: <API_KEY>"
```

Response: `204 No Content` (idempotent — deleting non-existent settings does not raise).

### Missing LLM configuration

If the user invokes an agent (e.g. `POST /api/v1/chat/{thread_id}`) **before** configuring an LLM provider, the agent factory raises `LlmNotConfiguredError` and the API returns:

```json
{"detail": "No LLM provider configured for user <user_id>. Configure via PUT /api/v1/settings/llm."}
```

with HTTP status **`422`**. The user must call `PUT /api/v1/settings/llm` first.

---

## Store Namespaces per User

The LangGraph Store (used by the [Store File API](#store-file-api) and the per-agent skill/memory namespaces) is **not** RLS-protected (see [Row-Level Security (RLS)](#row-level-security-rls)). Instead, per-user isolation is enforced at the **application layer** via namespace prefixes.

When a request is authenticated, the `LangGraphStoreFileRepository` resolves its namespace as:

```python
(user_id, "filesystem")
```

so a file written by `qa-user-1` is stored under the namespace `("qa-user-1", "filesystem")` and is invisible to `qa-user-2`. The same prefixing applies to the agent namespace copies (`/agents/{name}/skills/`, `/agents/{name}/memories/`).

When no user is authenticated (tests, system context), the namespace falls back to the static default `("filesystem",)` so existing tests stay green.

### Practical consequences

- `GET /api/v1/store/files?prefix=/skills/` lists **only the calling user's** skills.
- `PUT /api/v1/store/files/skills/my-skill/SKILL.md` writes into **the calling user's** namespace.
- An agent's `/agents/{name}/skills/` and `/agents/{name}/memories/` copies are scoped to the user who owns the agent config (RLS on `agent_configs` ensures the agent itself is per-user).

---

## Per-User API Keys

Per-user API keys allow a user to authenticate without a JWT (e.g. from a CI pipeline or a script). Keys are SHA-256 hashed at rest; the plaintext is returned **exactly once** at creation time.

### Endpoints

| Method | Path | Description | Success Status |
|---|---|---|---|
| `POST` | `/api/v1/api-keys` | Create a new API key (returns plaintext once) | `201` |
| `GET` | `/api/v1/api-keys` | List the user's API keys (no plaintext) | `200` |
| `DELETE` | `/api/v1/api-keys/{key_id}` | Revoke an API key (idempotent) | `204` |

> Creating an API key requires an existing authenticated session (JWT). In QA/local, keys are seeded directly in the database (see [Testing](#testing)).

### Create an API key

```bash
curl -X POST http://localhost:8000/api/v1/api-keys \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <jwt>" \
  -d '{"name": "my-ci-key"}'
```

Response (`201`):

```json
{
  "id": "8f3c1d2e-...",
  "name": "my-ci-key",
  "key_prefix": "cpk_abcde",
  "plaintext": "cpk_abcdefghijklmnopqrstuvwxyz0123456789...",
  "created_at": "2026-07-27T10:00:00Z"
}
```

### List API keys

```bash
curl http://localhost:8000/api/v1/api-keys \
  -H "X-API-Key: <API_KEY>"
```

Response (`200`) — a list of safe projections (no hash, no plaintext):

```json
[
  {
    "id": "8f3c1d2e-...",
    "name": "my-ci-key",
    "key_prefix": "cpk_abcde",
    "created_at": "2026-07-27T10:00:00Z",
    "last_used_at": "2026-07-27T11:30:00Z",
    "revoked_at": null
  }
]
```

### Revoke an API key

```bash
curl -X DELETE http://localhost:8000/api/v1/api-keys/8f3c1d2e-... \
  -H "X-API-Key: <API_KEY>"
```

Response: `204 No Content` (idempotent — revoking an already-revoked key returns 204). Revoking a key owned by another user returns `404` (RLS hides it).

---

## Architecture

composable-agents follows a strict **hexagonal architecture** (ports and adapters). The domain layer has zero dependencies on frameworks or infrastructure.

```
                    +---------------------------+
                    |      HTTP / WebSocket     |
                    |   (FastAPI application)   |
                    +------------+--------------+
                                 |
                    +------------+--------------+
                    |       Use Cases           |
                    | (application/use_cases/)  |
                    +------+----------+---------+
                           |          |
               +-----------+          +-----------+
               |                                  |
    +----------+---------+            +-----------+---------+
    |   Domain Ports     |            |   Domain Entities   |
    | (abstract classes) |            | AgentConfig, Thread |
    +----------+---------+            |   Message           |
               |                      +---------------------+
    +----------+---------+
    |  Infrastructure    |
    | (adapters)         |
    +--------------------+
    | - DeepAgentRunner  |
    | - DeepAgentRegistry|
    | - YamlConfigLoader |
    | - PostgresThreads  |
    | - Alembic (migrate)|
    +--------------------+
```

### File Tree

```
composable-agents/
  agents/                              # YAML agent configuration files
    example-agent.yaml                 #   Basic example agent
    minimal.yaml                       #   Minimal agent (name only)
    mcp-agent.yaml                     #   Agent with MCP server tools
    research-assistant.yaml            #   Research assistant with tools
    code-reviewer.yaml                 #   Code reviewer with HITL + subagents
  src/
    main.py                            # FastAPI app creation and lifespan (runs migrations)
    config.py                          # Pydantic Settings (env vars, DATABASE_URL normalization)
    dependencies.py                    # Dependency injection wiring
    alembic.ini                        # Alembic configuration
    alembic/
      env.py                           # Alembic env (async engine, model imports)
      versions/
        001_create_agent_configs_table.py
        002_create_threads_and_messages_tables.py
        005_create_trace_events_table.py
        006_migrate_messages_to_trace_events.py
        007_drop_messages_table.py
        010_add_description_to_agent_configs.py
        011_create_api_keys_table.py             # Per-user API keys (SHA-256 hashed)
        012_add_user_id_to_rls_tables.py         # Add user_id to agent_configs/threads/trace_events
        013_enable_rls_policies.py               # Enable + force RLS, create per-user policies
        014_create_user_llm_settings_table.py    # Per-user LLM settings (Fernet-encrypted)
        015_create_prompts_prompt_versions_models_tables.py
        016_add_thread_source_and_agent_registry_refs.py
        017_create_crons_and_cron_runs_tables.py
        018_make_legacy_rls_policies_bypass_aware.py
        019_create_cron_runs_single_active_index.py
    security.py                        # Dual-auth FastAPI dependency (JWT + per-user API key)
    application/
      requests/
        chat.py                        # Request models (ChatRequest, CreateThreadRequest, HITLDecisionRequest)
        api_key.py                     # CreateApiKeyRequest
        user_llm_settings.py           # UpsertUserLlmSettingsRequest
        guard.py                       # Explicit-null/non-null PATCH validation guard
      responses/
        thread_history.py              # ThreadHistory response DTO (thread + turns)
        prompt.py                      # Prompt/version API DTOs (no user_id)
        cron.py                        # Cron/run API DTOs (no user_id)
        traces.py                      # Global trace page DTO
      routes/
        health.py                      # GET /health (public)
        threads.py                     # CRUD /api/v1/threads
        chat.py                        # POST /api/v1/chat/{id} and /stream
        trace.py                       # GET /api/v1/threads/{id}/history and /trace
        agents.py                      # GET /api/v1/agents
        store.py                       # Store File API — /api/v1/store/files
        api_keys.py                    # POST/GET/DELETE /api/v1/api-keys (per-user API keys)
        user_llm_settings.py           # GET/PUT/DELETE /api/v1/settings/llm (per-user LLM)
        prompt.py                      # Local prompt/version registry
        model.py                       # Metadata-only model catalog
        cron.py                        # Cron CRUD/Run now/history
        traces.py                      # Global trace summaries
        websocket.py                   # WS /api/v1/ws/{id}
      use_cases/
        send_message.py                # Invoke agent synchronously
        stream_message.py              # Stream agent response
        get_thread_history.py          # Build ThreadHistory from trace_events (group by turn_id)
        manage_store_file.py           # ListStoreFiles / GetStoreFile / PutStoreFile / DeleteStoreFile use cases
        create_agent_config.py         # Create agent config (MinIO + Postgres)
        update_agent_config.py         # Update agent config
        delete_agent_config.py         # Delete agent config
        get_agent_config.py            # Get agent config from MinIO
        list_agent_configs.py          # List agent configs from Postgres
        load_agent_config.py           # Load and validate a YAML config
        seed_agents.py                 # Seed built-in agents from agents/ dir
        thread_management.py           # Create / get / list / delete threads
        api_key/                       # create / list / revoke per-user API keys
        user_llm_settings/             # get / upsert / delete per-user LLM settings
        prompt/                        # local prompt CRUD + immutable versions
        model/                         # metadata-only model CRUD
        cron/                          # cron CRUD/Run now/tick execution
        list_traces.py                 # bounded global trace summaries
    domain/
      entities/
        agent_config.py                # AgentConfig, BackendConfig, HITLConfig, SubAgentConfig
        agent_config_metadata.py       # AgentConfigMetadata (incl. description)
        mcp_server_config.py           # McpServerConfig, McpTransportType
        message.py                     # Message (role, content, timestamp, tool_calls) — projection model
        thread.py                      # Thread (id, agent_name, immutable source, user_id, timestamps)
        prompt.py                      # Prompt + PromptVersion (local registry)
        prompt_version_summary.py       # Prompt version history read-model
        model_metadata.py              # Metadata-only reusable model record
        cron.py                        # Cron definition
        cron_run.py                    # Cron run trigger/status/summary
        trace_event.py                 # TraceEvent entity + TraceEventType enum (6 types)
        trace_summary.py               # Global cursor-paginated trace summary
        tracing_config.py              # TracingConfig, TracingProviderType
        user_llm_settings.py           # UserLlmSettings, UserLlmSettingsInput
        auth/                          # AuthContext, ApiKeyView, CreatedApiKey
      ports/
        agent_config_loader.py         # Abstract: load config from file
        agent_config_repository.py     # Abstract: CRUD for agent config metadata
        agent_config_store.py          # Abstract: object storage for YAML blobs
        agent_registry.py              # Abstract: get_runner(name), list_agents(), close()
        agent_runner.py                # Abstract: invoke, stream, HITL operations
        mcp_tool_loader.py             # Abstract: load MCP tools
        store_file_repository.py       # Abstract: file CRUD on the LangGraph store (StoreFileRepository port)
        thread_repository.py           # Abstract: CRUD for threads
        prompt_repository.py           # Abstract: per-user prompt/version CRUD + resolution
        model_repository.py            # Abstract: metadata-only model CRUD + resolution
        cron_repository.py             # Abstract: cron definitions/durable runs/claim/reclaim
        trace_summary_repository.py    # Abstract: bounded global trace summaries
        runner_freshness.py            # Abstract: cross-worker runner dependency freshness
        trace_event_repository.py      # Abstract: persist/append/list TraceEvents
        tracing_provider.py            # Abstract: tracing lifecycle
        api_key_repository.py          # Abstract: per-user API key CRUD + hash lookup
        user_llm_settings_repository.py # Abstract: per-user LLM settings CRUD
        jwt_validator.py               # Abstract: JWT validation against Logto JWKS
      services/auth/                   # AuthService (dual JWT + API key orchestration)
      services/cron_schedule.py         # Cron expression/timezone validation + occurrence math
      errors/                          # DomainError hierarchy (incl. AuthenticationError, LlmNotConfiguredError)
    infrastructure/
      env_utils.py                     # ${VAR_NAME} + ${USER_JWT}/${USER_API_KEY} resolution
      database/
        rls_context.py                 # Per-request RLS contextvars + system_rls_context bypass
        rls_listener.py                # SQLAlchemy before_cursor_execute listener (SET LOCAL app.user_id)
        models/
          base.py                      # SQLAlchemy DeclarativeBase
          agent_config.py              # AgentConfigModel (ORM, incl. user_id)
          thread.py                    # ThreadModel (ORM, incl. user_id)
          trace_event.py               # TraceEventModel (ORM, incl. user_id)
          api_key.py                   # ApiKeyModel (ORM)
          user_llm_settings.py         # UserLlmSettingModel (ORM)
          prompt.py                    # Prompt/PromptVersionModel (ORM)
          model_config.py              # ModelMetadataModel (ORM, no credentials)
          cron.py                      # CronModel/CronRunModel (ORM)
      deepagent/
        adapter.py                     # DeepAgentRunner (LangGraph adapter) — emits TraceEvent
        factory.py                     # create_agent_from_config (per-user LLM credential resolution)
        registry.py                    # DeepAgentRegistry (lazy loading + caching from agents/ dir)
        example_tools.py               # Example tools: current_time, word_count
      mcp/
        adapter.py                     # LangchainMcpToolLoader (resolves ${USER_JWT}/${USER_API_KEY})
      minio_store/
        adapter.py                     # MinioAgentConfigStore (YAML blob storage)
      persistent_registry/
        adapter.py                     # PersistentAgentRegistry (MinIO + Postgres backed)
      store_file/
        adapter.py                     # LangGraphStoreFileRepository (LangGraph BaseStore adapter)
      postgres_repository/
        adapter.py                     # PostgresAgentConfigRepository
      postgres_thread/
        adapter.py                     # PostgresThreadRepository (thread persistence)
        models.py                     # Re-exports ThreadModel
      postgres_trace/
        adapter.py                     # PostgresTraceEventRepository (trace_events persistence)
        summary_repository.py          # Bounded global trace-summary aggregate query
      postgres_prompt/
        adapter.py                     # Per-user prompt/version repository
      postgres_model/
        adapter.py                     # Metadata-only model repository
      postgres_cron/
        adapter.py                     # Cron definitions, run history, atomic claims/reclaim
      runner_freshness/
        adapter.py                     # Cross-worker runner dependency fingerprints
      cron_scheduler.py                # Lifespan-owned background scheduler loop
      yaml_config/
        adapter.py                     # YamlAgentConfigLoader
      tracing/
        langfuse_adapter.py            # Langfuse tracing provider
        phoenix_adapter.py             # Phoenix tracing provider
        noop_adapter.py                # No-op tracing provider (default)
  tests/
    conftest.py
    fixtures/
      external.py                      # External service fixtures
      in_memory_thread_repository.py   # In-memory thread repository for tests
    unit/
      test_agent_config.py
      test_agent_crud.py
      test_deep_agent_runner.py
      test_env_utils.py
      test_factory.py
      test_factory_mcp_integration.py
      test_langfuse_adapter.py
      test_load_agent_config_use_case.py
      test_mcp_adapter.py
      test_mcp_lifecycle.py
      test_mcp_server_config.py
      test_message.py
      test_minio_store.py
      test_noop_tracing.py
      test_persistent_registry.py
      test_phoenix_adapter.py
      test_postgres_repository.py
      test_postgres_thread_repository.py
      test_registry.py
      test_routes.py
      test_runner_tracing.py
      test_seed_agents.py
      test_send_message.py
      test_thread.py
      test_thread_management.py
      test_tracing_config.py
      test_tracing_di.py
      test_tracing_lifecycle.py
      test_yaml_loader.py
  .env.example                         # Environment variable template
  Dockerfile                           # Container image
  pyproject.toml                       # Project metadata and dependencies
  CONTRIBUTING.md                      # Contributor guide
  uv.lock                             # Lockfile
```

---

## Examples

### Minimal Agent

`agents/minimal.yaml` -- the simplest possible agent. Uses all defaults (Claude Sonnet, no tools, state backend).

```yaml
name: minimal-agent
```

### Example Agent (OpenAI-compatible endpoint)

`agents/example-agent.yaml` -- a basic agent using an OpenAI-compatible model via OpenRouter.

```yaml
name: example-agent
model: "openai:anthropic/claude-haiku-4.5:nitro"
system_prompt: "You are a helpful assistant."
```

### MCP Agent

`agents/mcp-agent.yaml` -- an agent connected to an MCP filesystem server.

```yaml
name: mcp-agent
model: claude-sonnet-4-5-20250929
system_prompt: "You are an agent with MCP tool access."
mcp_servers:
  - name: filesystem
    transport: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

### Research Assistant with Tools

`agents/research-assistant.yaml` -- an agent with custom tools.

```yaml
name: research-assistant
model: "claude-sonnet-4-5-20250929"
system_prompt: |
  You are a research assistant specialized in technical documentation.
  Always cite your sources and provide structured summaries.
tools:
  - "src.infrastructure.deepagent.example_tools:current_time"
  - "src.infrastructure.deepagent.example_tools:word_count"
backend:
  type: state
debug: false
```

### Code Reviewer with HITL and Subagents

`agents/code-reviewer.yaml` -- a multi-agent system with human-in-the-loop approval. The Sub-Agent middleware is installed automatically because `subagents` is non-empty.

```yaml
name: code-reviewer
model: "claude-sonnet-4-5-20250929"
system_prompt: |
  You are an expert code reviewer. Analyze code for correctness,
  performance, security, and maintainability.
backend:
  type: state
hitl:
  rules:
    write_file: true
    execute:
      allowed_decisions:
        - approve
        - reject
subagents:
  - name: security-auditor
    description: "Specialized in security vulnerability analysis"
    instructions: "Focus on OWASP Top 10 and common security patterns"
  - name: performance-analyst
    description: "Specialized in performance optimization"
    instructions: "Analyze time complexity, memory usage, and bottlenecks"
```

---

## Database (PostgreSQL)

Thread, agent config, API key, and LLM-settings persistence is backed by PostgreSQL, accessed via SQLAlchemy's async ORM (`asyncpg` driver). Per-user tables are protected by Row-Level Security (see [Row-Level Security (RLS)](#row-level-security-rls)).

### Schema

The database uses a flat normalized schema. Thread persistence relies on a single `trace_events` table as the source of truth for all conversation activity (the legacy `messages` table has been dropped — see [Breaking changes](#breaking-changes)).

| Table | Description | RLS? |
|---|---|---|
| `threads` | One row per conversation thread. Columns: `id` (PK, VARCHAR 36), `agent_name`, immutable `source` (`chat`/`cron`), `user_id`, `created_at`, `updated_at`. | ✅ |
| `trace_events` | One row per trace event. Columns: `id` (PK), `thread_id` (FK to `threads.id`, CASCADE delete), `turn_id`, `type` (enum), `source`, `name`, `content`, `metadata` (JSONB), `timestamp`, `sequence`, `user_id`. | ✅ |
| `agent_configs` | Agent configuration metadata (incl. `description`, `user_id`). | ✅ |
| `api_keys` | Per-user API keys (SHA-256 hashed). Columns: `id` (PK), `user_id`, `name`, `key_hash`, `key_prefix`, `revoked_at`, `last_used_at`, `created_at`. Unique index on `key_hash`. | ✅ |
| `user_llm_settings` | Per-user LLM provider settings (Fernet-encrypted `api_key_encrypted`). PK is `user_id` — one configured provider per user. | ✅ |
| `prompts` | Per-user prompt metadata. Columns: `id` (UUID PK), `user_id`, `name`, `description`, `latest_version`, timestamps. | ✅ |
| `prompt_versions` | Immutable prompt versions. Columns: `prompt_id` (FK), `version`, `content`, `created_at`, `user_id`. Unique `(prompt_id, version)`. | ✅ |
| `models` | Identity-only per-user model catalog. Columns: `id` (UUID PK), `user_id`, `name`, `provider_model`, timestamps. Unique `(user_id, name)`. Provider/base URL/API key are owned by `user_llm_settings`, not by this table; it stores no credentials. | ✅ |
| `crons` | Per-user cron definitions. Columns: `id` (UUID PK), `user_id`, `name`, `agent_name`, `message`, `cron_expression`, `timezone`, `enabled`, scheduling/timestamp fields. | ✅ |
| `cron_runs` | Durable run history. Columns: `id` (UUID PK), `user_id`, `cron_id` (FK), `trigger`, `status`, `occurrence_at`, `thread_id`, timestamps, safe `error`. | ✅ |

Indexes on `trace_events`:

- `ix_trace_events_thread_id` — fast lookup of all events for a thread.
- `ix_trace_events_thread_id_sequence` — ordered retrieval of events within a thread (used by `/trace` and `/history`).
- `ix_trace_events_thread_id_turn_id` — grouping events by turn (used by `/history`).
- `ix_trace_events_user_id` — per-user RLS filter.

The `messages` table has been **dropped** (migration `007`). Its data was backfilled into `trace_events` by migration `006` (each old `Message` row became a `HUMAN_MESSAGE` or `AI_MESSAGE` event). The legacy `GET /api/v1/threads/{id}/messages` endpoint is preserved as a backward-compatible projection that filters `trace_events` to `HUMAN_MESSAGE` + `AI_MESSAGE` rows.

### Migrations (Alembic)

Alembic migrations live in `src/alembic/versions/` and run **automatically at startup** (via `asyncio.to_thread()` in the FastAPI lifespan). You never need to run `alembic upgrade` manually in normal operation.

Relevant migrations:

| Migration | Description |
|---|---|
| `005_create_trace_events_table` | Creates the `trace_events` table with the 3 indexes above. |
| `006_migrate_messages_to_trace_events` | Backfills `trace_events` from existing `messages` rows (`role = "human"` → `HUMAN_MESSAGE`, `role = "ai"` → `AI_MESSAGE`). |
| `007_drop_messages_table` | Drops the legacy `messages` table. |
| `010_add_description_to_agent_configs` | Adds a `description VARCHAR(500)` column to the `agent_configs` table. |
| `011_create_api_keys_table` | Creates the `api_keys` table (per-user API keys, SHA-256 hashed). Unique index on `key_hash`, index on `user_id`. |
| `012_add_user_id_to_rls_tables` | Adds a `user_id VARCHAR(255)` column to `agent_configs`, `threads`, and `trace_events` so RLS policies can filter rows per user. Adds an index on `user_id` for each table. |
| `013_enable_rls_policies` | Enables **and forces** Row-Level Security on `agent_configs`, `threads`, `trace_events`, and `api_keys`. Creates the per-user `user_isolation` policy on each table. |
| `014_create_user_llm_settings_table` | Creates the `user_llm_settings` table (per-user LLM provider settings, Fernet-encrypted API key) and enables RLS with a per-user policy. |
| `015_create_prompts_prompt_versions_models_tables` | Creates the local prompt registry, immutable prompt versions, and metadata-only model catalog with ownership indexes and forced RLS. |
| `016_add_thread_source_and_agent_registry_refs` | Adds immutable `threads.source` with `chat` backfill, plus validated `agent_configs.prompt_ref` / `model_ref` FK projections and aggregate/filter indexes. |
| `017_create_crons_and_cron_runs_tables` | Creates `crons` and `cron_runs`, CHECK constraints, run-history/due-scan indexes, forced per-user RLS, and the scheduled-occurrence uniqueness index. |
| `018_make_legacy_rls_policies_bypass_aware` | Makes legacy per-user policies honor the explicit `app.bypass_rls` context for migrations and scheduler system operations. |
| `019_create_cron_runs_single_active_index` | Adds DB-level uniqueness for at most one active run per cron and idempotent skipped occurrence records. |
| `024_drop_models_connection_columns` | Reduces the model catalog to its identity fields by dropping `models.provider` and `models.base_url`; Settings remains the single owner of the LLM connection. |

> The `mcp_servers` table is **not** managed here. It is owned by mcp-raganything's Alembic migration `001_create_mcp_servers_table` (tracked in the `raganything_alembic_version` table).

To create a new migration manually:

```bash
cd src
uv run alembic revision -m "describe_your_change"
```

To run migrations manually (useful for debugging):

```bash
cd src
uv run alembic upgrade head
```

To check current migration status:

```bash
cd src
uv run alembic current
```

### Architecture Decisions

- **Hexagonal architecture**: `ThreadRepository` (port) -> `PostgresThreadRepository` (adapter), `TraceEventRepository` (port) -> `PostgresTraceEventRepository` (adapter). The domain layer has no knowledge of SQLAlchemy.
- **Session-per-method**: Each repository method creates its own `AsyncSession` from the engine, ensuring thread-safety for concurrent FastAPI requests.
- **Connection pooling**: `AsyncAdaptedQueuePool` with `pool_size=20`, `max_overflow=20`, and `pool_pre_ping=True`.
- **Cascade deletes**: Deleting a thread automatically deletes all its `trace_events` via `ON DELETE CASCADE` at both the SQL and ORM level.
- **Event ordering**: `trace_events` are sorted by `sequence` (monotonic per thread). The adapter applies a defensive Python sort as well.
- **JSONB columns**: `metadata` is stored as PostgreSQL `JSONB`, allowing structured data (tool call IDs, structured responses) without additional join tables.
- **Single source of truth**: `trace_events` is the only persistence for conversation activity. `messages` is no longer a table; the `/messages` endpoint is a read-only projection.

---

## Breaking Changes

This release replaces the legacy `StreamEvent` / `messages`-based model with a unified `TraceEvent` model, and switches auth from a single master `X-API-Key` to **dual JWT + per-user API keys** with **Row-Level Security** and **per-user LLM credentials**.

### Dual auth replaces the master `X-API-Key`

The single master `X-API-Key` / `API_KEY` / `OPENAI_API_KEY` model is **removed** for authentication. Every protected endpoint now requires either `Authorization: Bearer <jwt>` (validated against Logto OIDC JWKS, `LOGTO_URL` + `JWT_AUDIENCE`) or `X-API-Key: cpk_...` (per-user, SHA-256 hashed, created via `POST /api/v1/api-keys`). See [Authentication](#authentication). Existing clients sending the old master key will receive `401 {"detail": "Invalid or missing credentials"}`.

### Per-user LLM credentials (no server-wide `OPENAI_API_KEY`)

The server-wide `OPENAI_API_KEY` env var is **no longer used** to call the LLM. Each authenticated user must configure their own provider via `PUT /api/v1/settings/llm`. If a user invokes an agent before configuring a provider, the API returns `422 LlmNotConfiguredError`. See [Per-User LLM Settings](#per-user-llm-settings).

### Row-Level Security on per-user tables

Migrations `011`–`014` add a `user_id` column to `agent_configs`, `threads`, and `trace_events`, create the `api_keys` and `user_llm_settings` tables, and enable **forced** Row-Level Security on all five tables. Existing rows become `user_id = ''` and are **invisible** under RLS (the policy compares against `current_setting('app.user_id', true)` which is NULL for unauthenticated sessions). Backfill existing rows to a real `user_id` before enabling RLS in production if you need to preserve access. See [Row-Level Security (RLS)](#row-level-security-rls).

### Phoenix prompt management removed

Prompt storage, listing, versioning, and agent lookup no longer use Phoenix. The old `PromptManager`/Phoenix prompt routes were replaced by the local `/api/v1/prompts` registry described above. Agents previously relying on an implicit Phoenix prompt-by-agent-name lookup must be migrated to either an inline/file prompt or a local `prompt_ref`. Phoenix remains only as an optional tracing provider.

### Local prompts and metadata-only models

`AgentConfig` accepts optional `prompt_ref` / `model_ref`. `prompt_ref` is mutually exclusive with `system_prompt` / `system_prompt_file`. `model_ref` takes precedence over legacy `model`, selects the catalog's `provider_model`, and stores no credential or connection data. Existing YAML without these references remains valid.

### Model catalog is identity-only

`models.provider` and `models.base_url` were removed (migration `024`). `POST` / `PATCH /api/v1/models` now accept only the editable fields `name` and `provider_model`; `provider`, `base_url`, and `base_url: null` are rejected with `422`. The user's LLM Settings is the **single source** of the provider connection (`provider`, `base_url`, `api_key`), so a Model can no longer override the Settings endpoint at runtime.

### Immutable thread source

`Thread.source` is added with values `chat` and `cron`. Existing rows are backfilled to `chat`. Public thread creation cannot override the source, and cron execution cannot accidentally create a chat thread.

### `StreamEvent` removed

The old SSE payload format (`{"type": "thinking" | "content" | "message" | "structured" | "error", "data": "..."}`) is **removed**. The `/stream` and WebSocket endpoints now emit `TraceEvent.model_dump_json()` objects (see [TraceEvent format](#traceevent-format)). Clients must be updated to parse the new schema. The only non-`TraceEvent` payload is the error object `{"type": "error", "data": "..."}` emitted on failure (followed by `[DONE]`).

### `messages` table dropped

The `messages` PostgreSQL table has been dropped (migration `007`). All conversation activity is now stored in `trace_events`. Migration `006` backfills `trace_events` from existing `messages` rows, so no data is lost when upgrading. The `GET /api/v1/threads/{id}/messages` endpoint is preserved as a backward-compatible projection (filters `trace_events` to `HUMAN_MESSAGE` + `AI_MESSAGE`).

### `AgentRunner` API

The `AgentRunner` port signatures have changed:

- `invoke(thread_id, message, turn_id) -> tuple[Message, list[TraceEvent]]`
- `stream(thread_id, message, turn_id) -> AsyncIterator[TraceEvent]`

Adapters and tests calling the old `invoke(thread_id, message) -> Message` / `stream(...) -> AsyncIterator[StreamEvent]` signatures must be updated.

### Migrations

Migrations `005`, `006`, `007`, `010`, `011`, `012`, `013`, `014` run automatically on startup. They are idempotent and safe to run on an existing database with data, **except** that `012`/`013` will make pre-existing rows with `user_id = ''` invisible under RLS — backfill them first.

---

## Environment Variables

Configured via `.env` file or environment variables. See `.env.example`.

### General

| Variable | Default | Description |
|---|---|---|
| `AGENTS_DIR` | `./agents` | Directory containing agent YAML configuration files. |
| `HOST` | `0.0.0.0` | Server bind host. |
| `PORT` | `8000` | Server bind port. |
| `LOG_LEVEL` | `INFO` | Application log level. |
| `UVICORN_LOG_LEVEL` | `info` | Uvicorn log level. |
| `ALLOWED_ORIGINS` | `["http://localhost:8080"]` | JSON array of CORS allowed origins. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Base URL for OpenAI-compatible endpoints. Set to use OpenRouter, LiteLLM, vLLM, etc. **Per-user** base URLs configured via `PUT /api/v1/settings/llm` override this for authenticated requests. |
| `MCP_RAGANYTHING_API_KEY` | -- | Shared API key for authenticating to mcp-raganything MCP servers via `${MCP_RAGANYTHING_API_KEY}` in agent YAML. Must match the `API_KEY` set on the raganything server. |

### Dual Authentication

| Variable | Default | Description |
|---|---|---|
| `LOGTO_URL` | `""` (empty) | Logto OIDC issuer URL for JWT validation (e.g. `https://logto.soludev.tech`). When empty, the JWT path is disabled and only the per-user API key path is active (QA/local mode). |
| `JWT_AUDIENCE` | `""` (empty) | Legacy single expected `aud` claim for incoming JWTs (typically the Logto oauth2-proxy app ID). Still honored; superseded by `JWT_AUDIENCES`. |
| `JWT_AUDIENCES` | `""` (empty) | Comma-separated list of accepted `aud` claims (e.g. `ui-app-id,opencode-app-id`). Merged with `JWT_AUDIENCE` (legacy first). Required when `LOGTO_URL` is set and several Logto apps authenticate. |
| `MCP_ALLOWED_HOSTS` | `127.0.0.1:*,localhost:*,[::1]:*` | Comma-separated allow-list for the MCP transport-security Host check and (fallback) discovery-origin derivation. DNS-rebinding protection stays ENABLED; a present-but-empty or JSON-style value refuses startup. Prod must add `composables-agent.soludev.tech`. |
| `MCP_PUBLIC_BASE_URL` | `""` (empty) | Explicit public origin for the MCP discovery documents (`WWW-Authenticate` + `/.well-known`). Always wins over the allow-listed request Host. Scheme-less/relative values refuse startup; non-https warns. Prod: `https://composables-agent.soludev.tech`. |
| `MCP_AUTHORIZATION_SERVER` | `""` (empty → `<LOGTO_URL>/oidc`) | Public AS advertised in `/.well-known/oauth-protected-resource` (`https://logto.soludev.tech/oidc` in prod). Relative values refuse startup; empty → no `authorization_servers` entry. |
| `SECRET_ENCRYPTION_KEY` | `""` (empty) | Fernet key used to encrypt per-user LLM API keys at rest. Generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. **Required** when per-user LLM settings are enabled. Must be stable across restarts. |

### Deprecated (no longer used for auth / LLM)

| Variable | Status | Replacement |
|---|---|---|
| `OPENAI_API_KEY` | **Deprecated** for both auth and LLM calls. Still read as an env fallback by `ChatOpenAI` when no authenticated user is present (tests). | Per-user LLM credentials via `PUT /api/v1/settings/llm` (see [Per-User LLM Settings](#per-user-llm-settings)). |
| `API_KEY` (master `X-API-Key`) | **Deprecated** for auth. | Dual JWT + per-user API keys (see [Authentication](#authentication)). |

### PostgreSQL Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | **required** | PostgreSQL connection URL (e.g. `postgresql://user:pass@host:5432/db`). Automatically normalized to `postgresql+asyncpg://` for async SQLAlchemy. `sslmode` and `channel_binding` query params are extracted and passed via `connect_args` (asyncpg doesn't accept them in the URL). |
| `POSTGRES_STATEMENT_CACHE_SIZE` | `100` (asyncpg default) | Set to `0` when using a connection pooler (Neon, PgBouncer, etc.). |
| `DATABASE_SSL_INSECURE` | `false` | Explicit opt-out for local/self-signed development. `false` (default) validates the TLS certificate and hostname for all database `sslmode` settings requiring encryption. Keep `false` in production. |

> **⚠️ Breaking change:** `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DATABASE` are no longer used. Migrate to `DATABASE_URL`.

### MinIO Variables

| Variable | Default | Description |
|---|---|---|
| `MINIO_ENDPOINT` | `localhost:9040` | MinIO server endpoint. |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key. |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key. |
| `MINIO_BUCKET` | `composable-agents` | Bucket for YAML config blob storage. |
| `MINIO_SECURE` | `false` | Use HTTPS for MinIO connections. |

### Tracing Variables

| Variable | Default | Description |
|---|---|---|
| `TRACING_PROVIDER` | `none` | Tracing backend: `none`, `langfuse`, or `phoenix`. |
| `TRACING_PROJECT_NAME` | `composable-agents` | Project name for the tracing backend. |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006` | Phoenix collector endpoint. |
| `PHOENIX_API_KEY` | -- | Phoenix API key. |

---

## Development

### Install dependencies (including dev tools)

```bash
uv sync
```

### Run the test suite

```bash
# Unit tests (no DB / no Docker required) — 667 tests
uv run pytest tests/unit -q

# Full suite (unit + integration)
uv run pytest tests/ -v
```

### Run tests with coverage

```bash
uv run pytest tests/ -v --cov=src
```

## Testing

The project has two test layers: **unit tests** (in this repo, no infrastructure) and a **QA suite** (in `soludev-compose-apps/bricks/qa`, runs against the local Docker compose stack).

### Unit tests

```bash
cd bricks/composable-agents
uv run pytest tests/unit -q
# → 667 passed
```

Unit tests use in-memory fixtures (`Base.metadata.create_all` on SQLite, in-memory repositories) — no PostgreSQL, no Logto, no MinIO. The RLS migrations do **not** run in unit tests (RLS is Postgres-only and validated in QA).

### QA suite (dual-auth + RLS + per-user LLM + store namespaces)

The QA suite exercises the full dual-auth + RLS + per-user LLM + store-namespace feature against a real PostgreSQL instance via Docker compose. It lives in `soludev-compose-apps/bricks/qa`.

#### 1. Start the stack

```bash
cd soludev-compose-apps/bricks
docker compose up -d --build composable-agents bricks-db minio composable-agents-qa-init
docker compose ps   # wait for composable-agents to be healthy
```

The `composable-agents-qa-init` one-shot service waits for `composable-agents` to be healthy (so the Alembic migrations that create `api_keys` have applied), then seeds two per-user API keys directly in `bricks-db`:

| `user_id` | plaintext `X-API-Key` | `id` |
|---|---|---|
| `qa-user-1` | `cpk_qa_test_key_12345` | `qa-key-id-1` |
| `qa-user-2` | `cpk_qa_test_key_67890` | `qa-key-id-2` |

The seed is idempotent (`ON CONFLICT DO UPDATE` re-activates revoked rows), so re-running `docker compose up -d composable-agents-qa-init` is safe. In QA, `LOGTO_URL` is empty, so only the per-user API key path is exercised (JWT validation is unit-tested separately).

#### 2. Run the QA tests

```bash
cd qa
uv run pytest                              # whole suite
# or just the dual-auth / RLS / per-user feature tests:
uv run pytest test_dual_auth.py test_api_keys_crud.py test_rls_isolation.py \
              test_llm_settings.py test_store_isolation.py test_api_keys.py \
              test_threads.py test_agents.py test_health.py --tb=short
```

The QA fixtures (`qa/conftest.py`) read `QA_API_KEY_USER_1` / `QA_API_KEY_USER_2` (defaulting to the seeded plaintext keys above) and send them as `X-API-Key` headers automatically. `COMPOSABLE_AGENTS_URL` defaults to `http://localhost:8010` (the port mapped in `docker-compose.yml`).

> **Note:** `test_mcp_bricks_endpoints.py`, `test_txt_support.py`, `test_file_endpoints.py`, and `test_extraction*.py` target the **raganything-api** service and require `API_KEY` to be set for that service — they are independent of the dual-auth feature.

### Lint

```bash
uv run ruff check .
```

### Type check

```bash
uv run mypy src/
```

### Validate all agent YAML files

```bash
for f in agents/*.yaml; do uv run python -m src validate "$f"; done
```

---

## Optional Dependencies

The project provides optional dependency groups for tracing support:

```bash
# Langfuse tracing only
uv sync --extra langfuse

# Phoenix tracing only
uv sync --extra phoenix

# All tracing providers
uv sync --extra tracing
```

---

## Deployment on Railway

[Railway](https://railway.app/) is a deployment platform that supports Docker-based services with managed PostgreSQL.

### Architecture on Railway

```
Railway project
├── PostgreSQL (Railway managed plugin)
├── composable-agents (Dockerfile deploy)
└── (mcp-raganything — separate Railway service or project)
```

### Step-by-step

1. **Create a Railway project** and add a PostgreSQL plugin.

2. **Deploy composable-agents**:
   - New Service → GitHub Repo → select this repository.
   - Railway detects the `Dockerfile` automatically.
   - Set the port to `8000`.

3. **Deploy mcp-raganything** (separate Railway service or project):
   - See the [mcp-raganything README](https://github.com/soludev/mcp-raganything#deployment-on-railway) for Railway deployment instructions.
   - Note the generated public domain (e.g. `mcp-raganything-production.up.railway.app`).

4. **Configure environment variables** in the Railway dashboard:

   | Variable | Example | Notes |
   |----------|---------|-------|
   | `DATABASE_URL` | `postgresql://postgres:pass@roundhouse.proxy.rlwy.net:33019/railway` | Railway PostgreSQL connection URL (required) |
   | `LOGTO_URL` | `https://logto.soludev.tech` | Logto OIDC issuer URL for JWT validation. Required in prod to enable the JWT auth path. |
   | `JWT_AUDIENCE` | `<logto-app-id>` | Expected `aud` claim for JWTs. Required when `LOGTO_URL` is set. |
   | `SECRET_ENCRYPTION_KEY` | `I32ylYwnej8p2Wa72G3FibHBoRNxWlVxWsC5F4LvXSU=` | Fernet key for encrypting per-user LLM API keys at rest. Generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. |
   | `AGENTS_DIR` | `./agents` | Directory containing agent YAML configs |
   | `MCP_RAGANYTHING_API_KEY` | `your-shared-secret` | Must match `API_KEY` on mcp-raganything |
   | `TRACING_PROVIDER` | `phoenix` | Tracing backend: `none`, `langfuse`, or `phoenix` |
   | `PHOENIX_COLLECTOR_ENDPOINT` | `https://phoenix.xxx.railway.app` | Phoenix collector URL |

   > **Note:** `OPENAI_API_KEY` and `API_KEY` (master) are **deprecated** for auth and LLM. Each end-user now configures their own LLM provider via `PUT /api/v1/settings/llm` (see [Per-User LLM Settings](#per-user-llm-settings)). `OPENAI_BASE_URL` is still read as a fallback for unauthenticated/test contexts.

5. **Update agent YAML configs** to point MCP server URLs to the Railway-deployed mcp-raganything domain:

   ```yaml
   mcp_servers:
     - name: bricks
       transport: http
       url: https://mcp-raganything-production.up.railway.app/bricks/mcp
       headers:
         X-API-Key: "${MCP_RAGANYTHING_API_KEY}"
         # Or, to forward the caller's identity to raganything:
         # Authorization: "Bearer ${USER_JWT}"
         # X-API-Key: "${USER_API_KEY}"
   ```

   The `${MCP_RAGANYTHING_API_KEY}` placeholder is resolved from the environment variable at runtime; `${USER_JWT}` / `${USER_API_KEY}` are resolved from the authenticated caller's credential (see [Per-user credential propagation](#per-user-credential-propagation-user_jwt--user_api_key)).

6. **MinIO** (optional — only if using MinIO for agent config storage):
   - Deploy MinIO as a separate Railway service or use an external S3-compatible service.
   - Set `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET` accordingly.

7. **Verify deployment**:

   ```bash
   curl https://composable-agents-production.up.railway.app/health
   ```

### Notes

- Railway automatically generates a public domain for each service.
- The `agents/` directory is baked into the Docker image at build time. To update agent configs without redeploying, use MinIO-backed agent config storage (see `AGENTS_DIR` and MinIO variables).
- Alembic migrations run automatically on startup.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on:

- Project architecture and dependency rules
- How to add custom tools and backends
- How the YAML schema works
- Running tests and linting
- Code style conventions

---

## License

This project does not currently include a license file. Contact the maintainers for licensing information.
