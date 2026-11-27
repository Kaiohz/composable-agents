"""Centralized error message templates.

The message text raised with every domain error is declared here so error
wording is discoverable, consistent and never duplicated. Callers format the
template with the runtime values::

    raise ConfigError(ErrorMessage.INVALID_AGENT_NAME.format(name))
"""

from enum import StrEnum
from string import Template


class ErrorMessage(StrEnum):
    """Catalog of centralized error message templates (``str.format`` style)."""

    # --- Configuration ---
    INVALID_AGENT_NAME = (
        "Invalid agent name '{name}'. Must match pattern: alphanumeric, dots, hyphens, underscores, 2-100 chars."
    )
    FILE_TOO_LARGE = "File too large. Maximum size is {max_size} bytes."
    FILE_NOT_UTF8 = "File must be valid UTF-8 encoded YAML."
    AGENT_NAME_MISMATCH = "Agent name in YAML '{yaml_name}' does not match provided name '{name}'"
    AGENT_NAME_MISMATCH_URL = "Agent name in YAML '{yaml_name}' does not match URL name '{name}'"
    YAML_INVALID = "Invalid YAML from {source}: {error}"
    YAML_NOT_MAPPING = "YAML from {source} must contain a YAML mapping, not {type}"
    YAML_VALIDATION_ERROR = "Validation error: {error}"
    YAML_EMPTY = "Empty YAML content from {source}"
    YAML_CONFIG_NOT_FOUND = "Config file not found: {path}"
    YAML_PROMPT_FILE_NOT_FOUND = "Prompt file not found: {path}"
    YAML_SYSTEM_PROMPT_FILE_DISALLOWED = (
        "system_prompt_file is not allowed in string-loaded YAML from {source}. "
        "Inline the prompt in system_prompt instead."
    )

    # --- Traces (spec §5.4) ---
    TRACE_CURSOR_INVALID = "Invalid pagination cursor"
    TRACE_LIMIT_INVALID = "limit must be between 1 and 100"

    # --- Agent ---
    AGENT_NOT_FOUND = "Agent not found: {name}"
    AGENT_NAME_RESERVED = (
        "Agent name '{name}' is reserved for the built-in Composable AI Assistant. "
        "Choose another name to create a custom agent."
    )
    BUILTIN_AGENT_READ_ONLY = (
        "Agent '{name}' is the built-in Composable AI Assistant and cannot be deleted. "
        "Personalize it with Update or restore its default definition with Reset."
    )
    BUILTIN_CUSTOMIZATION_FIELD_FORBIDDEN = (
        "The built-in assistant's '{field}' capability is managed by the platform and cannot be customized."
    )
    BUILTIN_CUSTOMIZATION_HITL_FLOOR = (
        "The built-in assistant's HITL rule '{tool}' must stay enabled because it protects a destructive delete."
    )
    BUILTIN_CUSTOMIZATION_USER_REQUIRED = "A built-in assistant customization requires an authenticated user context."
    AGENT_INVOCATION_REJECTED = "Agent invocation rejected: {reason}"
    AGENT_CONFIG_NOT_FOUND = "Agent config metadata not found: {name}"
    AGENT_CONFIG_NOT_FOUND_IN_STORE = "Agent config not found in store: {name}"
    AGENT_CONFIG_ALREADY_EXISTS = "Agent config already exists: {name}"
    AGENT_CONFIG_REFERENCE_RACE = (
        "A prompt or model referenced by agent config '{name}' is no longer available; the reference was not saved."
    )
    AGENT_NO_FINAL_MESSAGES = "Graph completed but no messages were found in the final state."
    AGENT_EXECUTION_ERROR = "Agent execution error: {error}"
    AGENT_STREAMING_ERROR = "Streaming error: {error}"
    AGENT_HITL_APPROVE_ERROR = "HITL approve error: {error}"
    AGENT_HITL_REJECT_ERROR = "HITL reject error: {error}"
    AGENT_HITL_EDIT_ERROR = "HITL edit error: {error}"
    AGENT_HITL_RESUME_ERROR = "HITL resume error: {error}"
    AGENT_HITL_NO_PENDING_INTERRUPT = (
        "HITL resume failed: no pending interrupt to resume (nothing to resume) for thread {thread_id}."
    )
    AGENT_HITL_UNKNOWN_TOOL_CALL = (
        "HITL resume failed: unknown tool_call_id {tool_call_id} (not found in pending interrupts) "
        "for thread {thread_id}."
    )
    AGENT_HITL_MISSING_DECISION = (
        "HITL resume failed: missing decision for {pending} interrupted tool call(s) "
        "(mismatch with provided {provided}) for thread {thread_id}."
    )
    AGENT_STREAM_IDLE_TIMEOUT = (
        "Agent stream idle for {timeout}s (thread={thread_id}); aborting — a tool result "
        "was likely lost (flaky transport)."
    )
    AGENT_INVOKE_TIMEOUT = "Agent invoke timed out after {timeout}s (thread={thread_id})"

    # --- Agent resource import (create-time YAML materialization) ---
    AGENT_IMPORT_MISSING_MODEL = (
        "A new custom agent must declare a model source: set `model` (materialized into the model catalog) "
        "or `model_ref` (an existing catalog model id). Agent '{name}' declares neither."
    )
    AGENT_IMPORT_MISSING_PROMPT = (
        "A new custom agent must declare a prompt source: set `system_prompt` (materialized into the prompt "
        "registry) or `prompt_ref` (an existing registry prompt id). Agent '{name}' declares neither."
    )
    AGENT_IMPORT_EXCLUSIVE_SOURCES = (
        "Agent '{name}' cannot declare {first} and {second} together: the sources are mutually exclusive."
    )
    AGENT_IMPORT_MODEL_DECLARATION_INVALID = "The inline model declaration must be a non-empty string (agent '{name}')."
    AGENT_IMPORT_MCP_DUPLICATE = (
        "Duplicate MCP server name '{name}' in mcp_servers: each server name must be unique within the import."
    )
    AGENT_IMPORT_MCP_REF_CONFLICT = (
        "MCP server name '{name}' appears as a reference (mcp_refs) and an inline definition (mcp_servers): "
        "a reference is never materialized again."
    )
    AGENT_IMPORT_MCP_ENTRY_INVALID = "Invalid inline MCP server definition (mcp_servers): {reason}"
    AGENT_IMPORT_MCP_REF_NOT_REGISTERED = (
        "MCP server(s) not registered for your account: {names}. Register them in the MCP Registry first."
    )
    AGENT_IMPORT_MCP_CONFLICT = (
        "MCP server '{name}' (mcp_servers) is already registered: the import never overwrites an existing "
        "registration. Rename the inline definition or reuse the registered one."
    )
    AGENT_IMPORT_SUBAGENT_CYCLE = "Circular sub-agent reference (cycle) detected before any creation: {names}"
    AGENT_IMPORT_SUBAGENT_SELF_REFERENCE = (
        "Sub-agent '{name}' references its own agent '{agent}' (self-reference is not allowed)."
    )
    AGENT_IMPORT_SUBAGENT_MUST_DECLARE = "Inline sub-agent '{name}' must declare '{field}'."
    AGENT_IMPORT_SUBAGENT_NAME_CONFLICT = (
        "An agent named '{name}' already exists: refusing to create the inline sub-agent (no overwrite)."
    )
    AGENT_IMPORT_SUBAGENT_NAME_RESERVED = "Sub-agent name '{name}' is reserved for the built-in assistant."
    AGENT_IMPORT_SKILL_PATH_INVALID = "Invalid skill entry ('{path}'): {reason}"
    AGENT_IMPORT_MEMORY_PATH_INVALID = "Invalid memory entry ('{path}'): {reason}"
    AGENT_IMPORT_SKILL_FIELD_UNEXPECTED = (
        "Inline skill definition accepts exactly `path` and `content`: unexpected field(s) {fields}."
    )
    AGENT_IMPORT_MEMORY_FIELD_UNEXPECTED = (
        "Inline memory definition accepts exactly `path` and `content`: unexpected field(s) {fields}."
    )
    AGENT_IMPORT_FILE_SHAPE_INVALID = (
        "Each inline {kind} definition must be a {{path, content}} object or an existing textual path reference."
    )
    AGENT_IMPORT_SKILL_EXISTS = (
        "A skill file already exists at '{path}': an inline skill definition must create a new skill (no overwrite)."
    )
    AGENT_IMPORT_MEMORY_EXISTS = (
        "A memory file already exists at '{path}': an inline memory definition must create a new memory (no overwrite)."
    )
    AGENT_IMPORT_SKILL_REF_MISSING = (
        "Skill reference '{path}' was not found in your store: create the skill before referencing it."
    )
    AGENT_IMPORT_MEMORY_REF_MISSING = (
        "Memory reference '{path}' was not found in your store: create the memory file before referencing it."
    )
    AGENT_IMPORT_UPDATE_INLINE_FORBIDDEN = (
        "Update refused for agent '{name}': the YAML introduces a new inline {field} definition, "
        "but automatic materialization is only supported at creation time. Reference registered resources "
        "instead ({alternatives})."
    )
    AGENT_IMPORT_TARGET_INVALID = "The imported agent configuration is invalid after materialization: {reason}"
    AGENT_IMPORT_RESOURCE_NAME_CONFLICT = (
        "A {kind} named '{name}' already exists for your account: the import never overwrites or duplicates "
        "an existing resource (REQ-TX-002). Rename the agent or reuse the registered resource."
    )
    AGENT_IMPORT_SUBAGENT_NAME_DUPLICATE = (
        "Duplicate sub-agent name '{name}' in subagents: each sub-agent name must be unique within the import."
    )
    AGENT_IMPORT_SUBAGENT_REF_FIELD_FORBIDDEN = (
        "Referenced sub-agent '{name}' must not declare an inline '{field}': a reference owns its own "
        "resources (it reuses the referenced agent's model, prompt and MCP selection)."
    )
    AGENT_IMPORT_MCP_VALIDATION_FAILED = (
        "MCP server '{name}' (mcp_servers) was rejected by the registry validation: the import refuses "
        "to persist an unvalidated server."
    )
    AGENT_UPDATE_REGISTRY_NOT_WIRED = "Agent '{name}' sets mcp_refs but no MCP registry is wired."

    # --- Thread ---
    THREAD_NOT_FOUND = "Thread not found: {thread_id}"
    THREAD_FAILED_CREATE = "Failed to create thread: {error}"
    THREAD_FAILED_GET = "Failed to get thread {thread_id}: {error}"
    THREAD_FAILED_LIST = "Failed to list threads: {error}"
    THREAD_FAILED_DELETE = "Failed to delete thread {thread_id}: {error}"
    THREAD_FAILED_ADD_MESSAGE = "Failed to add message to thread {thread_id}: {error}"

    # --- Trace events ---
    TRACE_FAILED_ADD = "Failed to add trace event to thread {thread_id}: {error}"
    TRACE_FAILED_LIST = "Failed to list trace events for thread {thread_id}: {error}"
    TRACE_FAILED_ADD_BATCH = "Failed to add batch trace events to thread {thread_id}: {error}"

    # --- LLM usage observability (dashboard spec §1-§5) ---
    TRACE_USAGE_AGGREGATE_FAILED = "Failed to aggregate LLM usage: {error}"
    TRACE_USAGE_TOP_FAILED = "Failed to aggregate the top LLM usage runs: {error}"
    LLM_PRICES_SEED_FAILED = "Failed to seed the LLM price table: {error}"

    # --- Storage / persistence ---
    STORAGE_REPO_NOT_INITIALIZED = "Thread repository not initialized. Check PostgreSQL connectivity."
    STORAGE_REGISTRY_NOT_INITIALIZED = "Agent registry not initialized. Check MinIO/PostgreSQL connectivity."
    STORAGE_PERSISTENCE_NOT_INITIALIZED = "Persistence layer not initialized. Check MinIO/PostgreSQL connectivity."
    STORAGE_FAILED_SAVE_AGENT_CONFIG = "Failed to save agent config metadata '{name}': {error}"
    STORAGE_FAILED_GET_AGENT_CONFIG = "Failed to get agent config metadata '{name}': {error}"
    STORAGE_FAILED_LIST_AGENT_CONFIG = "Failed to list agent config metadata: {error}"
    STORAGE_FAILED_DELETE_AGENT_CONFIG = "Failed to delete agent config metadata '{name}': {error}"
    STORAGE_FAILED_EXISTS_AGENT_CONFIG = "Failed to check existence of agent config '{name}': {error}"
    STORAGE_FAILED_GET_BUILTIN_OVERRIDE = "Failed to get the built-in assistant customization: {error}"
    STORAGE_FAILED_SAVE_BUILTIN_OVERRIDE = "Failed to save the built-in assistant customization: {error}"
    STORAGE_FAILED_DELETE_BUILTIN_OVERRIDE = "Failed to reset the built-in assistant customization: {error}"
    STORAGE_FAILED_PERSIST_STREAM = "Failed to persist AI message after stream: {error}"
    STORAGE_FAILED_API_KEY_OP = "Failed to perform API key operation: {error}"

    # --- HITL ---
    INVALID_HITL_ACTION = "Unsupported HITL action: {action}"

    # --- MCP ---
    MCP_CONNECTION_ERROR = "Failed to connect to MCP servers: {error}"
    MCP_TOOL_LOAD_ERROR = "Failed to load MCP tools: {error}"
    MCP_TOOL_CALL_TIMEOUT = "MCP tool '{name}' timed out after {timeout}s"
    MCP_REGISTRY_UNAVAILABLE = "MCP registry service is unavailable; please retry later."
    MCP_REGISTRY_REQUEST_FAILED = "MCP registry request failed with status {status}."
    MCP_REGISTRY_SERVER_NOT_FOUND = "Registered MCP server not found: {name}"
    MCP_TOOL_MESSAGE_EMPTY = "message must be a non-empty string"
    MCP_SKILL_NAME_INVALID = "Invalid skill name '{name}': use a plain directory name (no path separators, no '..')."
    MCP_MEMORY_NAME_INVALID = "Invalid memory filename '{name}': use a plain file name (no path separators, no '..')."
    MCP_MEMORY_AGENT_SEGMENT_INVALID = (
        "Invalid agent memory segment {slug!r}: must match ^[a-z0-9][a-z0-9-]{{0,99}}$ (exactly one agent segment)."
    )
    AGENT_SPACE_SLUG_COLLISION = (
        "RAG space slug '{slug}' (from '{name}') is already claimed by '{other}' — names mapping to the "
        "same scope slug would share one RAG space. Rename one of the agents or adjust the topology "
        "(there is no rename on agents: delete the unwanted one and re-create it) to resolve the conflict."
    )
    AGENT_SPACE_SLUG_RESERVED = (
        "RAG space slug '{slug}' is reserved by the platform (from '{name}') — the 'personal' scope is the "
        "user's own default space and can never be owned by an agent; use a distinct name."
    )

    # --- Prompt management (Phoenix, removed) ---
    # Legacy Phoenix prompt templates removed with the Phoenix prompt path.
    # (kept placeholder comment for section continuity)

    # --- Local prompt registry (spec §5.1) ---
    PROMPT_NOT_FOUND = "Prompt not found: {prompt_id}"
    PROMPT_VERSION_NOT_FOUND = "Prompt version {version} not found for prompt {prompt_id}"
    PROMPT_IN_USE = "Prompt is referenced by one or more agents and cannot be deleted: {prompt_id}"
    PROMPT_FAILED_CREATE = "Failed to create prompt: {error}"
    PROMPT_FAILED_GET = "Failed to get prompt {prompt_id}: {error}"
    PROMPT_FAILED_LIST = "Failed to list prompts: {error}"
    PROMPT_FAILED_DELETE = "Failed to delete prompt {prompt_id}: {error}"
    PROMPT_FAILED_ADD_VERSION = "Failed to add version to prompt {prompt_id}: {error}"
    PROMPT_FAILED_LIST_VERSIONS = "Failed to list versions for prompt {prompt_id}: {error}"

    # --- Model catalog (spec §5.2) ---
    MODEL_NOT_FOUND = "Model not found: {model_id}"
    MODEL_IN_USE = "Model is referenced by one or more agents and cannot be deleted: {model_id}"
    MODEL_REPOSITORY_NOT_WIRED = "Agent '{name}' sets model_ref but no model repository is wired."
    MODEL_FAILED_CREATE = "Failed to create model: {error}"
    MODEL_FAILED_GET = "Failed to get model {model_id}: {error}"
    MODEL_FAILED_LIST = "Failed to list models: {error}"
    MODEL_FAILED_DELETE = "Failed to delete model {model_id}: {error}"

    # --- Cron subsystem (spec §5.3) ---
    CRON_NOT_FOUND = "Cron not found: {cron_id}"
    CRON_INVALID_EXPRESSION = "Invalid cron expression '{expression}': {reason}"
    CRON_INVALID_TIMEZONE = "Invalid timezone '{timezone}'"
    CRON_AGENT_NOT_ACCESSIBLE = "Cron agent not found or not owned by you: {agent_name}"
    # Rendered as a 422 boundary error — the message deliberately carries no
    # client-supplied value (QA BUG-010: no Python ``None`` repr in payloads).
    CRON_AGENT_NAME_REQUIRED = (
        "agent_name must be provided as a non-empty string (an explicit null is not a valid agent reference)"
    )
    CRON_AGENT_UNATTENDED_PLACEHOLDER = (
        "Agent '{agent_name}' requires ${{USER_JWT}}/${{USER_API_KEY}} user credentials and cannot run "
        "unattended (scheduled executions never borrow an interactive user's credentials)."
    )
    CRON_FAILED_CREATE = "Failed to create cron: {error}"
    CRON_FAILED_GET = "Failed to get cron {cron_id}: {error}"
    CRON_FAILED_LIST = "Failed to list crons: {error}"
    CRON_FAILED_DELETE = "Failed to delete cron {cron_id}: {error}"
    CRON_FAILED_UPDATE = "Failed to update cron {cron_id}: {error}"
    CRON_RUN_CONFLICT = "Cron {cron_id} already has an active run"
    CRON_RUN_NOT_CREATED = "Failed to create cron run for cron {cron_id}: {error}"
    CRON_RUN_NOT_UPDATED = "Failed to update cron run {run_id}: {error}"
    CRON_RUN_LIST_FAILED = "Failed to list runs for cron {cron_id}: {error}"
    CRON_RUN_LIMIT_INVALID = "limit must be between 1 and 200"
    CRON_RUN_AGENT_MISSING = "Cron agent no longer available: {agent_name}"
    CRON_RUN_INVOCATION_FAILED = "Cron run invocation failed: {error}"
    CRON_RUN_STALE_RECLAIMED = (
        "Cron run was interrupted (worker terminated) and was reclaimed; it will not be retried automatically."
    )

    # --- Security ---
    API_KEY_UNAUTHORIZED = "The Api Key you provided is unauthorized"
    API_KEY_EMPTY = "The Api Key is empty"
    API_KEY_DISABLED = "Auth by Api key is disabled"
    API_KEY_NOT_FOUND = "API key not found: {key_id}"
    API_KEY_NAME_REQUIRED = "API key name is required"

    # --- Auth (dual JWT / API key) ---
    AUTH_INVALID_CREDENTIALS = "Invalid or missing credentials"
    AUTH_JWT_INVALID = "Invalid JWT token"
    AUTH_API_KEY_INVALID = "Invalid API key"

    # --- LLM credentials (per user) ---
    LLM_NOT_CONFIGURED = "No LLM provider configured for user {user_id}. Configure via PUT /api/v1/settings/llm."
    LLM_SETTINGS_NOT_FOUND = "LLM settings not found for user {user_id}"

    # --- Chat attachments (chat-pdf-attachments spec §5.1/§5.4) ---
    ATTACHMENT_EMPTY_FILE_NAME = "Attachment filename must be a non-empty string"
    ATTACHMENT_FILE_NAME_TRAVERSAL = "Attachment filename {filename!r} contains a forbidden path sequence"
    ATTACHMENT_INVALID_BASE64 = "Attachment {filename!r} content_base64 is not strictly decodable base64"
    ATTACHMENT_MIME_REQUIRED = "Attachment {filename!r} must declare a content_type"
    ATTACHMENT_IMAGE_TYPE_NOT_SUPPORTED = (
        "Attachment {filename!r} uses unsupported image type {content_type!r} (allowed: {allowed})"
    )
    ATTACHMENT_TYPE_NOT_COHERENT = (
        "Attachment {filename!r} content type {content_type!r} is not coherent with extension {extension!r}"
    )
    ATTACHMENT_DOCUMENT_TYPE_NOT_ALLOWED = (
        "Attachment {filename!r} uses type {content_type!r} outside the allowed document families"
    )
    ATTACHMENT_IMAGE_TOO_LARGE = "Attachment {filename!r} exceeds the maximum image size of {max_bytes} bytes"
    ATTACHMENT_DOCUMENT_TOO_LARGE = "Attachment {filename!r} exceeds the maximum document size of {max_bytes} bytes"
    ATTACHMENT_COUNT_EXCEEDED = "Too many attachments: {count} (maximum {max_count})"
    ATTACHMENT_TOTAL_TOO_LARGE = "Total attachment size {total_bytes} bytes exceeds the maximum of {max_bytes} bytes"
    ATTACHMENT_DOCUMENT_POLICY = (
        "This agent has no Files MCP server (an HTTP MCP server whose URL ends with /files/mcp): "
        "documents cannot be attached to its conversations."
    )
    ATTACHMENT_MCP_UPLOAD_FAILED = "Temporary document upload failed for {filename!r}: {error}"
    ATTACHMENT_MCP_DELETE_FAILED = "Temporary document cleanup failed (objects may remain in chat_tmp): {error}"
    ATTACHMENT_UNAVAILABLE = "Attachments are not supported for this deployment (attachment service not wired)."
    PDF_TOOL_ERROR = "create_pdf rejected the request: {reason}"


def tmpl(template: str) -> Template:
    """Return a ``string.Template`` for ``$name`` style templates when needed."""
    return Template(template)
