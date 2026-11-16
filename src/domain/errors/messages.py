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
    STORAGE_FAILED_PERSIST_STREAM = "Failed to persist AI message after stream: {error}"
    STORAGE_FAILED_API_KEY_OP = "Failed to perform API key operation: {error}"

    # --- HITL ---
    INVALID_HITL_ACTION = "Unsupported HITL action: {action}"

    # --- MCP ---
    MCP_CONNECTION_ERROR = "Failed to connect to MCP servers: {error}"
    MCP_TOOL_LOAD_ERROR = "Failed to load MCP tools: {error}"
    MCP_TOOL_CALL_TIMEOUT = "MCP tool '{name}' timed out after {timeout}s"

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


def tmpl(template: str) -> Template:
    """Return a ``string.Template`` for ``$name`` style templates when needed."""
    return Template(template)
