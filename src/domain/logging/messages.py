"""Centralized log message templates.

All log message strings used across the application are declared here so they
are discoverable, non-duplicated, and easy to audit. Callers reference these
constants instead of inlining message text, e.g.::

    logger.info(LogMessage.AGENT_CONFIG_CREATED, agent_name)

Each template may contain ``%s``/``%d`` style placeholders consumed by the
stdlib logging lazy interpolation.
"""

from enum import StrEnum


class LogMessage(StrEnum):
    """Catalog of centralized log message templates (stdlib ``%`` style)."""

    # --- Application lifecycle ---
    APP_STARTUP_INITIATED = "Application startup initiated"
    APP_STARTUP_COMPLETE = "Application startup complete"
    APP_PERSISTENCE_INITIALIZED = "Persistence initialized"
    APP_PERSISTENCE_INIT_FAILED = "Failed to initialize persistence, falling back to filesystem registry"
    APP_SHUTDOWN_INITIATED = "Application shutdown initiated"
    APP_SHUTDOWN_COMPLETE = "Application shutdown complete"
    APP_PERSISTENCE_CLOSED = "Persistence closed"
    APP_PERSISTENCE_CLOSE_FAILED = "Error closing persistence"
    APP_MCP_LOADER_CLOSED = "MCP tool loader closed"
    APP_MCP_LOADER_CLOSE_FAILED = "Error closing MCP tool loader"
    APP_TRACING_SHUTDOWN = "Tracing provider shut down"
    APP_TRACING_SHUTDOWN_FAILED = "Error shutting down tracing provider"
    APP_MIGRATIONS_RUNNING = "Running database migrations..."
    APP_MIGRATIONS_DONE = "Database migrations completed"
    APP_SCHEDULER_START_FAILED = (
        "Cron scheduler loop failed to start (non-fatal): the tick will not run until the next process start"
    )
    DEPENDENCIES_INITIALIZED = "Dependencies initialized"

    # --- Dependency wiring / persistence init ---
    TRACING_LANGFUSE_INIT = "Initializing Langfuse tracing provider (host=%s)"
    TRACING_PHOENIX_INIT = "Initializing Phoenix tracing provider (endpoint=%s)"
    TRACING_DISABLED = "Tracing disabled, using NoopTracingProvider"
    PERSISTENCE_INITIALIZING = "Initializing persistence layer"
    SQLALCHEMY_ENGINE_CREATED = (
        "SQLAlchemy async engine created (pool: AsyncAdaptedQueuePool, size=20, max_overflow=20)"
    )
    POSTGRES_REPOS_INITIALIZED = "PostgreSQL repositories initialized"
    RUSTFS_STORE_INITIALIZED = "RustFS store initialized (bucket=%s)"
    PERSISTENCE_REGISTRY_SET = "Persistence layer initialized, agent_registry set to PersistentAgentRegistry"
    PERSISTENT_REGISTRY_CLOSED = "Persistent registry closed"
    SQLALCHEMY_ENGINE_DISPOSED = "SQLAlchemy engine disposed"
    JWT_ADAPTER_CLOSED = "JWT adapter httpx client closed"
    PERSISTENCE_STORE_FILE_INITIALIZED = "Store file repository initialized (AsyncPostgresStore)"
    PERSISTENCE_STORE_FILE_INIT_FAILED = (
        "Failed to initialize store file repository with Postgres, falling back to InMemoryStore"
    )
    PERSISTENCE_STORE_FILE_FALLBACK_INMEMORY = "Store file repository initialized (InMemoryStore fallback)"

    # --- Agent config management ---
    AGENT_CONFIG_LISTED = "Listed %d agent configs"
    AGENT_CONFIG_LISTED_FROM_REPO = "Listed %d agent configs from repository"
    AGENT_CONFIG_GET = "Getting agent config: %s"
    AGENT_CONFIG_CREATING = "Creating agent config: %s"
    AGENT_CONFIG_CREATED = "Agent config created: %s"
    AGENT_CONFIG_CREATED_UC = "Created agent config '%s'"
    AGENT_CONFIG_UPDATING = "Updating agent config: %s"
    AGENT_CONFIG_UPDATED = "Agent config updated: %s"
    AGENT_CONFIG_UPDATED_UC = "Updated agent config '%s'"
    AGENT_CONFIG_DELETING = "Deleting agent config: %s"
    AGENT_CONFIG_DELETED = "Agent config deleted: %s"
    AGENT_CONFIG_DELETED_UC = "Deleted agent config '%s'"
    AGENT_CONFIG_METADATA_SAVED = "Saved agent config metadata '%s'"
    AGENT_CONFIG_METADATA_DELETED = "Deleted agent config metadata '%s'"
    AGENT_CONFIG_LOADED_FROM_STORE = "Loaded agent config '%s' from store"

    # --- Chat / messaging ---
    CHAT_RECEIVE = "[thread=%s] POST /chat - message=%s"
    CHAT_RESPONSE = "[thread=%s] Response status=%s content_len=%d"
    CHAT_STREAM_RECEIVE = "[thread=%s] POST /chat/stream - message=%s"
    CHAT_STREAM_COMPLETE = "[thread=%s] Stream complete, %d chunks"
    CHAT_STREAM_ERROR = "[thread=%s] Stream error after %d chunks"
    CHAT_SENDING = "Sending message to thread '%s' (agent=%s)"
    CHAT_SENT = "Chat completed [thread=%s][agent=%s] elapsed=%.2fs status=%s"
    CHAT_HITL = "HITL [thread=%s][agent=%s] elapsed=%.2fs status=%s"
    CHAT_SENDING_HUMAN = "[thread=%s][agent=%s] Sending human message"
    CHAT_SKIP_DUPLICATE_HUMAN = "[thread=%s] Skipping duplicate HUMAN message"
    CHAT_SENDING_HITL = "[thread=%s][agent=%s] Sending HITL decision (action=%s)"
    CHAT_STREAM_COMPLETE_UC = "Stream complete [thread=%s][agent=%s] %d chunks, elapsed=%.2fs, message=persisted"
    CHAT_STREAM_PERSIST_FAILED = "Failed to persist AI message after stream [thread=%s][agent=%s]"

    # --- Threads ---
    THREAD_CREATING = "Creating thread for agent=%s"
    THREAD_CREATED = "Thread created: id=%s agent=%s"
    THREAD_LISTED = "Listed %d threads"
    THREAD_GETTING = "Getting thread=%s"
    THREAD_DELETING = "Deleting thread=%s"
    THREAD_DELETED = "Thread deleted: %s"
    THREAD_MESSAGES_LISTED = "[thread=%s] Listed %d messages"

    # --- WebSocket ---
    WS_CONNECTED = "[thread=%s] WebSocket connected"
    WS_INVALID_JSON = "[thread=%s] Invalid JSON received: %s"
    WS_MESSAGE_RECEIVED = "[thread=%s] WS message received: %s"
    WS_STREAM_COMPLETE = "[thread=%s] WS stream complete, %d chunks"
    WS_STREAM_ERROR = "[thread=%s] WS stream error after %d chunks"
    WS_DISCONNECTED = "[thread=%s] WebSocket disconnected"
    WS_UNEXPECTED_ERROR = "[thread=%s] WebSocket unexpected error"

    # --- Prompt management ---
    PROMPT_CREATING = "Creating prompt: %s"
    PROMPT_CREATED = "Prompt created: %s"
    PROMPT_CREATED_SUCCESS = "Prompt created successfully: %s"
    PROMPT_UPDATING = "Updating prompt: %s"
    PROMPT_UPDATED = "Prompt updated: %s"
    PROMPT_UPDATED_SUCCESS = "Prompt updated successfully: %s"
    PROMPT_RETRIEVING = "Retrieving prompt: %s"
    PROMPT_RETRIEVING_CONTENT = "Retrieving prompt content: %s"
    PROMPT_RETRIEVED = "Retrieved prompt content for '%s' (version: %s, tags: %s)"
    PROMPT_CREATE_ERROR = "Error creating prompt '%s'"
    PROMPT_GET_ERROR = "Error getting prompt '%s'"
    PROMPT_GET_CONTENT_ERROR = "Error getting prompt content '%s'"
    PROMPT_UPDATE_ERROR = "Error updating prompt '%s'"
    PROMPT_TAG_ADDED = "Added tag '%s' to prompt '%s'"
    PROMPT_TAG_ADD_FAILED = "Failed to add tag '%s' to '%s': %s"
    PROMPT_PROVIDER_INITIALIZED = "Prompt provider initialized base_url=%s timeout=%ss"
    PROMPT_PROVIDER_INIT_FAILED = "Failed to initialize prompt provider client"
    PROMPT_DESC_UPDATE_UNSUPPORTED = (
        "Provider does not support updating description on existing prompts — description change ignored for '%s'"
    )

    # --- Validation warnings (Pydantic validators) ---
    VALIDATION_MSG_AND_HITL_EXCLUSIVE = "Provide either 'message' or HITL fields, not both"
    VALIDATION_ACTION_REQUIRED = "'action' is required for HITL decisions"
    VALIDATION_EDITS_REQUIRED = "'edits' is required for action 'edit'"
    VALIDATION_PROMPTS_MUTUALLY_EXCLUSIVE = "system_prompt and system_prompt_file are mutually exclusive"
    VALIDATION_COMMAND_REQUIRED = "'command' is required for stdio transport"
    VALIDATION_URL_REQUIRED = "'url' is required for http transport"

    # --- Infrastructure ---
    MCP_CONNECTING = "Connecting to MCP servers"
    MCP_TOOLS_LOADED = "Loaded %d MCP tools"
    MCP_TOOL_TIMEOUT = "MCP tool '%s' timed out after %ss"
    YAML_LOADED = "Loaded config from %s"
    DB_QUERY_FAILED = "Database operation failed: %s"

    # --- Chat use cases (send_message / stream_message) ---
    CHAT_INVOKE_COMPLETE = "[thread=%s][agent=%s] Invoke elapsed=%.2fs, status=%s, len=%d"
    CHAT_HITL_RECEIVED = "[thread=%s][agent=%s] HITL action=%s tool_call_id=%s"
    CHAT_HITL_COMPLETE = "[thread=%s][agent=%s] HITL elapsed=%.2fs, status=%s"
    CHAT_STREAM_STARTED = "[thread=%s][agent=%s] Stream started"
    CHAT_STREAM_ERROR_UC = "[thread=%s][agent=%s] Stream error after %d chunks"
    CHAT_STREAM_COMPLETE_PERSISTED = (
        "[thread=%s][agent=%s] Stream complete, %d chunks, elapsed=%.2fs, message=persisted"
    )

    # --- DeepAgent runner lifecycle ---
    AGENT_INVOKING = "[thread=%s] Invoking agent"
    AGENT_MESSAGE = "[thread=%s] Message: %s"
    AGENT_INVOKE_COMPLETE = "[thread=%s] Invoke complete, status=%s, elapsed=%.2fs"
    AGENT_EXECUTION_ERROR_LOG = "[thread=%s] Agent execution error"
    AGENT_FIRST_CHUNK = "[thread=%s] First chunk received, elapsed=%.2fs"
    AGENT_STREAMING = "[thread=%s] Streaming agent response"
    AGENT_STREAMING_ERROR_LOG = "[thread=%s] Streaming error"
    AGENT_STREAMING_WITH_MESSAGE = "[thread=%s] Streaming agent response with final message"
    AGENT_STREAM_WITH_MESSAGE_COMPLETE = "[thread=%s] Stream with message complete, %d chunks, elapsed=%.2fs, status=%s"
    AGENT_STREAM_COMPLETE = "[thread=%s] Stream complete, %d chunks, elapsed=%.2fs"
    AGENT_STREAM_IDLE_TIMEOUT = "[thread=%s] Agent stream idle timeout after %ss"
    AGENT_INVOKE_TIMEOUT = "[thread=%s] Agent invoke timeout after %ss"
    HITL_APPROVE = "[thread=%s] HITL approve"
    HITL_APPROVE_COMPLETE = "[thread=%s] HITL approve complete, elapsed=%.2fs"
    HITL_APPROVE_ERROR_LOG = "HITL approve error"
    HITL_REJECT = "[thread=%s] HITL reject, reason=%s"
    HITL_REJECT_COMPLETE = "[thread=%s] HITL reject complete, elapsed=%.2fs"
    HITL_REJECT_ERROR_LOG = "HITL reject error"
    HITL_EDIT = "[thread=%s] HITL edit, tool_call_id=%s"
    HITL_EDIT_COMPLETE = "[thread=%s] HITL edit complete, elapsed=%.2fs"
    HITL_EDIT_ERROR_LOG = "HITL edit error"
    HITL_RESUME = "[thread=%s] HITL resume, decisions=%d"
    HITL_RESUME_COMPLETE = "[thread=%s] HITL resume complete, elapsed=%.2fs, status=%s"
    HITL_RESUME_ERROR_LOG = "HITL resume error"
    HITL_NO_PENDING_INTERRUPT = "[thread=%s] HITL resume aborted: no pending interrupt"
    CHECKPOINTER_POSTGRES_UNAVAILABLE = "Postgres checkpointer unavailable, falling back to MemorySaver: %s"

    # --- DeepAgent runner / ToolNode patching ---
    TOOLS_NODE_MISSING = "No 'tools' node found in graph; cannot patch handle_tool_errors"
    TOOLS_NODE_NO_BOUND = "'tools' node has no 'bound' attribute; cannot patch handle_tool_errors"
    TOOLNODE_PATCHED = "Patched ToolNode handle_tool_errors=True"
    TOOLNODE_PATCH_MISSING_ATTR = "ToolNode bound object missing _handle_tool_errors; patch not applied"
    STRUCTURED_RESPONSE_MISSING = "Structured response missing despite response_format being configured"
    STRUCTURED_RESPONSE_VALIDATION_FAILED = "Failed to validate structured_response against schema, returning raw data"
    STRUCTURED_FIELD_STRIPPED = "Stripped extra field from structured_response: '%s'"
    STRUCTURED_NESTED_FIELD_STRIPPED = "Stripped extra nested field: '%s.%s'"

    # --- Agent factory ---
    AGENT_CREATING = "Creating agent '%s' (model=%s)"
    AGENT_MCP_TOOLS_LOADING = "Loading MCP tools for agent '%s' (%d servers)"
    AGENT_MCP_TOOLS_LOADED = "Loaded %d MCP tools for agent '%s'"
    AGENT_TOOLS_TOTAL = "Agent '%s' tools: %d total"
    AGENT_SUBAGENTS = "Agent '%s' has %d subagents"
    AGENT_CREATE_ERROR = "Error creating agent '%s'"
    AGENT_CREATED = "Agent '%s' created successfully"
    TOOL_FORMAT_INVALID = "Invalid tool format '%s'"
    TOOL_MODULE_NOT_FOUND = "Module not found for tool '%s'"
    TOOL_ATTRIBUTE_NOT_FOUND = "Attribute '%s' not found in '%s'"
    SUBAGENT_PROMPT_LOAD_FAILED = (
        "Could not load system prompt for sub-agent '%s' from Phoenix, using YAML instructions if available."
    )

    # --- Persistent agent registry ---
    AGENT_CACHE_HIT = "Agent '%s' loaded from cache"
    AGENT_BUILDING = "Building agent '%s' from persistent store"
    AGENT_CACHED = "Agent '%s' ready and cached"
    AGENT_CACHE_INVALIDATED = "Invalidated cached agent '%s'"
    REGISTRY_CLOSING = "Closing persistent registry, clearing %d cached agents"

    # --- RustFS store ---
    RUSTFS_CONFIG_UPLOADED = "RustFS config uploaded: %s (bucket=%s)"
    RUSTFS_CONFIG_DELETED = "RustFS config deleted: %s (bucket=%s)"
    RUSTFS_BUCKET_CREATED = "Created RustFS bucket '%s'"
    RUSTFS_BUCKET_EXISTS = "RustFS bucket already created: %s"

    # --- MCP loader ---
    MCP_CONNECT_FAILED = "Failed to connect to MCP servers"
    MCP_TOOLS_LOAD_FAILED = "Failed to load MCP tools"

    # --- Phoenix prompt adapter (removed with the Phoenix prompt path) ---

    # --- Local prompt registry (spec §5.1) ---
    PROMPT_REGISTRY_CREATING = "Creating prompt %s"
    PROMPT_REGISTRY_CREATED = "Prompt created: id=%s latest_version=%s"
    PROMPT_REGISTRY_METADATA_UPDATED = "Prompt metadata updated: id=%s"
    PROMPT_REGISTRY_DELETED = "Prompt deleted: id=%s"
    PROMPT_REGISTRY_VERSION_ADDED = "Prompt version added: prompt=%s version=%s"
    PROMPT_REGISTRY_VERSIONS_LISTED = "Listed %d versions for prompt %s"
    PROMPT_REGISTRY_FETCHED = "Fetched prompt: id=%s"

    # --- Model catalog (spec §5.2) ---
    MODEL_CREATING = "Creating model %s"
    MODEL_CREATED = "Model created: id=%s"
    MODEL_UPDATED = "Model updated: id=%s"
    MODEL_DELETED = "Model deleted: id=%s"
    MODEL_FETCHED = "Fetched model: id=%s"
    MODEL_DELETED_IN_USE = "Refused to delete in-use model: id=%s"
    PROMPT_DELETED_IN_USE = "Refused to delete in-use prompt: id=%s"

    # --- Cron subsystem (spec §5.3) ---
    CRON_CREATING = "Creating cron %s (agent=%s)"
    CRON_CREATED = "Cron created: id=%s next_run_at=%s"
    CRON_UPDATED = "Cron updated: id=%s next_run_at=%s"
    CRON_DELETED = "Cron deleted: id=%s"
    CRON_FETCHED = "Fetched cron: id=%s"
    CRON_TICK_STARTED = "Scheduler tick: %d due cron(s)"
    CRON_TICK_CLAIMED = "Scheduler claimed occurrence: cron=%s occurrence=%s"
    CRON_TICK_CLAIM_CONFLICT = "Scheduler lost occurrence claim: cron=%s occurrence=%s"
    CRON_TICK_OVERLAP_SKIPPED = "Overlapping occurrence skipped: cron=%s occurrence=%s"
    CRON_RUN_NOW = "Manual run requested: cron=%s"
    CRON_RUN_STARTING = "Starting cron run %s (cron=%s trigger=%s)"
    CRON_RUN_PLACEHOLDER_REJECTED = "Cron run %s rejected: agent requires user placeholders"
    CRON_RUN_FINISHED = "Cron run %s finished: status=%s elapsed=%.2fs"
    CRON_RUN_FRONTIER_ADVANCED = "Cron %s next_run_at advanced to %s"
    CRON_RUN_STALE_RECLAIMED = "Reclaimed %d stale cron run(s) left by a terminated worker"
    CRON_SCHEDULER_LOOP_STARTED = "Cron scheduler loop started (interval=%ss)"
    CRON_SCHEDULER_LOOP_STOPPED = "Cron scheduler loop stopped"
    CRON_SCHEDULER_TICK_FAILED = "Scheduler tick failed"
    CRON_RUN_INVOCATION_FAILED = "Cron run invocation failed: %s"

    # --- RLS (Row-Level Security) ---
    RLS_CONTEXT_SET = "RLS context set: app.user_id=%s"
    RLS_BYPASS_ENABLED = "RLS bypass enabled (row_security=off)"

    # --- Global traces (spec §5.4) ---
    TRACE_SUMMARIES_LISTED = "Listed %d trace summaries (agent_name=%s, cursor=%s)"
    TRACE_SUMMARIES_UNSCOPED = "Trace summaries requested without an authenticated owner — returning an empty page"

    # --- LLM usage observability (dashboard spec §1-§5) ---
    TRACE_USAGE_LISTED = "Aggregated LLM usage (agent_name=%s, run_count=%d, error_count=%d)"
    TRACE_USAGE_TOP_LISTED = "Listed top LLM usage runs (limit=%d, items=%d)"
    TRACE_USAGE_UNSCOPED = "LLM usage requested without an authenticated owner — returning an empty result"
    TRACE_USAGE_UNEXPECTED_SHAPE = "LLM usage statement returned an unexpected shape — returning an empty summary"
    LLM_PRICES_SEEDED = "Seeded/refreshed %d known LLM price rows"
    LLM_PRICES_SEED_FAILED = "LLM price seeding failed (non-fatal): usage costs will resolve as unpriced"

    # --- Registry freshness (CACHE-002) ---
    RUNNER_FRESHNESS_REBUILD = "Dependency fingerprint changed for agent '%s' (user=%s); rebuilding runner"
    RUNNER_FRESHNESS_STALE_EVICTED = "Evicted stale cached runner for agent '%s' (user=%s)"
    AGENT_MCP_TOOLS_SKIPPED_SERVER = (
        "Agent '%s': MCP server '%s' unreachable during tool loading (%s); skipped — "
        "the agent boots with the remaining servers' tools"
    )

    # --- Phoenix tracing provider ---
    PHOENIX_PROVIDER_INIT = "Initializing PhoenixTracingProvider with endpoint=%s, project_name=%s"
    PHOENIX_TRACING_INITIALIZED = "Phoenix tracing initialized successfully"
    PHOENIX_SPANS_FLUSHED = "Flushed pending spans to Phoenix"
    PHOENIX_FLUSH_FAILED = "Error flushing spans to Phoenix"
    PHOENIX_TRACING_SHUTDOWN = "Phoenix tracing provider shutdown complete"
    TRACER_SHUTDOWN_FAILED = "Error shutting down tracer provider"

    # --- YAML config loader ---
    YAML_PARSE_FAILED = "Invalid YAML from %s"
    YAML_NOT_MAPPING_LOG = "YAML from %s must contain a YAML mapping, not %s"
    YAML_VALIDATION_FAILED = "Validation error from %s"
    CONFIG_FILE_NOT_FOUND_LOG = "Config file not found: %s"
    PROMPT_FILE_NOT_FOUND_LOG = "Prompt file not found: %s"
    YAML_EMPTY_LOG = "Empty YAML content from %s"
    YAML_SYSTEM_PROMPT_FILE_DISALLOWED = "system_prompt_file is not allowed in string-loaded YAML from %s"

    # --- Exception handler log lines (main.py) ---
    LOG_AGENT_CONFIG_ALREADY_EXISTS = "Agent config already exists: %s"
    LOG_STORAGE_ERROR = "Storage error: %s"
    LOG_AGENT_NOT_FOUND = "Agent not found: %s"
    LOG_CONFIG_NOT_FOUND = "Config not found: %s"
    LOG_THREAD_NOT_FOUND = "Thread not found: %s"
    LOG_PROMPT_NOT_FOUND = "Prompt not found: %s"
    LOG_PROMPT_ALREADY_EXISTS = "Prompt already exists: %s"
    LOG_PROMPT_MANAGER_UNAVAILABLE = "Prompt manager unavailable: %s"
    LOG_INVALID_HITL_ACTION = "Invalid HITL action: %s"
    LOG_CONFIG_VALIDATION_ERROR = "Config validation error: %s errors=%s"
    LOG_CONFIG_ERROR = "Config error: %s"
    LOG_AGENT_ERROR = "Agent error: %s"
    LOG_MCP_ERROR = "MCP error: %s"
    LOG_UNHANDLED_DOMAIN_ERROR = "Unhandled domain error: %s"
    LOG_PROMPT_IN_USE = "Prompt in use: %s"
    LOG_MODEL_NOT_FOUND = "Model not found: %s"
    LOG_MODEL_IN_USE = "Model in use: %s"
    LOG_CRON_NOT_FOUND = "Cron not found: %s"
    LOG_CRON_VALIDATION_ERROR = "Cron validation error: %s"
    LOG_CRON_RUN_CONFLICT = "Cron run conflict: %s"
    LOG_AGENT_CONFIG_REFERENCE_ERROR = "Agent config reference error: %s"
    LOG_ATTACHMENT_VALIDATION = "Attachment validation error: %s"
    LOG_DOCUMENTS_NOT_ALLOWED = "Documents not allowed error: %s"
    LOG_FILES_MCP_TEMPORARY_ERROR = "Files MCP temporary tool error: %s"

    # --- Chat attachments (chat-pdf-attachments spec §5.1-§5.4) ---
    CHAT_ATTACHMENTS_VALIDATING = "Validating %d attachment(s) for thread %s"
    CHAT_ATTACHMENTS_IMAGE_STORED = "Stored image attachment %s at store path %s (thread=%s)"
    CHAT_ATTACHMENTS_DOCUMENT_UPLOADING = "Uploading temporary document %s via Files MCP server %s (thread=%s)"
    CHAT_ATTACHMENTS_DOCUMENT_UPLOADED = "Temporary document %s stored at %s (thread=%s)"
    CHAT_ATTACHMENTS_DOCUMENT_POLICY_FAILED = (
        "Document attachment refused for thread %s (agent=%s): no /files/mcp server"
    )
    CHAT_ATTACHMENTS_MCP_UPLOAD_FAILED = "Files MCP temporary_upload_file failed for %s (thread=%s)"
    CHAT_ATTACHMENTS_MCP_TOOL_FAILED = "Files MCP temporary tool call failed for %s (server=%s)"
    CHAT_TEMP_CLEANUP_STARTED = "Cleaning up %d temporary document(s) for thread %s"
    CHAT_TEMP_CLEANUP_SUCCEEDED = "Temporary documents deleted for thread %s (count=%d)"
    CHAT_TEMP_CLEANUP_FAILED = "Temporary document cleanup failed for object %s (thread=%s): %s"
    CHAT_TEMP_CLEANUP_RETRY = "Retrying temporary document cleanup for object %s (thread=%s)"
    CHAT_TEMP_CLEANUP_SKIPPED_HITL = "Turn ended in AWAITING_HITL: temporary documents kept for thread %s"
    CHAT_TEMP_CLEANUP_CANCELLED_RETRY = "Cleanup interrupted by cancellation for thread %s; retrying once"
    CHAT_ATTACHMENTS_UNAVAILABLE = (
        "Chat attachment preparation is unavailable (persistence not initialized); "
        "attachment-bearing requests will be refused while text-only chat keeps working"
    )
    CHAT_TEMP_CLEANER_UNAVAILABLE = (
        "Temporary document cleaner is unavailable (agent persistence not initialized); "
        "temp objects may remain in chat_tmp after turns"
    )

    # --- create_pdf builtin tool (chat-pdf-attachments spec §5.5) ---
    CREATE_PDF_GENERATING = "create_pdf generating %s for thread %s (markdown_chars=%d)"
    CREATE_PDF_ERROR = "create_pdf failed: %s"

    # --- Thread deletion store cascade (chat-pdf-attachments spec §5.2.1) ---
    THREAD_STORE_CASCADE_STARTED = "Thread store cascade started for thread %s (prefixes=%d)"
    THREAD_STORE_CASCADE_FAILED = (
        "Thread store cascade failed for thread %s under prefix %s: %s (thread row still deleted)"
    )
    THREAD_STORE_CASCADE_RETRY = "Retrying thread store cascade for thread %s under prefix %s"
    THREAD_STORE_CASCADE_COMPLETED = "Thread store cascade completed for thread %s (deleted=%d)"
    THREAD_STORE_CASCADE_UNAVAILABLE = (
        "Thread store cascade unavailable (store repository not initialized): "
        "deleting threads will leave orphan store objects under chat_attachments/documents"
    )
    STORE_FILE_LISTING_TRUNCATED = (
        "Store file listing may be truncated at the %d-item namespace cap for prefix %s "
        "(matches beyond the cap are missed)"
    )

    # --- Security ---
    LOG_INVALID_API_KEY = "Invalid API key: %s"
    # --- Auth (dual JWT / API key) ---
    # The enum value is a stable prefix kept verbatim in the formatted log line
    # (no ``%s`` inside it) so tests can assert ``LogMessage.X in r.getMessage()``.
    # No PII is interpolated here — only error type / message for diagnostics.
    AUTH_JWT_DECODE_FAILED = "JWT decode failed"
    AUTH_JWKS_FETCH_FAILED = "JWKS fetch failed"
    AUTH_CREDENTIALS_VALIDATED = "Credentials validated for user_id=%s method=%s"

    # --- Sonar security (TLS opt-out flag) ---
    DATABASE_SSL_INSECURE_ENABLED = (
        "DATABASE_SSL_INSECURE=true: PostgreSQL TLS in mode ``%s`` runs WITHOUT certificate/hostname validation "
        "(explicit unsafe opt-out). Never enable this against a public endpoint."
    )

    # --- API key management (per-user) ---
    API_KEY_CREATED = "API key created: id=%s user_id=%s"
    API_KEY_REVOKED = "API key revoked: id=%s user_id=%s"
    API_KEY_LISTED = "Listed %d API keys for user_id=%s"

    # --- LLM settings (per user) ---
    LLM_SETTINGS_DECRYPT_FAILED = "Failed to decrypt LLM API key for user_id=%s; returning masked=None"
    LLM_SETTINGS_REPO_INITIALIZED = "User LLM settings repository initialized"
    LLM_CRYPTO_KEY_EMPTY = (
        "SECRET_ENCRYPTION_KEY is empty — generated a throwaway in-memory Fernet key. "
        "Set SECRET_ENCRYPTION_KEY in production to persist encrypted API keys across restarts."
    )

    # --- Composable MCP control plane (feat-composable-mcp) ---
    MCP_SERVER_MOUNTED = "Composable MCP control plane mounted at /mcp (streamable HTTP)"
    MCP_SESSION_MANAGER_RUNNING = "Composable MCP session manager running"
    MCP_SESSION_MANAGER_STOPPED = "Composable MCP session manager stopped"
    MCP_REGISTRY_CLIENT_CLOSED = "Trusted MCP-registry HTTP client closed"
    MCP_REGISTRY_CLIENT_CLOSE_FAILED = "Error closing the trusted MCP-registry HTTP client"
    MCP_SESSION_MANAGER_ALREADY_RAN = (
        "Composable MCP session manager already ran once in this process; skipping re-entry "
        "(repeat-lifespan context, e.g. unit tests) — REST/SSE surfaces unaffected"
    )
    MCP_REQUEST_DENIED = "MCP request denied (missing or invalid credentials)"
    MCP_TOOL_EXECUTED = "MCP tool executed (user=%s tool=%s duration=%.3fs ok=%s)"
    MCP_TOOL_FAILED = "MCP tool failed (user=%s tool=%s error=%s)"
    MCP_TOOL_FAILURE_TYPE = "MCP tool failed with exception type: %s"
    MCP_REGISTRY_FORWARDING = "Registry request forwarded (method=%s path=%s)"
    MCP_REGISTRY_FORWARD_FAILED = "Registry request failed (status=%s)"
    BUILTIN_AGENT_RESOLVING = "Resolving built-in assistant definition (user=%s)"
    BUILTIN_AGENT_TOOLS_RESOLVED = "Built-in assistant tools resolved (user=%s tools=%d degraded=%d)"
    BUILTIN_AGENT_OPTIONAL_TOOL_FAILED = (
        "Built-in assistant optional MCP source '%s' failed (%s); the toolset degrades without it"
    )
    BUILTIN_TOOLNAME_COLLISION = (
        "Tool name collision on '%s' for MCP server '%s' — deterministically namespaced to '%s'"
    )
    AGENT_ONE_SHOT_STARTED = "One-shot agent run started (agent=%s source=mcp chain=%s)"
    AGENT_ONE_SHOT_COMPLETED = "One-shot agent run completed (thread=%s agent=%s status=%s duration=%.3fs)"
    AGENT_ONE_SHOT_REJECTED = "One-shot agent run rejected (agent=%s chain=%s reason=%s)"
    LOG_BUILTIN_AGENT_READONLY = "Built-in agent definition is read-only: %s"
