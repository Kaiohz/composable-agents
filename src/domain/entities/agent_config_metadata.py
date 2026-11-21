from datetime import datetime

from pydantic import BaseModel


class AgentConfigMetadata(BaseModel):
    """Metadata for a persisted agent configuration."""

    name: str
    model: str
    minio_path: str
    created_at: datetime
    updated_at: datetime
    description: str | None = None
    # Registry reference projections from the agent YAML (spec §5.5). None for
    # legacy agents; used for reference-safe deletion (SEC-002) and runner
    # freshness fingerprints (CACHE-002).
    prompt_ref: str | None = None
    model_ref: str | None = None
    # Owner of the configuration — set by the repository from the
    # ``current_user_id`` contextvar on save. Defaults to ``""`` so existing
    # constructions (no auth context) keep working.
    user_id: str = ""
    # Overlay-computed marker (REQ-033): ``True`` only for the virtual built-in
    # Composable AI Assistant resolved by the built-in overlay — never
    # persisted and never restored as a database column.
    is_builtin: bool = False
    # Overlay-computed per-user marker: ``True`` only when the built-in
    # assistant resolves to the current user's customization snapshot.
    is_customized: bool = False
