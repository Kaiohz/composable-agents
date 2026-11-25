"""auth_mode business rule for registered MCP servers (feat-mcp-oauth2 REQ-004).

Pure domain rule: given the configured ``auth_mode`` of a registered MCP
server and the resolved :class:`~src.domain.entities.auth.auth_context.AuthContext`
of the originating request, return the credential headers the backend must
inject when loading/validating that server's tools. The credential source
travels as a plain domain entity — no contextvar, transport or framework
dependency; the connection-dict assembly (placeholder resolution, merge
order) stays in the infrastructure loader.
"""

from src.domain.entities.auth.auth_context import AuthContext
from src.domain.entities.mcp_server_config import McpAuthMode


def user_credential_headers(auth_mode: str | None, context: AuthContext | None) -> dict[str, str]:
    """Return the credential headers to inject for ``auth_mode: "user_token"``.

    The connected user's credential is transmitted as:

    * ``Authorization: Bearer <jwt>`` when the originating request was
      JWT-authenticated (OAuth/JWT);
    * ``X-API-Key: <key>`` when it was API-key-authenticated;
    * nothing when the server uses the default ``headers`` mode or when no
      user credential exists (unattended runs) — the caller's empty-value
      dropping then never transmits a credential (``resolve_headers_drop_empty``
      semantics, REQ-004 error case).

    Args:
        auth_mode: The registered server's ``auth_mode`` (``"headers"`` or
            ``"user_token"``; absent values mean the default).
        context: The authenticated context of the originating request.

    Returns:
        The single injected credential header (empty mapping = no injection).
    """
    if auth_mode != McpAuthMode.USER_TOKEN or context is None or not context.raw_credential:
        return {}
    if context.method == "jwt":
        return {"Authorization": f"Bearer {context.raw_credential}"}
    if context.method == "api_key":
        return {"X-API-Key": context.raw_credential}
    return {}
