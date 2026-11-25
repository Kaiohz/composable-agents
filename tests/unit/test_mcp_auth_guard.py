"""Unit tests for the ``/mcp`` ASGI auth guard (REQ-002/003/004, C2 remediation).

Drives :class:`McpAuthGuard` with recorded send/receive channels and a stub
authenticator, without the full app: HTTP 401 on missing/invalid credentials
and on authenticator failures, forwarding (with the guard's path
normalization) on success, and non-HTTP ASGI pass-through.
"""

import json
from unittest.mock import MagicMock

from src.application.mcp.server import McpAuthGuard, public_discovery_origin, www_authenticate_challenge


class _Recorder:
    """Collects the ASGI ``send`` calls and the forwarded scope path."""

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.forwarded_paths: list[str] = []

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        self.forwarded_paths.append(scope.get("path"))
        await send({"type": "asgi.record", "forwarded": True})


def _http_scope(headers: dict[str, str] | None = None, path: str = "/mcp") -> dict:
    return {
        "type": "http",
        "path": path,
        "headers": [(k.encode("latin-1"), v.encode("latin-1")) for k, v in (headers or {}).items()],
    }


_LOCALHOST_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*", "composables-agent.soludev.tech"]
PROD_HOST = "composables-agent.soludev.tech"


def _send_collector(messages: list[dict]):
    async def _send(message: dict) -> None:
        messages.append(message)

    return _send


async def _receive_noop() -> dict:  # pragma: no cover — never read by the guard
    return {"type": "http.request", "body": b""}


def _guard(recorder: _Recorder, authenticate) -> McpAuthGuard:
    return McpAuthGuard(recorder, authenticate)


class TestAuthGuardDenials:
    async def test_missing_credentials_returns_401_json(self):
        recorder, messages = _Recorder(), []

        async def _auth(authorization, api_key):  # noqa: ARG001 — stub signature matches the port
            return None

        await _guard(recorder, _auth)(_http_scope(), _receive_noop, _send_collector(messages))

        assert recorder.forwarded_paths == [], "the MCP app must not see an unauthenticated request"
        assert messages[0]["status"] == 401
        body = json.loads(messages[1]["body"])
        assert body == {"detail": "Invalid or missing credentials"}

    async def test_authenticator_failure_returns_401(self):
        recorder, messages = _Recorder(), []

        async def _auth(authorization, api_key):  # noqa: ARG001 — stub signature matches the port
            raise RuntimeError("auth internals")

        await _guard(recorder, _auth)(_http_scope(), _receive_noop, _send_collector(messages))

        assert recorder.forwarded_paths == []
        assert messages[0]["status"] == 401
        assert json.loads(messages[1]["body"])["detail"], "a generic, non-secret JSON body is returned"


class TestAuthGuardForwarding:
    async def test_valid_credentials_forward_with_path_normalized(self):
        recorder, messages = _Recorder(), []

        from src.domain.entities.auth.auth_context import AuthContext

        async def _auth(authorization, api_key):  # noqa: ARG001 — stub signature matches the port
            return AuthContext(user_id="user-a", method="api_key", raw_credential="cpk")

        await _guard(recorder, _auth)(_http_scope({"X-API-Key": "cpk"}), _receive_noop, _send_collector(messages))

        assert messages == [{"type": "asgi.record", "forwarded": True}]
        assert recorder.forwarded_paths == ["/"], "the exact /mcp path is rewritten to / for the inner app"

    async def test_sub_paths_pass_through_unmodified(self):
        recorder, messages = _Recorder(), []

        from src.domain.entities.auth.auth_context import AuthContext

        async def _auth(authorization, api_key):  # noqa: ARG001 — stub signature matches the port
            return AuthContext(user_id="user-a", method="jwt", raw_credential="token")

        scope = _http_scope({}, path="/mcp")

        await _guard(recorder, _auth)(scope, _receive_noop, _send_collector(messages))
        assert recorder.forwarded_paths == ["/"]

    async def test_non_http_scope_passes_through_without_auth(self):
        recorder, messages = _Recorder(), []

        async def _auth(authorization, api_key):  # noqa: ARG001 — stub signature matches the port  # pragma: no cover — must not run
            raise AssertionError("lifespan scopes are never authenticated")

        await _guard(recorder, _auth)(
            {"type": "lifespan"},
            _receive_noop,
            _send_collector(messages),
        )

        assert recorder.forwarded_paths == [None]
        # No 401 was produced; the recorder's forwarding marker is the only output.
        assert messages == [{"type": "asgi.record", "forwarded": True}]
        assert all(message.get("status") != 401 for message in messages)


class TestPublicDiscoveryOriginDerivation:
    """M-finding M-2: the discovery URL never echoes a raw, spoofable request
    Host — the configured `MCP_PUBLIC_BASE_URL` wins, and only an explicitly
    allow-listed Host may be used as a fallback."""

    def test_configured_base_url_always_wins(self):
        origin = public_discovery_origin(
            "dns-rebind.evil.example",
            configured_base_url="https://composables-agent.soludev.tech/",
            allowed_hosts=_LOCALHOST_HOSTS,
        )
        assert origin == "https://composables-agent.soludev.tech"

    def test_allowed_host_is_trusted_without_configuration(self):
        origin = public_discovery_origin(PROD_HOST, configured_base_url="", allowed_hosts=_LOCALHOST_HOSTS)
        assert origin == f"https://{PROD_HOST}"

    def test_wildcard_port_host_is_allowed(self):
        origin = public_discovery_origin("127.0.0.1:8012", configured_base_url="", allowed_hosts=_LOCALHOST_HOSTS)
        assert origin == "https://127.0.0.1:8012"

    def test_foreign_host_is_never_echoed(self):
        assert (
            public_discovery_origin("dns-rebind.evil.example", configured_base_url="", allowed_hosts=_LOCALHOST_HOSTS)
            is None
        )

    def test_empty_host_is_never_echoed(self):
        assert public_discovery_origin("", configured_base_url="", allowed_hosts=_LOCALHOST_HOSTS) is None

    def test_empty_allowlist_matches_nothing(self):
        assert public_discovery_origin("127.0.0.1:80", configured_base_url="", allowed_hosts=[]) is None

    def test_challenge_embeds_the_full_path(self):
        challenge = www_authenticate_challenge(f"https://{PROD_HOST}")
        assert challenge == f'Bearer resource_metadata="https://{PROD_HOST}/.well-known/oauth-protected-resource"'


class TestDefaultPortHostNormalization:
    """QA BUG-002: the default-port Host forms reach the derivation.

    An ingress may forward the port the client dialed (``Host: <host>:443``);
    behind the TLS-terminating proxy the port-less and default-port forms are
    the SAME endpoint. Only the DEFAULT-port suffixes (``:443`` https /
    ``:80`` http) are normalized — any other port still requires the
    ``host:*`` wildcard and a foreign host keeps deriving no origin
    (fail-closed).
    """

    def test_https_default_port_form_is_trusted_with_bare_entry(self):
        origin = public_discovery_origin(f"{PROD_HOST}:443", configured_base_url="", allowed_hosts=_LOCALHOST_HOSTS)
        assert origin == f"https://{PROD_HOST}", "the advertised origin drops the default https port"

    def test_http_default_port_form_is_trusted_with_bare_entry(self):
        origin = public_discovery_origin("soludev.dev:80", configured_base_url="", allowed_hosts=["soludev.dev"])
        assert origin == "https://soludev.dev", "the advertised origin drops the default http port"

    def test_default_port_form_matches_the_host_star_wildcard(self):
        origin = public_discovery_origin(
            f"{PROD_HOST}:443", configured_base_url="", allowed_hosts=[f"{PROD_HOST}:*", "localhost:*"]
        )
        assert origin == f"https://{PROD_HOST}"

    def test_non_default_port_form_still_requires_the_wildcard(self):
        assert public_discovery_origin(f"{PROD_HOST}:8443", configured_base_url="", allowed_hosts=[PROD_HOST]) is None

    def test_foreign_host_with_default_port_is_still_rejected(self):
        assert (
            public_discovery_origin(
                "dns-rebind.evil.example:443", configured_base_url="", allowed_hosts=_LOCALHOST_HOSTS
            )
            is None
        )


class TestAuthGuardDiscoveryOrigin:
    """Behavior of the guard's 401 challenge against spoofed/foreign Hosts."""

    async def _run(self, guard, headers: dict[str, str] | None) -> tuple[_Recorder, list[dict]]:
        recorder, messages = _Recorder(), []

        async def _auth(authorization, api_key):  # noqa: ARG001 — stub signature matches the port
            return None

        await guard(_http_scope(headers), _receive_noop, _send_collector(messages))
        return recorder, messages

    async def test_allowed_host_on_401_carries_allowlisted_challenge(self):
        guard = McpAuthGuard(MagicMock(), _deny, configured_base_url="", allowed_hosts=_LOCALHOST_HOSTS)
        _recorder, messages = await self._run(guard, {"Host": PROD_HOST})
        start = messages[0]
        headers = dict(start["headers"])
        assert start["status"] == 401
        assert (
            headers[b"www-authenticate"].decode()
            == f'Bearer resource_metadata="https://{PROD_HOST}/.well-known/oauth-protected-resource"'
        )

    async def test_configured_base_url_wins_over_spoofed_host(self):
        guard = McpAuthGuard(
            MagicMock(), _deny, configured_base_url="https://public.example", allowed_hosts=_LOCALHOST_HOSTS
        )
        _recorder, messages = await self._run(guard, {"Host": "dns-rebind.evil.example"})
        headers = dict(messages[0]["headers"])
        assert (
            headers[b"www-authenticate"].decode()
            == 'Bearer resource_metadata="https://public.example/.well-known/oauth-protected-resource"'
        )

    async def test_foreign_host_on_401_carries_no_discovery_url(self):
        """A spoofable (non-allow-listed) Host is NEVER echoed: the 401 goes
        out WITHOUT the WWW-Authenticate header (no discovery pointer to an
        attacker-controlled origin)."""
        guard = McpAuthGuard(MagicMock(), _deny, configured_base_url="", allowed_hosts=_LOCALHOST_HOSTS)
        _recorder, messages = await self._run(guard, {"Host": "dns-rebind.evil.example"})
        start = messages[0]
        header_names = {name for name, _ in start["headers"]}
        assert start["status"] == 401
        assert b"www-authenticate" not in header_names

    async def test_401_is_never_cacheable(self):
        """M-finding M-2 hardening: every 401 carries Cache-Control: no-store
        so a shared cache cannot replay a challenge to another client."""
        guard = McpAuthGuard(MagicMock(), _deny, configured_base_url="", allowed_hosts=_LOCALHOST_HOSTS)
        _recorder, messages = await self._run(guard, {"Host": PROD_HOST})
        assert dict(messages[0]["headers"])[b"cache-control"] == b"no-store"

    async def test_default_port_host_on_401_carries_the_challenge(self):
        """QA BUG-002: ``Host: <allowed-host>:443`` still produces the
        discovery pointer — derived from the bare, canonical origin (the
        default-port suffix is normalized away)."""
        guard = McpAuthGuard(MagicMock(), _deny, configured_base_url="", allowed_hosts=_LOCALHOST_HOSTS)
        _recorder, messages = await self._run(guard, {"Host": f"{PROD_HOST}:443"})
        start = messages[0]
        headers = dict(start["headers"])
        assert start["status"] == 401
        assert (
            headers[b"www-authenticate"].decode()
            == f'Bearer resource_metadata="https://{PROD_HOST}/.well-known/oauth-protected-resource"'
        )


async def _deny(authorization, api_key):  # noqa: ANN001, ARG001 — stub authenticator shape (unused args match the port)
    """An authenticator that always refuses (matches the port signature)."""
    return None
