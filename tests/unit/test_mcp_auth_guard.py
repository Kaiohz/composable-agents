"""Unit tests for the ``/mcp`` ASGI auth guard (REQ-002/003/004, C2 remediation).

Drives :class:`McpAuthGuard` with recorded send/receive channels and a stub
authenticator, without the full app: HTTP 401 on missing/invalid credentials
and on authenticator failures, forwarding (with the guard's path
normalization) on success, and non-HTTP ASGI pass-through.
"""

import json

from src.application.mcp.server import McpAuthGuard


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
