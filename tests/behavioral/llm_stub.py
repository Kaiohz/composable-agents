"""In-process stub of the external OpenAI-compatible LLM API.

This mock lives at the ONLY external boundary of the agent invocation chain
(the LLM provider HTTP API). Everything inside the chain — router, use cases,
PersistentAgentRegistry, DeepAgent factory, DeepAgentRunner, repositories —
runs real, so tests observe:

* the resolved SYSTEM PROMPT (first ``messages[0]`` with role ``system``),
  proving Local-registry prompt resolution (Latest vs pinned),
* the resolved MODEL NAME (``body["model"]``), proving model_ref/catalog
  resolution and legacy custom-string behavior,
* the ``Authorization: Bearer <key>`` header, proving per-user credentials are
  never shared across users (CACHE-001 user-scoped runners).

The stub supports three modes:

* ``normal``  — respond immediately with a single assistant completion.
* ``slow:<seconds>`` — sleep before responding (drives overlap / concurrency).
* ``fail``    — respond 500 (drives failed-run / no-retry scenarios).

Streaming requests (``"stream": true``) receive an SSE chunked reply; the
request body is still recorded so assertions work for both paths.
"""

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import uvicorn

STUB_HOST = "127.0.0.1"

# Deliberately distinct from the QA stack stub's default (18777) so backend
# behavioral tests never route LLM traffic to a concurrently running QA stub.
TEST_STUB_LLM_PORT = int(os.environ.get("TEST_STUB_LLM_PORT", "28777"))


@dataclass
class RecordedRequest:
    auth_header: str | None
    body: dict[str, Any]

    @property
    def system_messages(self) -> list[str]:
        return [
            str(m.get("content", ""))
            for m in self.body.get("messages", [])
            if m.get("role") == "system"
        ]

    @property
    def model(self) -> str | None:
        return self.body.get("model")


class StubLlmApp:
    """Raw ASGI application recording every chat-completion request."""

    def __init__(self) -> None:
        self.requests: list[RecordedRequest] = []
        self.mode: str = "normal"
        self._lock = asyncio.Lock()

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] != "http":
            return
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if message["type"] != "http.request" or not message.get("more_body"):
                break
        path = scope["path"]
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}

        if path.endswith("/chat/completions"):
            await self._chat(scope, body, headers, send)
            return

        await self._json(send, 200, {"data": [], "object": "list"})

    # ------------------------------------------------------------------
    async def _chat(self, scope, raw: bytes, headers: dict[str, str], send) -> None:
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {}

        if self.mode == "fail":
            await self._json(send, 500, {"error": {"message": "stub LLM failure", "type": "stub_error"}})
            return
        if self.mode.startswith("slow:"):
            await asyncio.sleep(float(self.mode.split(":", 1)[1]))

        await self._record(headers, payload)

        model = payload.get("model", "stub-model")
        created = int(time.time())
        if payload.get("stream"):
            await self._stream_reply(send, model, created)
        else:
            await self._json(send, 200, self._completion(model, created))

    async def _record(self, headers: dict[str, str], payload: dict) -> None:
        async with self._lock:
            self.requests.append(RecordedRequest(auth_header=headers.get("authorization"), body=payload))

    @staticmethod
    def _completion(model: str, created: int) -> dict:
        return {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "stub-ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    @staticmethod
    def _chunk(model: str, created: int, delta: dict, finish: str | None) -> str:
        chunk = {
            "id": "chatcmpl-stub",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    async def _stream_reply(self, send, model: str, created: int) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/event-stream"),
                    (b"cache-control", b"no-cache"),
                ],
            }
        )
        chunks = [
            self._chunk(model, created, {"role": "assistant"}, None),
            self._chunk(model, created, {"content": "stub-ok"}, None),
            self._chunk(model, created, {}, "stop"),
            "data: [DONE]\n\n",
        ]
        for chunk in chunks:
            await send({"type": "http.response.body", "body": chunk.encode(), "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    @staticmethod
    async def _json(send, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    # ------------------------------------------------------------------
    def reset(self, mode: str = "normal") -> None:
        """Clear recordings and set the response mode (per-test reset)."""
        self.requests = []
        self.mode = mode


class StubLlmServer:
    """Runs the stub ASGI app on a dedicated uvicorn server thread."""

    def __init__(self) -> None:
        self.app = StubLlmApp()
        self.port = TEST_STUB_LLM_PORT
        self.base_url = f"http://{STUB_HOST}:{self.port}/v1"
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._started = threading.Event()

    def start(self) -> None:
        config = uvicorn.Config(self.app, host=STUB_HOST, port=self.port, log_level="error")
        self._server = uvicorn.Server(config)

        def _run() -> None:
            # uvicorn sets its own signal handlers; opt out for the thread.
            self._server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
            self._started.set()
            self._server.run()

        self._thread = threading.Thread(target=_run, name="stub-llm-server", daemon=True)
        self._thread.start()

    def reset(self, mode: str = "normal") -> None:
        """Delegate to the in-process app reset (tests reset the server)."""
        self.app.reset(mode)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
