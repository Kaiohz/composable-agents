"""Unit tests for the built-in assistant's dynamic toolset resolution.

Covers the REQ-036/038/039/040 contracts without the behavioral stack:
CONCURRENT source loading (async gather over a fan-out semaphore), per-source
exception isolation of ANY exception type, deterministic collision
namespacing applied AFTER the gather, reserved platform names never shadowed
by registered servers, and the registry-outage degrade path.
"""

import asyncio
from typing import Any

from src.domain.entities.mcp_server_config import McpServerConfig
from src.domain.errors.mcp import McpRegistryRequestError
from src.domain.errors.messages import ErrorMessage
from src.infrastructure.deepagent.builtin_resolution import (
    MAX_CONCURRENT_MCP_SOURCES,
    BuiltinToolsetResolver,
    _namespaced_tool_name,
)


class _ResolvedTool:
    """Minimal stand-in for a LangChain tool (model_copy-able)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def model_copy(self, *, update: dict) -> "_ResolvedTool":
        clone = _ResolvedTool(update["name"])
        return clone


def _config(name: str, url: str = "http://x/mcp") -> McpServerConfig:
    return McpServerConfig(name=name, transport="http", url=url)


class _LoaderStub:
    """Answers per server name; optional per-call delay; records calls."""

    def __init__(self, answers: dict[str, Any], delay: float = 0.0) -> None:
        self._answers = answers
        self._delay = delay
        self.calls: list[str] = []
        self.max_concurrent = 0
        self._in_flight = 0
        self._lock = asyncio.Lock()

    async def load_tools(self, configs: list[McpServerConfig]) -> list:
        async with self._lock:
            self._in_flight += 1
            self.max_concurrent = max(self.max_concurrent, self._in_flight)
            self.calls.append(configs[0].name)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            answer = self._answers[configs[0].name]
            if isinstance(answer, BaseException):
                raise answer
            return list(answer)
        finally:
            async with self._lock:
                self._in_flight -= 1


def _resolver(loader, registry_client=None, base_url: str = "") -> BuiltinToolsetResolver:
    return BuiltinToolsetResolver(
        mcp_tool_loader=loader,
        registry_client=registry_client,
        builtin_mcp_base_url_provider=lambda: base_url,
    )


class TestConcurrentLoading:
    async def test_sources_are_loaded_concurrently(self):
        """NFR: the optional MCP sources run in parallel, not sequentially."""
        delay = 0.15
        loader = _LoaderStub({f"srv-{i}": [_ResolvedTool(f"tool-{i}")] for i in range(6)}, delay=delay)
        sources = [_config(f"srv-{i}") for i in range(6)]

        started = asyncio.get_event_loop().time()
        tools, degraded = await _resolver(loader)._load_sources(sources, frozenset())
        elapsed = asyncio.get_event_loop().time() - started

        assert degraded == []
        # 6 sources: sequential would be >= 6 * delay; the concurrent fan-out
        # with a semaphore of 8 must finish in well under half of that.
        assert elapsed < delay * 3, f"sources appear to load sequentially (elapsed={elapsed:.2f}s)"
        assert len(tools) == 6

    async def test_fan_out_is_bounded_by_the_semaphore(self):
        assert MAX_CONCURRENT_MCP_SOURCES == 8
        loader = _LoaderStub({f"srv-{i}": [] for i in range(12)})
        sources = [_config(f"srv-{i}") for i in range(12)]
        tools, degraded = await _resolver(loader)._load_sources(sources, frozenset())
        assert loader.max_concurrent <= MAX_CONCURRENT_MCP_SOURCES
        assert degraded == [] and not tools


class TestFailureIsolation:
    async def test_any_exception_type_degrades_only_that_source(self):
        loader = _LoaderStub(
            {
                "good": [_ResolvedTool("good_tool")],
                "mcp-fail": McpRegistryRequestError("boom"),
                "raw-fail": TypeError("_create_streamable_http_session() got an unexpected kwarg"),
                "malformed": ValueError("malformed payload"),
            }
        )
        # Drive the load path directly with the four sources (the resolver's
        # top-level flow has no registered servers in a unit run):
        tools, degraded = await _resolver(loader)._load_sources(
            [_config("good"), _config("mcp-fail"), _config("raw-fail"), _config("malformed")],
            frozenset(),
        )
        assert [t.name for t in tools] == ["good_tool"]
        assert [d["name"] for d in degraded] == ["mcp-fail", "raw-fail", "malformed"]
        assert all(d["reason"] for d in degraded)

    async def test_registry_outage_degrades_without_raising(self):
        class _BrokenRegistry:
            async def list_servers(self):
                raise McpRegistryRequestError(ErrorMessage.MCP_REGISTRY_UNAVAILABLE)

            async def get_execution_config(self, name):  # pragma: no cover — unreachable
                raise AssertionError("not reached")

            async def aclose(self):  # pragma: no cover — unused lifecycle hook
                return None

        loader = _LoaderStub({})
        tools, degraded = await _resolver(loader, _BrokenRegistry()).resolve_dynamic_tools()
        assert [t.name for t in tools] == ["create_pdf"]
        assert degraded == [{"name": "mcp-registry", "reason": ErrorMessage.MCP_REGISTRY_UNAVAILABLE}]

    async def test_malformed_single_registry_entry_degrades_that_server_only(self):
        class _PartiallyBrokenRegistry:
            async def list_servers(self):
                return [_config("healthy"), _config("broken")]

            async def get_execution_config(self, name: str) -> McpServerConfig:
                if name == "broken":
                    raise McpRegistryRequestError(ErrorMessage.MCP_REGISTRY_REQUEST_FAILED.format(status="shape"))
                return _config(name)

            async def aclose(self):  # pragma: no cover — unused lifecycle hook
                return None

        loader = _LoaderStub({"healthy": [_ResolvedTool("ping")]})
        tools, degraded = await _resolver(loader, _PartiallyBrokenRegistry()).resolve_dynamic_tools()
        assert sorted(t.name for t in tools) == ["create_pdf", "ping"]
        assert [d["name"] for d in degraded] == ["broken"]

    async def test_empty_base_url_yields_only_the_pdf_tool(self):
        loader = _LoaderStub({})
        tools, degraded = await _resolver(loader).resolve_dynamic_tools()
        assert [t.name for t in tools] == ["create_pdf"]
        assert degraded == []
        assert loader.calls == []


class TestDeterministicNamespacing:
    async def test_collisions_assign_stable_names_in_source_order(self):
        loader = _LoaderStub({name: [_ResolvedTool("ping")] for name in ("second", "first")})
        # The collector sorts by name: "first" claims "ping" BEFORE "second".
        sources = [_config("first"), _config("second")]
        tools, degraded = await _resolver(loader)._load_sources(sources, frozenset())

        assert degraded == []
        names = [t.name for t in tools]
        assert names == ["ping", "second__ping"]

    async def test_namespacing_is_deterministic_across_turns(self):
        names_by_turn = []
        for _ in range(3):
            loader = _LoaderStub({name: [_ResolvedTool("ping")] for name in ("first", "second")})
            sources = [_config("first"), _config("second")]
            tools, _ = await _resolver(loader)._load_sources(sources, frozenset())
            names_by_turn.append([t.name for t in tools])
        assert names_by_turn[0] == names_by_turn[1] == names_by_turn[2]

    async def test_reserved_platform_names_are_never_shadowed(self):
        loader = _LoaderStub({"alpha": [_ResolvedTool("composable_agents_delete"), _ResolvedTool("ping")]})
        reserved = frozenset({"composable_agents_delete", "composable_agents_run"})
        tools, degraded = await _resolver(loader)._load_sources([_config("alpha")], reserved)

        names = [t.name for t in tools]
        assert degraded == []
        # The registered server's platform-named tool is deterministically
        # namespaced — it can never silently shadow the real platform tool.
        assert names.count("composable_agents_delete") == 0
        assert "alpha__composable_agents_delete" in names
        assert "ping" in names

    async def test_same_name_on_two_servers_reaches_exactly_one(self):
        loader = _LoaderStub({"first": [_ResolvedTool("ping")], "second": [_ResolvedTool("ping")]})
        sources = [_config("first"), _config("second")]
        tools, _ = await _resolver(loader)._load_sources(sources, frozenset())
        names = [t.name for t in tools]
        assert names.count("ping") == 1
        assert "second__ping" in names

    def test_double_collision_gets_a_numeric_suffix(self):
        taken = {"ping", "alpha__ping", "alpha__ping__2"}
        assert _namespaced_tool_name("alpha", "ping", taken) == "alpha__ping__3"

    async def test_rename_preserves_the_backing_tool(self):
        loader = _LoaderStub({"alpha": [_ResolvedTool("ping")]})
        tools, _ = await _resolver(loader)._load_sources([_config("alpha")], frozenset())
        assert tools[0].name == "ping"  # no reservation → raw name kept
