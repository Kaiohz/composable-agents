"""Typed, JSON-safe result DTOs for the Composable MCP tools (REQ-008).

MCP structured tool outputs must be JSON objects (the protocol forbids bare
arrays as ``structuredContent``), so every list-shaped result is wrapped in an
explicit object DTO whose field name mirrors the REST contract. Single-object
results reuse the existing domain entities / application responses directly —
the same fields and masking semantics as REST (REQ-011), never looser MCP-only
shapes.
"""

from pydantic import BaseModel

from src.application.responses.cron import CronResponse, CronRunResponse
from src.application.responses.mcp_threads import ThreadSummary
from src.application.responses.prompt import PromptResponse, PromptVersionSummary
from src.application.responses.trace_usage import TopRunsResponse, TraceUsageResponse
from src.application.responses.traces import TraceListResponse
from src.domain.entities.agent_config_metadata import AgentConfigMetadata
from src.domain.entities.llm_usage_run import LlmUsageRun
from src.domain.entities.model_metadata import ModelMetadata
from src.domain.entities.registered_mcp_server import RegisteredMcpServer
from src.domain.entities.trace_event import TraceEvent


class McpAgentsList(BaseModel):
    """``composable_agents_list`` result (built-in assistant first)."""

    agents: list[AgentConfigMetadata]


class McpRegisteredServersList(BaseModel):
    """``composable_mcp_servers_list`` result — masked entries only (REQ-018)."""

    servers: list[RegisteredMcpServer]


class McpNamesList(BaseModel):
    """Generic name/path listing (skill/memory file paths, usage agent names)."""

    items: list[str]


class McpStoreFile(BaseModel):
    """One store file with its full content."""

    path: str
    content: str


class McpSkillUsage(BaseModel):
    """``composable_skills_usage`` result."""

    agents: list[str]


class McpDeleteResult(BaseModel):
    """Confirmation payload of a destructive delete tool."""

    deleted: str


class McpUpsertResult(BaseModel):
    """Confirmation payload of a store upsert tool."""

    path: str


class McpPromptsList(BaseModel):
    prompts: list[PromptResponse]


class McpPromptVersionsList(BaseModel):
    versions: list[PromptVersionSummary]


class McpModelsList(BaseModel):
    models: list[ModelMetadata]


class McpCronsList(BaseModel):
    crons: list[CronResponse]


class McpCronRunsList(BaseModel):
    runs: list[CronRunResponse]


class McpThreadsList(BaseModel):
    threads: list[ThreadSummary]


class McpTraceEventsList(BaseModel):
    events: list[TraceEvent]


class McpTopRunsList(BaseModel):
    """``composable_traces_top_runs`` result."""

    runs: list[LlmUsageRun]


__all__ = [
    "McpAgentsList",
    "McpCronRunsList",
    "McpCronsList",
    "McpDeleteResult",
    "McpModelsList",
    "McpNamesList",
    "McpPromptVersionsList",
    "McpPromptsList",
    "McpRegisteredServersList",
    "McpSkillUsage",
    "McpStoreFile",
    "McpThreadsList",
    "McpTopRunsList",
    "McpTraceEventsList",
    "McpUpsertResult",
    "ThreadSummary",
    "TopRunsResponse",
    "TraceListResponse",
    "TraceUsageResponse",
]
