"""LLM-usage series projection shared by the REST route and the MCP tool.

Extracted from the trace route so the MCP ``composable_traces_usage`` tool
builds the exact same response shape (no REST/MCP drift).
"""

from typing import Literal

from src.application.responses.trace_usage import UsageSeries
from src.domain.entities.llm_usage_summary import LlmUsageSummary


def usage_series(summary: LlmUsageSummary, group_by: Literal["agent", "model"] | None) -> UsageSeries:
    """Build the ``series`` DTO for the requested group-by narrowing.

    Args:
        summary: The aggregated usage summary (both breakdowns populated).
        group_by: ``agent`` / ``model`` narrowing, or ``None`` for both.

    Returns:
        A :class:`UsageSeries` carrying only the requested breakdown; the
        unrequested one stays ``None`` (dropped from the serialized payload).
    """
    match group_by:
        case "agent":
            return UsageSeries(by_agent=summary.series_by_agent, by_model=None)
        case "model":
            return UsageSeries(by_agent=None, by_model=summary.series_by_model)
        case _:
            return UsageSeries(by_agent=summary.series_by_agent, by_model=summary.series_by_model)
