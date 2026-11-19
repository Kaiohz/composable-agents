"""Toolset guard middleware for the built-in Composable AI Assistant.

REQ-020/REQ-030: settings, API-key, registry-reveal and self-invocation
capabilities are NEVER passed to the built-in assistant. When the model
nevertheless attempts to call a capability that is not part of its resolved
toolset, the call must fail SERVER-SIDE with an explicit refusal instead of
silently resolving.

The middleware hooks LangChain's ``awrap_tool_call`` (wired by
``create_agent(middleware=[...])`` through deepagents): every tool execution
passes through it. A call whose tool is unknown to the graph's ToolNode
(``request.tool is None``) is refused with a stable, actionable message; the
ToolNode converts the raised error into an error ToolMessage the model sees.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langgraph.prebuilt.tool_node import ToolMessage
from langgraph.types import Command

from src.domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)

#: Refusal wording — stable and actionable; includes the word "Excluded" so
#: downstream surfaces (tool results, one-shot traces) can recognize it.
_EXCLUDED_MESSAGE = (
    "Excluded capability '{name}' requested — this capability is not available to the "
    "built-in assistant (settings, credentials and self-invocation are excluded)."
)


class BuiltInToolsetGuardMiddleware(AgentMiddleware):
    """Refuse tool calls outside the resolved built-in assistant toolset."""

    name = "builtin_toolset_guard"

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command[Any]:
        """Refuse unknown capabilities, otherwise delegate transparently.

        Args:
            request: The pending tool call (``request.tool`` is ``None`` when
                the graph has no tool bound under the requested name).
            handler: The continuation executing the tool.

        Returns:
            The handler's result for allowed capabilities.
        """
        tool_name = str((request.tool_call or {}).get("name", ""))
        if request.tool is None:
            logger.warning(LogMessage.BUILTIN_AGENT_OPTIONAL_TOOL_FAILED, tool_name, "not in the resolved toolset")
            raise ValueError(_EXCLUDED_MESSAGE.format(name=tool_name))
        return await handler(request)
