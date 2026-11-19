"""Agent-related domain errors."""

from src.domain.errors.base import DomainError
from src.domain.errors.codes import ErrorCode


class AgentError(DomainError):
    """Agent execution error (LLM failure, graph error)."""

    status_code = ErrorCode.BAD_GATEWAY


class AgentNotFoundError(AgentError):
    """Agent not found."""

    status_code = ErrorCode.NOT_FOUND


class AgentConfigAlreadyExistsError(DomainError):
    """Agent configuration already exists."""

    status_code = ErrorCode.CONFLICT


class AgentConfigReferenceError(DomainError):
    """Agent configuration references a missing/inaccessible prompt or model.

    Raised at save time (refs must resolve for the owner, §9) and when a
    concurrent prompt/model deletion raced the reference insert (SEC-002:
    no dangling reference may be accepted).
    """

    status_code = ErrorCode.UNPROCESSABLE_ENTITY


class BuiltinAgentError(DomainError):
    """Read-only built-in agent definition violation (REQ-034).

    The built-in Composable AI Assistant's definition is centrally managed:
    updating or deleting it is refused (HTTP 403). Its name is reserved for
    the overlay; creating a custom agent with that name raises
    :class:`AgentConfigAlreadyExistsError` instead.
    """

    status_code = ErrorCode.FORBIDDEN


class AgentInvocationRejectedError(AgentError):
    """Agent-to-agent one-shot invocation rejected (REQ-029).

    Direct self-invocation, an invocation cycle or an invocation deeper than
    four nested ``composable_agents_run`` calls is rejected with a safe
    error BEFORE another agent runner is invoked. The message carries the
    stable ``AGENT_INVOCATION_REJECTED`` marker so nested one-shot chains can
    propagate the rejection through the calling agent's turn.
    """

    status_code = ErrorCode.UNPROCESSABLE_ENTITY
