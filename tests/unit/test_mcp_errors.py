"""Unit tests for the MCP tool error mapping (REQ-009 — C2 remediation).

Table-driven over the ``_ERROR_FAMILIES`` registry: every family member maps
to its stable code, specificity ordering holds (subclasses win over bases),
pydantic validation failures NEVER echo the rejected input, and unexpected
exceptions collapse into a generic message.
"""

import pytest
from pydantic import ValidationError

from src.application.mcp.errors import (
    _CODE_CONFLICT,
    _CODE_INVOCATION,
    _CODE_NOT_FOUND,
    _CODE_READ_ONLY,
    _CODE_REGISTRY,
    _CODE_STORAGE,
    _CODE_UNEXPECTED,
    _CODE_VALIDATION,
    _ERROR_FAMILIES,
    error_detail,
)
from src.domain.errors.agent import (
    AgentConfigAlreadyExistsError,
    AgentError,
    AgentInvocationRejectedError,
    AgentNotFoundError,
    BuiltinAgentError,
)
from src.domain.errors.config import ConfigError, ConfigValidationError
from src.domain.errors.cron import CronNotFoundError
from src.domain.errors.mcp import McpRegistryRequestError
from src.domain.errors.model import ModelNotFoundError
from src.domain.errors.prompt import PromptNotFoundError, PromptVersionNotFoundError
from src.domain.errors.storage import StorageError
from src.domain.errors.store_file import StoreFileNotFoundError
from src.domain.errors.thread import ThreadNotFoundError


def _family_for(exc: BaseException):
    """Find the family entry that would claim ``exc`` (first match wins)."""
    for family in _ERROR_FAMILIES:
        if isinstance(exc, family.exception_type):
            return family
    return None


class TestFamilySpecificity:
    def test_subclasses_precede_their_bases(self):
        """Every except-subtype must map to its OWN rule, not a base's."""
        for family in _ERROR_FAMILIES:
            base = family.exception_type
            for other in _ERROR_FAMILIES:
                sub = other.exception_type
                if sub is not base and issubclass(sub, base):
                    # ``sub`` must appear EARLIER in the tuple than ``base``.
                    assert _ERROR_FAMILIES.index(other) < _ERROR_FAMILIES.index(family), (
                        f"{sub.__name__} (subclass) must be matched before its base {base.__name__}"
                    )

    def test_agent_not_found_wins_over_agent_error(self):
        assert _family_for(AgentNotFoundError("x")).code == _CODE_NOT_FOUND


class TestErrorFamilyMapping:
    @pytest.mark.parametrize(
        ("exc", "expected_code", "expected_detail_fragment"),
        [
            (AgentInvocationRejectedError("Agent invocation rejected: cycle"), _CODE_INVOCATION, "cycle"),
            (AgentConfigAlreadyExistsError("Agent config already exists: a"), _CODE_CONFLICT, "already exists"),
            (BuiltinAgentError("read-only built-in"), _CODE_READ_ONLY, "read-only"),
            (AgentNotFoundError("Agent not found: a"), _CODE_NOT_FOUND, "Agent not found"),
            (AgentError("Agent execution error: graph"), _CODE_NOT_FOUND, "graph"),
            (StoreFileNotFoundError("File not found: /skills/x"), _CODE_NOT_FOUND, "File not found"),
            (ThreadNotFoundError("Thread not found: t"), _CODE_NOT_FOUND, "Thread not found"),
            (PromptNotFoundError("Prompt not found: p"), _CODE_NOT_FOUND, "Prompt not found"),
            (PromptVersionNotFoundError("Prompt version 2 not found"), _CODE_NOT_FOUND, "version"),
            (ModelNotFoundError("Model not found: m"), _CODE_NOT_FOUND, "Model not found"),
            (CronNotFoundError("Cron not found: c"), _CODE_NOT_FOUND, "Cron not found"),
            (ConfigValidationError([{"loc": ("yaml",), "msg": "Invalid mapping"}]), _CODE_VALIDATION, "yaml"),
            (ConfigError("Invalid agent name"), _CODE_VALIDATION, "Invalid agent name"),
            (McpRegistryRequestError("MCP registry request failed with status 502."), _CODE_REGISTRY, "status"),
        ],
    )
    def test_domain_family_maps_to_its_code(self, exc, expected_code, expected_detail_fragment):
        detail = error_detail(exc)
        assert detail.startswith(f"[{expected_code}]")
        assert expected_detail_fragment in detail

    def test_registry_unavailable_uses_the_safe_generic_detail(self):
        from src.domain.errors.mcp import McpRegistryUnavailableError
        from src.domain.errors.messages import ErrorMessage

        detail = error_detail(McpRegistryUnavailableError("raw socket noise"))
        assert detail.startswith(f"[{_CODE_REGISTRY}]")
        assert detail == f"[{_CODE_REGISTRY}] {ErrorMessage.MCP_REGISTRY_UNAVAILABLE}"

    def test_storage_error_is_generic_without_driver_detail(self):
        detail = error_detail(StorageError("Failed to create thread: (asyncpg) select * from secrets"))
        assert detail.startswith(f"[{_CODE_STORAGE}]")
        assert "asyncpg" not in detail
        assert "secrets" not in detail


class TestPydanticRedaction:
    def test_validation_error_lists_locations_but_never_the_rejected_value(self):
        from pydantic import BaseModel, Field

        class Probe(BaseModel):
            api_key: str = Field(..., min_length=40)

        with pytest.raises(ValidationError) as exc_info:
            Probe(api_key="sk-live-short-secret-credential")
        detail = error_detail(exc_info.value)
        assert detail.startswith(f"[{_CODE_VALIDATION}]")
        assert "Invalid tool arguments" in detail
        # The rejected INPUT value never appears:
        assert "sk-live-short-secret-credential" not in detail
        assert "input_value" not in detail

    def test_extra_forbidden_field_name_is_visible_but_values_are_not(self):
        from pydantic import BaseModel, ConfigDict

        class Probe(BaseModel):
            model_config = ConfigDict(extra="forbid")
            name: str = ""

        with pytest.raises(ValidationError) as exc_info:
            Probe.model_validate({"name": "", "headers": {"Authorization": "Bearer super-secret-token"}})
        detail = error_detail(exc_info.value)
        assert "extra_forbidden" in detail or "headers" in detail
        assert "Bearer super-secret-token" not in detail


class TestUnexpectedFallback:
    def test_unknown_exception_collapses_to_generic_message(self):
        detail = error_detail(RuntimeError("boom with credentials sk-abc"))
        assert detail == f"[{_CODE_UNEXPECTED}] Tool failed unexpectedly; check the server logs."

    def test_unmapped_domain_error_falls_back_to_the_validation_code(self):
        """Domain families without a dedicated rule keep their actionable detail."""
        from src.domain.errors.mcp import McpError

        detail = error_detail(McpError("Failed to load MCP tools: connection refused"))
        assert detail.startswith(f"[{_CODE_VALIDATION}]")
        assert "Failed to load MCP tools" in detail
