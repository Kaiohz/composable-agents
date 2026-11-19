from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.domain.entities.hitl_decision import HitlDecision


class ChatAttachmentIn(BaseModel):
    """One attachment payload (REQ-001 shape: filename/content_type/base64).

    This model enforces only the STRUCTURAL contract; the strict business
    rules (strict base64 decode, MIME coherence, size limits, traversal
    guards) are enforced by the domain attachment validator BEFORE any agent
    invocation (REQ-006..008, NFR-010).
    """

    filename: str = Field(..., min_length=1)
    content_type: str = Field(..., min_length=1)
    content_base64: str = Field(..., min_length=1)


def _reject_mixed_payloads(has_message: bool, has_hitl: bool, has_decisions: bool, has_attachments: bool) -> None:
    """Reject mutually exclusive payload shapes (REQ-005, unchanged wording)."""
    if has_attachments and (has_hitl or has_decisions):
        raise ValueError("Attachments are only accepted with a 'message', not with HITL fields.")
    if has_message and (has_hitl or has_decisions):
        raise ValueError("Provide either 'message' or HITL fields (tool_call_id + action / decisions), not both.")
    if has_decisions and has_hitl:
        raise ValueError("'decisions' is mutually exclusive with legacy 'tool_call_id' + 'action'.")


def _reject_empty_hitl(decisions: list[HitlDecision] | None) -> None:
    """Reject a non-None but empty decisions list."""
    if decisions is not None and len(decisions) == 0:
        raise ValueError("'decisions' must be a non-empty list.")


def _reject_missing_hitl_details(hitl_requested: bool, action: str | None, edits: dict | None) -> None:
    """Complete the legacy HITL contract: action and edit payloads."""
    if hitl_requested and action is None:
        raise ValueError("'action' is required for HITL decisions.")
    if action == "edit" and edits is None:
        raise ValueError("'edits' is required for action 'edit'.")


class ChatRequest(BaseModel):
    """Request body for sending a chat message or an HITL decision.

    Exactly one of the following must be provided:
        * ``message`` — a human message (optionally with ``attachments``), or
        * ``decisions`` — a non-empty list of HITL decisions (new contract), or
        * ``tool_call_id`` + ``action`` — a single legacy HITL decision.

    REQ-004: ``message`` may be absent ONLY when a non-empty ``attachments``
    list is provided. REQ-005 remains untouched (message XOR HITL fields).
    """

    message: str | None = Field(default=None, min_length=1)
    tool_call_id: str | None = Field(default=None, min_length=1)
    action: Literal["approve", "reject", "edit"] | None = None
    reason: str | None = None
    edits: dict | None = None
    decisions: list[HitlDecision] | None = None
    # REQ-001: optional attachments (absent means strictly legacy behaviour).
    attachments: list[ChatAttachmentIn] | None = None

    @model_validator(mode="after")
    def validate_input(self) -> Self:
        has_message = self.message is not None
        has_hitl = self.tool_call_id is not None
        has_decisions = self.decisions is not None
        has_attachments = self.attachments is not None
        _reject_mixed_payloads(has_message, has_hitl, has_decisions, has_attachments)
        if not has_message and not has_decisions and not has_hitl:
            # REQ-004: message-less turns are allowed WITH a non-empty
            # attachments list only; an empty list is not an attachment.
            if has_attachments and len(self.attachments) > 0:  # type: ignore[arg-type]
                return self
            raise ValueError("Provide either 'message', 'decisions', or HITL fields (tool_call_id + action).")
        _reject_empty_hitl(self.decisions)
        _reject_missing_hitl_details(has_hitl, self.action, self.edits)
        return self


class CreateThreadRequest(BaseModel):
    """Request body for creating a new thread.

    SRC-002 spoof prevention: only the existing public creation inputs are
    accepted — ``source`` is NOT client-controllable (extra fields such as a
    supplied ``source`` are rejected outright).
    """

    model_config = ConfigDict(extra="forbid")

    agent_name: str = Field(..., min_length=1)
