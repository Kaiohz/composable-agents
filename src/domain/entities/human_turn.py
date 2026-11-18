"""Human turn input carrying multimodal content blocks (REQ-018).

Without attachments the runner keeps accepting the plain ``str`` message and
builds the exact legacy graph input (REQ-017). When attachments were prepared,
the use case passes a :class:`HumanTurnContent` whose ``content_blocks`` is the
standard LangChain block list (text first, then image blocks, then document
reference text blocks).
"""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.domain.entities.prepared_chat_turn import PreparedChatTurn


class HumanTurnContent(BaseModel, frozen=True):
    """A human turn expressed as standard content blocks.

    Attributes:
        text: The user text (``None`` for attachments-only turns).
        content_blocks: Standard LangChain content blocks (REQ-012 —
            ``{"type": "image", "base64": ..., "mime_type": ...}`` for vision
            payload; plain text blocks for the user message and the document
            read_file references).
        attachments: Projection-safe attachment metadata entries persisted on
            the HUMAN_MESSAGE trace event (REQ-020, no base64 REQ-021).
    """

    text: str | None = None
    content_blocks: list[dict[str, Any]] = Field(default_factory=list)
    attachments: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_prepared(cls, prepared: "PreparedChatTurn") -> "HumanTurnContent":
        """Build the runner input from a prepared attachment turn (REQ-018).

        Single shared conversion so both chat entry points (invoke and stream)
        cannot drift.

        Args:
            prepared: The prepared attachment turn.

        Returns:
            The runner content-block input.
        """
        return cls(
            text=prepared.text,
            content_blocks=prepared.content_blocks or [],
            attachments=prepared.attachments_metadata,
        )
