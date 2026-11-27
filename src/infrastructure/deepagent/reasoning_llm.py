"""OpenAI-compatible chat model that PRESERVES provider reasoning deltas.

langchain-openai 1.x drops the OpenRouter/GLM ``delta.reasoning`` field (and
the DeepSeek legacy ``delta.reasoning_content``) on the floor: the thinking
never reaches the chat UI even when reasoning is ENABLED in the request
payload. The override copies those fields into
``additional_kwargs["reasoning_content"]`` so :class:`DeepAgentRunner`'s
thinking classifier streams them as THINKING trace events (chat spec —
``reasoning_details`` is intentionally ignored: rich/encrypted payloads).
"""

import logging
from typing import Any

from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

#: Key the classifier reads (DeepSeek-style legacy field — our canonical
#: additional_kwargs carrier for thinking text).
REASONING_CONTENT_KEY = "reasoning_content"


class ReasoningAwareChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` that surfaces provider reasoning deltas in chunk metadata.

    Otherwise identical: streaming, ``stream_usage``, tools, structured
    output and the whole OpenAI-compatible surface are inherited unchanged.
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation is None:
            return None
        choices = chunk.get("choices") or []
        delta: dict[str, Any] = (choices[0].get("delta") if choices else None) or {}
        reasoning = delta.get("reasoning") or delta.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            message = generation.message
            message.additional_kwargs = {
                **(message.additional_kwargs or {}),
                REASONING_CONTENT_KEY: reasoning,
            }
            return ChatGenerationChunk(
                message=message, generation_info=generation.generation_info
            )
        return generation
