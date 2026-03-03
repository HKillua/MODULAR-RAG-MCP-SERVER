"""DeepSeek LLM placeholder implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from libs.llm.base_llm import BaseLLM


class DeepSeekLLM(BaseLLM):
    """DeepSeek provider placeholder.

    The concrete API call implementation is deferred to stage B7.
    """

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        raise NotImplementedError("DeepSeek LLM provider is not implemented yet")
