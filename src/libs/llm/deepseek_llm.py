"""DeepSeek OpenAI-compatible LLM implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from libs.llm.base_llm import BaseLLM
from libs.llm.openai_llm import _extract_content, _maybe_add, _post_json, _validate_messages

HttpPost = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]


class DeepSeekLLM(BaseLLM):
    """DeepSeek provider using OpenAI-compatible API."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self._provider = "deepseek"
        self._http_post: HttpPost | None = self.config.get("http_post")

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Generate a response using DeepSeek chat API."""
        _validate_messages(messages, self._provider)

        api_key = self.config.get("api_key")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError(f"{self._provider} missing api_key")

        model = self.config.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(f"{self._provider} missing model")

        base_url = self.config.get("base_url") or self.config.get("api_base")
        if not isinstance(base_url, str) or not base_url:
            base_url = "https://api.deepseek.com/v1"

        url = base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {"model": model, "messages": list(messages)}
        _maybe_add(payload, self.config, ["temperature", "max_tokens"])

        response = _post_json(url, headers, payload, self._provider, self._http_post)
        return _extract_content(response, self._provider)
