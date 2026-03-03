"""Ollama LLM implementation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from libs.llm.base_llm import BaseLLM

HttpPost = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]


class OllamaLLM(BaseLLM):
    """Ollama provider using local HTTP endpoint."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self._provider = "ollama"
        self._http_post: HttpPost | None = self.config.get("http_post")

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        _validate_messages(messages, self._provider)

        model = self.config.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(f"{self._provider} missing model")

        base_url = self.config.get("base_url") or "http://localhost:11434"
        if not isinstance(base_url, str) or not base_url:
            raise ValueError(f"{self._provider} missing base_url")

        url = base_url.rstrip("/") + "/api/chat"
        headers = {"Content-Type": "application/json"}
        payload: dict[str, Any] = {"model": model, "messages": list(messages), "stream": False}

        response = _post_json(url, headers, payload, self._provider, self._http_post)
        return _extract_content(response, self._provider)


def _validate_messages(messages: Sequence[Mapping[str, str]], provider: str) -> None:
    if not isinstance(messages, Sequence):
        raise ValueError(f"{provider} invalid messages: must be a sequence")
    for idx, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"{provider} invalid messages[{idx}]: must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"{provider} invalid messages[{idx}].role")
        if not isinstance(content, str):
            raise ValueError(f"{provider} invalid messages[{idx}].content")


def _post_json(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    provider: str,
    http_post: HttpPost | None,
) -> Mapping[str, Any]:
    try:
        if http_post is not None:
            return http_post(url, headers, payload, 30.0)

        data = json.dumps(payload).encode("utf-8")
        request = Request(url, data=data, headers=dict(headers), method="POST")
        with urlopen(request, timeout=30.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(f"{provider} request failed (URLError): {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - keep error context for caller
        raise RuntimeError(
            f"{provider} request failed ({exc.__class__.__name__}): {exc}"
        ) from exc


def _extract_content(response: Mapping[str, Any], provider: str) -> str:
    message = response.get("message")
    if not isinstance(message, Mapping):
        raise ValueError(f"{provider} response missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError(f"{provider} response missing content")
    return content
