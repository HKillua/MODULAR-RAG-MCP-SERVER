"""OpenAI-compatible LLM implementation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from libs.llm.base_llm import BaseLLM

HttpPost = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]


class OpenAILLM(BaseLLM):
    """OpenAI provider using the OpenAI-compatible chat API."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self._provider = "openai"
        self._http_post: HttpPost | None = self.config.get("http_post") # 供测试覆盖 http_post 分支

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Generate a response using OpenAI-compatible API."""
        _validate_messages(messages, self._provider) # 验证消息格式是否正确

        api_key = self.config.get("api_key")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError(f"{self._provider} missing api_key")

        model = self.config.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(f"{self._provider} missing model")

        base_url = self.config.get("base_url") or self.config.get("api_base")
        if not isinstance(base_url, str) or not base_url:
            base_url = "https://api.openai.com/v1"

        url = base_url.rstrip("/") + "/chat/completions" # 构建请求 URL，确保没有重复的斜杠
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        } # 构建请求头，使用 Bearer 认证方式，并指定内容类型为 JSON
        payload: dict[str, Any] = {"model": model, "messages": list(messages)} # 构建请求负载，包含模型名称和消息列表
        _maybe_add(payload, self.config, ["temperature", "max_tokens"]) # 可选地添加 temperature 和 max_tokens 参数到负载中

        response = _post_json(url, headers, payload, self._provider, self._http_post) # 发送 HTTP POST 请求并获取响应
        return _extract_content(response, self._provider) # 从响应中提取生成的内容并返回


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


def _maybe_add(payload: dict[str, Any], config: Mapping[str, Any], keys: list[str]) -> None:
    for key in keys:
        value = config.get(key)
        if value is not None:
            payload[key] = value


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
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"{provider} response missing choices")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str):
        raise ValueError(f"{provider} response missing content")
    return content
