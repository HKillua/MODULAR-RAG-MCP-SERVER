"""OpenAI embedding implementation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from libs.embedding.base_embedding import BaseEmbedding

HttpPost = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding provider using the OpenAI-compatible embeddings API."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self._provider = "openai"
        self._http_post: HttpPost | None = self.config.get("http_post")

    def embed(
        self,
        texts: list[str],
        trace: "TraceContext | None" = None,
    ) -> list[list[float]]:
        _validate_texts(texts, self._provider)

        api_key = self.config.get("api_key")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError(f"{self._provider} missing api_key")

        model = self.config.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(f"{self._provider} missing model")

        base_url = self.config.get("base_url") or self.config.get("api_base")
        if not isinstance(base_url, str) or not base_url:
            base_url = "https://api.openai.com/v1"

        url = base_url.rstrip("/") + "/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {"model": model, "input": list(texts)}
        _maybe_add(payload, self.config, ["dimensions", "encoding_format"])

        response = _post_json(url, headers, payload, self._provider, self._http_post)
        return _extract_embeddings(response, self._provider)


def _validate_texts(texts: Sequence[str], provider: str) -> None:
    if not isinstance(texts, Sequence):
        raise ValueError(f"{provider} invalid texts: must be a sequence")
    if len(texts) == 0:
        raise ValueError(f"{provider} invalid texts: empty input")
    for idx, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(f"{provider} invalid texts[{idx}]")


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


def _extract_embeddings(response: Mapping[str, Any], provider: str) -> list[list[float]]:
    data = response.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError(f"{provider} response missing data")
    embeddings: list[list[float]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise ValueError(f"{provider} response invalid item[{idx}]")
        vector = item.get("embedding")
        if not isinstance(vector, list):
            raise ValueError(f"{provider} response missing embedding[{idx}]")
        embeddings.append([float(value) for value in vector])
    return embeddings
