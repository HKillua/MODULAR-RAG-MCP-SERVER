"""Smoke tests for OpenAI embedding provider."""

from __future__ import annotations

import pytest

from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import OpenAIEmbedding


def _fake_http_post(url: str, headers: dict[str, str], payload: dict, timeout: float):
    input_items = payload.get("input", [])
    return {
        "data": [
            {"embedding": [float(idx), float(len(str(item)))]} for idx, item in enumerate(input_items)
        ]
    }


@pytest.mark.unit
def test_factory_routes_openai_embedding() -> None:
    embedding = EmbeddingFactory.create(
        {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "test",
            "http_post": _fake_http_post,
        }
    )
    assert isinstance(embedding, OpenAIEmbedding)
    vectors = embedding.embed(["hello", "world"])
    assert vectors == [[0.0, 5.0], [1.0, 5.0]]


@pytest.mark.unit
def test_missing_model_raises_value_error() -> None:
    embedding = EmbeddingFactory.create(
        {"provider": "openai", "api_key": "test", "http_post": _fake_http_post}
    )
    with pytest.raises(ValueError, match="openai missing model"):
        embedding.embed(["hello"])


@pytest.mark.unit
def test_empty_texts_raise_value_error() -> None:
    embedding = EmbeddingFactory.create(
        {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "test",
            "http_post": _fake_http_post,
        }
    )
    with pytest.raises(ValueError, match="openai invalid texts: empty input"):
        embedding.embed([])


@pytest.mark.unit
def test_invalid_texts_raise_value_error() -> None:
    embedding = EmbeddingFactory.create(
        {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "test",
            "http_post": _fake_http_post,
        }
    )
    with pytest.raises(ValueError, match="openai invalid texts"):
        embedding.embed(["ok", 1])  # type: ignore[list-item]
