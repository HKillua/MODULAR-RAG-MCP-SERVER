"""Unit tests for local embedding provider."""

from __future__ import annotations

import pytest

from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.local_embedding import LocalEmbedding


@pytest.mark.unit
def test_factory_routes_local_provider() -> None:
    embedding = EmbeddingFactory.create({"provider": "local", "dimension": 4})
    assert isinstance(embedding, LocalEmbedding)


@pytest.mark.unit
def test_local_embedding_dimension_is_stable() -> None:
    embedding = EmbeddingFactory.create({"provider": "local", "dimension": 4})
    vectors = embedding.embed(["hello", "world"])
    assert len(vectors) == 2
    assert all(len(vector) == 4 for vector in vectors)


@pytest.mark.unit
def test_local_embedding_is_deterministic() -> None:
    embedding = EmbeddingFactory.create({"provider": "local", "dimension": 4})
    first = embedding.embed(["hello"])
    second = embedding.embed(["hello"])
    assert first == second


@pytest.mark.unit
def test_empty_texts_returns_empty_list() -> None:
    embedding = EmbeddingFactory.create({"provider": "local", "dimension": 4})
    assert embedding.embed([]) == []


@pytest.mark.unit
def test_invalid_texts_raise_value_error() -> None:
    embedding = EmbeddingFactory.create({"provider": "local", "dimension": 4})
    with pytest.raises(ValueError, match="local invalid texts"):
        embedding.embed(["ok", 1])  # type: ignore[list-item]
