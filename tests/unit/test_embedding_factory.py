"""Unit tests for embedding factory routing."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import OpenAIEmbedding


class FakeEmbedding(BaseEmbedding):
    """Simple fake embedding provider for factory tests."""

    def embed(
        self,
        texts: list[str],
        trace: "TraceContext | None" = None,
    ) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


@dataclass(frozen=True)
class _EmbeddingSection:
    provider: str
    model: str | None = None


@dataclass(frozen=True)
class _Settings:
    embedding: _EmbeddingSection


@pytest.fixture(autouse=True)
def _register_fake_provider() -> None:
    EmbeddingFactory.register_provider("fake", FakeEmbedding)
    yield
    EmbeddingFactory.unregister_provider("fake")


@pytest.mark.unit
def test_create_from_embedding_section_mapping_routes_to_provider() -> None:
    embedding = EmbeddingFactory.create({"provider": "fake", "model": "demo"})
    assert isinstance(embedding, FakeEmbedding)


@pytest.mark.unit
def test_create_from_top_level_settings_routes_to_provider() -> None:
    settings = _Settings(embedding=_EmbeddingSection(provider="fake", model="demo"))
    embedding = EmbeddingFactory.create(settings)
    assert isinstance(embedding, FakeEmbedding)


@pytest.mark.unit
def test_fake_embedding_returns_stable_vectors() -> None:
    embedding = EmbeddingFactory.create({"provider": "fake"})
    vectors = embedding.embed(["a", "abcd"])
    assert vectors == [[1.0], [4.0]]


@pytest.mark.unit
def test_create_unknown_provider_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding provider: missing"):
        EmbeddingFactory.create({"provider": "missing"})


@pytest.mark.unit
def test_create_openai_uses_builtin_provider() -> None:
    embedding = EmbeddingFactory.create({"provider": "openai"})
    assert isinstance(embedding, OpenAIEmbedding)
