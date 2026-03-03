"""Unit tests for reranker factory routing."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from libs.reranker.base_reranker import BaseReranker, NoneReranker
from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from libs.reranker.llm_reranker import LLMReranker
from libs.reranker.reranker_factory import RerankerFactory


class FakeReranker(BaseReranker):
    """Fake reranker for factory tests."""

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, object]],
        trace: "TraceContext | None" = None,
    ) -> list[dict[str, object]]:
        return list(candidates)


@dataclass(frozen=True)
class _RerankSection:
    provider: str
    enabled: bool = True


@dataclass(frozen=True)
class _Settings:
    rerank: _RerankSection


@pytest.fixture(autouse=True)
def _register_fake_provider() -> None:
    RerankerFactory.register_provider("fake", FakeReranker)
    yield
    RerankerFactory.unregister_provider("fake")


@pytest.mark.unit
def test_none_reranker_preserves_order() -> None:
    reranker = NoneReranker()
    candidates = [{"id": 1}, {"id": 2}]
    assert reranker.rerank("q", candidates) == candidates


@pytest.mark.unit
def test_factory_routes_to_fake_provider() -> None:
    reranker = RerankerFactory.create({"provider": "fake"})
    assert isinstance(reranker, FakeReranker)


@pytest.mark.unit
def test_factory_routes_from_settings_object() -> None:
    settings = _Settings(rerank=_RerankSection(provider="fake"))
    reranker = RerankerFactory.create(settings)
    assert isinstance(reranker, FakeReranker)


@pytest.mark.unit
def test_factory_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported reranker provider: missing"):
        RerankerFactory.create({"provider": "missing"})


@pytest.mark.unit
def test_factory_missing_provider_raises() -> None:
    with pytest.raises(ValueError, match="reranker.provider"):
        RerankerFactory.create({"rerank": {}})


@pytest.mark.unit
def test_factory_builtin_none_provider() -> None:
    reranker = RerankerFactory.create({"provider": "none"})
    assert isinstance(reranker, NoneReranker)


@pytest.mark.unit
def test_factory_builtin_llm_provider() -> None:
    reranker = RerankerFactory.create({"provider": "llm"})
    assert isinstance(reranker, LLMReranker)


@pytest.mark.unit
def test_factory_builtin_cross_encoder_provider() -> None:
    reranker = RerankerFactory.create({"provider": "cross_encoder"})
    assert isinstance(reranker, CrossEncoderReranker)
