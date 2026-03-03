"""Contract tests for vector store interface and factory."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.chroma_store import ChromaStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


class FakeVectorStore(BaseVectorStore):
    """Fake vector store to validate contract shape."""

    def upsert(
        self,
        records: list[dict[str, object]],
        trace: "TraceContext | None" = None,
    ) -> int:
        for record in records:
            assert isinstance(record.get("id"), str)
            assert isinstance(record.get("vector"), list)
        return len(records)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
        trace: "TraceContext | None" = None,
    ) -> list[dict[str, object]]:
        return [{"id": "doc-1", "score": 0.5, "payload": {"source": "fake"}}][:top_k]


@dataclass(frozen=True)
class _VectorStoreSection:
    provider: str


@dataclass(frozen=True)
class _Settings:
    vector_store: _VectorStoreSection


@pytest.fixture(autouse=True)
def _register_fake_provider() -> None:
    VectorStoreFactory.register_provider("fake", FakeVectorStore)
    yield
    VectorStoreFactory.unregister_provider("fake")


@pytest.mark.unit
def test_upsert_returns_count() -> None:
    store = FakeVectorStore()
    count = store.upsert([{"id": "doc-1", "vector": [0.1, 0.2]}])
    assert count == 1


@pytest.mark.unit
def test_query_result_shape() -> None:
    store = FakeVectorStore()
    results = store.query([0.1, 0.2], top_k=1)
    assert isinstance(results, list)
    assert isinstance(results[0]["id"], str)
    assert isinstance(results[0]["score"], float)


@pytest.mark.unit
def test_factory_routes_to_fake_provider() -> None:
    store = VectorStoreFactory.create({"provider": "fake"})
    assert isinstance(store, FakeVectorStore)


@pytest.mark.unit
def test_factory_routes_from_settings_object() -> None:
    settings = _Settings(vector_store=_VectorStoreSection(provider="fake"))
    store = VectorStoreFactory.create(settings)
    assert isinstance(store, FakeVectorStore)


@pytest.mark.unit
def test_factory_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported vector_store provider: missing"):
        VectorStoreFactory.create({"provider": "missing"})


@pytest.mark.unit
def test_factory_missing_provider_raises() -> None:
    with pytest.raises(ValueError, match="vector_store.provider"):
        VectorStoreFactory.create({"vector_store": {}})


@pytest.mark.unit
def test_factory_builtin_chroma_provider() -> None:
    store = VectorStoreFactory.create({"provider": "chroma"})
    assert isinstance(store, ChromaStore)
