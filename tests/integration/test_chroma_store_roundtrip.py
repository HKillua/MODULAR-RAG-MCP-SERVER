"""Integration test for Chroma vector store roundtrip."""

from __future__ import annotations

import pytest

from libs.vector_store.chroma_store import ChromaStore


@pytest.mark.integration
def test_chroma_store_roundtrip(tmp_path) -> None:
    chromadb = pytest.importorskip("chromadb")
    _ = chromadb  # silence unused

    store = ChromaStore({"persist_dir": tmp_path, "collection": "test"})
    records = [
        {"id": "doc-1", "vector": [0.1, 0.2], "payload": {"text": "alpha"}},
        {"id": "doc-2", "vector": [0.2, 0.1], "payload": {"text": "beta"}},
    ]
    assert store.upsert(records) == 2

    results = store.query([0.1, 0.2], top_k=1)
    assert results
    assert results[0]["id"] == "doc-1"
