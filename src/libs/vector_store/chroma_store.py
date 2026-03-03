"""Chroma vector store placeholder implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from libs.vector_store.base_vector_store import BaseVectorStore


class ChromaStore(BaseVectorStore):
    """Chroma store placeholder.

    Concrete persistence and query logic is deferred to phase B7.6.
    """

    def upsert(
        self,
        records: list[Mapping[str, Any]],
        trace: "TraceContext | None" = None,
    ) -> int:
        raise NotImplementedError("Chroma vector store is not implemented yet")

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: "TraceContext | None" = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("Chroma vector store is not implemented yet")
