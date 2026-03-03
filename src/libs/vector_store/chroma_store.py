"""Chroma vector store implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from libs.vector_store.base_vector_store import BaseVectorStore


class ChromaStore(BaseVectorStore):
    """Chroma vector store with minimal upsert/query support."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self._collection_name = str(self.config.get("collection", "default"))
        self._persist_dir = self.config.get("persist_dir")
        self._collection = None
        self._client = None

    def _ensure_collection(self) -> Any:
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
        except Exception as exc:  # noqa: BLE001 - optional dependency
            raise RuntimeError("chroma backend requires 'chromadb' package") from exc

        if self._persist_dir:
            self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        else:
            self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(self._collection_name)
        return self._collection

    def upsert(
        self,
        records: list[Mapping[str, Any]],
        trace: "TraceContext | None" = None,
    ) -> int:
        if not records:
            return 0
        collection = self._ensure_collection()

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        documents: list[str] = []

        for record in records:
            record_id = record.get("id")
            vector = record.get("vector")
            if not isinstance(record_id, str):
                raise ValueError("chroma record missing id")
            if not isinstance(vector, Sequence):
                raise ValueError("chroma record missing vector")
            ids.append(record_id)
            embeddings.append([float(value) for value in vector])

            payload = record.get("payload", {})
            if payload is None:
                payload = {}
            if not isinstance(payload, Mapping):
                raise ValueError("chroma record payload must be a mapping")
            metadatas.append(dict(payload))
            documents.append(str(payload.get("text", "")))

        collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        return len(ids)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: "TraceContext | None" = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        collection = self._ensure_collection()
        query = collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=dict(filters) if filters else None,
            include=["metadatas", "documents", "distances", "ids"],
        )
        return _format_results(query)


def _format_results(query: Mapping[str, Any]) -> list[dict[str, Any]]:
    ids = query.get("ids") or [[]]
    distances = query.get("distances") or [[]]
    metadatas = query.get("metadatas") or [[]]
    documents = query.get("documents") or [[]]

    result_ids = list(ids[0])
    result_distances = list(distances[0])
    result_metadatas = list(metadatas[0])
    result_documents = list(documents[0])

    results: list[dict[str, Any]] = []
    for idx, record_id in enumerate(result_ids):
        distance = result_distances[idx] if idx < len(result_distances) else None
        score = _distance_to_score(distance)
        payload = result_metadatas[idx] if idx < len(result_metadatas) else {}
        document = result_documents[idx] if idx < len(result_documents) else ""
        if isinstance(payload, Mapping):
            payload = dict(payload)
        else:
            payload = {"meta": payload}
        if "text" not in payload and isinstance(document, str) and document:
            payload["text"] = document
        results.append({"id": str(record_id), "score": score, "payload": payload})
    return results


def _distance_to_score(distance: Optional[float]) -> float:
    if distance is None:
        return 0.0
    return 1.0 / (1.0 + float(distance))
