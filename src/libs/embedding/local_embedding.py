"""Local embedding implementation."""

from __future__ import annotations

import hashlib

from libs.embedding.base_embedding import BaseEmbedding


class LocalEmbedding(BaseEmbedding):
    """Local embedding provider with deterministic fake vectors."""

    def embed(
        self,
        texts: list[str],
        trace: "TraceContext | None" = None,
    ) -> list[list[float]]:
        if not isinstance(texts, list):
            raise ValueError("local invalid texts: must be a list")
        if not texts:
            return []

        dimension = self._get_dimension()
        vectors: list[list[float]] = []
        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"local invalid texts[{idx}]")
            vectors.append(_hash_to_vector(text, dimension))
        return vectors

    def _get_dimension(self) -> int:
        value = self.config.get("dimension", 8)
        if not isinstance(value, int) or value <= 0:
            raise ValueError("local invalid dimension")
        return value


def _hash_to_vector(text: str, dimension: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    for idx in range(dimension):
        values.append(digest[idx % len(digest)] / 255.0)
    return values
