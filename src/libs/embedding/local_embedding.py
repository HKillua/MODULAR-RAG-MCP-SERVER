"""Local embedding placeholder implementation."""

from __future__ import annotations

from libs.embedding.base_embedding import BaseEmbedding


class LocalEmbedding(BaseEmbedding):
    """Local embedding provider placeholder.

    Concrete model integration is deferred to phase B7.4.
    """

    def embed(
        self,
        texts: list[str],
        trace: "TraceContext | None" = None,
    ) -> list[list[float]]:
        raise NotImplementedError("Local embedding provider is not implemented yet")
