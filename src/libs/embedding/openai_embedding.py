"""OpenAI embedding placeholder implementation."""

from __future__ import annotations

from libs.embedding.base_embedding import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding provider placeholder.

    Concrete API integration is deferred to phase B7.3.
    """

    def embed(
        self,
        texts: list[str],
        trace: "TraceContext | None" = None,
    ) -> list[list[float]]:
        raise NotImplementedError("OpenAI embedding provider is not implemented yet")
