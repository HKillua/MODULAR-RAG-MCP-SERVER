"""Cross-encoder reranker placeholder implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from libs.reranker.base_reranker import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker placeholder."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        trace: "TraceContext | None" = None,
    ) -> list[Mapping[str, Any]]:
        raise NotImplementedError("Cross-encoder reranker is not implemented yet")
