"""Base contract for reranker providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any


class BaseReranker(ABC):
    """Abstract interface for reranking candidates."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        trace: "TraceContext | None" = None,
    ) -> list[Mapping[str, Any]]:
        """Rerank candidates and return ordered results."""


class NoneReranker(BaseReranker):
    """No-op reranker that preserves input ordering."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        trace: "TraceContext | None" = None,
    ) -> list[Mapping[str, Any]]:
        return list(candidates)
