"""Base contract for vector store providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class BaseVectorStore(ABC):
    """Abstract interface for vector storage backends."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    @abstractmethod
    def upsert(
        self,
        records: list[Mapping[str, Any]],
        trace: "TraceContext | None" = None,
    ) -> int:
        """Upsert vector records and return count.

        Each record should contain at least:
        - id: str
        - vector: list[float]
        Optional fields:
        - payload: dict[str, Any]
        """

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: "TraceContext | None" = None,
    ) -> list[dict[str, Any]]:
        """Query vectors and return ranked results.

        Each result should contain at least:
        - id: str
        - score: float
        Optional fields:
        - payload: dict[str, Any]
        """
