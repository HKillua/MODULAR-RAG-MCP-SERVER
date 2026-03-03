"""Base contract for evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any


class BaseEvaluator(ABC):
    """Abstract interface for evaluation backends."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: Sequence[str],
        golden_ids: Sequence[str],
    ) -> dict[str, float]:
        """Evaluate retrieval results and return metric scores."""
