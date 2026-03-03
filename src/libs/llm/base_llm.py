"""Base contract for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any


class BaseLLM(ABC):
    """Abstract interface that all LLM providers must implement."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    @abstractmethod
    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Generate text response from a chat message sequence."""
