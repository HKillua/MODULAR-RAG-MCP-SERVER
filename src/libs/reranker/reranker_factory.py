"""Factory for creating reranker providers from configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from libs.reranker.base_reranker import BaseReranker, NoneReranker


class RerankerFactory:
    """Registry-backed factory for reranker providers."""

    _registry: dict[str, type[BaseReranker]] = {}

    @classmethod
    def register_provider(
        cls,
        provider: str,
        implementation: type[BaseReranker],
    ) -> None:
        """Register or override a reranker provider implementation."""
        normalized = provider.strip().lower()
        if not normalized:
            raise ValueError("Reranker provider name cannot be empty")
        cls._registry[normalized] = implementation

    @classmethod
    def unregister_provider(cls, provider: str) -> None:
        """Remove a provider from registry if it exists."""
        cls._registry.pop(provider.strip().lower(), None)

    @classmethod
    def create(cls, settings: Any) -> BaseReranker:
        """Create a reranker instance from full settings or section data."""
        config = _coerce_reranker_config(settings)
        provider_value = config.get("provider")
        if not isinstance(provider_value, str) or not provider_value.strip():
            raise ValueError("Missing required field: reranker.provider")

        provider = provider_value.strip().lower()
        implementation = cls._registry.get(provider)
        if implementation is None:
            raise ValueError(f"Unsupported reranker provider: {provider}")
        return implementation(config=config)


def _coerce_reranker_config(settings: Any) -> dict[str, Any]:
    """Normalize settings input into plain reranker config mapping."""
    if isinstance(settings, Mapping):
        if "provider" in settings:
            return dict(settings)
        section = settings.get("rerank")
        if isinstance(section, Mapping):
            return dict(section)
        raise ValueError("Missing required field: rerank")

    section = getattr(settings, "rerank", settings)
    if is_dataclass(section):
        return asdict(section)
    if hasattr(section, "__dict__"):
        return {
            key: value
            for key, value in vars(section).items()
            if not key.startswith("_")
        }

    raise ValueError("Unsupported settings type for reranker factory")


def _register_builtin_providers() -> None:
    """Register built-in provider classes once at import time."""
    from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
    from libs.reranker.llm_reranker import LLMReranker

    builtins = {
        "none": NoneReranker,
        "llm": LLMReranker,
        "cross_encoder": CrossEncoderReranker,
    }
    for provider, implementation in builtins.items():
        RerankerFactory._registry.setdefault(provider, implementation)


_register_builtin_providers()
