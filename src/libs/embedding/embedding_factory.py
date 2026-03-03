"""Factory for creating embedding providers from configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from libs.embedding.base_embedding import BaseEmbedding


class EmbeddingFactory:
    """Registry-backed factory for embedding providers."""

    _registry: dict[str, type[BaseEmbedding]] = {}

    @classmethod
    def register_provider(
        cls,
        provider: str,
        implementation: type[BaseEmbedding],
    ) -> None:
        """Register or override an embedding provider implementation."""
        normalized = provider.strip().lower()
        if not normalized:
            raise ValueError("Embedding provider name cannot be empty")
        cls._registry[normalized] = implementation

    @classmethod
    def unregister_provider(cls, provider: str) -> None:
        """Remove a provider from registry if it exists."""
        cls._registry.pop(provider.strip().lower(), None)

    @classmethod
    def create(cls, settings: Any) -> BaseEmbedding:
        """Create an embedding instance from either full settings or section data."""
        config = _coerce_embedding_config(settings)
        provider_value = config.get("provider")
        if not isinstance(provider_value, str) or not provider_value.strip():
            raise ValueError("Missing required field: embedding.provider")

        provider = provider_value.strip().lower()
        implementation = cls._registry.get(provider)
        if implementation is None:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        return implementation(config=config)


def _coerce_embedding_config(settings: Any) -> dict[str, Any]:
    """Normalize settings input into plain embedding config mapping."""
    if isinstance(settings, Mapping):
        if "provider" in settings:
            return dict(settings)
        embedding_section = settings.get("embedding")
        if isinstance(embedding_section, Mapping):
            return dict(embedding_section)
        raise ValueError("Missing required field: embedding")

    embedding_section = getattr(settings, "embedding", settings)
    if is_dataclass(embedding_section):
        return asdict(embedding_section)
    if hasattr(embedding_section, "__dict__"):
        return {
            key: value
            for key, value in vars(embedding_section).items()
            if not key.startswith("_")
        }

    raise ValueError("Unsupported settings type for embedding factory")


def _register_builtin_providers() -> None:
    """Register built-in provider classes once at import time."""
    from libs.embedding.local_embedding import LocalEmbedding
    from libs.embedding.openai_embedding import OpenAIEmbedding

    builtins = {
        "openai": OpenAIEmbedding,
        "local": LocalEmbedding,
    }
    for provider, implementation in builtins.items():
        EmbeddingFactory._registry.setdefault(provider, implementation)


_register_builtin_providers()
