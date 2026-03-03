"""Factory for creating vector store providers from configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from libs.vector_store.base_vector_store import BaseVectorStore


class VectorStoreFactory:
    """Registry-backed factory for vector store providers."""

    _registry: dict[str, type[BaseVectorStore]] = {}

    @classmethod
    def register_provider(
        cls,
        provider: str,
        implementation: type[BaseVectorStore],
    ) -> None:
        """Register or override a vector store provider implementation."""
        normalized = provider.strip().lower()
        if not normalized:
            raise ValueError("Vector store provider name cannot be empty")
        cls._registry[normalized] = implementation

    @classmethod
    def unregister_provider(cls, provider: str) -> None:
        """Remove a provider from registry if it exists."""
        cls._registry.pop(provider.strip().lower(), None)

    @classmethod
    def create(cls, settings: Any) -> BaseVectorStore:
        """Create a vector store instance from full settings or section data."""
        config = _coerce_vector_store_config(settings)
        provider_value = config.get("provider")
        if not isinstance(provider_value, str) or not provider_value.strip():
            raise ValueError("Missing required field: vector_store.provider")

        provider = provider_value.strip().lower()
        implementation = cls._registry.get(provider)
        if implementation is None:
            raise ValueError(f"Unsupported vector_store provider: {provider}")
        return implementation(config=config)


def _coerce_vector_store_config(settings: Any) -> dict[str, Any]:
    """Normalize settings input into plain vector_store config mapping."""
    if isinstance(settings, Mapping):
        if "provider" in settings:
            return dict(settings)
        section = settings.get("vector_store")
        if isinstance(section, Mapping):
            return dict(section)
        raise ValueError("Missing required field: vector_store")

    section = getattr(settings, "vector_store", settings)
    if is_dataclass(section):
        return asdict(section)
    if hasattr(section, "__dict__"):
        return {
            key: value
            for key, value in vars(section).items()
            if not key.startswith("_")
        }

    raise ValueError("Unsupported settings type for vector_store factory")


def _register_builtin_providers() -> None:
    """Register built-in provider classes once at import time."""
    from libs.vector_store.chroma_store import ChromaStore

    VectorStoreFactory._registry.setdefault("chroma", ChromaStore)


_register_builtin_providers()
