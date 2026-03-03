"""Factory for creating splitter providers from configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from libs.splitter.base_splitter import BaseSplitter


class SplitterFactory:
    """Registry-backed factory for splitter providers."""

    _registry: dict[str, type[BaseSplitter]] = {}

    @classmethod
    def register_provider(
        cls,
        provider: str,
        implementation: type[BaseSplitter],
    ) -> None:
        """Register or override a splitter provider implementation."""
        normalized = provider.strip().lower()
        if not normalized:
            raise ValueError("Splitter provider name cannot be empty")
        cls._registry[normalized] = implementation

    @classmethod
    def unregister_provider(cls, provider: str) -> None:
        """Remove a provider from registry if it exists."""
        cls._registry.pop(provider.strip().lower(), None)

    @classmethod
    def create(cls, settings: Any) -> BaseSplitter:
        """Create a splitter instance from either full settings or section data."""
        config = _coerce_splitter_config(settings)
        provider_value = config.get("provider")
        if not isinstance(provider_value, str) or not provider_value.strip():
            raise ValueError("Missing required field: splitter.provider")

        provider = provider_value.strip().lower()
        implementation = cls._registry.get(provider)
        if implementation is None:
            raise ValueError(f"Unsupported splitter provider: {provider}")
        return implementation(config=config)


def _coerce_splitter_config(settings: Any) -> dict[str, Any]:
    """Normalize settings input into plain splitter config mapping."""
    if isinstance(settings, Mapping):
        if "provider" in settings:
            return dict(settings)
        splitter_section = settings.get("splitter")
        if isinstance(splitter_section, Mapping):
            return dict(splitter_section)
        raise ValueError("Missing required field: splitter")

    splitter_section = getattr(settings, "splitter", settings)
    if is_dataclass(splitter_section):
        return asdict(splitter_section)
    if hasattr(splitter_section, "__dict__"):
        return {
            key: value
            for key, value in vars(splitter_section).items()
            if not key.startswith("_")
        }

    raise ValueError("Unsupported settings type for splitter factory")


def _register_builtin_providers() -> None:
    """Register built-in provider classes once at import time."""
    from libs.splitter.fixed_length_splitter import FixedLengthSplitter
    from libs.splitter.recursive_splitter import RecursiveSplitter
    from libs.splitter.semantic_splitter import SemanticSplitter

    builtins = {
        "recursive": RecursiveSplitter,
        "semantic": SemanticSplitter,
        "fixed": FixedLengthSplitter,
    }
    for provider, implementation in builtins.items():
        SplitterFactory._registry.setdefault(provider, implementation)


_register_builtin_providers()
