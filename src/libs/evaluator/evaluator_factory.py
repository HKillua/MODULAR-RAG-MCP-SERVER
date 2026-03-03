"""Factory for creating evaluator providers from configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from libs.evaluator.base_evaluator import BaseEvaluator


class EvaluatorFactory:
    """Registry-backed factory for evaluator providers."""

    _registry: dict[str, type[BaseEvaluator]] = {}

    @classmethod
    def register_provider(
        cls,
        provider: str,
        implementation: type[BaseEvaluator],
    ) -> None:
        """Register or override an evaluator provider implementation."""
        normalized = provider.strip().lower()
        if not normalized:
            raise ValueError("Evaluator provider name cannot be empty")
        cls._registry[normalized] = implementation

    @classmethod
    def unregister_provider(cls, provider: str) -> None:
        """Remove a provider from registry if it exists."""
        cls._registry.pop(provider.strip().lower(), None)

    @classmethod
    def create(cls, settings: Any) -> BaseEvaluator:
        """Create an evaluator instance from full settings or section data."""
        config = _coerce_evaluator_config(settings)
        provider_value = config.get("provider")
        if not isinstance(provider_value, str) or not provider_value.strip():
            raise ValueError("Missing required field: evaluation.provider")

        provider = provider_value.strip().lower()
        implementation = cls._registry.get(provider)
        if implementation is None:
            raise ValueError(f"Unsupported evaluator provider: {provider}")
        return implementation(config=config)


def _coerce_evaluator_config(settings: Any) -> dict[str, Any]:
    """Normalize settings input into plain evaluation config mapping."""
    if isinstance(settings, Mapping):
        if "provider" in settings:
            return dict(settings)
        section = settings.get("evaluation")
        if isinstance(section, Mapping):
            return dict(section)
        raise ValueError("Missing required field: evaluation")

    section = getattr(settings, "evaluation", settings)
    if is_dataclass(section):
        return asdict(section)
    if hasattr(section, "__dict__"):
        return {
            key: value
            for key, value in vars(section).items()
            if not key.startswith("_")
        }

    raise ValueError("Unsupported settings type for evaluator factory")


def _register_builtin_providers() -> None:
    """Register built-in provider classes once at import time."""
    from libs.evaluator.custom_evaluator import CustomEvaluator

    EvaluatorFactory._registry.setdefault("custom", CustomEvaluator)


_register_builtin_providers()
