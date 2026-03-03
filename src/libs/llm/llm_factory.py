"""Factory for creating LLM providers from configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from libs.llm.base_llm import BaseLLM


class LLMFactory:
    """Registry-backed factory for LLM providers."""

    _registry: dict[str, type[BaseLLM]] = {}

    @classmethod
    def register_provider(cls, provider: str, implementation: type[BaseLLM]) -> None:
        """Register or override an LLM provider implementation."""
        normalized = provider.strip().lower()
        if not normalized:
            raise ValueError("LLM provider name cannot be empty")
        cls._registry[normalized] = implementation

    @classmethod
    def unregister_provider(cls, provider: str) -> None:
        """Remove a provider from the registry if it exists."""
        cls._registry.pop(provider.strip().lower(), None)

    @classmethod
    def create(cls, settings: Any) -> BaseLLM:
        """Create an LLM instance from either full settings or llm section."""
        config = _coerce_llm_config(settings)
        provider_value = config.get("provider")
        if not isinstance(provider_value, str) or not provider_value.strip():
            raise ValueError("Missing required field: llm.provider")

        provider = provider_value.strip().lower()
        implementation = cls._registry.get(provider)
        if implementation is None:
            raise ValueError(f"Unsupported llm provider: {provider}")
        return implementation(config=config)


def _coerce_llm_config(settings: Any) -> dict[str, Any]:
    """Normalize different settings shapes into a plain llm config mapping."""
    if isinstance(settings, Mapping):
        if "provider" in settings:
            return dict(settings)
        llm_section = settings.get("llm")
        if isinstance(llm_section, Mapping):
            return dict(llm_section)
        raise ValueError("Missing required field: llm")

    llm_section = getattr(settings, "llm", settings)
    if is_dataclass(llm_section):
        return asdict(llm_section)
    if hasattr(llm_section, "__dict__"):
        return {
            key: value
            for key, value in vars(llm_section).items()
            if not key.startswith("_")
        }

    raise ValueError("Unsupported settings type for llm factory")


def _register_builtin_providers() -> None:
    """Register built-in provider classes once at import time."""
    from libs.llm.azure_llm import AzureLLM
    from libs.llm.deepseek_llm import DeepSeekLLM
    from libs.llm.ollama_llm import OllamaLLM
    from libs.llm.openai_llm import OpenAILLM

    builtins = {
        "openai": OpenAILLM,
        "azure": AzureLLM,
        "deepseek": DeepSeekLLM,
        "ollama": OllamaLLM,
    }
    for provider, implementation in builtins.items():
        LLMFactory._registry.setdefault(provider, implementation)


_register_builtin_providers()
