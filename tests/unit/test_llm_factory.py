"""Unit tests for LLM factory routing."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import pytest

from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import OpenAILLM


class FakeLLM(BaseLLM):
    """Simple fake provider for factory tests."""

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        return f"ok:{len(messages)}"


@dataclass(frozen=True)
class _LLMSection:
    provider: str
    model: str | None = None


@dataclass(frozen=True)
class _Settings:
    llm: _LLMSection


@pytest.fixture(autouse=True)
def _register_fake_provider() -> None:
    LLMFactory.register_provider("fake", FakeLLM)
    yield
    LLMFactory.unregister_provider("fake")


@pytest.mark.unit
def test_create_from_llm_section_mapping_routes_to_provider() -> None:
    llm = LLMFactory.create({"provider": "fake", "model": "test-model"})
    assert isinstance(llm, FakeLLM)
    assert llm.config["model"] == "test-model"


@pytest.mark.unit
def test_create_from_top_level_settings_routes_to_provider() -> None:
    settings = _Settings(llm=_LLMSection(provider="fake", model="demo"))
    llm = LLMFactory.create(settings)
    assert isinstance(llm, FakeLLM)
    assert llm.chat([{"role": "user", "content": "hello"}]) == "ok:1"


@pytest.mark.unit
def test_create_unknown_provider_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported llm provider: missing"):
        LLMFactory.create({"provider": "missing"})


@pytest.mark.unit
def test_create_openai_uses_builtin_provider() -> None:
    llm = LLMFactory.create({"provider": "openai"})
    assert isinstance(llm, OpenAILLM)
