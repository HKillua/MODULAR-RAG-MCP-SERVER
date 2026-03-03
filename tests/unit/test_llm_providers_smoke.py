"""Smoke tests for OpenAI-compatible LLM providers."""

from __future__ import annotations

import pytest

from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import OpenAILLM
from libs.llm.azure_llm import AzureLLM
from libs.llm.deepseek_llm import DeepSeekLLM


def _fake_http_post(url: str, headers: dict[str, str], payload: dict, timeout: float):
    return {"choices": [{"message": {"content": "ok"}}]}


@pytest.mark.unit
def test_factory_routes_openai_provider() -> None:
    llm = LLMFactory.create(
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test",
            "http_post": _fake_http_post,
        }
    )
    assert isinstance(llm, OpenAILLM)
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok"


@pytest.mark.unit
def test_factory_routes_deepseek_provider() -> None:
    llm = LLMFactory.create(
        {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "api_key": "test",
            "http_post": _fake_http_post,
        }
    )
    assert isinstance(llm, DeepSeekLLM)
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok"


@pytest.mark.unit
def test_factory_routes_azure_provider() -> None:
    llm = LLMFactory.create(
        {
            "provider": "azure",
            "endpoint": "https://example.openai.azure.com",
            "deployment_name": "gpt-4o-mini",
            "api_key": "test",
            "http_post": _fake_http_post,
        }
    )
    assert isinstance(llm, AzureLLM)
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok"


@pytest.mark.unit
def test_openai_missing_model_raises() -> None:
    llm = LLMFactory.create({"provider": "openai", "api_key": "test", "http_post": _fake_http_post})
    with pytest.raises(ValueError, match="openai missing model"):
        llm.chat([{"role": "user", "content": "hi"}])


@pytest.mark.unit
def test_invalid_messages_raise_value_error() -> None:
    llm = LLMFactory.create(
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test",
            "http_post": _fake_http_post,
        }
    )
    with pytest.raises(ValueError, match="openai invalid messages"):
        llm.chat(["invalid"])  # type: ignore[list-item]
