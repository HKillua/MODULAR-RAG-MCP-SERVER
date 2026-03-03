"""Unit tests for Ollama LLM provider."""

from __future__ import annotations

import pytest

from libs.llm.llm_factory import LLMFactory
from libs.llm.ollama_llm import OllamaLLM


def _fake_http_post(url: str, headers: dict[str, str], payload: dict, timeout: float):
    return {"message": {"content": "ok"}}


@pytest.mark.unit
def test_factory_routes_ollama_provider() -> None:
    llm = LLMFactory.create(
        {
            "provider": "ollama",
            "model": "llama3",
            "base_url": "http://localhost:11434",
            "http_post": _fake_http_post,
        }
    )
    assert isinstance(llm, OllamaLLM)
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok"


@pytest.mark.unit
def test_missing_model_raises_value_error() -> None:
    llm = LLMFactory.create(
        {"provider": "ollama", "base_url": "http://localhost:11434", "http_post": _fake_http_post}
    )
    with pytest.raises(ValueError, match="ollama missing model"):
        llm.chat([{"role": "user", "content": "hi"}])


@pytest.mark.unit
def test_invalid_messages_raise_value_error() -> None:
    llm = LLMFactory.create(
        {
            "provider": "ollama",
            "model": "llama3",
            "base_url": "http://localhost:11434",
            "http_post": _fake_http_post,
        }
    )
    with pytest.raises(ValueError, match="ollama invalid messages"):
        llm.chat(["invalid"])  # type: ignore[list-item]
