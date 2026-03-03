"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import Settings, load_settings


def _write_settings(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_load_settings_success() -> None:
    settings = load_settings("config/settings.yaml")
    assert isinstance(settings, Settings)
    assert settings.embedding.provider == "openai"


def test_load_settings_missing_required_field_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    _write_settings(
        config_file,
        """
llm:
  provider: openai
embedding:
  model: text-embedding-3-small
vector_store:
  provider: chroma
retrieval:
  mode: hybrid
  top_k: 5
rerank:
  enabled: false
  provider: none
evaluation:
  enabled: false
  backends: []
observability:
  enabled: true
  trace_file: logs/traces.jsonl
""",
    )

    with pytest.raises(ValueError, match="embedding.provider"):
        load_settings(config_file)


def test_load_settings_rejects_invalid_top_k(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    _write_settings(
        config_file,
        """
llm:
  provider: openai
embedding:
  provider: openai
vector_store:
  provider: chroma
retrieval:
  mode: hybrid
  top_k: 0
rerank:
  enabled: false
  provider: none
evaluation:
  enabled: false
  backends: []
observability:
  enabled: true
  trace_file: logs/traces.jsonl
""",
    )

    with pytest.raises(ValueError, match="retrieval.top_k"):
        load_settings(config_file)
