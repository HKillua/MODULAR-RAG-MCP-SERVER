"""Configuration loading and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LLMSettings:
	"""LLM provider configuration."""

	provider: str
	model: str | None = None


@dataclass(frozen=True)
class EmbeddingSettings:
	"""Embedding provider configuration."""

	provider: str
	model: str | None = None


@dataclass(frozen=True)
class VectorStoreSettings:
	"""Vector store configuration."""

	provider: str
	collection: str | None = None


@dataclass(frozen=True)
class RetrievalSettings:
	"""Retrieval pipeline configuration."""

	mode: str
	top_k: int


@dataclass(frozen=True)
class RerankSettings:
	"""Reranker configuration."""

	enabled: bool
	provider: str


@dataclass(frozen=True)
class EvaluationSettings:
	"""Evaluation configuration."""

	enabled: bool
	backends: list[str]


@dataclass(frozen=True)
class ObservabilitySettings:
	"""Observability configuration."""

	enabled: bool
	trace_file: str


@dataclass(frozen=True)
class Settings:
	"""Top-level application settings."""

	llm: LLMSettings
	embedding: EmbeddingSettings
	vector_store: VectorStoreSettings
	retrieval: RetrievalSettings
	rerank: RerankSettings
	evaluation: EvaluationSettings
	observability: ObservabilitySettings


def _required_str(mapping: dict[str, Any], field_path: str) -> str:
	value = mapping.get(field_path.split(".")[-1])
	if not isinstance(value, str) or not value.strip():
		raise ValueError(f"Missing required field: {field_path}")
	return value


def _required_bool(mapping: dict[str, Any], field_path: str) -> bool:
	value = mapping.get(field_path.split(".")[-1])
	if not isinstance(value, bool):
		raise ValueError(f"Missing required field: {field_path}")
	return value


def _required_int(mapping: dict[str, Any], field_path: str) -> int:
	value = mapping.get(field_path.split(".")[-1])
	if not isinstance(value, int):
		raise ValueError(f"Missing required field: {field_path}")
	return value


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
	section = data.get(name)
	if not isinstance(section, dict):
		raise ValueError(f"Missing required field: {name}")
	return section


def validate_settings(settings: Settings) -> None:
	"""Validate required settings fields.

	Args:
		settings: Parsed settings object.

	Raises:
		ValueError: If any required field is missing.
	"""

	if not settings.llm.provider:
		raise ValueError("Missing required field: llm.provider")
	if not settings.embedding.provider:
		raise ValueError("Missing required field: embedding.provider")
	if not settings.vector_store.provider:
		raise ValueError("Missing required field: vector_store.provider")
	if not settings.retrieval.mode:
		raise ValueError("Missing required field: retrieval.mode")
	if settings.retrieval.top_k <= 0:
		raise ValueError("Invalid field value: retrieval.top_k must be > 0")
	if not settings.rerank.provider:
		raise ValueError("Missing required field: rerank.provider")
	if not settings.observability.trace_file:
		raise ValueError("Missing required field: observability.trace_file")


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
	"""Load and validate settings from yaml file.

	Args:
		path: Path to yaml configuration.

	Returns:
		Parsed and validated settings object.

	Raises:
		FileNotFoundError: If file does not exist.
		ValueError: If yaml structure or required fields are invalid.
	"""

	settings_path = Path(path)
	if not settings_path.is_file():
		raise FileNotFoundError(f"Settings file not found: {settings_path}")

	with settings_path.open("r", encoding="utf-8") as file:
		raw = yaml.safe_load(file) or {}

	if not isinstance(raw, dict):
		raise ValueError("Invalid settings format: root must be a mapping")

	llm_raw = _section(raw, "llm")
	embedding_raw = _section(raw, "embedding")
	vector_store_raw = _section(raw, "vector_store")
	retrieval_raw = _section(raw, "retrieval")
	rerank_raw = _section(raw, "rerank")
	evaluation_raw = _section(raw, "evaluation")
	observability_raw = _section(raw, "observability")

	settings = Settings(
		llm=LLMSettings(
			provider=_required_str(llm_raw, "llm.provider"),
			model=llm_raw.get("model"),
		),
		embedding=EmbeddingSettings(
			provider=_required_str(embedding_raw, "embedding.provider"),
			model=embedding_raw.get("model"),
		),
		vector_store=VectorStoreSettings(
			provider=_required_str(vector_store_raw, "vector_store.provider"),
			collection=vector_store_raw.get("collection"),
		),
		retrieval=RetrievalSettings(
			mode=_required_str(retrieval_raw, "retrieval.mode"),
			top_k=_required_int(retrieval_raw, "retrieval.top_k"),
		),
		rerank=RerankSettings(
			enabled=_required_bool(rerank_raw, "rerank.enabled"),
			provider=_required_str(rerank_raw, "rerank.provider"),
		),
		evaluation=EvaluationSettings(
			enabled=_required_bool(evaluation_raw, "evaluation.enabled"),
			backends=evaluation_raw.get("backends", []),
		),
		observability=ObservabilitySettings(
			enabled=_required_bool(observability_raw, "observability.enabled"),
			trace_file=_required_str(observability_raw, "observability.trace_file"),
		),
	)

	validate_settings(settings)
	return settings
