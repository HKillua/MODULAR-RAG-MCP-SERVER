"""Unit tests for evaluator factory routing."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory


class FakeEvaluator(BaseEvaluator):
    """Fake evaluator for factory tests."""

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
    ) -> dict[str, float]:
        return {"hit_rate": 1.0, "mrr": 1.0}


@dataclass(frozen=True)
class _EvaluationSection:
    provider: str
    enabled: bool = True


@dataclass(frozen=True)
class _Settings:
    evaluation: _EvaluationSection


@pytest.fixture(autouse=True)
def _register_fake_provider() -> None:
    EvaluatorFactory.register_provider("fake", FakeEvaluator)
    yield
    EvaluatorFactory.unregister_provider("fake")


@pytest.mark.unit
def test_factory_routes_to_fake_provider() -> None:
    evaluator = EvaluatorFactory.create({"provider": "fake"})
    assert isinstance(evaluator, FakeEvaluator)


@pytest.mark.unit
def test_factory_routes_from_settings_object() -> None:
    settings = _Settings(evaluation=_EvaluationSection(provider="fake"))
    evaluator = EvaluatorFactory.create(settings)
    assert isinstance(evaluator, FakeEvaluator)


@pytest.mark.unit
def test_factory_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported evaluator provider: missing"):
        EvaluatorFactory.create({"provider": "missing"})


@pytest.mark.unit
def test_factory_missing_provider_raises() -> None:
    with pytest.raises(ValueError, match="evaluation.provider"):
        EvaluatorFactory.create({"evaluation": {}})


@pytest.mark.unit
def test_factory_builtin_custom_provider() -> None:
    evaluator = EvaluatorFactory.create({"provider": "custom"})
    assert isinstance(evaluator, CustomEvaluator)
