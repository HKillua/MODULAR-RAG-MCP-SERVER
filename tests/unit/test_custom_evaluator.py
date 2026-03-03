"""Unit tests for CustomEvaluator metrics."""

from __future__ import annotations

import pytest

from libs.evaluator.custom_evaluator import CustomEvaluator


@pytest.mark.unit
def test_evaluator_returns_zero_when_no_golden_ids() -> None:
    evaluator = CustomEvaluator()
    metrics = evaluator.evaluate("q", ["a"], [])
    assert metrics["hit_rate"] == 0.0
    assert metrics["mrr"] == 0.0


@pytest.mark.unit
def test_evaluator_hit_rate_when_match_exists() -> None:
    evaluator = CustomEvaluator()
    metrics = evaluator.evaluate("q", ["a", "b"], ["b"])
    assert metrics["hit_rate"] == 1.0


@pytest.mark.unit
def test_evaluator_mrr_for_first_match() -> None:
    evaluator = CustomEvaluator()
    metrics = evaluator.evaluate("q", ["x", "y", "z"], ["y", "w"])
    assert metrics["mrr"] == pytest.approx(0.5)
