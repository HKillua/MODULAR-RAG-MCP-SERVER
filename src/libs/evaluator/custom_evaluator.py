"""Custom evaluator with lightweight metrics."""

from __future__ import annotations

from collections.abc import Sequence

from libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """Compute basic retrieval metrics like hit_rate and mrr."""

    def evaluate(
        self,
        query: str,
        retrieved_ids: Sequence[str],
        golden_ids: Sequence[str],
    ) -> dict[str, float]:
        if not golden_ids:
            return {"hit_rate": 0.0, "mrr": 0.0}

        golden_set = set(golden_ids)
        hit = any(item_id in golden_set for item_id in retrieved_ids)
        hit_rate = 1.0 if hit else 0.0

        mrr = 0.0
        for idx, item_id in enumerate(retrieved_ids, start=1):
            if item_id in golden_set:
                mrr = 1.0 / idx
                break

        return {"hit_rate": hit_rate, "mrr": mrr}
