"""Semantic splitter placeholder implementation."""

from __future__ import annotations

from libs.splitter.base_splitter import BaseSplitter


class SemanticSplitter(BaseSplitter):
    """Semantic splitter placeholder.中文解释：语义分割器占位符实现。"""

    def split_text(
        self,
        text: str,
        trace: "TraceContext | None" = None,
    ) -> list[str]:
        raise NotImplementedError("Semantic splitter is not implemented yet")
