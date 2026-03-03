"""Fixed-length splitter placeholder implementation."""

from __future__ import annotations

from libs.splitter.base_splitter import BaseSplitter


class FixedLengthSplitter(BaseSplitter):
    """Fixed-length splitter placeholder."""

    def split_text(
        self,
        text: str,
        trace: "TraceContext | None" = None,
    ) -> list[str]:
        raise NotImplementedError("Fixed-length splitter is not implemented yet")
