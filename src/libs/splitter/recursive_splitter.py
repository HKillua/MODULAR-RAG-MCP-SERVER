"""Recursive splitter placeholder implementation."""

from __future__ import annotations

from libs.splitter.base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """Recursive splitter placeholder.

    Concrete LangChain integration is deferred to phase B7.5.
    """

    def split_text(
        self,
        text: str,
        trace: "TraceContext | None" = None,
    ) -> list[str]:
        raise NotImplementedError("Recursive splitter is not implemented yet")
