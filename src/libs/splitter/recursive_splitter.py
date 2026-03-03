"""Recursive splitter implementation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from libs.splitter.base_splitter import BaseSplitter

_CODE_BLOCK_PATTERN = re.compile(r"```.*?```", re.DOTALL)


class RecursiveSplitter(BaseSplitter):
    """Recursive splitter with markdown-aware heuristics.

    Uses LangChain's RecursiveCharacterTextSplitter if available,
    otherwise falls back to a minimal recursive splitter.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self._chunk_size = int(self.config.get("chunk_size", 800))
        self._chunk_overlap = int(self.config.get("chunk_overlap", 100))
        self._separators = list(
            self.config.get(
                "separators",
                ["\n\n", "\n", " ", ""],
            )
        )

    def split_text(
        self,
        text: str,
        trace: "TraceContext | None" = None,
    ) -> list[str]:
        if not isinstance(text, str):
            raise ValueError("recursive splitter expects text string")
        if not text:
            return []

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                separators=self._separators,
            )
            return splitter.split_text(text)
        except Exception:
            return _fallback_split(
                text=text,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                separators=self._separators,
            )


def _fallback_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str],
) -> list[str]:
    sections = _split_on_headings(text)
    chunks: list[str] = []
    for section in sections:
        for segment, is_code in _split_preserving_code_blocks(section):
            if is_code:
                chunks.append(segment)
            else:
                chunks.extend(_recursive_split(segment, chunk_size, separators))
    return _apply_overlap(chunks, chunk_overlap)


def _split_on_headings(text: str) -> list[str]:
    sections: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#") and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))
    return sections


def _split_preserving_code_blocks(text: str) -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    last_idx = 0
    for match in _CODE_BLOCK_PATTERN.finditer(text):
        if match.start() > last_idx:
            segments.append((text[last_idx : match.start()], False))
        segments.append((match.group(0), True))
        last_idx = match.end()
    if last_idx < len(text):
        segments.append((text[last_idx:], False))
    return segments


def _recursive_split(text: str, chunk_size: int, separators: Sequence[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        return _split_fixed(text, chunk_size)

    separator = separators[0]
    if separator and separator in text:
        parts = text.split(separator)
    else:
        parts = [text]

    if len(parts) == 1 and len(separators) > 1:
        return _recursive_split(text, chunk_size, separators[1:])

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else current + separator + part
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = part
        else:
            chunks.extend(_recursive_split(part, chunk_size, separators[1:]))
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _split_fixed(text: str, chunk_size: int) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _apply_overlap(chunks: list[str], chunk_overlap: int) -> list[str]:
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks
    adjusted: list[str] = [chunks[0]]
    for idx in range(1, len(chunks)):
        prefix = chunks[idx - 1][-chunk_overlap:]
        adjusted.append(prefix + chunks[idx])
    return adjusted
