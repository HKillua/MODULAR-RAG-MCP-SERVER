"""Unit tests for recursive splitter implementation."""

from __future__ import annotations

import pytest

from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.splitter_factory import SplitterFactory


@pytest.mark.unit
def test_factory_routes_recursive_provider() -> None:
    splitter = SplitterFactory.create({"provider": "recursive", "chunk_size": 20})
    assert isinstance(splitter, RecursiveSplitter)


@pytest.mark.unit
def test_recursive_splitter_preserves_code_blocks() -> None:
    text = "# Title\n\nIntro text.\n\n```python\nprint('hi')\n```\n\nMore text."
    splitter = RecursiveSplitter({"chunk_size": 20, "chunk_overlap": 0})
    chunks = splitter.split_text(text)
    code_chunks = [chunk for chunk in chunks if "```python" in chunk]
    assert len(code_chunks) == 1
    assert "print('hi')" in code_chunks[0]
    assert code_chunks[0].count("```") == 2


@pytest.mark.unit
def test_recursive_splitter_keeps_heading_with_section() -> None:
    text = "# Heading\nContent line one.\nContent line two."
    splitter = RecursiveSplitter({"chunk_size": 200, "chunk_overlap": 0})
    chunks = splitter.split_text(text)
    assert len(chunks) == 1
    assert chunks[0].startswith("# Heading")
