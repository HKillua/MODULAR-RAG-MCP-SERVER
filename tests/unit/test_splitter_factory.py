"""Unit tests for splitter factory routing."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.fixed_length_splitter import FixedLengthSplitter
from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.semantic_splitter import SemanticSplitter
from libs.splitter.splitter_factory import SplitterFactory


class FakeSplitter(BaseSplitter):
    """Simple fake splitter for factory tests."""

    def split_text(
        self,
        text: str,
        trace: "TraceContext | None" = None,
    ) -> list[str]:
        return [text]


@dataclass(frozen=True)
class _SplitterSection:
    provider: str


@dataclass(frozen=True)
class _Settings:
    splitter: _SplitterSection


@pytest.fixture(autouse=True)
def _register_fake_provider() -> None:
    SplitterFactory.register_provider("fake", FakeSplitter)
    yield
    SplitterFactory.unregister_provider("fake")


@pytest.mark.unit
def test_create_from_splitter_section_mapping_routes_to_provider() -> None:
    splitter = SplitterFactory.create({"provider": "fake"})
    assert isinstance(splitter, FakeSplitter)


@pytest.mark.unit
def test_create_from_top_level_settings_routes_to_provider() -> None:
    settings = _Settings(splitter=_SplitterSection(provider="fake"))
    splitter = SplitterFactory.create(settings)
    assert isinstance(splitter, FakeSplitter)


@pytest.mark.unit
def test_fake_splitter_returns_single_chunk() -> None:
    splitter = SplitterFactory.create({"provider": "fake"})
    assert splitter.split_text("hello") == ["hello"]


@pytest.mark.unit
def test_create_unknown_provider_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported splitter provider: missing"):
        SplitterFactory.create({"provider": "missing"})


@pytest.mark.unit
def test_create_recursive_uses_builtin_provider() -> None:
    splitter = SplitterFactory.create({"provider": "recursive"})
    assert isinstance(splitter, RecursiveSplitter)


@pytest.mark.unit
def test_create_semantic_uses_builtin_provider() -> None:
    splitter = SplitterFactory.create({"provider": "semantic"})
    assert isinstance(splitter, SemanticSplitter)


@pytest.mark.unit
def test_create_fixed_uses_builtin_provider() -> None:
    splitter = SplitterFactory.create({"provider": "fixed"})
    assert isinstance(splitter, FixedLengthSplitter)
