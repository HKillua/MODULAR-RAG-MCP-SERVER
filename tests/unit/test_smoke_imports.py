"""Smoke tests for core package imports."""

import importlib

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_name",
    ["mcp_server", "core", "ingestion", "libs", "observability"],
)
def test_top_level_packages_importable(module_name: str) -> None:
    """Verify top-level project packages are importable in test runtime."""
    module = importlib.import_module(module_name)
    assert module is not None
