#!/usr/bin/env python3
"""Project entry point for Modular RAG MCP Server."""

from pathlib import Path


def main() -> None:
    """Run the minimal bootstrap for the project."""
    config_path = Path("config/settings.yaml")
    print(f"Modular RAG MCP Server bootstrap ready. Config: {config_path}")


if __name__ == "__main__":
    main()
