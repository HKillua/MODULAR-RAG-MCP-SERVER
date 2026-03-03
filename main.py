#!/usr/bin/env python3
"""Project entry point for Modular RAG MCP Server."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.is_dir() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from core.settings import load_settings
from observability.logger import get_logger


def main() -> None:
    """Run bootstrap and fail fast when configuration is invalid."""
    logger = get_logger("bootstrap")

    try:
        settings = load_settings()
    except (FileNotFoundError, ValueError) as error:
        logger.error("Configuration error: %s", error)
        raise SystemExit(1) from error

    logger.info(
        "Bootstrap ready with llm=%s embedding=%s vector_store=%s",
        settings.llm.provider,
        settings.embedding.provider,
        settings.vector_store.provider,
    )
    print("Modular RAG MCP Server bootstrap ready.")


if __name__ == "__main__":
    main()
