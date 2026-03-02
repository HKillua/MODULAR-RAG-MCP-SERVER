"""Pytest shared configuration for test runtime."""

from pathlib import Path
import sys

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if SRC_PATH.is_dir() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
