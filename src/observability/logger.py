"""Logging helper for application bootstrap."""

import logging
import sys


def get_logger(name: str = "modular_rag_mcp") -> logging.Logger:
	"""Create or retrieve a logger configured to stderr.

	Args:
		name: Logger name.

	Returns:
		Configured logger instance.
	"""

	logger = logging.getLogger(name)
	if logger.handlers:
		return logger

	logger.setLevel(logging.INFO)
	handler = logging.StreamHandler(stream=sys.stderr)
	handler.setFormatter(
		logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
	)
	logger.addHandler(handler)
	logger.propagate = False
	return logger
