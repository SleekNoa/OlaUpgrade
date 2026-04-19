# mcp-server/utils/logging_config.py
"""Shared logging setup for the MCP subprocess and related components."""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure root logging once.

    stderr is intentional here: MCP uses stdout for JSON-RPC messages.
    """
    root = logging.getLogger()
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(handler)
    root.setLevel(level)

    # Reduce noisy third-party logs so tool traces stay readable.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
