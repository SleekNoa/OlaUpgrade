# utils/logging_config.py
import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure global logging for the application.

    - Consistent format across MCP server, API, and agents
    - Outputs to stdout (important for MCP subprocess visibility)
    - Prevents duplicate handlers
    """

    root_logger = logging.getLogger()

    # Prevent duplicate logs if configure_logging() is called multiple times
    if root_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stderr)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Optional: reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.WARNING)