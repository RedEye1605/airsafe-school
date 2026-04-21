"""
Shared utility helpers for the AirSafe project.

Provides file I/O, directory management, and logging setup used
across all sub-packages.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist.

    Args:
        path: Directory path to ensure.

    Returns:
        The same path (for chaining).
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path, *, indent: int = 2) -> None:
    """Save data to a JSON file.

    Args:
        data: JSON-serialisable object.
        path: Output file path.
        indent: Indentation level (default 2).
    """
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=str)
    logger.debug("Saved JSON: %s", path)


def load_json(path: Path) -> Any:
    """Load data from a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Deserialised Python object.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a consistent format.

    Args:
        level: Logging level (default ``logging.INFO``).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
