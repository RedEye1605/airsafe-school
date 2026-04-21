#!/usr/bin/env python3
"""Download DKI Jakarta school registries from Kemendikdasmen portal.

Scrapes SD, SMP, SMA, SMK listings and saves three CSV files to
``data/raw/schools/``.
"""

import logging

from src.data.school_registry import download_all_schools
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Download all school registries."""
    setup_logging()
    counts = download_all_schools()
    for name, n in counts.items():
        logger.info("%s: %s rows", name, f"{n:,}")


if __name__ == "__main__":
    main()
