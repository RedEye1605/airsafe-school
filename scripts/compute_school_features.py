#!/usr/bin/env python3
"""Compute school context features: elevation, road proximity, land use, building density.

Usage:
    python scripts/compute_school_features.py
    python scripts/compute_school_features.py --skip land_use,buildings
    python scripts/compute_school_features.py --force-refresh
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.features.school_features import compute_all_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = project_root / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute school context features")
    parser.add_argument(
        "--skip", default="",
        help="Comma-separated steps to skip: elevation, roads, land_use, buildings",
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Re-download all OSM data (ignore cache)",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Override OSM cache directory",
    )
    args = parser.parse_args()

    skip_steps = set(args.skip.split(",")) if args.skip else set()

    schools_path = DATA_DIR / "processed" / "schools" / "schools_geocoded.csv"
    output_path = DATA_DIR / "processed" / "schools" / "school_features.csv"
    cache_dir = Path(args.cache_dir) if args.cache_dir else DATA_DIR / "processed" / "osm_cache"

    logger.info("=== School Feature Engineering ===")
    logger.info("Schools: %s", schools_path)
    logger.info("Output: %s", output_path)
    if skip_steps:
        logger.info("Skipping: %s", skip_steps)
    if args.force_refresh:
        logger.info("Force refresh: OSM data will be re-downloaded")

    compute_all_features(
        schools_path=schools_path,
        output_path=output_path,
        cache_dir=cache_dir,
        skip_steps=skip_steps,
        force_refresh=args.force_refresh,
    )
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
