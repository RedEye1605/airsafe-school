#!/usr/bin/env python3
"""Geocode DKI Jakarta schools via Nominatim.

Geocodes unique kelurahan centres first, then assigns coordinates
to all schools. Results are saved to ``data/processed/schools/schools_geocoded.csv``.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_DIR,
    NOMINATIM_RATE_LIMIT,
    NOMINATIM_URL,
    NOMINATIM_USER_AGENT,
    PROCESSED_DIR,
    RAW_DIR,
)
from src.data.school_registry import geocode_query
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def _load_cache(cache_path: Path) -> dict:
    """Load geocode cache from disk."""
    if cache_path.exists():
        with open(cache_path, "r") as fh:
            return json.load(fh)
    return {}


def _save_cache(cache: dict, cache_path: Path) -> None:
    """Persist geocode cache to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as fh:
        json.dump(cache, fh, indent=2)


def main() -> None:
    """Run the geocoding pipeline."""
    setup_logging()

    input_dir = Path(RAW_DIR) / "schools"
    output_dir = Path(PROCESSED_DIR) / "schools"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(PROCESSED_DIR) / "geocode_cache.json"

    cache = _load_cache(cache_path)

    # Load all school data
    frames: list[pd.DataFrame] = []
    for name in ("sd_dki.csv", "smp_dki.csv", "sma_smk_dki.csv"):
        path = input_dir / name
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            logger.warning("Skipping %s (not found)", name)

    if not frames:
        logger.error("No school CSV files found in %s", input_dir)
        return

    schools = pd.concat(frames, ignore_index=True)
    logger.info("Total schools: %d", len(schools))

    # Unique kelurahan
    kel_list = schools[["kota_kab", "kecamatan", "kelurahan"]].drop_duplicates()
    logger.info("Unique kelurahan: %d", len(kel_list))

    # Clear stale cache entries
    stale_keys = [k for k, v in cache.items() if v.get("lat") is None]
    for k in stale_keys:
        del cache[k]
    if stale_keys:
        logger.info("Cleared %d stale cache entries", len(stale_keys))

    # Geocode each kelurahan
    kel_coords: dict[str, dict] = {}
    for idx, (_, row) in enumerate(kel_list.iterrows()):
        query = f"{row['kelurahan']}, {row['kecamatan']}, DKI Jakarta, Indonesia"
        result = geocode_query(query, cache)
        key = f"{row['kota_kab']}|{row['kecamatan']}|{row['kelurahan']}"
        kel_coords[key] = result

        if (idx + 1) % 20 == 0:
            found = sum(1 for v in kel_coords.values() if v.get("lat"))
            logger.info("  [%d/%d] Found: %d/%d", idx + 1, len(kel_list), found, idx + 1)
            _save_cache(cache, cache_path)

    _save_cache(cache, cache_path)

    # Assign coordinates to schools
    def _get_coords(row: pd.Series) -> pd.Series:
        key = f"{row['kota_kab']}|{row['kecamatan']}|{row['kelurahan']}"
        c = kel_coords.get(key, {})
        return pd.Series({
            "latitude": c.get("lat"),
            "longitude": c.get("lon"),
            "geocode_display": c.get("display_name", ""),
            "geocode_source": c.get("source", ""),
        })

    coords_df = schools.apply(_get_coords, axis=1)
    schools = pd.concat([schools, coords_df], axis=1)

    output_path = output_dir / "schools_geocoded.csv"
    schools.to_csv(output_path, index=False, encoding="utf-8-sig")

    found = schools["latitude"].notna().sum()
    logger.info(
        "Done: %d/%d geocoded (%.1f%%)", found, len(schools),
        found / len(schools) * 100 if len(schools) else 0,
    )
    logger.info("Saved: %s", output_path)


if __name__ == "__main__":
    main()
