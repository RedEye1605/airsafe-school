"""Fetch elevation data for school locations via Open-Meteo API.

Uses POST requests to avoid URL length limits, with exponential backoff
on rate limiting (429) and server errors.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_ELEVATION_API = "https://api.open-meteo.com/v1/elevation"
_BATCH_SIZE = 100
_MAX_RETRIES = 5


def fetch_elevations(
    schools_df: pd.DataFrame,
    batch_size: int = _BATCH_SIZE,
) -> pd.Series:
    """Fetch elevation for each school via Open-Meteo Elevation API.

    Args:
        schools_df: DataFrame with latitude, longitude columns.
            Schools with NaN coordinates are skipped.
        batch_size: Number of coordinates per API request.

    Returns:
        Series indexed by original position with elevation_m values.
    """
    valid = schools_df.dropna(subset=["latitude", "longitude"]).copy()
    n_valid = len(valid)
    logger.info("Fetching elevation for %d schools (batch_size=%d)", n_valid, batch_size)

    elevations = pd.Series(np.nan, index=schools_df.index, name="elevation_m")
    if n_valid == 0:
        return elevations

    lats = valid["latitude"].values
    lons = valid["longitude"].values
    orig_idx = valid.index

    for start in range(0, n_valid, batch_size):
        end = min(start + batch_size, n_valid)
        # Use POST to avoid URL length limits with batched coordinates
        payload = {
            "latitude": ",".join(f"{v:.6f}" for v in lats[start:end]),
            "longitude": ",".join(f"{v:.6f}" for v in lons[start:end]),
        }

        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.post(_ELEVATION_API, data=payload, timeout=30)
                if resp.status_code == 429:
                    delay = 10 * (2 ** attempt)
                    logger.warning("Rate limited (attempt %d) — waiting %.0fs", attempt + 1, delay)
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                data = resp.json()
                batch_elev = data.get("elevation", [])
                if len(batch_elev) != end - start:
                    logger.warning("Elevation count mismatch: expected %d, got %d",
                                   end - start, len(batch_elev))
                    batch_elev = batch_elev[:end - start]
                for j, elev in enumerate(batch_elev):
                    elevations.loc[orig_idx[start + j]] = elev
                break
            except Exception as exc:
                if attempt < _MAX_RETRIES - 1:
                    delay = 5 * (2 ** attempt)
                    logger.warning("Elevation API error (attempt %d): %s — retrying in %.0fs",
                                   attempt + 1, exc, delay)
                    time.sleep(delay)
                else:
                    logger.error("Elevation API failed for batch %d-%d: %s", start, end, exc)

        if end < n_valid:
            time.sleep(1.0)

    n_fetched = elevations.notna().sum()
    if n_fetched > 0:
        logger.info("Elevation fetched: %d / %d schools (range: %.1f - %.1f m)",
                     n_fetched, n_valid, elevations.min(), elevations.max())
    else:
        logger.warning("Elevation: 0 / %d schools fetched", n_valid)
    return elevations
