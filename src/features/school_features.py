"""School context features orchestrator.

Computes static spatial features for all schools: elevation, road proximity,
land use composition, and building density. Produces school_features.csv.

Usage:
    from src.features.school_features import compute_all_features
    compute_all_features()
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.features.elevation_features import fetch_elevations
from src.features.osm_features import (
    compute_building_density,
    compute_land_use_features,
    compute_road_distances,
    download_buildings,
    download_land_use,
    download_road_network,
)

logger = logging.getLogger(__name__)

_DEFAULT_SCHOOLS_PATH = Path("data/processed/schools/schools_geocoded.csv")
_DEFAULT_OUTPUT_PATH = Path("data/processed/schools/school_features.csv")
_DEFAULT_CACHE_DIR = Path("data/processed/osm_cache")


def get_feature_columns() -> list[str]:
    return [
        "elevation_m",
        "dist_nearest_road_m", "dist_nearest_primary_road_m",
        "road_density_500m", "nearest_road_type",
        "dominant_land_use_500m",
        "landuse_residential_frac_500m", "landuse_commercial_frac_500m",
        "landuse_industrial_frac_500m", "landuse_green_frac_500m",
        "landuse_water_frac_500m",
        "dominant_land_use_1000m",
        "landuse_residential_frac_1000m", "landuse_commercial_frac_1000m",
        "landuse_industrial_frac_1000m", "landuse_green_frac_1000m",
        "landuse_water_frac_1000m",
        "dist_to_industrial_m",
        "building_count_500m", "building_density_500m",
        "building_count_1000m", "building_density_1000m",
    ]


def compute_all_features(
    schools_path: str | Path = _DEFAULT_SCHOOLS_PATH,
    output_path: str | Path = _DEFAULT_OUTPUT_PATH,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    skip_steps: Optional[set[str]] = None,
    force_refresh: bool = False,
) -> Path:
    """Compute all school context features and save to CSV.

    Args:
        schools_path: Path to geocoded schools CSV.
        output_path: Output CSV path.
        cache_dir: Directory for OSM data cache.
        skip_steps: Set of steps to skip: {'elevation', 'roads', 'land_use', 'buildings'}.
        force_refresh: Re-download all OSM data.

    Returns:
        Path to the output CSV.
    """
    skip_steps = skip_steps or set()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    schools = pd.read_csv(schools_path)
    n_total = len(schools)
    valid_mask = schools["latitude"].notna() & schools["longitude"].notna()
    n_valid = valid_mask.sum()
    logger.info("Loaded %d schools (%d with valid coordinates)", n_total, n_valid)

    # Identity columns to keep
    identity_cols = [c for c in ["npsn", "latitude", "longitude", "jenjang", "status", "kota_kab", "kecamatan"] if c in schools.columns]
    result = schools[identity_cols].copy()

    # Step 1: Elevation
    if "elevation" not in skip_steps:
        logger.info("=== Step 1/4: Elevation ===")
        try:
            result["elevation_m"] = fetch_elevations(schools)
        except Exception as exc:
            logger.error("Elevation failed: %s", exc)
            result["elevation_m"] = pd.NA
    else:
        logger.info("Skipping elevation")
        result["elevation_m"] = pd.NA

    # Step 2: Road distances
    if "roads" not in skip_steps:
        logger.info("=== Step 2/4: Road distances ===")
        try:
            edges = download_road_network(cache_dir, force=force_refresh)
            road_features = compute_road_distances(schools, edges)
            for col in road_features.columns:
                result[col] = pd.NA
                result.loc[road_features.index, col] = road_features[col].values
            del edges
            gc.collect()
        except Exception as exc:
            logger.error("Road features failed: %s", exc)
            for col in ["dist_nearest_road_m", "dist_nearest_primary_road_m",
                         "road_density_500m", "nearest_road_type"]:
                result[col] = pd.NA
    else:
        logger.info("Skipping roads")

    # Step 3: Land use
    if "land_use" not in skip_steps:
        logger.info("=== Step 3/4: Land use ===")
        try:
            land_use_gdf = download_land_use(cache_dir, force=force_refresh)
            if land_use_gdf is not None:
                lu_features = compute_land_use_features(schools, land_use_gdf)
                # Initialize feature columns (skip npsn identity column)
                feature_cols = [c for c in lu_features.columns if c != "npsn"]
                for col in feature_cols:
                    result[col] = pd.NA
                if "npsn" in result.columns and "npsn" in lu_features.columns:
                    lu_indexed = lu_features.set_index("npsn")
                    for col in feature_cols:
                        result[col] = result["npsn"].map(lu_indexed[col])
                else:
                    # Fallback: positional alignment
                    valid_idx = schools[valid_mask].index
                    for col in feature_cols:
                        result.loc[valid_idx, col] = lu_features[col].values
                del land_use_gdf, lu_features
                gc.collect()
        except Exception as exc:
            logger.error("Land use features failed: %s", exc)
    else:
        logger.info("Skipping land use")

    # Step 4: Building density
    if "buildings" not in skip_steps:
        logger.info("=== Step 4/4: Building density ===")
        try:
            buildings_gdf = download_buildings(cache_dir, force=force_refresh)
            if buildings_gdf is not None:
                bld_features = compute_building_density(schools, buildings_gdf)
                for col in bld_features.columns:
                    result[col] = pd.NA
                    result.loc[bld_features.index, col] = bld_features[col].values
                del buildings_gdf, bld_features
                gc.collect()
        except Exception as exc:
            logger.error("Building density failed: %s", exc)
    else:
        logger.info("Skipping buildings")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("Saved: %s (%d rows, %d columns)", output_path, len(result), len(result.columns))
    return output_path
