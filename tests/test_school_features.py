"""Unit tests for school feature modules."""

import numpy as np
import pandas as pd
import pytest

from src.features.elevation_features import fetch_elevations
from src.features.osm_features import (
    LANDUSE_MAP,
    LANDUSE_CATEGORIES,
    _haversine_batch,
    compute_building_density,
    compute_road_distances,
)


def _make_schools(n: int = 5, with_nan: bool = False) -> pd.DataFrame:
    np.random.seed(42)
    data = {
        "npsn": [f"S{i:04d}" for i in range(n)],
        "latitude": np.linspace(-6.15, -6.35, n),
        "longitude": np.linspace(106.75, 106.95, n),
    }
    df = pd.DataFrame(data)
    if with_nan:
        df.loc[0, "latitude"] = np.nan
        df.loc[0, "longitude"] = np.nan
    return df


class TestHaversine:
    def test_known_distance(self):
        # Jakarta HI to Kelapa Gading: ~12.5 km
        dist = _haversine_batch(
            np.array([-6.1955]), np.array([106.8227]),
            np.array([-6.1541]), np.array([106.9082]),
        )
        assert 10000 < dist[0] < 15000

    def test_zero_distance(self):
        dist = _haversine_batch(
            np.array([-6.2]), np.array([106.85]),
            np.array([-6.2]), np.array([106.85]),
        )
        assert dist[0] < 1.0  # < 1 meter

    def test_vectorized(self):
        n = 10
        dist = _haversine_batch(
            np.full(n, -6.2), np.full(n, 106.85),
            np.full(n, -6.15), np.full(n, 106.90),
        )
        assert len(dist) == n
        assert np.all(dist > 0)


class TestLandUseMapping:
    def test_common_categories(self):
        assert LANDUSE_MAP["residential"] == "residential"
        assert LANDUSE_MAP["industrial"] == "industrial"
        assert LANDUSE_MAP["forest"] == "green_space"
        assert LANDUSE_MAP["commercial"] == "commercial"

    def test_all_categories_in_map(self):
        for cat in LANDUSE_CATEGORIES:
            assert cat in LANDUSE_MAP.values() or cat == "other"

    def test_unknown_falls_back(self):
        result = LANDUSE_MAP.get("unknown_type", "other")
        assert result == "other"


class TestElevation:
    def test_elevation_returns_series(self):
        df = _make_schools(3)
        elev = fetch_elevations(df)
        assert isinstance(elev, pd.Series)
        assert len(elev) == 3
        assert elev.notna().all()

    def test_elevation_range_jakarta(self):
        df = _make_schools(5)
        elev = fetch_elevations(df)
        # Jakarta elevation: 0-100m
        assert elev.min() >= -10
        assert elev.max() < 200

    def test_nan_coords_get_nan_elevation(self):
        df = _make_schools(5, with_nan=True)
        elev = fetch_elevations(df)
        assert pd.isna(elev.iloc[0])
        assert elev.iloc[1:].notna().all()

    def test_empty_df_returns_nan(self):
        df = pd.DataFrame({"latitude": [], "longitude": []})
        elev = fetch_elevations(df)
        assert len(elev) == 0


class TestRoadDistances:
    def test_with_synthetic_edges(self):
        import geopandas as gpd
        from shapely.geometry import LineString

        schools = _make_schools(3)
        edges = gpd.GeoDataFrame({
            "highway": ["residential", "primary", "tertiary"],
            "geometry": [
                LineString([(106.75, -6.20), (106.76, -6.20)]),
                LineString([(106.80, -6.15), (106.81, -6.15)]),
                LineString([(106.90, -6.30), (106.91, -6.30)]),
            ],
        }, crs="EPSG:4326")

        result = compute_road_distances(schools, edges)
        assert "dist_nearest_road_m" in result.columns
        assert "dist_nearest_primary_road_m" in result.columns
        assert "road_density_500m" in result.columns
        assert "nearest_road_type" in result.columns
        assert len(result) == 3
        assert (result["dist_nearest_road_m"] > 0).all()


class TestBuildingDensity:
    def test_with_synthetic_buildings(self):
        import geopandas as gpd
        from shapely.geometry import Polygon

        schools = _make_schools(3)
        buildings = gpd.GeoDataFrame({
            "building": ["yes", "yes", "yes"],
            "geometry": [
                Polygon([(106.751, -6.201), (106.752, -6.201), (106.752, -6.202), (106.751, -6.202)]),
                Polygon([(106.81, -6.151), (106.811, -6.151), (106.811, -6.152), (106.81, -6.152)]),
                Polygon([(106.95, -6.301), (106.951, -6.301), (106.951, -6.302), (106.95, -6.302)]),
            ],
        }, crs="EPSG:4326")

        result = compute_building_density(schools, buildings)
        assert "building_count_500m" in result.columns
        assert "building_density_500m" in result.columns
        assert "building_count_1000m" in result.columns
        assert "building_density_1000m" in result.columns
        assert len(result) == 3
        assert (result["building_count_1000m"] >= result["building_count_500m"]).all()
        assert (result["building_density_1000m"] >= 0).all()

    def test_nan_coords_excluded(self):
        import geopandas as gpd
        from shapely.geometry import Polygon

        schools = _make_schools(5, with_nan=True)
        buildings = gpd.GeoDataFrame({
            "building": ["yes"],
            "geometry": [Polygon([(106.8, -6.2), (106.801, -6.2), (106.801, -6.21), (106.8, -6.21)])],
        }, crs="EPSG:4326")

        result = compute_building_density(schools, buildings)
        assert len(result) == 4  # 5 schools minus 1 NaN
