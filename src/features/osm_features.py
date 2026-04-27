"""OSM-derived school features: road distances, land use, building density.

Downloads OpenStreetMap data for Jakarta via OSMnx (roads) and Overpass API
(land use, buildings), then computes spatial features for each school location.

All downloaded data is cached to disk for re-runs.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, Polygon
from shapely.validation import make_valid

logger = logging.getLogger(__name__)

# Jakarta bounding box (padded slightly beyond DKI boundary)
_BBOX_NORTH = -5.63
_BBOX_SOUTH = -6.38
_BBOX_EAST = 106.98
_BBOX_WEST = 106.51

# BallTree requires radians for haversine — precompute factor
_KM_TO_M = 1000.0
_EARTH_RADIUS_KM = 6371.0

# Standardized land use categories
LANDUSE_MAP = {
    "residential": "residential",
    "commercial": "commercial",
    "retail": "commercial",
    "industrial": "industrial",
    "brownfield": "industrial",
    "forest": "green_space",
    "grass": "green_space",
    "meadow": "green_space",
    "park": "green_space",
    "cemetery": "green_space",
    "village_green": "green_space",
    "recreation_ground": "green_space",
    "construction": "construction",
    "military": "institutional",
    "institutional": "institutional",
    "education": "institutional",
    "religious": "institutional",
    "water": "water",
    "reservoir": "water",
    "basin": "water",
    "salt_pond": "water",
    "railway": "transportation",
    "highway": "transportation",
    "airport": "transportation",
    "port": "transportation",
    "quarry": "industrial",
    "landfill": "industrial",
    "farmland": "agriculture",
    "farmyard": "agriculture",
    "orchard": "agriculture",
    "vineyard": "agriculture",
}

LANDUSE_CATEGORIES = ["residential", "commercial", "industrial", "green_space", "water"]

_PRIMARY_ROAD_TYPES = {"primary", "trunk", "motorway", "motorway_link", "trunk_link", "primary_link"}


def _coords_to_rad(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    return np.radians(np.column_stack([lats, lons]))


def _haversine_batch(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine distance in meters."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return _EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a)) * _KM_TO_M


# ── Road Network ──────────────────────────────────────────────────────────


def download_road_network(cache_dir: Path, force: bool = False) -> gpd.GeoDataFrame:
    """Download Jakarta drive road network via OSMnx, cache as GraphML.

    Returns GeoDataFrame of edges with geometry and highway type.
    """
    import osmnx as ox

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    graph_path = cache_dir / "jakarta_drive_graph.graphml"

    if graph_path.exists() and not force:
        logger.info("Loading cached road network: %s", graph_path)
        G = ox.load_graphml(graph_path)
    else:
        logger.info("Downloading Jakarta road network via OSMnx...")
        G = ox.graph_from_bbox(
            bbox=(_BBOX_WEST, _BBOX_SOUTH, _BBOX_EAST, _BBOX_NORTH),
            network_type="drive",
        )
        ox.save_graphml(G, graph_path)
        logger.info("Cached road network: %s (%d edges)", graph_path, len(G.edges))

    edges = ox.graph_to_gdfs(G, nodes=False)
    del G
    gc.collect()

    logger.info("Road network: %d edges", len(edges))
    return edges


def compute_road_distances(
    schools_df: pd.DataFrame,
    edges_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Compute road proximity features for each school using BallTree.

    Returns DataFrame with columns: dist_nearest_road_m,
    dist_nearest_primary_road_m, road_density_500m, nearest_road_type.
    """
    from sklearn.neighbors import BallTree

    valid = schools_df.dropna(subset=["latitude", "longitude"])
    n_schools = len(valid)
    logger.info("Computing road distances for %d schools...", n_schools)

    # Extract edge midpoints (compute once)
    edges = edges_gdf.copy()
    centroids = edges.geometry.centroid
    edges["mid_lat"] = centroids.y
    edges["mid_lon"] = centroids.x
    del centroids
    edge_coords = _coords_to_rad(edges["mid_lat"].values, edges["mid_lon"].values)

    tree = BallTree(edge_coords, metric="haversine")

    school_coords = _coords_to_rad(valid["latitude"].values, valid["longitude"].values)

    # Nearest road (any type)
    dist_any, idx_any = tree.query(school_coords, k=1)
    dist_any_m = dist_any.flatten() * _EARTH_RADIUS_KM * _KM_TO_M

    # Nearest road types — normalize list values to first element
    def _normalize_highway(h):
        if isinstance(h, list):
            return h[0]
        return str(h)

    highway_normalized = edges["highway"].apply(_normalize_highway)
    nearest_types = highway_normalized.iloc[idx_any.flatten()].values

    # Nearest primary road — filter edges to primary types
    primary_mask = edges["highway"].apply(
        lambda h: any(t in _PRIMARY_ROAD_TYPES for t in (h if isinstance(h, list) else [h]))
    )
    primary_edges = edges[primary_mask]

    if len(primary_edges) > 0:
        primary_coords = _coords_to_rad(
            primary_edges["mid_lat"].values, primary_edges["mid_lon"].values,
        )
        primary_tree = BallTree(primary_coords, metric="haversine")
        dist_primary, _ = primary_tree.query(school_coords, k=1)
        dist_primary_m = dist_primary.flatten() * _EARTH_RADIUS_KM * _KM_TO_M
    else:
        dist_primary_m = np.full(n_schools, np.nan)

    # Road density — count edges within 500m
    radius_rad = (0.5 / _EARTH_RADIUS_KM)  # 500m in radians
    counts_500 = tree.query_radius(school_coords, r=radius_rad, count_only=True)

    del tree, edge_coords
    if len(primary_edges) > 0:
        del primary_tree, primary_coords
    del edges
    gc.collect()

    result = pd.DataFrame({
        "dist_nearest_road_m": dist_any_m,
        "dist_nearest_primary_road_m": dist_primary_m,
        "road_density_500m": counts_500.astype(int),
        "nearest_road_type": nearest_types,
    }, index=valid.index)

    logger.info(
        "Road distances: median=%.0fm, max=%.0fm",
        result["dist_nearest_road_m"].median(),
        result["dist_nearest_road_m"].max(),
    )
    return result


# ── Buildings ──────────────────────────────────────────────────────────────


_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_OVERPASS_HEADERS = {"Accept": "application/json", "User-Agent": "airsafe-school/1.0"}


def _overpass_query(query: str, timeout: int = 300) -> Optional[dict]:
    """Execute Overpass API query with proper headers. Returns JSON or None."""
    try:
        resp = requests.post(
            _OVERPASS_URL,
            data={"data": query},
            headers=_OVERPASS_HEADERS,
            timeout=timeout + 30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("Overpass query failed: %s", exc)
        return None


def _parse_overpass_ways(data: dict) -> tuple[list[dict], dict]:
    """Parse Overpass JSON into way polygons. Returns (polygons, node_lookup)."""
    nodes = {}
    for el in data.get("elements", []):
        if el.get("type") == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])

    ways = []
    for el in data.get("elements", []):
        if el.get("type") == "way" and "tags" in el:
            ways.append(el)
    return ways, nodes


def download_buildings(cache_dir: Path, force: bool = False) -> Optional[gpd.GeoDataFrame]:
    """Download Jakarta building centroids via Overpass API, cache as GeoJSON.

    Uses `out center` to get centroids only — sufficient for density computation
    and avoids downloading millions of full polygon geometries.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = cache_dir / "jakarta_buildings.geojson"

    if geojson_path.exists() and not force:
        logger.info("Loading cached buildings: %s", geojson_path)
        return gpd.read_file(geojson_path)

    logger.info("Downloading Jakarta building centroids via Overpass API...")
    query = """
        [out:json][timeout:300];
        way["building"](%(south)f,%(west)f,%(north)f,%(east)f);
        out center;
    """ % {"north": _BBOX_NORTH, "south": _BBOX_SOUTH, "east": _BBOX_EAST, "west": _BBOX_WEST}

    data = _overpass_query(query)
    if data is None:
        return None

    points = []
    for el in data.get("elements", []):
        if el.get("type") == "way" and "center" in el:
            c = el["center"]
            points.append({
                "geometry": Point(c["lon"], c["lat"]),
                "building": el.get("tags", {}).get("building", "yes"),
            })

    if not points:
        logger.warning("No building centroids found")
        return None

    gdf = gpd.GeoDataFrame(points, crs="EPSG:4326")
    gdf.to_file(geojson_path, driver="GeoJSON")
    logger.info("Cached buildings: %s (%d centroids)", geojson_path, len(gdf))
    return gdf


def compute_building_density(
    schools_df: pd.DataFrame,
    buildings_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Count buildings within 500m and 1km of each school.

    Returns DataFrame with building_count and building_density for 500m/1000m.
    """
    from sklearn.neighbors import BallTree

    valid = schools_df.dropna(subset=["latitude", "longitude"])
    n_schools = len(valid)
    logger.info("Computing building density for %d schools...", n_schools)

    # Use geometry directly (points for centroids, or compute centroids for polygons)
    if buildings_gdf.geometry.iloc[0].geom_type == "Point":
        bld_lats = buildings_gdf.geometry.y.values
        bld_lons = buildings_gdf.geometry.x.values
    else:
        centroids = buildings_gdf.geometry.centroid
        bld_lats = centroids.y.values
        bld_lons = centroids.x.values

    bld_coords = _coords_to_rad(bld_lats, bld_lons)
    tree = BallTree(bld_coords, metric="haversine")

    school_coords = _coords_to_rad(valid["latitude"].values, valid["longitude"].values)

    radius_500 = 0.5 / _EARTH_RADIUS_KM
    radius_1000 = 1.0 / _EARTH_RADIUS_KM

    counts_500 = tree.query_radius(school_coords, r=radius_500, count_only=True)
    counts_1000 = tree.query_radius(school_coords, r=radius_1000, count_only=True)

    del tree, bld_coords
    gc.collect()

    # Density = count / area (km2)
    area_500 = np.pi * 0.5 ** 2
    area_1000 = np.pi * 1.0 ** 2

    result = pd.DataFrame({
        "building_count_500m": counts_500.astype(int),
        "building_density_500m": counts_500 / area_500,
        "building_count_1000m": counts_1000.astype(int),
        "building_density_1000m": counts_1000 / area_1000,
    }, index=valid.index)

    logger.info(
        "Building density: median=%.0f buildings/500m",
        result["building_count_500m"].median(),
    )
    return result


# ── Land Use ──────────────────────────────────────────────────────────────


def download_land_use(cache_dir: Path, force: bool = False) -> Optional[gpd.GeoDataFrame]:
    """Download Jakarta land use polygons via Overpass API, cache as GeoJSON."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = cache_dir / "jakarta_land_use.geojson"

    if geojson_path.exists() and not force:
        logger.info("Loading cached land use: %s", geojson_path)
        return gpd.read_file(geojson_path)

    logger.info("Downloading Jakarta land use via Overpass API...")
    query = """
        [out:json][timeout:300];
        (
            way["landuse"](%(south)f,%(west)f,%(north)f,%(east)f);
            relation["landuse"](%(south)f,%(west)f,%(north)f,%(east)f);
        );
        out body;
        >;
        out skel qt;
    """ % {"north": _BBOX_NORTH, "south": _BBOX_SOUTH, "east": _BBOX_EAST, "west": _BBOX_WEST}

    data = _overpass_query(query)
    if data is None:
        return None

    # Build node lookup
    nodes = {}
    relations = []
    for el in data.get("elements", []):
        if el.get("type") == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])
        elif el.get("type") == "relation" and "tags" in el:
            relations.append(el)

    # Parse ways
    ways, _ = _parse_overpass_ways(data)

    polygons = []
    for way in ways:
        coords = [nodes[nid] for nid in way.get("nodes", []) if nid in nodes]
        if len(coords) >= 4:
            landuse_raw = way["tags"].get("landuse", "unknown")
            polygons.append({
                "geometry": Polygon(coords),
                "landuse_raw": landuse_raw,
                "landuse_category": LANDUSE_MAP.get(landuse_raw, "other"),
            })

    # Handle relations (multipolygons) — build way lookup for member resolution
    way_lookup = {el["id"]: el for el in data.get("elements", []) if el.get("type") == "way"}
    for rel in relations:
        try:
            outer_coords = []
            for member in rel.get("members", []):
                if member.get("role") == "outer" and member.get("type") == "way":
                    member_way = way_lookup.get(member["ref"])
                    if member_way:
                        outer_coords.extend(
                            nodes[nid] for nid in member_way.get("nodes", []) if nid in nodes
                        )
            if len(outer_coords) >= 4:
                landuse_raw = rel["tags"].get("landuse", "unknown")
                polygons.append({
                    "geometry": Polygon(outer_coords),
                    "landuse_raw": landuse_raw,
                    "landuse_category": LANDUSE_MAP.get(landuse_raw, "other"),
                })
        except Exception:
            pass

    if not polygons:
        logger.warning("No land use polygons found")
        return None

    gdf = gpd.GeoDataFrame(polygons, crs="EPSG:4326")
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        logger.info("Fixing %d invalid land use polygons", invalid.sum())
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)
    gdf.to_file(geojson_path, driver="GeoJSON")
    logger.info("Cached land use: %s (%d polygons)", geojson_path, len(gdf))
    return gdf


def _compute_land_use_for_buffer(
    school_point: Point,
    buffer_m: float,
    land_use_utm: gpd.GeoDataFrame,
    utm_crs: str,
) -> dict[str, float | str]:
    """Compute land use fractions within buffer of a single school.

    Args:
        land_use_utm: Must already be reprojected to UTM CRS.
    """
    school_gdf = gpd.GeoDataFrame(
        {"geometry": [school_point]}, crs="EPSG:4326",
    ).to_crs(utm_crs)
    buf = school_gdf.geometry.iloc[0].buffer(buffer_m)

    intersected = land_use_utm[land_use_utm.geometry.intersects(buf)]

    if intersected.empty:
        return {"dominant": "unknown"}

    fractions = {}
    for _, row in intersected.iterrows():
        cat = row["landuse_category"]
        geom = row.geometry
        if not geom.is_valid:
            geom = make_valid(geom)
        overlap = geom.intersection(buf)
        area = overlap.area
        fractions[cat] = fractions.get(cat, 0) + area

    total = sum(fractions.values())
    if total == 0:
        return {"dominant": "unknown"}

    result = {k: v / total for k, v in fractions.items()}
    result["dominant"] = max(fractions, key=fractions.get)
    return result


def compute_land_use_features(
    schools_df: pd.DataFrame,
    land_use_gdf: gpd.GeoDataFrame,
    chunk_size: int = 500,
) -> pd.DataFrame:
    """Compute land use composition within 500m and 1km of each school.

    Returns DataFrame with dominant land use and fraction columns per buffer.
    """
    valid = schools_df.dropna(subset=["latitude", "longitude"]).copy()
    n_schools = len(valid)
    logger.info("Computing land use features for %d schools (chunk_size=%d)...", n_schools, chunk_size)

    # UTM zone 48S for Jakarta — reproject once
    utm_crs = "EPSG:32748"
    land_use_utm = land_use_gdf.to_crs(utm_crs)

    rows = []
    for start in range(0, n_schools, chunk_size):
        end = min(start + chunk_size, n_schools)
        chunk = valid.iloc[start:end]
        logger.info("Land use: %d / %d schools", end, n_schools)

        for idx, row in chunk.iterrows():
            point = Point(row["longitude"], row["latitude"])
            row_data = {"npsn": row.get("npsn", idx)}

            for buffer_m, suffix in [(500, "500m"), (1000, "1000m")]:
                lu = _compute_land_use_for_buffer(point, buffer_m, land_use_utm, utm_crs)
                row_data[f"dominant_land_use_{suffix}"] = lu.get("dominant", "unknown")
                for cat in LANDUSE_CATEGORIES:
                    row_data[f"landuse_{cat}_frac_{suffix}"] = round(lu.get(cat, 0.0), 4)

            rows.append(row_data)

    # Distance to nearest industrial polygon
    industrial = land_use_gdf[land_use_gdf["landuse_category"] == "industrial"]
    if len(industrial) > 0:
        from sklearn.neighbors import BallTree

        ind_centroids = industrial.geometry.centroid
        ind_coords = _coords_to_rad(ind_centroids.y.values, ind_centroids.x.values)
        ind_tree = BallTree(ind_coords, metric="haversine")

        school_coords = _coords_to_rad(valid["latitude"].values, valid["longitude"].values)
        dist_ind, _ = ind_tree.query(school_coords, k=1)
        dist_ind_m = dist_ind.flatten() * _EARTH_RADIUS_KM * _KM_TO_M

        dist_series = pd.Series(dist_ind_m, index=valid.index, name="dist_to_industrial_m")
        del ind_tree, ind_coords
    else:
        dist_series = pd.Series(np.nan, index=valid.index, name="dist_to_industrial_m")

    result = pd.DataFrame(rows)

    # Merge distance to industrial — align by npsn
    result["dist_to_industrial_m"] = dist_series.values

    del land_use_utm
    gc.collect()
    logger.info("Land use features computed for %d schools", len(result))
    return result
