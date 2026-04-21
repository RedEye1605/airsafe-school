"""
Coverage Map — SPKU sensor coverage vs school locations.

Generates an interactive Folium map showing PM2.5 monitoring station
coverage circles (2 km radius) overlaid on geocoded school locations,
with colour-coded risk categories.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import folium
import pandas as pd

from src.config import (
    COVERAGE_RADIUS_M,
    DATA_DIR,
    ISPU_STATION_CITY,
    JAKARTA_CENTER,
    PROCESSED_DIR,
    RAW_DIR,
)
from src.data.transforms import risk_to_color

logger = logging.getLogger(__name__)

_STATUS_COLORS: dict[str, str] = {
    "Baik": "green",
    "Sedang": "orange",
    "Tidak Sehat": "red",
}


def build_coverage_map(
    schools_path: Optional[Path] = None,
    stations_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Build an interactive coverage-gap HTML map.

    Args:
        schools_path: Geocoded schools CSV.
        stations_path: SPKU stations catalog CSV.
        output_path: Where to save the HTML file.

    Returns:
        Path to the saved HTML file.
    """
    if schools_path is None:
        schools_path = PROCESSED_DIR / "schools" / "schools_geocoded.csv"
    if stations_path is None:
        stations_path = RAW_DIR / "spku" / "spku_stations_catalog.csv"
    if output_path is None:
        output_path = DATA_DIR / "eda_04_pm25_coverage_gap.html"

    # Load & filter PM25 stations
    spku = pd.read_csv(stations_path)
    pm25_stations = spku[spku["parameter"] == "PM25"].copy()
    logger.info("PM2.5 stations: %d", len(pm25_stations))

    # Load schools
    schools = pd.read_csv(schools_path).dropna(subset=["latitude", "longitude"])
    logger.info("Schools with coords: %d", len(schools))

    # Classify schools
    covered_count = 0
    school_cats: list[str] = []
    for _, s in schools.iterrows():
        is_covered = False
        for _, st in pm25_stations.iterrows():
            dist = (
                (s["latitude"] - float(st["latitude"])) ** 2
                + (s["longitude"] - float(st["longitude"])) ** 2
            ) ** 0.5 * 111.19
            if dist <= 2.0:
                is_covered = True
                break
        if is_covered:
            covered_count += 1
            school_cats.append("covered")
        else:
            school_cats.append("outside")

    schools = schools.copy()
    schools["coverage_cat"] = school_cats
    outside_count = len(schools) - covered_count

    # ── Build map ─────────────────────────────────────────────────────────
    m = folium.Map(location=JAKARTA_CENTER, zoom_start=11, tiles="OpenStreetMap")

    # Station markers + radius circles
    station_group = folium.FeatureGroup(name="PM25 Stations (2 km radius)")
    for _, s in pm25_stations.iterrows():
        lat, lon = float(s["latitude"]), float(s["longitude"])
        name = s.get("station_name", "")
        value = s.get("latest_value", 0)
        status = s.get("status", "")
        color = _STATUS_COLORS.get(status, "blue")

        folium.Marker(
            location=[lat, lon],
            popup=f"<b>{name}</b><br>PM2.5: {value} µg/m³<br>Status: {status}",
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(station_group)

        folium.Circle(
            location=[lat, lon],
            radius=COVERAGE_RADIUS_M,
            color="blue", fill=True, fill_opacity=0.1, weight=1,
        ).add_to(station_group)

    station_group.add_to(m)

    # School markers
    cat_style: dict[str, dict] = {
        "covered": {"color": "green"},
        "outside": {"color": "orange"},
    }
    for cat, style in cat_style.items():
        grp = folium.FeatureGroup(name=f"Schools ({cat.title()})")
        subset = schools[schools["coverage_cat"] == cat]
        for _, s in subset.iterrows():
            folium.CircleMarker(
                location=[s["latitude"], s["longitude"]],
                radius=3, fill=True, fill_opacity=0.5, weight=0.5,
                popup=(
                    f"<b>{s['nama_sekolah']}</b><br>"
                    f"{s['jenjang']} — {s['kota_kab']}<br>"
                    f"{s['kecamatan']}"
                ),
                **style,
            ).add_to(grp)
        grp.add_to(m)

    folium.LayerControl().add_to(m)

    # Title overlay
    title = (
        f'<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
        f'z-index:9999;font-size:16px;font-weight:bold;background:white;'
        f'padding:10px 15px;border-radius:8px;border:2px solid #333;">'
        f"AirSafe School — PM2.5 Coverage Gap Map<br>"
        f'<span style="font-size:12px;font-weight:normal;">'
        f"🔵 {len(pm25_stations)} PM2.5 Stations (2 km radius)<br>"
        f"🟢 Covered: {covered_count:,} schools<br>"
        f"🟠 Outside: {outside_count:,} schools<br>"
        f"<small>(Total: {len(schools):,})</small>"
        f"</span></div>"
    )
    m.get_root().html.add_child(folium.Element(title))

    m.save(str(output_path))
    logger.info("Coverage map saved: %s", output_path)
    return output_path
