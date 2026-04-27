"""
Spatial error map for LOSOCV results.

Generates an interactive Folium map with sensors color-coded by
cross-validation absolute error, with popup details and aggregate
statistics overlay.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import folium
import pandas as pd

from src.config import DATA_DIR, JAKARTA_CENTER
from src.spatial.losolocv import LosoMetrics

logger = logging.getLogger(__name__)

_ERROR_COLORS: list[tuple[float, str, str]] = [
    (3.0, "green", "< 3 µg/m³ (Excellent)"),
    (6.0, "#2ecc71", "3–6 (Good)"),
    (10.0, "orange", "6–10 (Moderate)"),
    (15.0, "#e74c3c", "10–15 (Poor)"),
    (float("inf"), "#8b0000", "> 15 (Bad)"),
]


def _color_for(error: float) -> str:
    for threshold, color, _ in _ERROR_COLORS:
        if error < threshold:
            return color
    return "#8b0000"


def build_error_map(
    loso_results: pd.DataFrame,
    metrics: LosoMetrics,
    output_path: Optional[Path] = None,
) -> Path:
    """Build an interactive HTML map of per-sensor LOSOCV errors.

    Args:
        loso_results: Per-sensor DataFrame from losocv_validate().
        metrics: Aggregate LosoMetrics.
        output_path: Where to save the HTML. Defaults to
            data/losocv_error_map.html.

    Returns:
        Path to the saved HTML file.
    """
    if output_path is None:
        output_path = DATA_DIR / "losocv_error_map.html"

    m = folium.Map(location=JAKARTA_CENTER, zoom_start=11, tiles="OpenStreetMap")

    sensor_group = folium.FeatureGroup(name="Sensor Error")
    for _, row in loso_results.iterrows():
        if pd.isna(row.get("abs_error")):
            continue
        lat = row["latitude"]
        lon = row["longitude"]
        error = row["abs_error"]
        color = _color_for(error)

        radius = max(50, min(error * 40, 800))

        folium.CircleMarker(
            location=[lat, lon],
            radius=max(5, min(error * 0.8, 18)),
            color=color, fill=True, fill_opacity=0.7, weight=1,
            popup=(
                f"<b>{row['sensor_id']}</b><br>"
                f"Actual: {row['actual_pm25']:.1f} µg/m³<br>"
                f"Predicted: {row['predicted_pm25']:.1f} µg/m³<br>"
                f"Abs Error: {error:.2f} µg/m³<br>"
                f"Model: {row['variogram_used']}"
            ),
        ).add_to(sensor_group)

    sensor_group.add_to(m)
    folium.LayerControl().add_to(m)

    # Build legend
    legend_items = "".join(
        f'<span style="color:{color}">&#9679;</span> {label}<br>'
        for _, color, label in _ERROR_COLORS
    )
    title = (
        f'<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
        f"z-index:9999;font-size:14px;font-weight:bold;background:white;"
        f'padding:10px 15px;border-radius:8px;border:2px solid #333;">'
        f"AirSafe — LOSOCV Spatial Error Map<br>"
        f'<span style="font-size:12px;font-weight:normal;">'
        f"MAE: {metrics.mae:.2f} | RMSE: {metrics.rmse:.2f} | "
        f"R²: {metrics.r_squared:.3f} | Bias: {metrics.bias:.2f}<br>"
        f"Sensors: {metrics.n_sensors} | Fallback: {metrics.n_fallback}<br><br>"
        f"{legend_items}"
        f"</span></div>"
    )
    m.get_root().html.add_child(folium.Element(title))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.info("Error map saved: %s", output_path)
    return output_path
