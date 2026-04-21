"""
SPKU API Client — Jakarta air quality sensor network.

Fetches real-time PM2.5 and other pollutant readings from Jakarta's
SPKU (Stasiun Pemantau Kualitas Udara) network via the DataTables-
style POST endpoint.

API endpoint:
    https://udara.jakarta.go.id/api/lokasi_stasiun_udara

Example:
    >>> data = fetch_all_stations()
    >>> pm25 = extract_pm25_readings(data)
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import requests

from src.config import (
    SPKU_API_URL,
    SPKU_REQUEST_TIMEOUT,
    SPKU_STALE_DAYS,
    SPKU_MAX_STATIONS,
    ISPU_STATION_CITY,
    ISPU_STATION_COORDS,
)

logger = logging.getLogger(__name__)

_HEADERS: dict[str, str] = {
    "User-Agent": "AirSafeSchool/1.0",
    "X-Requested-With": "XMLHttpRequest",
    "Content-Type": "application/x-www-form-urlencoded",
}


def fetch_all_stations(stale_days: int = SPKU_STALE_DAYS) -> dict[str, Any]:
    """Fetch all SPKU station readings from the Jakarta API.

    Args:
        stale_days: Days threshold to consider a station active.

    Returns:
        Dictionary with keys ``collected_at``, ``total``, ``active``,
        ``active_pm25``, and ``stations``.

    Raises:
        requests.HTTPError: On non-2xx responses.
    """
    data = {
        "draw": "1",
        "start": "0",
        "length": str(SPKU_MAX_STATIONS),
        "columns[0][data]": "tgl",
        "columns[0][name]": "tgl",
        "columns[0][searchable]": "true",
        "columns[0][search][value]": "",
        "columns[0][search][regex]": "false",
        "order[0][column]": "0",
        "order[0][dir]": "desc",
        "search[value]": "",
        "search[regex]": "false",
    }

    resp = requests.post(
        SPKU_API_URL,
        headers=_HEADERS,
        data=data,
        timeout=SPKU_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    result = resp.json()

    stations_raw: list[dict] = result.get("data", [])
    now = datetime.now()
    cutoff = now - timedelta(days=stale_days)

    stations: list[dict[str, Any]] = []
    for s in stations_raw:
        try:
            ts = datetime.strptime(s.get("tgl", ""), "%m/%d/%Y %H:%M:%S")
            is_active = ts >= cutoff
        except (ValueError, TypeError):
            ts = None
            is_active = False

        stations.append({
            "station_id": s.get("dataSourceID", ""),
            "station_name": s.get("dataSourceName", ""),
            "latitude": float(s.get("latitude", 0)),
            "longitude": float(s.get("longitude", 0)),
            "parameter": s.get("matricName", ""),
            "value": _parse_float(s.get("value")),
            "status": s.get("status", ""),
            "timestamp": s.get("tgl", ""),
            "timestamp_parsed": ts.isoformat() if ts else None,
            "is_active": is_active,
        })

    active_count = sum(1 for s in stations if s["is_active"])
    pm25_active = [s for s in stations if s["is_active"] and s["parameter"] == "PM25"]

    logger.info(
        "SPKU: %d total, %d active, %d active PM2.5",
        len(stations),
        active_count,
        len(pm25_active),
    )

    return {
        "collected_at": now.isoformat(),
        "total": len(stations),
        "active": active_count,
        "active_pm25": len(pm25_active),
        "stations": stations,
    }


def extract_pm25_readings(spku_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract PM2.5 readings from an SPKU snapshot.

    Args:
        spku_data: Output of :func:`fetch_all_stations`.

    Returns:
        List of flat record dicts with PM2.5 readings.
    """
    records: list[dict[str, Any]] = []
    collected_at = spku_data["collected_at"]

    for s in spku_data["stations"]:
        if s["parameter"] != "PM25":
            continue
        records.append({
            "collection_time": collected_at,
            "station_name": s["station_name"],
            "station_id": s["station_id"],
            "latitude": s["latitude"],
            "longitude": s["longitude"],
            "pm25": s["value"],
            "status": s["status"],
            "is_active": s["is_active"],
            "station_timestamp": s["timestamp"],
        })

    logger.info("Extracted %d PM2.5 records", len(records))
    return records


def _parse_float(val: Any) -> Optional[float]:
    """Safely coerce a value to float.

    Args:
        val: Raw value from API response.

    Returns:
        Float value or ``None`` if parsing fails.
    """
    if val is None or val == "" or val == "-":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
