"""
BMKG Weather Client — official Indonesian Meteorological Agency API.

Fetches weather forecasts from BMKG's public JSON endpoint for Jakarta
administrative areas (kelurahan-level via adm4 codes).

API endpoint:
    https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={kode}

Returns 3-day forecasts, 8 time-slots per day, updated twice daily.
Rate limit: 60 requests/minute per IP.

Example:
    >>> data = fetch_bmkg_forecast()
    >>> df = bmkg_to_dataframe(data)
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any

import requests

from src.config import BMKG_BASE_URL, BMKG_ADM4_CODES, BMKG_REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

_SESSION_HEADERS: dict[str, str] = {
    "User-Agent": "AirSafeSchool/1.0",
    "Accept": "application/json",
}


def fetch_bmkg_forecast(
    adm4_codes: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch BMKG weather forecasts for one or more kelurahan areas.

    Args:
        adm4_codes: List of BMKG adm4 administrative codes.
            Defaults to ``BMKG_ADM4_CODES`` from config.

    Returns:
        Dictionary with ``collected_at``, ``count``, and ``areas``
        containing per-area forecast data.

    Raises:
        requests.HTTPError: On non-2xx responses.
    """
    if adm4_codes is None:
        adm4_codes = [c.strip() for c in BMKG_ADM4_CODES.split(",") if c.strip()]

    if not adm4_codes:
        logger.warning("No BMKG adm4 codes configured, skipping fetch")
        return {"collected_at": datetime.now(timezone.utc).isoformat(), "count": 0, "areas": []}

    session = requests.Session()
    session.headers.update(_SESSION_HEADERS)

    areas: list[dict[str, Any]] = []
    now_utc = datetime.now(timezone.utc)

    for i, code in enumerate(adm4_codes):
        url = f"{BMKG_BASE_URL}?adm4={code}"
        try:
            resp = session.get(url, timeout=BMKG_REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            logger.error("BMKG fetch failed for adm4=%s: %s", code, exc)
            areas.append({
                "adm4": code,
                "ok": False,
                "error": str(exc),
                "fetched_at_utc": now_utc.isoformat(),
            })
            continue

        areas.append({
            "adm4": code,
            "ok": True,
            "request_url": url,
            "fetched_at_utc": now_utc.isoformat(),
            "payload": payload,
        })
        logger.info("BMKG adm4=%s: OK", code)

        # Rate limit: stay well under 60 req/min
        if i < len(adm4_codes) - 1:
            time.sleep(1.1)

    return {
        "source": "bmkg",
        "collected_at": now_utc.isoformat(),
        "count": len(areas),
        "ok_count": sum(1 for a in areas if a.get("ok")),
        "areas": areas,
    }


def bmkg_to_dataframe(bmkg_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract flat forecast records from BMKG response for DataFrame use.

    BMKG response structure:
        payload.data[].lokasi.{provinsi, kotkab, kecamatan, desa, lon, lat}
        payload.data[].cuaca[day][slot].{datetime, t, hu, ws, wd, ...}

    Args:
        bmkg_data: Output of :func:`fetch_bmkg_forecast`.

    Returns:
        List of flat record dicts, one per forecast time-slot.
    """
    records: list[dict[str, Any]] = []

    for area in bmkg_data.get("areas", []):
        if not area.get("ok"):
            continue

        adm4 = area["adm4"]
        payload = area.get("payload", {})

        for data_entry in payload.get("data", []):
            lokasi = data_entry.get("lokasi", {})
            area_info = {
                "adm4": adm4,
                "provinsi": lokasi.get("provinsi", ""),
                "kota": lokasi.get("kotkab", ""),
                "kecamatan": lokasi.get("kecamatan", ""),
                "desa": lokasi.get("desa", ""),
                "lon": lokasi.get("lon"),
                "lat": lokasi.get("lat"),
            }

            # cuaca is a list of days, each day is a list of time-slots
            for day_slots in data_entry.get("cuaca", []):
                if not isinstance(day_slots, list):
                    continue
                for slot in day_slots:
                    records.append({
                        **area_info,
                        "datetime": slot.get("datetime", ""),
                        "utc_datetime": slot.get("utc_datetime", ""),
                        "local_datetime": slot.get("local_datetime", ""),
                        "weather_code": slot.get("weather"),
                        "weather_desc": slot.get("weather_desc", ""),
                        "weather_desc_en": slot.get("weather_desc_en", ""),
                        "temp_c": slot.get("t"),
                        "humidity_pct": slot.get("hu"),
                        "wind_speed_kmph": slot.get("ws"),
                        "wind_dir": slot.get("wd"),
                        "wind_dir_to": slot.get("wd_to"),
                        "precipitation_mm": slot.get("tp"),
                        "cloud_cover_pct": slot.get("tcc"),
                        "visibility_m": slot.get("vs"),
                        "visibility_text": slot.get("vs_text", ""),
                        "time_index": slot.get("time_index", ""),
                        "analysis_date": slot.get("analysis_date", ""),
                        "icon": slot.get("image", ""),
                    })

    logger.info("BMKG: %d forecast records extracted", len(records))
    return records
