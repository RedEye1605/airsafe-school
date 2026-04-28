"""Rendahemisi scraper for hourly multi-pollutant data from 5 ISPU stations.

Downloads hourly PM2.5, PM10, SO2, CO, O3, NO2, HC data from
rendahemisi.jakarta.go.id ISPU detail pages. Each page shows one day of
hourly readings for one station as an HTML table.

Ported from Adit's Scraper.ipynb.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_RENDAHEMISI_BASE = "https://rendahemisi.jakarta.go.id/ispu-detail"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
}

POLLUTANT_COLS = ["pm10", "pm25", "so2", "co", "o3", "no2", "hc"]

STATIONS = [
    {"station_id": 4, "slug": "dki1-bundaran-hi",   "station_name": "DKI1 Bundaran HI"},
    {"station_id": 5, "slug": "dki2-kelapa-gading", "station_name": "DKI2 Kelapa Gading"},
    {"station_id": 6, "slug": "dki3-jagakarsa",     "station_name": "DKI3 Jagakarsa"},
    {"station_id": 7, "slug": "dki4-lubang-buaya",  "station_name": "DKI4 Lubang Buaya"},
    {"station_id": 8, "slug": "dki5-kebun-jeruk",   "station_name": "DKI5 Kebun Jeruk"},
]

_MAX_RETRIES = 3
_TIMEOUT = 40


def _build_url(station_id: int, slug: str, dt: datetime) -> str:
    return f"{_RENDAHEMISI_BASE}/{station_id}/{slug}/{dt.strftime('%d-%m-%Y')}"


def _extract_regex(text: str, pattern: str, flags: int = re.S) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _parse_location_and_update(page_text: str) -> tuple[Optional[str], Optional[str]]:
    last_update = _extract_regex(
        page_text,
        r"Terakhir Update:\s*(.*?)\s*(?:Lokasi:|Sumber:|Keterangan:|$)",
    )
    lokasi = _extract_regex(
        page_text,
        r"Lokasi:\s*(.*?)\s*(?:Sumber:|Keterangan:|$)",
    )
    return lokasi, last_update


def _norm_col(c: str) -> str:
    c = str(c).strip().lower()
    return c.replace(" ", "").replace(".", "")


def _parse_hourly_table(html: str, station: dict, dt: datetime, url: str) -> Optional[pd.DataFrame]:
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text("\n", strip=True)

    try:
        tables = pd.read_html(StringIO(html))
    except (ValueError, KeyError, IndexError):
        return None

    target = None
    for tbl in tables:
        cols = [_norm_col(c) for c in tbl.columns]
        if "waktu" in cols:
            target = tbl.copy()
            target.columns = cols
            break

    if target is None:
        return None

    # Rename 'waktu' → 'hour'
    if "waktu" in target.columns and "hour" not in target.columns:
        target = target.rename(columns={"waktu": "hour"})

    keep_cols = ["hour", "pm10", "pm25", "so2", "co", "o3", "no2", "hc", "kategori"]
    target = target[[c for c in keep_cols if c in target.columns]].copy()

    for col in POLLUTANT_COLS:
        if col in target.columns:
            target[col] = (
                target[col]
                .astype(str)
                .str.strip()
                .replace({"-": pd.NA, "": pd.NA, "nan": pd.NA, "None": pd.NA})
            )
            target[col] = pd.to_numeric(target[col], errors="coerce")

    if "kategori" in target.columns:
        target["kategori"] = (
            target["kategori"]
            .astype(str)
            .str.strip()
            .replace({"-": pd.NA, "": pd.NA, "nan": pd.NA, "None": pd.NA})
        )

    lokasi, last_update = _parse_location_and_update(page_text)

    target["date"] = pd.to_datetime(dt.date())
    target["station_id"] = station["station_id"]
    target["station_slug"] = station["slug"]
    target["station_name"] = station["station_name"]
    target["lokasi"] = lokasi
    target["last_update"] = last_update
    target["source_url"] = url

    target["datetime"] = pd.to_datetime(
        dt.strftime("%d-%m-%Y") + " " + target["hour"].astype(str),
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )

    target = target.drop_duplicates(
        subset=["datetime", "station_slug"], keep="first",
    ).reset_index(drop=True)

    front = [
        "datetime", "date", "hour",
        "station_id", "station_slug", "station_name", "lokasi",
        "pm25", "pm10", "so2", "co", "o3", "no2", "hc",
        "kategori", "last_update", "source_url",
    ]
    front = [c for c in front if c in target.columns]
    other = [c for c in target.columns if c not in front]
    return target[front + other]


def fetch_station_day(
    station_id: int,
    slug: str,
    dt: datetime,
    station_name: str = "",
) -> Optional[pd.DataFrame]:
    """Fetch hourly data for one station on one date.

    Args:
        station_id: Rendahemisi station ID (4-8).
        slug: Station URL slug (e.g. 'dki1-bundaran-hi').
        dt: Date to fetch.
        station_name: Human-readable station name.

    Returns:
        DataFrame with hourly pollutant data, or None if unavailable.
    """
    station = {"station_id": station_id, "slug": slug, "station_name": station_name}
    url = _build_url(station_id, slug, dt)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)

            if resp.status_code == 404:
                logger.debug("No data for %s on %s (404)", slug, dt.date())
                return None

            if resp.status_code != 200:
                if attempt < _MAX_RETRIES:
                    time.sleep(2 ** attempt)
                    continue
                logger.warning("HTTP %d for %s on %s", resp.status_code, slug, dt.date())
                return None

            df = _parse_hourly_table(resp.text, station, dt, url)
            return df

        except Exception as exc:
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            logger.error("Failed %s on %s: %s", slug, dt.date(), exc)
            return None

    return None


def fetch_recent_hours(hours: int = 72) -> pd.DataFrame:
    """Fetch the last N hours of data for all 5 ISPU stations.

    Scrapes today and yesterday (plus day before if needed) to cover
    the requested time range.

    Args:
        hours: Number of recent hours to fetch (default 72).

    Returns:
        Combined DataFrame with all station data.
    """
    now = datetime.now()
    n_days = (hours // 24) + 2  # buffer
    dates = [now - timedelta(days=d) for d in range(n_days)]

    all_dfs = []
    for station in STATIONS:
        for dt in dates:
            df = fetch_station_day(
                station["station_id"],
                station["slug"],
                dt,
                station_name=station["station_name"],
            )
            if df is not None and not df.empty:
                all_dfs.append(df)
            time.sleep(0.3)

    if not all_dfs:
        logger.warning("No data fetched from rendahemisi")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["datetime", "station_slug"], keep="first",
    )

    # Filter to requested time range
    cutoff = now - timedelta(hours=hours)
    combined["datetime"] = pd.to_datetime(combined["datetime"])
    combined = combined[combined["datetime"] >= cutoff].reset_index(drop=True)

    logger.info(
        "Rendahemisi: fetched %d rows across %d stations (last %dh)",
        len(combined), combined["station_slug"].nunique(), hours,
    )
    return combined
