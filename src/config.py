"""
AirSafe School — Configuration Module.

Reads settings from environment variables with sensible defaults.
Copy `.env.example` to `.env` and fill in values for local development.
"""

import os
from pathlib import Path
from typing import Final

# ── Project Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT: Final[Path] = Path(
    os.environ.get("AIRSAFE_ROOT", Path(__file__).resolve().parent.parent.parent)
)
DATA_DIR: Final[Path] = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
OUTPUT_DIR: Final[Path] = Path(
    os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "output")
)

# ── SPKU API ─────────────────────────────────────────────────────────────────

SPKU_API_URL: Final[str] = os.environ.get(
    "SPKU_API_URL",
    "https://udara.jakarta.go.id/api/lokasi_stasiun_udara",
)
SPKU_STALE_DAYS: Final[int] = int(os.environ.get("SPKU_STALE_DAYS", "7"))
SPKU_REQUEST_TIMEOUT: Final[int] = int(os.environ.get("SPKU_REQUEST_TIMEOUT", "30"))
SPKU_MAX_STATIONS: Final[int] = int(os.environ.get("SPKU_MAX_STATIONS", "500"))

# ── Open-Meteo Weather API ───────────────────────────────────────────────────

OPEN_METEO_ARCHIVE_URL: Final[str] = os.environ.get(
    "OPEN_METEO_ARCHIVE_URL",
    "https://archive-api.open-meteo.com/v1/archive",
)
OPEN_METEO_FORECAST_URL: Final[str] = os.environ.get(
    "OPEN_METEO_FORECAST_URL",
    "https://api.open-meteo.com/v1/forecast",
)
JAKARTA_LAT: Final[float] = float(os.environ.get("JAKARTA_LAT", "-6.2"))
JAKARTA_LON: Final[float] = float(os.environ.get("JAKARTA_LON", "106.85"))
WEATHER_REQUEST_TIMEOUT: Final[int] = int(
    os.environ.get("WEATHER_REQUEST_TIMEOUT", "60")
)

# ── Nominatim Geocoding ──────────────────────────────────────────────────────

NOMINATIM_URL: Final[str] = os.environ.get(
    "NOMINATIM_URL",
    "https://nominatim.openstreetmap.org/search",
)
NOMINATIM_USER_AGENT: Final[str] = os.environ.get(
    "NOMINATIM_USER_AGENT",
    "AirSafeSchool/1.0 (airsafe@example.com)",
)
NOMINATIM_RATE_LIMIT: Final[float] = float(
    os.environ.get("NOMINATIM_RATE_LIMIT", "1.1")
)

# ── School Data ──────────────────────────────────────────────────────────────

SCHOOL_REGISTRY_BASE_URL: Final[str] = os.environ.get(
    "SCHOOL_REGISTRY_BASE_URL",
    "https://referensi.data.kemendikdasmen.go.id",
)

# ── ISPU Station Mapping ─────────────────────────────────────────────────────

ISPU_STATION_CITY: Final[dict[str, str]] = {
    "DKI1": "Jakarta Pusat",
    "DKI2": "Jakarta Utara",
    "DKI3": "Jakarta Selatan",
    "DKI4": "Jakarta Timur",
    "DKI5": "Jakarta Barat",
}

ISPU_STATION_COORDS: Final[dict[str, tuple[float, float]]] = {
    "DKI1": (-6.1753, 106.8272),   # Bunderan HI
    "DKI2": (-6.1578, 106.9067),   # Kelapa Gading
    "DKI3": (-6.3692, 106.8186),   # Jagakarsa
    "DKI4": (-6.2942, 106.8897),   # Lubang Buaya
    "DKI5": (-6.1909, 106.7369),   # Kebon Jeruk
}

# ── Map Defaults ─────────────────────────────────────────────────────────────

JAKARTA_CENTER: Final[tuple[float, float]] = (-6.2088, 106.8456)
COVERAGE_RADIUS_M: Final[int] = 2000  # 2 km in metres

# ── BMKG Weather API ────────────────────────────────────────────────────────

BMKG_BASE_URL: Final[str] = os.environ.get(
    "BMKG_BASE_URL",
    "https://api.bmkg.go.id/publik/prakiraan-cuaca",
)
BMKG_ADM4_CODES: Final[str] = os.environ.get(
    "BMKG_ADM4_CODES",
    "31.71.03.1001",  # Kemayoran, Jakarta Pusat
)
BMKG_REQUEST_TIMEOUT: Final[int] = int(
    os.environ.get("BMKG_REQUEST_TIMEOUT", "60")
)

# ── Azure Blob Storage ──────────────────────────────────────────────────────

AIRSAFE_BLOB_CONNECTION_STRING: Final[str] = os.environ.get(
    "AIRSAFE_BLOB_CONNECTION_STRING", ""
)
AIRSAFE_RAW_CONTAINER: Final[str] = os.environ.get(
    "AIRSAFE_RAW_CONTAINER", "raw"
)
AIRSAFE_PROCESSED_CONTAINER: Final[str] = os.environ.get(
    "AIRSAFE_PROCESSED_CONTAINER", "processed"
)
AIRSAFE_PREDICT_CONTAINER: Final[str] = os.environ.get(
    "AIRSAFE_PREDICT_CONTAINER", "predictions"
)
AIRSAFE_LOG_CONTAINER: Final[str] = os.environ.get(
    "AIRSAFE_LOG_CONTAINER", "logs"
)
AIRSAFE_MODELS_CONTAINER: Final[str] = os.environ.get(
    "AIRSAFE_MODELS_CONTAINER", "models"
)
AIRSAFE_FEATURES_CONTAINER: Final[str] = os.environ.get(
    "AIRSAFE_FEATURES_CONTAINER", "features"
)
AIRSAFE_REFERENCE_CONTAINER: Final[str] = os.environ.get(
    "AIRSAFE_REFERENCE_CONTAINER", "reference"
)
AIRSAFE_SCRATCH_CONTAINER: Final[str] = os.environ.get(
    "AIRSAFE_SCRATCH_CONTAINER", "scratch"
)

# ── ETL Schedule ────────────────────────────────────────────────────────────

ETL_SCHEDULE: Final[str] = os.environ.get(
    "ETL_SCHEDULE",
    "0 5 * * * *",  # every hour at :05 (UTC)
)

# ── Local / Azure Mode ──────────────────────────────────────────────────────

LOCAL_MODE: Final[bool] = os.environ.get("AIRSAFE_LOCAL_MODE", "0") == "1"

# ── OpenRouter API ──────────────────────────────────────────────────────────

OPENROUTER_API_KEY: Final[str] = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: Final[str] = os.environ.get(
    "OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
)
