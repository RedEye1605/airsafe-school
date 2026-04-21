"""Tests for configuration module."""

import os
from pathlib import Path

from src.config import (
    COVERAGE_RADIUS_M,
    DATA_DIR,
    ISPU_STATION_CITY,
    JAKARTA_CENTER,
    JAKARTA_LAT,
    JAKARTA_LON,
    PROJECT_ROOT,
    SPKU_API_URL,
)


def test_project_root_is_path() -> None:
    assert isinstance(PROJECT_ROOT, Path)


def test_data_dir_under_project() -> None:
    assert str(DATA_DIR).startswith(str(PROJECT_ROOT))


def test_jakarta_coords_reasonable() -> None:
    assert -7.0 < JAKARTA_LAT < -5.0
    assert 106.0 < JAKARTA_LON < 108.0


def test_ispu_stations_five() -> None:
    assert len(ISPU_STATION_CITY) == 5


def test_coverage_radius_positive() -> None:
    assert COVERAGE_RADIUS_M > 0


def test_spku_url_is_https() -> None:
    assert SPKU_API_URL.startswith("https://")
