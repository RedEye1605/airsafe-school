"""Tests for AirSafe School data transforms."""

import pandas as pd
import pytest

from src.data.transforms import (
    classify_pm25_risk,
    enrich_spku_with_risk,
    risk_to_color,
    spku_to_dataframe,
    weather_daily_to_dataframe,
    weather_hourly_to_dataframe,
)


# ── classify_pm25_risk ────────────────────────────────────────────────────


class TestClassifyPm25Risk:
    """Tests for ISPU PM2.5 risk classification."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0, "BAIK"),
            (35, "BAIK"),
            (36, "SEDANG"),
            (75, "SEDANG"),
            (76, "TIDAK SEHAT"),
            (115, "TIDAK SEHAT"),
            (116, "SANGAT TIDAK SEHAT"),
            (150, "SANGAT TIDAK SEHAT"),
            (151, "BERBAHAYA"),
            (500, "BERBAHAYA"),
        ],
    )
    def test_risk_categories(self, value: float, expected: str) -> None:
        assert classify_pm25_risk(value) == expected

    def test_nan_returns_no_data(self) -> None:
        assert classify_pm25_risk(float("nan")) == "TIDAK ADA DATA"


# ── risk_to_color ─────────────────────────────────────────────────────────


class TestRiskToColor:
    """Tests for risk-to-colour mapping."""

    def test_known_risk(self) -> None:
        assert risk_to_color("BAIK") == "#22c55e"

    def test_unknown_risk_returns_grey(self) -> None:
        assert risk_to_color("UNKNOWN") == "#9ca3af"


# ── spku_to_dataframe ────────────────────────────────────────────────────


class TestSpkuToDataframe:
    """Tests for SPKU record cleaning."""

    def test_empty_list(self) -> None:
        df = spku_to_dataframe([])
        assert df.empty

    def test_valid_records(self) -> None:
        records = [
            {"latitude": -6.2, "longitude": 106.8, "pm25": 42.0},
            {"latitude": -6.3, "longitude": 106.9, "pm25": 80.0},
        ]
        df = spku_to_dataframe(records)
        assert len(df) == 2

    def test_filters_zero_coords(self) -> None:
        records = [{"latitude": 0, "longitude": 0, "pm25": 42.0}]
        df = spku_to_dataframe(records)
        assert df.empty

    def test_filters_negative_pm25(self) -> None:
        records = [{"latitude": -6.2, "longitude": 106.8, "pm25": -1}]
        df = spku_to_dataframe(records)
        assert df.empty

    def test_filters_extreme_pm25(self) -> None:
        records = [{"latitude": -6.2, "longitude": 106.8, "pm25": 999}]
        df = spku_to_dataframe(records)
        assert df.empty


# ── weather_hourly_to_dataframe ───────────────────────────────────────────


class TestWeatherHourlyToDataframe:
    """Tests for Open-Meteo hourly data conversion."""

    def test_empty_dict(self) -> None:
        df = weather_hourly_to_dataframe({})
        assert df.empty

    def test_rename_columns(self) -> None:
        data = {
            "time": ["2024-01-01T00:00"],
            "temperature_2m": [28.5],
            "relative_humidity_2m": [75.0],
            "wind_speed_10m": [3.2],
            "precipitation": [0.0],
            "surface_pressure": [1013.0],
        }
        df = weather_hourly_to_dataframe(data)
        assert "temperature" in df.columns
        assert "humidity" in df.columns
        assert len(df) == 1


# ── weather_daily_to_dataframe ────────────────────────────────────────────


class TestWeatherDailyToDataframe:
    """Tests for Open-Meteo daily data conversion."""

    def test_empty_dict(self) -> None:
        df = weather_daily_to_dataframe({})
        assert df.empty

    def test_rename_columns(self) -> None:
        data = {
            "time": ["2024-01-01"],
            "temperature_2m_max": [32.0],
            "temperature_2m_min": [25.0],
        }
        df = weather_daily_to_dataframe(data)
        assert "temp_max" in df.columns
        assert "temp_min" in df.columns


# ── enrich_spku_with_risk ────────────────────────────────────────────────


class TestEnrichSpkuWithRisk:
    """Tests for SPKU DataFrame enrichment."""

    def test_adds_risk_columns(self) -> None:
        df = pd.DataFrame({"pm25": [30, 60, 120]})
        result = enrich_spku_with_risk(df)
        assert "risk_level" in result.columns
        assert "risk_color" in result.columns
        assert result["risk_level"].tolist() == ["BAIK", "SEDANG", "TIDAK SEHAT"]

    def test_does_not_mutate_input(self) -> None:
        df = pd.DataFrame({"pm25": [30]})
        original_cols = set(df.columns)
        enrich_spku_with_risk(df)
        assert set(df.columns) == original_cols
