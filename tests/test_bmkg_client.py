"""Unit tests for BMKG weather client."""

from unittest.mock import MagicMock, patch

import pytest

from src.data.bmkg_client import bmkg_to_dataframe, fetch_bmkg_forecast


class TestFetchBmkgForecast:
    """Tests for fetch_bmkg_forecast()."""

    @patch("src.data.bmkg_client.BMKG_ADM4_CODES", "")
    def test_returns_empty_when_no_codes(self) -> None:
        result = fetch_bmkg_forecast()
        assert result["count"] == 0
        assert result["areas"] == []

    @patch("src.data.bmkg_client.BMKG_ADM4_CODES", "31.71.03.1001")
    @patch("src.data.bmkg_client.requests.Session")
    def test_fetches_single_area(self, mock_session_cls: MagicMock) -> None:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"lokasi": {}, "cuaca": []}]}
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        result = fetch_bmkg_forecast(adm4_codes=["31.71.03.1001"])
        assert result["count"] == 1
        assert result["ok_count"] == 1
        assert result["areas"][0]["adm4"] == "31.71.03.1001"
        assert result["areas"][0]["ok"] is True


class TestBmkgToDataframe:
    """Tests for bmkg_to_dataframe()."""

    def test_extracts_forecast_records(self) -> None:
        bmkg_data = {
            "areas": [{
                "ok": True,
                "adm4": "31.71.03.1001",
                "payload": {
                    "data": [{
                        "lokasi": {
                            "adm4": "31.71.03.1001",
                            "provinsi": "DKI Jakarta",
                            "kotkab": "Kota Adm. Jakarta Pusat",
                            "kecamatan": "Kemayoran",
                            "desa": "Kemayoran",
                            "lon": 106.845,
                            "lat": -6.165,
                        },
                        "cuaca": [
                            # Day 1: 2 slots
                            [
                                {
                                    "datetime": "2026-04-23T09:00:00Z",
                                    "utc_datetime": "2026-04-23 09:00:00",
                                    "local_datetime": "2026-04-23 16:00:00",
                                    "t": 29,
                                    "hu": 76,
                                    "ws": 5.9,
                                    "wd": "N",
                                    "wd_to": "S",
                                    "tp": 1.1,
                                    "tcc": 92,
                                    "weather": 61,
                                    "weather_desc": "Hujan Ringan",
                                    "weather_desc_en": "Light Rain",
                                    "vs": 23687,
                                    "vs_text": "> 10 km",
                                    "time_index": "8-9",
                                    "analysis_date": "2026-04-23T00:00:00",
                                    "image": "https://example.com/icon.svg",
                                },
                                {
                                    "datetime": "2026-04-23T12:00:00Z",
                                    "utc_datetime": "2026-04-23 12:00:00",
                                    "local_datetime": "2026-04-23 19:00:00",
                                    "t": 26,
                                    "hu": 88,
                                    "ws": 5.6,
                                    "wd": "S",
                                    "wd_to": "N",
                                    "tp": 0.2,
                                    "tcc": 96,
                                    "weather": 3,
                                    "weather_desc": "Berawan",
                                    "weather_desc_en": "Mostly Cloudy",
                                    "vs": 10967,
                                    "vs_text": "> 10 km",
                                    "time_index": "11-12",
                                    "analysis_date": "2026-04-23T00:00:00",
                                    "image": "https://example.com/icon2.svg",
                                },
                            ]
                        ],
                    }]
                }
            }]
        }
        records = bmkg_to_dataframe(bmkg_data)
        assert len(records) == 2
        assert records[0]["adm4"] == "31.71.03.1001"
        assert records[0]["desa"] == "Kemayoran"
        assert records[0]["weather_desc"] == "Hujan Ringan"
        assert records[0]["temp_c"] == 29
        assert records[1]["weather_desc"] == "Berawan"

    def test_skips_failed_areas(self) -> None:
        bmkg_data = {
            "areas": [{"ok": False, "adm4": "31.71.03.1001", "error": "timeout"}]
        }
        records = bmkg_to_dataframe(bmkg_data)
        assert len(records) == 0

    def test_handles_empty_data(self) -> None:
        records = bmkg_to_dataframe({"areas": []})
        assert len(records) == 0
