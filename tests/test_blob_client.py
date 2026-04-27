"""Unit tests for Blob Storage client."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.blob_client import (
    is_blob_configured,
    save_dataframe_dual,
    save_json_dual,
    upload_dataframe,
    upload_json,
    upload_text,
)


class TestIsBlobConfigured:
    """Tests for is_blob_configured()."""

    @patch("src.data.blob_client.AIRSAFE_BLOB_CONNECTION_STRING", "")
    def test_returns_false_when_empty(self) -> None:
        assert is_blob_configured() is False

    @patch("src.data.blob_client.AIRSAFE_BLOB_CONNECTION_STRING", "DefaultEndpointsProtocol=https;AccountName=test")
    def test_returns_true_when_set(self) -> None:
        assert is_blob_configured() is True


class TestSaveJsonDual:
    """Tests for save_json_dual() — local-only mode."""

    @patch("src.data.blob_client.AIRSAFE_BLOB_CONNECTION_STRING", "")
    def test_saves_locally(self, tmp_path: Path) -> None:
        local = tmp_path / "output" / "test.json"
        result = save_json_dual({"key": "value"}, "raw/test.json", local)
        assert local.exists()
        assert json.loads(local.read_text()) == {"key": "value"}
        assert str(local) in result

    @patch("src.data.blob_client.AIRSAFE_BLOB_CONNECTION_STRING", "")
    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        local = tmp_path / "deep" / "nested" / "test.json"
        save_json_dual({"a": 1}, "raw/test.json", local)
        assert local.exists()


class TestSaveDataframeDual:
    """Tests for save_dataframe_dual() — local-only mode."""

    @patch("src.data.blob_client.AIRSAFE_BLOB_CONNECTION_STRING", "")
    def test_saves_csv_locally(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        local = tmp_path / "data.csv"
        result = save_dataframe_dual(df, "processed/data.csv", local)
        assert local.exists()
        loaded = pd.read_csv(local)
        assert list(loaded.columns) == ["x", "y"]
        assert len(loaded) == 2
