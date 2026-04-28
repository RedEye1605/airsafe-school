"""
Upload initial data to Azure Blob Storage.

Uploads models, reference data, historical datasets, and features to the
appropriate Blob containers. Run once to seed the production environment.

Usage:
    python scripts/upload_to_blob.py
    python scripts/upload_to_blob.py --models-only
    python scripts/upload_to_blob.py --reference-only
    python scripts/upload_to_blob.py --all
"""

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os_pre = Path(__file__).resolve().parent.parent
import os
os.environ.setdefault("AIRSAFE_ROOT", str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from src.config import (
    AIRSAFE_BLOB_CONNECTION_STRING,
    AIRSAFE_FEATURES_CONTAINER,
    AIRSAFE_MODELS_CONTAINER,
    AIRSAFE_PROCESSED_CONTAINER,
    AIRSAFE_REFERENCE_CONTAINER,
    DATA_DIR,
)
from src.data.blob_client import is_blob_configured

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _get_svc():
    from azure.storage.blob import BlobServiceClient
    return BlobServiceClient.from_connection_string(AIRSAFE_BLOB_CONNECTION_STRING)


def _upload_small(container: str, blob_name: str, file_path: Path) -> None:
    if not file_path.exists():
        logger.warning("File not found, skipping: %s", file_path)
        return
    from azure.storage.blob import ContentSettings

    data = file_path.read_bytes()
    svc = _get_svc()
    cc = svc.get_container_client(container)
    try:
        cc.get_container_properties()
    except Exception:
        cc.create_container()
        logger.info("Created container: %s", container)

    ct = "application/octet-stream"
    if blob_name.endswith(".json"):
        ct = "application/json"
    elif blob_name.endswith(".csv"):
        ct = "text/csv"

    bc = svc.get_blob_client(container=container, blob=blob_name)
    bc.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type=ct))
    logger.info("Uploaded %s → %s/%s (%.1f KB)", file_path.name, container, blob_name, len(data) / 1024)


def _upload_large(container: str, blob_name: str, file_path: Path) -> None:
    """Upload a large file using streaming with az CLI fallback."""
    if not file_path.exists():
        logger.warning("File not found, skipping: %s", file_path)
        return

    import subprocess
    from azure.storage.blob import ContentSettings

    size_mb = file_path.stat().st_size / 1024 / 1024
    logger.info("Uploading %s (%.1f MB) → %s/%s ...", file_path.name, size_mb, container, blob_name)

    # Use az CLI for large files (handles chunking automatically)
    conn_str = AIRSAFE_BLOB_CONNECTION_STRING
    result = subprocess.run(
        [
            "az", "storage", "blob", "upload",
            "--account-name", conn_str.split("AccountName=")[1].split(";")[0],
            "--account-key", conn_str.split("AccountKey=")[1].split(";")[0],
            "--container-name", container,
            "--name", blob_name,
            "--file", str(file_path),
            "--overwrite",
        ],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode == 0:
        logger.info("Uploaded %s → %s/%s (%.1f MB)", file_path.name, container, blob_name, size_mb)
    else:
        logger.error("Failed to upload %s: %s", file_path.name, result.stderr)


def upload_models() -> None:
    models_dir = _PROJECT_ROOT / "models"
    for f in ["final_lgbm_h6.pkl", "final_lgbm_h12.pkl", "final_lgbm_h24.pkl",
              "hourly_residual_corrector.pkl", "station_stats_lookup.json"]:
        _upload_small(AIRSAFE_MODELS_CONTAINER, f, models_dir / f)


def upload_reference() -> None:
    _upload_small(
        AIRSAFE_REFERENCE_CONTAINER,
        "schools/schools_geocoded.csv",
        DATA_DIR / "processed" / "schools" / "schools_geocoded.csv",
    )
    _upload_large(
        AIRSAFE_REFERENCE_CONTAINER,
        "historical/dataset_master_spku_weather.csv",
        DATA_DIR / "dataset_master_spku_weather.csv",
    )


def upload_lag_datasets() -> None:
    lag_dir = DATA_DIR / "friend_model" / "Data Lag"
    for f in ["dataset_h6.csv", "dataset_h12.csv", "dataset_h24.csv"]:
        _upload_large(AIRSAFE_REFERENCE_CONTAINER, f"lag-datasets/{f}", lag_dir / f)


def upload_features() -> None:
    _upload_small(
        AIRSAFE_FEATURES_CONTAINER,
        "schools/school_features.csv",
        DATA_DIR / "processed" / "schools" / "school_features.csv",
    )


def main():
    parser = argparse.ArgumentParser(description="Upload data to Azure Blob Storage")
    parser.add_argument("--models-only", action="store_true", help="Only upload models")
    parser.add_argument("--reference-only", action="store_true", help="Only upload reference data")
    parser.add_argument("--lag-only", action="store_true", help="Only upload lag datasets")
    parser.add_argument("--features-only", action="store_true", help="Only upload features")
    parser.add_argument("--all", action="store_true", help="Upload everything")
    args = parser.parse_args()

    if not is_blob_configured():
        logger.error("AIRSAFE_BLOB_CONNECTION_STRING not set. Check .env file.")
        sys.exit(1)

    if args.all:
        upload_models()
        upload_reference()
        upload_lag_datasets()
        upload_features()
    elif args.models_only:
        upload_models()
    elif args.reference_only:
        upload_reference()
    elif args.lag_only:
        upload_lag_datasets()
    elif args.features_only:
        upload_features()
    else:
        upload_models()
        upload_reference()
        upload_lag_datasets()
        upload_features()

    logger.info("Upload complete.")


if __name__ == "__main__":
    main()
