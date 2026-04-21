# 🏫 AirSafe School

**Air quality monitoring and analysis system for schools in DKI Jakarta.**

AirSafe School integrates real-time SPKU sensor data, Open-Meteo weather
forecasts, and government school registries to assess and visualise PM2.5
exposure risks at school locations across Jakarta.

---

## Features

- **SPKU Data Pipeline** — Fetches and cleans real-time PM2.5 readings from
  Jakarta's air quality sensor network.
- **Weather Integration** — Historical (ERA5) and forecast weather via the
  free Open-Meteo API.
- **School Registry** — Downloads and geocodes DKI Jakarta school data.
- **Coverage Gap Analysis** — Interactive Folium maps showing which schools
  fall within/outside sensor coverage radii.
- **ISPU Risk Classification** — Automatic PM2.5 → ISPU category mapping.
- **Azure Functions Backend** — Serverless timer-triggered ETL for
  automated data collection.

## Project Structure

```
airsafe-school/
├── src/                        # Core Python package
│   ├── __init__.py
│   ├── config.py               # Centralised configuration
│   ├── exceptions.py           # Custom exception classes
│   ├── data/
│   │   ├── spku_client.py      # SPKU API client
│   │   ├── weather_client.py   # Open-Meteo weather client
│   │   ├── school_registry.py  # School data helpers
│   │   └── transforms.py       # Data cleaning & aggregation
│   ├── features/               # Feature engineering (ML)
│   ├── visualization/
│   │   └── coverage_map.py     # Interactive Folium maps
│   └── utils/
│       ├── config.py           # (alias for src.config)
│       └── helpers.py          # File I/O, logging setup
├── azure-functions/            # Azure Functions app
│   ├── function_app.py         # Timer & HTTP triggers
│   ├── etl/                    # Re-exports from src.data
│   └── requirements.txt
├── scripts/                    # Runnable one-off scripts
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── .env.example                # Environment variable template
├── pyproject.toml              # Project metadata & build config
├── requirements.txt            # Production dependencies
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/RedEye1605/airsafe-school.git
cd airsafe-school

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# For development (tests, linting)
pip install -r requirements.txt pytest ruff
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env with your settings (optional — defaults work out of the box)
```

## Usage

### Run the ETL Pipeline

```bash
# Local ETL run (no Azure Functions required)
AIRSAFE_LOCAL_MODE=1 python azure-functions/function_app.py --local

# Or use individual clients
python -c "from src.data.spku_client import fetch_all_stations; print(fetch_all_stations()['active_pm25'])"
```

### Generate Coverage Map

```python
from src.visualization.coverage_map import build_coverage_map

output = build_coverage_map()
print(f"Map saved to: {output}")
```

### Run Tests

```bash
python -m pytest tests/ -v
```

## Configuration

All settings are read from environment variables with sensible defaults.
See `.env.example` for the full list. Key settings:

| Variable | Default | Description |
|---|---|---|
| `SPKU_API_URL` | Jakarta SPKU endpoint | Air quality API URL |
| `SPKU_STALE_DAYS` | `7` | Days before station marked inactive |
| `JAKARTA_LAT` | `-6.2` | Default latitude for weather queries |
| `JAKARTA_LON` | `106.85` | Default longitude for weather queries |
| `COVERAGE_RADIUS_M` | `2000` | Sensor coverage radius in metres |

## Tech Stack

- **Python 3.10+** — pandas, numpy, requests, folium
- **Azure Functions** — Serverless ETL pipeline
- **Open-Meteo API** — Free weather data (no API key)
- **SPKU Jakarta** — Real-time air quality sensors

## Team

Competition project for a data analytics challenge.

## License

[MIT](LICENSE)
