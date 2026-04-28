# AirSafe School

**End-to-end air quality prediction and recommendation system for schools in DKI Jakarta.**

AirSafe School collects hourly PM2.5 data from Jakarta's 5 ISPU monitoring stations, enriches it with weather data, trains LightGBM models to predict PM2.5 at 6h/12h/24h horizons, spatially interpolates predictions to 4,215 school locations using Ordinary Kriging, and generates actionable Bahasa Indonesia recommendations via LLM — all deployed as Azure Functions.

---

## Architecture

Three pipelines run as Azure Functions:

1. **ETL** (hourly) — Scrapes Rendahemisi ISPU data + Open-Meteo weather → merged dataset
2. **Predict** (daily) — LightGBM station prediction → Kriging interpolation to 4,215 schools
3. **Recommend** (on-demand) — OpenRouter LLM / template fallback → Bahasa Indonesia recommendations

```
Rendahemisi (5 stations)  ──┐
                             ├──► ETL Merge ──► Azure Blob Storage
Open-Meteo Weather ─────────┘                        │
                                                      ▼
                                    Feature Engineering (lag, rolling, temporal)
                                                      │
                                                      ▼
                                    LightGBM Prediction (h6, h12, h24)
                                    at 5 ISPU stations
                                                      │
                                                      ▼
                                    Ordinary Kriging (5 stations → 4,215 schools)
                                    + Residual Correction (LightGBM)
                                                      │
                                                      ▼
                                    Risk Classification (BMKG thresholds)
                                                      │
                                                      ▼
                                    OpenRouter LLM → Bahasa Indonesia Recommendations
```

## Project Structure

```
airsafe-school/
├── azure-functions/                # Azure Functions deployment root
│   ├── function_app.py             # ETL + Predict + Recommend functions
│   ├── host.json                   # Azure Functions runtime config
│   ├── requirements.txt            # Azure dependencies
│   ├── prompts -> ../prompts       # Symlink for LLM prompt deployment
│   ├── src -> ../src               # Symlink for source code
│   └── models -> ../models         # Symlink for ML models
│
├── src/                            # Core source code
│   ├── config.py                   # Centralized configuration (env vars)
│   ├── exceptions.py               # Custom exception hierarchy
│   ├── data/
│   │   ├── blob_client.py          # Azure Blob Storage (dual-write: local + blob)
│   │   ├── spku_client.py          # Jakarta SPKU sensor network client
│   │   ├── rendahemisi_client.py   # ISPU hourly data scraper (5 stations)
│   │   ├── bmkg_client.py          # BMKG weather forecast API
│   │   ├── weather_client.py       # Open-Meteo historical/forecast weather
│   │   ├── transforms.py           # PM2.5 risk classification, data transforms
│   │   └── school_registry.py      # School data helpers
│   ├── features/
│   │   ├── lag_features.py         # Temporal lag + rolling features for LightGBM
│   │   ├── school_features.py      # Spatial context features orchestrator
│   │   ├── elevation_features.py   # Elevation data via Open-Meteo
│   │   └── osm_features.py         # OSM road/land-use/building features
│   ├── spatial/
│   │   ├── kriging.py              # Ordinary Kriging (PyKrige) + IDW fallback
│   │   ├── hourly_kriging.py       # Per-hour Kriging pipeline
│   │   ├── lag_kriging.py          # Temporal lag Kriging pipeline
│   │   ├── residual_corrector.py   # LightGBM Kriging bias correction
│   │   ├── hourly_losocv.py        # Hourly leave-one-sensor-out CV
│   │   ├── losolocv.py            # Leave-one-sensor-out CV
│   │   └── error_map.py           # Folium error visualization map
│   ├── recommendations/
│   │   ├── engine.py               # OpenRouter LLM + template fallback
│   │   └── quality.py              # Output quality validation
│   ├── visualization/
│   │   └── coverage_map.py         # Interactive Folium coverage map
│   └── utils/
│       └── helpers.py              # JSON I/O, logging, directory helpers
│
├── models/                         # Trained ML models
│   ├── final_lgbm_h6.pkl           # LightGBM 6h PM2.5 predictor
│   ├── final_lgbm_h12.pkl          # LightGBM 12h PM2.5 predictor
│   ├── final_lgbm_h24.pkl          # LightGBM 24h PM2.5 predictor
│   ├── hourly_residual_corrector.pkl
│   └── station_stats_lookup.json
│
├── prompts/                        # LLM prompt templates
│   ├── airsafe_recommendation_system.txt
│   └── airsafe_recommendation_user.txt
│
├── data/                           # Local data storage
├── notebooks/                      # Jupyter notebooks (EDA, modelling, SHAP)
├── scripts/                        # Standalone utility scripts
├── tests/                          # Unit tests (89 passing)
└── docs/
    └── PROJECT_BRIEF.md            # Comprehensive project documentation
```

## Setup

```bash
# Clone and enter project
git clone https://github.com/RedEye1605/airsafe-school.git
cd airsafe-school

# Create virtual environment (requires uv)
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install -r azure-functions/requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — add OPENROUTER_API_KEY and AIRSAFE_BLOB_CONNECTION_STRING
```

## Running

### Run Azure Functions Locally

```bash
python azure-functions/function_app.py --local etl        # ETL pipeline
python azure-functions/function_app.py --local predict    # Predict pipeline
python azure-functions/function_app.py --local recommend  # Recommend pipeline
```

### Run Tests

```bash
pytest tests/ -q
```

## API Endpoints (Azure Functions)

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/api/etl` | Trigger ETL pipeline (scrape + merge) |
| `POST` | `/api/predict` | Trigger prediction pipeline |
| `GET`  | `/api/recommend` | Get recommendations (see query params below) |
| `GET`  | `/api/recommend?npsn=10001234` | Single school (uses OpenRouter LLM) |
| `GET`  | `/api/recommend?district=Cempaka Putih` | District filter (template-based) |
| `GET`  | `/api/recommend` | Summary view (risk counts, no LLM) |

### Timer Triggers

| Function | Schedule | Description |
|----------|----------|-------------|
| `etl_timer` | `0 5 * * * *` (hourly at :05 UTC) | Automatic ETL |
| `predict_timer` | `0 0 15 * * *` (daily 15:00 UTC) | Automatic prediction |

## Configuration

All settings from environment variables (see `.env.example` for full list):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | LLM API key for recommendations |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | LLM model |
| `AIRSAFE_BLOB_CONNECTION_STRING` | — | Azure Blob Storage connection |
| `ETL_SCHEDULE` | `0 5 * * * *` | ETL timer CRON (UTC) |
| `AIRSAFE_LOCAL_MODE` | `0` | Set `1` for local development |
| `JAKARTA_LAT` | `-6.2` | Default latitude |
| `JAKARTA_LON` | `106.85` | Default longitude |
| `COVERAGE_RADIUS_M` | `2000` | Sensor coverage radius (metres) |

## ML Models

| Model | File | Purpose |
|-------|------|---------|
| LightGBM h6 | `final_lgbm_h6.pkl` | Predict PM2.5 6 hours ahead |
| LightGBM h12 | `final_lgbm_h12.pkl` | Predict PM2.5 12 hours ahead |
| LightGBM h24 | `final_lgbm_h24.pkl` | Predict PM2.5 24 hours ahead |
| Residual Corrector | `hourly_residual_corrector.pkl` | Correct Kriging spatial bias |

Feature engineering includes: PM2.5 lags, rolling statistics (mean/std/min/max), temporal features (hour, day, month, seasonal), wind components, pollutant cross-features, and precomputed station statistics.

Spatial interpolation uses Ordinary Kriging (PyKrige) with auto-selected variogram model (spherical/exponential/gaussian/linear) via LOOCV, with IDW fallback when <3 sensors available.

## Risk Classification

PM2.5 predictions are classified using BMKG hourly thresholds:

| Category | PM2.5 Range (ug/m3) | Action |
|----------|---------------------|--------|
| BAIK | 0.0 – 15.5 | Aman (Safe) |
| SEDANG | 15.6 – 55.4 | Waspada (Caution) |
| TIDAK SEHAT | 55.5 – 150.4 | Batasi (Restrict) |
| SANGAT TIDAK SEHAT | 150.5 – 250.4 | Bahaya (Danger) |
| BERBAHAYA | > 250.4 | Bahaya (Danger) |

## Tech Stack

- **Python 3.11** — pandas, numpy, lightgbm, pykrige, scikit-learn
- **Azure Functions** — Serverless pipeline (Python v2 model, East Asia)
- **Azure Blob Storage** — Dual-write data persistence (8 containers)
- **OpenRouter** — LLM API proxy (nvidia/nemotron free tier)
- **Open-Meteo** — Free historical/forecast weather (ERA5 reanalysis)
- **Rendahemisi** — Jakarta ISPU hourly pollutant data (5 stations)
- **PyKrige** — Ordinary Kriging spatial interpolation
- **GitHub Actions** — CI/CD deployment to Azure Functions

## Team

- **Rhendy Japelhendal Saragih Sumbayak** — Data Engineering, Backend, Azure Infrastructure
- **Aditya** — ML Modeling (LightGBM, feature engineering, SHAP)
- **Aufi** — BI / Visualization

## Documentation

Full project documentation: [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md)

## License

[MIT](LICENSE)
