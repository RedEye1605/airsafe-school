# AirSafe School — Project Brief

## 1. Project Overview

**AirSafe School** is an end-to-end air quality monitoring and recommendation system for schools in Jakarta, Indonesia. It combines real-time air quality data collection, machine learning prediction, spatial interpolation, and LLM-powered recommendations to help school administrators make informed decisions about outdoor activities based on PM2.5 forecasts.

**Team:** Rhendy Japelhendal Saragih Sumbayak (Data Engineering / Backend / Azure), Aditya (ML Modeling), Aufi (BI / Visualization)

**Deployment:** Azure Functions (Python v2 model) on East Asia region — `func-airsafe-dev-01`

---

## 2. Problem Statement

Jakarta experiences hazardous air quality episodes where PM2.5 levels exceed safe thresholds. Schools lack real-time, actionable guidance on whether to hold outdoor activities, sports, or flag ceremonies. Existing monitoring covers only 5 ISPU stations — there is no school-level data for the ~4,215 schools across Jakarta's 6 administrative regions.

AirSafe School bridges this gap by:
1. Collecting hourly pollutant data from 5 Jakarta monitoring stations
2. Predicting PM2.5 levels at 6h, 12h, and 24h horizons using LightGBM
3. Interpolating predictions to 4,215 school locations using Ordinary Kriging
4. Generating actionable recommendations in Bahasa Indonesia via LLM

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES (External)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │  Rendahemisi  │  │  Open-Meteo   │  │    BMKG      │  │    SPKU   │  │
│  │  (ISPU Hourly)│  │  (Weather)    │  │  (Forecast)  │  │  (Sensor) │  │
│  │  5 stations   │  │  Historical   │  │  3-day       │  │  Network  │  │
│  │  Web scrape   │  │  + Forecast   │  │  API         │  │  API      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘  └───────────┘  │
│         │                  │                                           │
└─────────┼──────────────────┼───────────────────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ETL PIPELINE (Azure Function)                   │
│                                                                         │
│  Step 1: Rendahemisi Scrape      → PM2.5, PM10, SO2, CO, O3, NO2, HC  │
│          (72h lookback, 5 stations)                                     │
│                                                                         │
│  Step 2: Open-Meteo Weather       → Temperature, Humidity, Wind, etc.  │
│          (per-station, 3-day historical)                                │
│                                                                         │
│  Step 3: Merge + Temporal Cols    → datetime, hour, day, weekend flags  │
│          → dataset_master_spku_weather format                           │
│                                                                         │
│  Output: merged CSV → Azure Blob (raw + processed containers)           │
│  Schedule: Every hour at :05 UTC                                        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PREDICT PIPELINE (Azure Function)                  │
│                                                                         │
│  Step 1: Load LightGBM Models     → h6, h12, h24 (3 horizon models)   │
│          + Residual Corrector     → LightGBM systematic error model    │
│                                                                         │
│  Step 2: Feature Engineering      → Lag features, rolling stats,       │
│          (build_prediction_         temporal features, wind components  │
│           features)                                                     │
│                                                                         │
│  Step 3: Station Prediction       → LightGBM predict at 5 ISPU stations│
│                                                                         │
│  Step 4: Spatial Interpolation    → Ordinary Kriging (PyKrige)         │
│          (kriging_interpolate)       5 stations → 4,215 schools        │
│          + Residual Correction     → LightGBM corrects Kriging bias    │
│                                                                         │
│  Step 5: Risk Classification      → BMKG hourly thresholds             │
│          (classify_pm25_hourly)       BAIK/SEDANG/TIDAK SEHAT/BAHAYA   │
│                                                                         │
│  Output: predictions JSON → Azure Blob (predictions container)          │
│  Schedule: Daily at 15:00 UTC                                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    RECOMMEND PIPELINE (Azure Function)                   │
│                                                                         │
│  Load predictions + school metadata (4,215 schools)                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Single School (npsn=?):                                     │        │
│  │    OpenRouter LLM → JSON recommendation                      │        │
│  │    (Bahasa Indonesia, risk-specific action items)             │        │
│  │    Fallback: Template-based if LLM unavailable               │        │
│  ├─────────────────────────────────────────────────────────────┤        │
│  │  District Filter (district=?):                               │        │
│  │    Template-based for all matching schools                   │        │
│  │    (avoids LLM rate limits on bulk queries)                  │        │
│  ├─────────────────────────────────────────────────────────────┤        │
│  │  Summary (no params):                                        │        │
│  │    Direct computation — risk counts across all schools        │        │
│  │    (no LLM call, instant response)                           │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                         │
│  Output: GET /recommend → JSON response                                 │
│  No timer trigger — HTTP-only, on-demand                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Pipeline — Detailed Documentation

### 4.1 ETL Pipeline

#### Purpose
Collects hourly multi-pollutant data from Jakarta's 5 ISPU monitoring stations and enriches it with weather data from Open-Meteo. Produces a merged dataset matching the format used for ML model training.

#### Data Sources

| Source | Type | Coverage | Frequency |
|--------|------|----------|-----------|
| Rendahemisi (Jakarta Provincial Gov) | Web scraper | 5 stations, 72h lookback | Per ETL run |
| Open-Meteo Archive API | REST API | Per-station, 3-day historical | Per ETL run |
| BMKG Weather Forecast | REST API | 3-day forecast | Available but not used in current ETL |
| SPKU Sensor Network | REST API | Jakarta-wide sensor network | Available but not used in current ETL |

#### Step-by-Step Flow

```
1. RENDAHEMISI SCRAPE
   URL: https://rendahemisi.jakarta.go.id/ispu-detail/{id}/{slug}/{date}
   Stations:
     - DKI1: Bundaran HI (id=4, slug=bundaran-hi)
     - DKI2: Kelapa Gading (id=5, slug=kelapa-gading)
     - DKI3: Jagakarsa (id=6, slug=jagakarsa)
     - DKI4: Lubang Buaya (id=7, slug=lubang-buaya)
     - DKI5: Kebun Jeruk (id=8, slug=kebun-jeruk)

   Fetches: Last 72 hours of hourly readings
   Pollutants: PM2.5, PM10, SO2, CO, O3, NO2, HC, Category
   Method: requests + BeautifulSoup HTML table parsing
   Retry: Up to 3 attempts with exponential backoff (1s, 2s, 4s)

   Output: DataFrame with columns:
     datetime, date, hour, station_id, station_slug, station_name,
     lokasi, pm25, pm10, so2, co, o3, no2, hc, kategori, last_update, source_url

2. OPEN-METEO WEATHER FETCH
   URL: https://archive-api.open-meteo.com/v1/archive
   Per-station coordinates (matching ISPU station locations):
     - dki1-bundaran-hi:   (-6.1931, 106.8230)
     - dki2-kelapa-gading: (-6.1586, 106.9050)
     - dki3-jagakarsa:     (-6.3346, 106.8228)
     - dki4-lubang-buaya:  (-6.2908, 106.9019)
     - dki5-kebun-jeruk:   (-6.1951, 106.7694)

   Variables: temperature_2m, relative_humidity_2m, precipitation, rain,
              surface_pressure, wind_speed_10m, wind_direction_10m
   Range: 3 days back from current date
   Retry: 2 attempts on server errors (5xx), 5s delay

   Output: Per-station DataFrame with hourly weather aligned to pollutant timestamps

3. DATA MERGE
   Join: LEFT JOIN on (station_slug, station_name, datetime)
   Datetime alignment: Both sides floored to hourly precision
   Additional columns:
     - date (normalized datetime)
     - year, month, day, hour_num
     - dayofweek (0=Monday), is_weekend (Sat/Sun = 1)

   Output: dataset_master_spku_weather format (Adit's training format)

4. STORAGE
   Files written (dual-write: local filesystem + Azure Blob):
     - raw/spku/date=YYYY-MM-DD/spku_TIMESTAMP.csv          → Raw container
     - processed/daily/dataset_master_spku_weather_DATE.csv   → Processed container
     - processed/daily/dataset_master_spku_weather_latest.csv → Processed container
     - logs/etl/date=YYYY-MM-DD/run_TIMESTAMP.json           → Logs container
```

#### Azure Function Endpoints

| Trigger | Type | Schedule/Route | Description |
|---------|------|----------------|-------------|
| `etl_timer` | Timer | `0 5 * * * *` (every hour :05 UTC) | Automatic ETL execution |
| `etl_http` | HTTP | `POST /api/etl` | Manual trigger, returns manifest |

---

### 4.2 Predict Pipeline

#### Purpose
Loads trained LightGBM models, builds prediction features from the latest ETL data, predicts PM2.5 at 5 ISPU stations, then spatially interpolates to all 4,215 Jakarta schools using Ordinary Kriging with optional residual correction.

#### Model Architecture

```
Training Pipeline (offline, in notebooks):
  Historical ISPU data (2010-2021) + Weather → Feature Engineering → LightGBM

Production Models:
  ├── final_lgbm_h6.pkl   — 6-hour PM2.5 forecast
  ├── final_lgbm_h12.pkl  — 12-hour PM2.5 forecast
  ├── final_lgbm_h24.pkl  — 24-hour PM2.5 forecast
  └── hourly_residual_corrector.pkl — LightGBM correction for Kriging bias
```

#### Feature Engineering

The `build_prediction_features()` function in `src/features/lag_features.py` produces:

| Feature Category | Examples | Count |
|------------------|----------|-------|
| PM2.5 Lags | pm25_lag_1h, pm25_lag_3h, pm25_lag_6h | Horizon-dependent |
| Pollutant Lags | pm10_lag_1h, co_lag_1h, so2_lag_1h | Per pollutant |
| Weather Lags | temperature_lag_1h, humidity_lag_1h | Per weather var |
| Rolling Stats | pm25_rolling_mean_6h, pm25_rolling_std_12h | Mean, std, min, max |
| Temporal | hour_sin, hour_cos, month_sin, month_cos, is_weekend | Cyclical encoding |
| Wind | wind_speed, wind_dir_sin, wind_dir_cos | Decomposed |
| Station Stats | pm25_station_mean, pm25_station_std (from JSON lookup) | Precomputed |

Configuration per horizon (`HORIZON_CONFIG`):
- **h6**: Short lags (1-6h), rolling windows (3-6h)
- **h12**: Medium lags (1-12h), rolling windows (6-12h)
- **h24**: Long lags (1-24h), rolling windows (12-24h)

#### Spatial Interpolation

```
Station Predictions (5 points)          School Predictions (4,215 points)
┌─────────────────────┐                 ┌─────────────────────────────┐
│ DKI1 Bundaran HI    │                 │ SDN Cempaka Putih 01        │
│ DKI2 Kelapa Gading  │  ──Kriging──►   │ SMPN 5 Jakarta              │
│ DKI3 Jagakarsa      │                 │ SMAN 1 Jakarta              │
│ DKI4 Lubang Buaya   │                 │ ... (4,215 schools total)   │
│ DKI5 Kebun Jeruk    │                 └─────────────────────────────┘
└─────────────────────┘
         │                                        │
         │  Ordinary Kriging                      │  Residual Correction
         │  (PyKrige)                             │  (LightGBM)
         │  - Variogram: spherical/exponential/   │  - Learns systematic
         │    gaussian/linear (auto-selected)      │    spatial patterns
         │  - nlags=4, coordinates=geographic     │    missed by variogram
         │  - Fallback: IDW if <3 sensors         │  - Features: distance,
         │                                         │    density, variogram type
         ▼                                         ▼
    pm25_kriging                              pm25_corrected
    kriging_variance                          = pm25_kriging + predicted_residual
    kriging_std
    variogram_model
```

**Kriging Configuration:** `KrigingConfig(nlags=4, min_sensors=3, max_output_pm25=1000.0)`
- Requires minimum 3 sensors for Kriging; falls back to IDW below that
- Auto-selects best variogram model via LOOCV (lowest cR statistic)
- Clips output to max 1000 µg/m³

**Residual Correction:** Optional two-stage correction using `ResidualCorrector`
- Trained on LOSOCV residuals from historical data
- Adds `pm25_corrected = pm25_kriging + predicted_residual`
- 15 features including sensor distances, density, variogram type

#### Risk Classification

Uses BMKG hourly thresholds (from `src/data/transforms.py`):

| Category | PM2.5 Range (µg/m³) | Action |
|----------|---------------------|--------|
| BAIK | 0.0 – 15.5 | Aman (Safe) |
| SEDANG | 15.6 – 55.4 | Waspada (Caution) |
| TIDAK SEHAT | 55.5 – 150.4 | Batasi (Restrict) |
| SANGAT TIDAK SEHAT | 150.5 – 250.4 | Bahaya (Danger) |
| BERBAHAYA | > 250.4 | Bahaya (Danger) |

#### Output Schema

```json
{
  "timestamp_utc": "2026-04-28T12:05:00Z",
  "timestamp_wib": "2026-04-28T19:05:00+07:00",
  "station_predictions": [
    {"station": "DKI1 Bundaran HI", "slug": "dki1-bundaran-hi", "horizon": 6, "pm25_predicted": 42.3}
  ],
  "n_schools": 4215,
  "school_predictions": [
    {
      "npsn": "10001234",
      "latitude": -6.1862,
      "longitude": 106.8352,
      "pm25_h6": 38.2,
      "pm25_h12": 41.5,
      "pm25_h24": 35.1,
      "risk_h6": "SEDANG",
      "risk_h12": "SEDANG",
      "risk_h24": "SEDANG"
    }
  ]
}
```

#### Azure Function Endpoints

| Trigger | Type | Schedule/Route | Description |
|---------|------|----------------|-------------|
| `predict_timer` | Timer | `0 0 15 * * *` (daily 15:00 UTC) | Automatic prediction |
| `predict_http` | HTTP | `POST /api/predict` | Manual trigger, returns manifest |

---

### 4.3 Recommend Pipeline

#### Purpose
Transforms PM2.5 predictions into actionable, Bahasa Indonesia recommendations for school administrators and parents. Uses a three-tier strategy to balance quality, speed, and cost.

#### Recommendation Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│  GET /api/recommend                                             │
│                                                                 │
│  Query Param: npsn=10001234 (single school)                    │
│  ─────────────────────────────────────                          │
│  → OpenRouter LLM (nvidia/nemotron-3-super-120b-a12b:free)     │
│    - System prompt: AirSafe persona, Bahasa Indonesia           │
│    - User prompt: School data + PM2.5 values + risk level       │
│    - Temperature: 0.3, JSON response format                     │
│    - Timeout: 30s                                               │
│  → Template fallback if LLM fails                               │
│                                                                 │
│  Query Param: district=Cempaka Putih (district filter)          │
│  ──────────────────────────────────────────────────────          │
│  → Template-based for ALL matching schools                      │
│    - Predefined policies per risk level                          │
│    - Avoids LLM rate limits on bulk queries (~10-50 schools)    │
│                                                                 │
│  No params (summary view)                                       │
│  ──────────────────────                                          │
│  → Direct computation from predictions JSON                     │
│    - Risk counts across all 4,215 schools                       │
│    - Station prediction summary                                 │
│    - No LLM call — instant response                             │
└─────────────────────────────────────────────────────────────────┘
```

#### Recommendation Output Schema

```json
{
  "school_id": "10001234",
  "school_name": "SDN Cempaka Putih 01",
  "district": "Cempaka Putih",
  "risk_level": "Waspada",
  "pm25_summary": "PM2.5 di sekitar SDN Cempaka Putih 01 diprediksi 38.2 µg/m³ dalam 6 jam, 41.5 µg/m³ dalam 12 jam, dan 35.1 µg/m³ dalam 24 jam.",
  "headline": "Waspada kualitas udara, kurangi paparan luar ruang yang terlalu lama",
  "recommendation": "...",
  "action_items": [
    "Kurangi durasi aktivitas luar ruang yang panjang.",
    "Pantau siswa yang memiliki sensitivitas pernapasan.",
    "Siapkan alternatif kegiatan di dalam ruangan jika kondisi memburuk."
  ],
  "reasoning_summary": "Faktor model yang mendukung rekomendasi ini meliputi: ...",
  "parent_message": "Kualitas udara di sekitar SDN Cempaka Putih 01 dalam tingkat waspada. ...",
  "generation_mode": "openrouter"
}
```

#### Risk-to-Action Mapping

| BMKG Label | Action Label | Policy |
|------------|-------------|--------|
| BAIK | Aman | Outdoor activities normal, monitor air quality |
| SEDANG | Waspada | Reduce long outdoor exposure, monitor sensitive students |
| TIDAK SEHAT | Batasi | Move sports indoors, avoid outdoor ceremonies, masks for sensitive students |
| SANGAT TIDAK SEHAT | Bahaya | Cancel all outdoor activities, keep students indoors, masks mandatory, notify parents |
| BERBAHAYA | Bahaya | Same as above |

#### Quality Validation (`src/recommendations/quality.py`)

Every recommendation is validated against:
1. All `REQUIRED_OUTPUT_KEYS` present
2. `school_name` matches input
3. `risk_level` matches expected BMKG mapping
4. PM2.5 values mentioned in output text
5. `action_items` is a list with ≥3 items
6. No medical diagnosis terms (diagnosis, obat, terapi medis, menyembuhkan, penyakit pasti)

#### Azure Function Endpoints

| Trigger | Type | Route | Description |
|---------|------|-------|-------------|
| `recommend_http` | HTTP | `GET /api/recommend` | Query params: `npsn`, `district` |
| (no timer) | — | — | On-demand only |

---

## 5. Azure Infrastructure

### 5.1 Azure Functions Configuration

| Setting | Value |
|---------|-------|
| Function App Name | `func-airsafe-dev-01` |
| Region | East Asia |
| Runtime | Python 3.11 |
| Model | Azure Functions v2 (Python) |
| SKU | Flex Consumption |
| Concurrency | Dynamic (enabled) |
| Application Insights | Enabled (sampling) |

### 5.2 Azure Blob Storage

**Account:** `airsafedata01`

**Containers:**

| Container | Purpose | Key Paths |
|-----------|---------|-----------|
| `raw` | Raw scraped data | `spku/date=YYYY-MM-DD/spku_TIMESTAMP.csv` |
| `processed` | Cleaned/merged datasets | `daily/dataset_master_spku_weather_*.csv` |
| `predictions` | ML outputs | `daily/predict_*.json`, `daily/recommend_*.json` |
| `logs` | Pipeline manifests | `etl/date=YYYY-MM-DD/run_TIMESTAMP.json` |
| `reference` | Static reference data | `schools/schools_geocoded.csv`, `historical/dataset_master_spku_weather.csv` |
| `models` | Model artifacts | (backup, primarily deployed with function) |
| `features` | Feature datasets | (computed on-demand) |
| `scratch` | Temporary files | (ephemeral) |

### 5.3 Storage Pattern: Dual Write

All pipeline outputs use a dual-write pattern via `save_json_dual()` and `save_dataframe_dual()`:
1. **Always** write to local filesystem first (guaranteed)
2. **If Blob configured**, also upload to Azure Blob Storage
3. If Blob upload fails, log the error and continue (local file is still valid)
4. On read: try local first, fall back to Blob download if local missing

### 5.4 Deployment Pipeline

```
GitHub (main branch) → GitHub Actions → Azure Functions (flexconsumption)
                         │
                         ├── Checkout
                         ├── Setup Python 3.11
                         ├── Install deps (requirements.txt)
                         ├── Stage: function_app.py + host.json + requirements.txt
                         │         + src/ + models/ + prompts/ (symlinked)
                         ├── Zip artifact
                         └── Deploy via Azure/functions-action
```

**Trigger:** Push to `main` branch or manual workflow dispatch

### 5.5 Environment Variables (App Settings)

| Variable | Purpose | Example |
|----------|---------|---------|
| `AIRSAFE_BLOB_CONNECTION_STRING` | Azure Blob Storage auth | `DefaultEndpointsProtocol=https;AccountName=airsafedata01;...` |
| `AIRSAFE_RAW_CONTAINER` | Raw data container name | `raw` |
| `AIRSAFE_PROCESSED_CONTAINER` | Processed data container | `processed` |
| `AIRSAFE_PREDICT_CONTAINER` | Predictions container | `predictions` |
| `AIRSAFE_LOG_CONTAINER` | Logs container | `logs` |
| `AIRSAFE_REFERENCE_CONTAINER` | Reference data container | `reference` |
| `ETL_SCHEDULE` | ETL timer CRON | `0 5 * * * *` |
| `OPENROUTER_API_KEY` | LLM API key | `sk-or-v1-...` |
| `OPENROUTER_MODEL` | LLM model | `nvidia/nemotron-3-super-120b-a12b:free` |
| `AIRSAFE_LOCAL_MODE` | Local development flag | `0` or `1` |

---

## 6. Codebase Structure

```
airsafe-school/
├── azure-functions/                    # Azure Functions deployment root
│   ├── function_app.py                 # Main app: ETL, Predict, Recommend functions
│   ├── host.json                       # Azure Functions runtime config
│   ├── requirements.txt                # Python dependencies for Azure
│   ├── prompts -> ../prompts           # Symlink for prompt deployment
│   ├── src -> ../src                   # Symlink for source code
│   ├── models -> ../models             # Symlink for ML models
│   └── etl/                            # Azure-specific ETL helpers (unused, kept for reference)
│
├── src/                                # Core source code
│   ├── config.py                       # Centralized configuration (env vars)
│   ├── exceptions.py                   # Custom exception hierarchy
│   ├── data/                           # Data acquisition & storage
│   │   ├── blob_client.py              # Azure Blob Storage client (dual-write)
│   │   ├── spku_client.py              # Jakarta SPKU sensor network client
│   │   ├── rendahemisi_client.py       # ISPU hourly data scraper (5 stations)
│   │   ├── bmkg_client.py              # BMKG weather forecast API client
│   │   ├── weather_client.py           # Open-Meteo weather API client
│   │   ├── transforms.py              # PM2.5 risk classification, data transforms
│   │   └── school_registry.py          # School data helpers (placeholder)
│   ├── features/                       # Feature engineering
│   │   ├── lag_features.py             # Temporal lag + rolling features for ML
│   │   ├── school_features.py          # Spatial context features orchestrator
│   │   ├── elevation_features.py       # Elevation data via Open-Meteo
│   │   └── osm_features.py             # OSM road/land-use/building features
│   ├── spatial/                        # Spatial interpolation
│   │   ├── kriging.py                  # Ordinary Kriging (PyKrige) + IDW fallback
│   │   ├── hourly_kriging.py           # Per-hour Kriging pipeline
│   │   ├── lag_kriging.py              # Temporal lag Kriging pipeline
│   │   ├── residual_corrector.py       # LightGBM Kriging bias correction
│   │   ├── hourly_losocv.py            # Hourly leave-one-sensor-out CV
│   │   ├── losolocv.py                # Leave-one-sensor-out CV
│   │   └── error_map.py               # Folium error visualization map
│   ├── recommendations/                # Recommendation engine
│   │   ├── engine.py                   # OpenRouter LLM + template fallback
│   │   └── quality.py                  # Output quality validation
│   ├── utils/                          # Shared utilities
│   │   └── helpers.py                  # JSON I/O, logging setup, directory helpers
│   └── visualization/                  # Map visualizations
│       └── coverage_map.py             # School coverage gap map (folium)
│
├── models/                             # Trained ML models
│   ├── final_lgbm_h6.pkl               # LightGBM 6h PM2.5 predictor
│   ├── final_lgbm_h12.pkl              # LightGBM 12h PM2.5 predictor
│   ├── final_lgbm_h24.pkl              # LightGBM 24h PM2.5 predictor
│   ├── hourly_residual_corrector.pkl    # Kriging residual correction model
│   └── station_stats_lookup.json        # Precomputed station statistics
│
├── prompts/                            # LLM prompt templates
│   ├── airsafe_recommendation_system.txt   # System prompt (persona + rules)
│   └── airsafe_recommendation_user.txt     # User prompt template (data slots)
│
├── data/                               # Local data storage
│   ├── raw/                            # Raw scraped data
│   │   ├── spku/                       # SPKU snapshots + station catalog
│   │   ├── ispu/                       # Historical ISPU data (2010-2021)
│   │   ├── schools/                    # School registries (SD, SMP, SMA/SMK)
│   │   └── weather/                    # Weather archives
│   ├── processed/                      # Processed/pipeline outputs
│   │   ├── daily/                      # ETL outputs + predictions + recommendations
│   │   ├── schools/                    # Geocoded school data + features
│   │   └── hourly_losocv_*             # Cross-validation results
│   └── test/                           # Test outputs
│
├── tests/                              # Unit tests (89 passing)
│   ├── test_blob_client.py
│   ├── test_bmkg_client.py
│   ├── test_config.py
│   ├── test_helpers.py
│   ├── test_hourly_kriging.py
│   ├── test_hourly_losocv.py
│   ├── test_kriging.py
│   ├── test_lag_kriging.py
│   ├── test_losocv.py
│   ├── test_recommendations.py
│   ├── test_residual_corrector.py
│   ├── test_school_features.py
│   ├── test_spku_client.py
│   └── test_transforms.py
│
├── notebooks/                          # Jupyter notebooks
│   ├── Scraper.ipynb                   # Initial SPKU exploration
│   ├── prepo-eda.ipynb                 # Exploratory data analysis
│   ├── prepo-modelling-data.ipynb      # Feature engineering + training data prep
│   ├── modelling-baseline.ipynb        # Baseline model experiments
│   ├── modelling-compare.ipynb         # Model comparison
│   ├── modelling-optimize.ipynb        # Hyperparameter tuning
│   ├── modelling-ablation.ipynb        # Feature ablation study
│   └── modelling-shap.ipynb            # SHAP explainability analysis
│
├── scripts/                            # Standalone utility scripts
│   ├── collect_spku_snapshot.py        # SPKU data collector
│   ├── compute_school_features.py      # School spatial features computation
│   ├── download_schools.py             # School registry downloader
│   ├── fetch_training_weather.py       # Historical weather data fetcher
│   ├── geocode_schools.py              # Nominatim geocoding pipeline
│   ├── run_hourly_kriging.py           # Hourly Kriging pipeline runner
│   ├── run_lag_kriging.py              # Lag Kriging pipeline runner
│   ├── test_recommendation_templates.py # Template testing script
│   └── upload_to_blob.py              # Bulk Blob upload utility
│
├── .github/workflows/
│   └── deploy-azure-functions.yml      # CI/CD: GitHub → Azure Functions
├── .env                                # Local environment (not committed)
├── .env.example                        # Environment template
├── pyproject.toml                      # Project metadata
└── requirements.txt                    # Full dependency list
```

---

## 7. External API Dependencies

| Service | URL | Purpose | Rate Limit | Auth |
|---------|-----|---------|------------|------|
| Rendahemisi (Jakarta Gov) | `https://rendahemisi.jakarta.go.id/ispu-detail` | Hourly pollutant data (5 stations) | None documented | None |
| Open-Meteo Archive | `https://archive-api.open-meteo.com/v1/archive` | Historical weather data | 10,000 req/day (free) | None |
| Open-Meteo Forecast | `https://api.open-meteo.com/v1/forecast` | Weather forecasts | 10,000 req/day (free) | None |
| BMKG | `https://api.bmkg.go.id/publik/prakiraan-cuaca` | Weather forecasts (Indonesia) | ~60 req/min | None |
| SPKU Jakarta | `https://udara.jakarta.go.id/api/lokasi_stasiun_udara` | Sensor network data | Unknown | None |
| Nominatim | `https://nominatim.openstreetmap.org/search` | Geocoding | 1 req/sec | User-Agent header |
| OpenRouter | `https://openrouter.ai/api/v1/chat/completions` | LLM API proxy | Free tier: rate-limited | API key |
| Overpass API | `https://overpass-api.de/api/interpreter` | OSM building/land-use data | 2 req/sec | None |

---

## 8. Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| OpenRouter over Gemini | Gemini region-blocked on Azure East Asia; OpenRouter has no region restrictions |
| `nvidia/nemotron-3-super-120b-a12b:free` | Only free OpenRouter model not rate-limited during testing |
| Template fallback for batch | Avoids LLM rate limits when generating recommendations for 4,215+ schools |
| Dual-write (local + Blob) | Ensures data availability even when Blob Storage is unavailable |
| Symlinks for deployment | `src/`, `models/`, `prompts/` symlinked into `azure-functions/` to avoid duplication |
| Kriging + Residual Correction | Two-stage approach improves accuracy over plain Kriging by correcting systematic spatial bias |
| LightGBM over neural networks | Handles tabular data well, fast inference, feature importance for XAI |
| BMKG hourly thresholds | Uses official Indonesian government classification system for consistency |
| Timer + HTTP triggers | ETL/Predict run on schedule but can be manually triggered; Recommend is on-demand only |

---

## 9. Data Flow Summary (End-to-End)

```
                    ┌──────────────┐
                    │  Rendahemisi │
                    │  (5 stations)│
                    └──────┬───────┘
                           │ PM2.5, PM10, SO2, CO, O3, NO2, HC
                           ▼
                    ┌──────────────┐     ┌──────────────┐
                    │  Open-Meteo  │────►│  ETL Merge   │
                    │  (Weather)   │     │  + Temporal   │
                    └──────────────┘     └──────┬───────┘
                                                │ dataset_master_spku_weather
                                                ▼
                                         ┌──────────────┐
                                         │  Azure Blob  │
                                         │  (processed)  │
                                         └──────┬───────┘
                                                │ Latest CSV
                                                ▼
┌──────────┐    ┌──────────────┐    ┌───────────────────────┐
│ LightGBM │◄───│   Feature    │◄───│  Historical Context   │
│ h6/h12/  │    │ Engineering  │    │  (last 72h from ETL)  │
│ h24      │    └──────┬───────┘    └───────────────────────┘
└────┬─────┘           │
     │ Station PM2.5   │
     │ predictions     │
     ▼                 ▼
┌──────────────┐    ┌──────────────┐
│   Kriging    │    │   School     │
│ Interpolation│◄───│  Locations   │
│ (5 → 4,215)  │    │  (geocoded)  │
└──────┬───────┘    └──────────────┘
       │ School PM2.5 + risk classification
       ▼
┌──────────────┐    ┌──────────────┐
│   Residual   │    │  Azure Blob  │
│  Correction  │───►│(predictions) │
└──────┬───────┘    └──────┬───────┘
       │                    │ predict_latest.json
       ▼                    ▼
┌──────────────────────────────────────┐
│        Recommend Pipeline            │
│                                      │
│  Single → OpenRouter LLM            │
│  Batch  → Template fallback         │
│  Summary → Direct computation       │
│                                      │
│  Output: Bahasa Indonesia            │
│  recommendations + action items      │
└──────────────────────────────────────┘
```

---

## 10. Testing & Quality Assurance

| Component | Test Coverage | Test Count |
|-----------|--------------|------------|
| Data clients (SPKU, BMKG, blob) | Unit tests with mocked HTTP | 16 tests |
| Spatial (Kriging, LOSOCV, residual) | Unit tests with synthetic data | 94 tests |
| Recommendations (engine, quality) | Template + LLM output validation | 22 tests |
| Transforms (risk classification) | Edge cases and boundary values | 15 tests |
| Config | Environment variable handling | 6 tests |
| Features (lag, school) | Feature engineering correctness | 13 tests |
| Helpers | File I/O, logging | 3 tests |
| **Total** | | **169 tests** |

### Running Tests
```bash
pytest tests/ -q
# Note: spatial tests (kriging, losocv, residual) require pykrige C extensions
# 169 test functions total; 89 pass without pykrige, all pass with full deps
```

### Local Pipeline Testing
```bash
python azure-functions/function_app.py --local etl        # Run ETL pipeline
python azure-functions/function_app.py --local predict    # Run Predict pipeline
python azure-functions/function_app.py --local recommend  # Run Recommend pipeline
```

---

## 11. Local Development Setup

```bash
# 1. Clone and enter project
cd airsafe-school

# 2. Create virtual environment
uv venv --python 3.11

# 3. Activate
source .venv/bin/activate

# 4. Install dependencies
uv pip install -r requirements.txt
uv pip install -r azure-functions/requirements.txt

# 5. Configure environment
cp .env.example .env
# Edit .env with your API keys and Blob connection string

# 6. Run tests
pytest tests/ -q

# 7. Run pipeline locally
python azure-functions/function_app.py --local etl        # ETL pipeline
python azure-functions/function_app.py --local predict    # Predict pipeline
python azure-functions/function_app.py --local recommend  # Recommend pipeline
```
