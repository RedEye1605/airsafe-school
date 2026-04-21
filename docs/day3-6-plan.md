# AirSafe School — Day 3-6 Research & Implementation Plan

**Generated:** 2026-04-20 23:52 WIB  
**Deadline:** April 30, 2026 (10 days remaining)

---

## Research Findings

### Task 1: Weather Data Sources

#### Option A: Open-Meteo Historical Weather API ⭐ RECOMMENDED
- **URL:** `https://archive-api.open-meteo.com/v1/archive`
- **Free:** Yes, non-commercial use. No API key required.
- **Data source:** ERA5 reanalysis (ECMWF), 0.25° resolution, hourly since 1940
- **ERA5-Land:** 0.1° resolution (~11 km), hourly since 1950
- **Available hourly variables:**
  - `temperature_2m` — Air temperature at 2m
  - `relative_humidity_2m` — Relative humidity
  - `wind_speed_10m` — Wind speed at 10m
  - `wind_direction_10m` — Wind direction
  - `precipitation` — Rainfall
  - `surface_pressure` — Atmospheric pressure
  - `cloud_cover` — Total cloud cover
- **Example API call for Jakarta:**
  ```
  https://archive-api.open-meteo.com/v1/archive?latitude=-6.2&longitude=106.85&start_date=2020-01-01&end_date=2025-12-31&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,surface_pressure&timezone=Asia/Jakarta
  ```
- **Pros:** Free, reliable, structured JSON, good temporal coverage, no auth needed
- **Cons:** ~25km spatial resolution (ERA5); for finer resolution use ERA5-Land (~11km)
- **Strategy:** Fetch daily aggregated weather for Jakarta center point. For LightGBM features, we aggregate to daily means/max/min.

#### Option B: BMKG Open Data
- **Forecast API:** `https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4=<kode_wilayah>` — only 3-day forecast, NOT historical
- **GIS API:** `https://gis.bmkg.go.id/arcgis/rest/services/` — rainfall maps, wind potential, but limited programmatic historical access
- **datacuaca.bmkg.go.id** — has download capability (CSV, GeoJSON) but appears to be current/forecast data
- **iklim.bmkg.go.id** — climate portal, has PM2.5 concentration info and disease prediction (DBD), but mostly dashboard-based
- **Verdict:** BMKG open data is primarily forecast/realtime, NOT historical weather. Open-Meteo is far superior for our use case.
- **Note:** We should credit BMKG as data source where applicable (for SPKU/ISPU data)

#### Option C: Open-Meteo Air Quality API
- **URL:** `https://air-quality-api.open-meteo.com/v1/air-quality`
- **Variables:** PM2.5, PM10, NO2, SO2, O3, etc.
- **Could supplement SPKU data but lower resolution. Keep as backup.**

### Task 2: Jakarta Health Data (ISPA)

#### Primary Source: Surveilans Dinkes DKI Jakarta
- **Portal:** `https://surveilans-dinkes.jakarta.go.id/`
- **Data available:** ISPA cases (daily, weekly, monthly), filterable by kecamatan/kelurahan
- **Dashboard shows:** Current month ISPA cases (~1,454), Pneumonia, etc.
- **Data granularity:** Per kecamatan/kelurahan, from RS (hospitals) and PKM (puskesmas)
- **Key pages:**
  - `surveilans-dinkes.jakarta.go.id/sarsbaru/rs_rekap.php` — Hospital recap
  - `surveilans-dinkes.jakarta.go.id/sarsbaru/rs_banyakmatigabung.php` — Combined data
- **Strategy:** Manual CSV export from dashboard or scrape the tables. The portal has filter/download capability.
- **ISPA Data points found:**
  - H1 2023: 638,291 cases (Dinkes DKI)
  - Jan-Oct 2025: 1,966,308 cases
  - Jakarta Selatan alone (Jan-Oct 2025): 453,725 cases

#### Secondary Source: BPS Jakarta
- **Table:** "Jumlah Kasus Penyakit Menurut Provinsi/Kabupaten/Kota dan Jenis Penyakit"
- **URL:** `https://jakarta.bps.go.id/id/statistics-table/2/NTA0IzI=/`
- **Contains:** ISPA, TB, Pneumonia, Malaria by kabupaten/kota
- **Format:** Annual table, can download

#### Tertiary Source: Profil Kesehatan DKI Jakarta 2023
- **Available on Scribd:** Comprehensive health profile with disease statistics
- **Use for:** Context/validation of health data

#### Health Correlation Strategy:
1. Collect monthly ISPA cases per kecamatan (from Surveilans portal)
2. Aggregate PM2.5 data monthly per kecamatan (from our ISPU/SPKU data)
3. Compute Pearson/Spearman correlation
4. Create scatter plot + time-series overlay chart
5. **If granular kecamatan data unavailable:** Use city-level monthly ISPA totals vs monthly avg PM2.5

### Task 3: Azure Functions Research

#### Timer Trigger (Python v2 programming model)
```python
import azure.functions as func
import logging

app = func.FunctionApp()

@app.timer_trigger(schedule="0 0 2 * * *",  # Daily at 2 AM UTC
                   arg_name="myTimer",
                   run_on_startup=False) 
def daily_etl(myTimer: func.TimerRequest) -> None:
    logging.info('AirSafe ETL function executed.')
    # ETL logic here
```

#### Azure Blob Storage Integration
```python
from azure.storage.blob import BlobServiceClient
import os

# Connection via environment variable or App Settings
connect_str = os.environ["AzureWebJobsStorage"]
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

# Upload
blob_client = blob_service_client.get_blob_client(container="airsafe-data", blob="daily/pm25_2026-04-20.csv")
blob_client.upload_blob(csv_data, overwrite=True)

# Download
blob_client = blob_service_client.get_blob_client(container="airsafe-data", blob="daily/pm25_2026-04-20.csv")
data = blob_client.download_blob().readall()
```

#### Azure Functions Pricing (Consumption Plan)
- **Free grant:** 1 million executions + 400,000 GB-s per month per subscription
- **Cost after free tier:** $0.000016/GB-s + $0.20/million executions
- **Our daily ETL estimate:** 1 execution/day × ~30 sec × 512MB = ~15,360 GB-s/month → WELL within free tier
- **Blob Storage:** ~$0.018/GB/month (hot), ~$0.004/GB/month (cool)
- **Total estimated cost:** Essentially FREE for our use case

#### Project Structure
```
airsafe-function/
├── function_app.py          # Main function entry point
├── requirements.txt         # Dependencies
├── host.json               # Functions runtime config
├── local.settings.json     # Local dev settings
└── etl/
    ├── __init__.py
    ├── fetch_spku.py       # Fetch SPKU data
    ├── fetch_weather.py    # Fetch Open-Meteo weather
    ├── run_kriging.py      # Run Kriging interpolation
    └── upload_results.py   # Upload to Blob
```

### Task 4: pykrige Research

#### Core API
```python
from pykrige.ok import OrdinaryKriging

# Create Kriging object
OK = OrdinaryKriging(
    x=stations_lon,           # Station longitudes
    y=stations_lat,           # Station latitudes  
    z=stations_pm25,          # PM2.5 values
    variogram_model='gaussian',  # Best for air quality per research
    verbose=True,
    enable_plotting=False
)

# Interpolate to grid or specific points
z_interp, ss = OK.execute('points', school_lons, school_lats)
```

#### Variogram Model Selection
- **Gaussian:** Best for smooth spatial fields like PM2.5
- **Spherical:** Good for bounded spatial correlation
- **Exponential:** Fast decay, good for heterogeneous data
- **Recommendation:** Test Gaussian and Spherical, pick best via CV

#### Cross-Validation with sklearn GridSearchCV
```python
from pykrige.rk import Krige
from sklearn.model_selection import GridSearchCV

param_dict = {
    "method": ["ordinary", "universal"],
    "variogram_model": ["linear", "power", "gaussian", "spherical"],
}
estimator = GridSearchCV(Krige(), param_dict, verbose=True)
estimator.fit(X=coords, y=pm25_values)
print(f"Best: {estimator.best_params_}, R²: {estimator.best_score_:.3f}")
```

#### Leave-One-Sensor-Out (LOSO) Validation
```python
from sklearn.metrics import mean_absolute_error, r2_score

predictions = []
actuals = []

for i in range(len(stations)):
    # Leave out station i
    train_idx = [j for j in range(len(stations)) if j != i]
    
    OK = OrdinaryKriging(
        lons[train_idx], lats[train_idx], pm25[train_idx],
        variogram_model='gaussian'
    )
    
    pred, _ = OK.execute('points', [lons[i]], [lats[i]])
    predictions.append(pred[0])
    actuals.append(pm25[i])

mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
```

#### Key Research Findings for Jakarta PM2.5 Interpolation
- A specific paper exists: "Identifying the best spatial interpolation method for estimating spatial distribution of PM2.5 in Jakarta" (ResearchGate, 2021)
- Seven methods were compared — we should reference this paper in our documentation
- Ordinary Kriging with Gaussian variogram is a solid baseline
- **Critical:** With only 21 active PM2.5 stations, spatial coverage is sparse — Kriging uncertainty will be HIGH in areas far from stations. Document this limitation.

#### Best Practices
1. **Coordinate system:** Use decimal degrees directly (pykrige supports geographic coords)
2. **Temporal aggregation:** Use daily/hourly averages per station before interpolation
3. **Minimum stations:** Need ≥10 for meaningful variogram fitting (we have 21 — just enough)
4. **Anisotropy:** Consider if PM2.5 has directional dependence (e.g., wind-driven)
5. **Variogram fitting:** Use `weight=True` to emphasize short-range structure
6. **Validation:** Always report MAE, RMSE, and R² from LOSO validation

---

## Day 3-6 Implementation Plan

### Day 3 (April 21) — ETL Pipeline + Weather Data Integration

**Goal:** Build the data processing pipeline backbone

#### Morning: Weather Data Collection
- [ ] **Script:** `scripts/fetch_weather.py`
  - Fetch Jakarta historical weather from Open-Meteo API (2020-2025)
  - Variables: temperature, humidity, wind_speed, precipitation, pressure
  - Daily aggregation (mean, max, min for temp; sum for precip)
  - Save to `data/raw/weather/jakarta_daily_2020_2025.csv`
  - API call template:
    ```
    https://archive-api.open-meteo.com/v1/archive?latitude=-6.2&longitude=106.85&start_date=2020-01-01&end_date=2025-12-31&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,wind_speed_10m_max,precipitation_sum,surface_pressure_mean&timezone=Asia/Jakarta
    ```

#### Afternoon: ISPU Data Cleaning
- [ ] **Script:** `scripts/clean_ispu.py`
  - Load 6 ISPU CSVs from `data/raw/ispu/`
  - Standardize columns: date, station, pm25, pm10, so2, co, o3, no2
  - Handle missing values (flag, don't impute yet)
  - Merge into single `data/processed/ispu/ispu_clean_2010_2025.parquet`
  - Summary stats: coverage per station, per year

#### Evening: Feature Engineering Foundation
- [ ] **Script:** `scripts/build_features.py`
  - Merge ISPU + weather on date
  - Create lag features: pm25_lag1, pm25_lag7, pm25_lag30
  - Create rolling features: pm25_rolling_7d, pm25_rolling_30d
  - Temporal features: day_of_week, month, is_weekend, is_dry_season
  - Weather interaction features: temp_x_humidity, wind_x_precip
  - Output: `data/processed/features/ml_features.parquet`

**Deliverables:**
- 3 working scripts with CLI args
- 3 processed data files (weather, clean ISPU, ML features)
- Data quality report: row counts, coverage, missing % per column

**Validation:**
- Weather data: Spot check against BMKG website for 3 random dates
- ISPU: Row count matches sum of source files (deduped)
- Features: No NaN in date column, lag features properly shifted

---

### Day 4 (April 22) — Kriging Interpolation + LightGBM Training

**Goal:** Spatial interpolation engine + initial ML model

#### Morning: Kriging Implementation
- [ ] **Script:** `scripts/run_kriging.py`
  - Load SPKU hourly data, filter to PM2.5 active stations (21)
  - Aggregate to daily mean per station
  - Run Ordinary Kriging with Gaussian variogram
  - Interpolate to all 3,985 school locations
  - Implement LOSO cross-validation
  - Output: `data/processed/kriging/schools_pm25_daily.csv`
  - Output: `data/processed/kriging/loso_validation.json`

#### Afternoon: LightGBM Model v1
- [ ] **Script:** `scripts/train_lightgbm.py`
  - Load `ml_features.parquet`
  - Train/val/test split: temporal (last 6 months = test)
  - Features: weather, temporal, lag, rolling, station metadata
  - Target: next-day PM2.5 (or daily avg)
  - LightGBM with early stopping
  - Hyperparameter search: learning_rate, num_leaves, max_depth
  - Log metrics: MAE, RMSE, R² on test set
  - Save model: `models/lightgbm_v1.pkl`
  - Feature importance chart

#### Evening: Kriging + ML Ensemble
- [ ] **Script:** `scripts/ensemble_predict.py`
  - Use Kriging for spatial interpolation
  - Use LightGBM for temporal prediction
  - Combine: predict PM2.5 at each school for tomorrow
  - Generate risk categories (ISPU scale): Baik, Sedang, Tidak Sehat, dll.
  - Output: `data/processed/predictions/tomorrow_predictions.csv`

**Deliverables:**
- Kriging interpolation with LOSO validation report
- LightGBM model with feature importance
- Ensemble prediction pipeline

**Validation:**
- LOSO Kriging: MAE < 15 µg/m³, R² > 0.3 (realistic for 21 stations)
- LightGBM: Beat persistence baseline (yesterday's PM2.5) by ≥10% MAE
- Predictions: All 3,985 schools have non-null PM2.5 predictions

---

### Day 5 (April 23) — Health Correlation + Azure Functions Skeleton

**Goal:** Health impact chart + cloud deployment scaffolding

#### Morning: ISPA Health Data Collection & Correlation
- [ ] **Manual task:** Export ISPA data from `surveilans-dinkes.jakarta.go.id`
  - Navigate to disease statistics
  - Filter by ISPA, monthly, per kecamatan
  - Export 2023-2025 data to CSV
  - Save to `data/raw/health/ispa_dinkes_2023_2025.csv`
- [ ] **If granular data unavailable:** Use aggregate numbers from news reports
  - H1 2023: 638,291 total ISPA cases
  - Jan-Oct 2025: 1,966,308 total ISPA cases
  - Extrapolate monthly from available data points

- [ ] **Script:** `scripts/health_correlation.py`
  - Merge ISPA monthly data with PM2.5 monthly averages
  - Compute correlation (Pearson + Spearman)
  - Generate charts:
    1. PM2.5 vs ISPA scatter plot (monthly aggregates)
    2. Dual-axis time series: PM2.5 trend + ISPA cases
    3. Heatmap: ISPA cases by kecamatan × month
  - Output: `outputs/charts/health_correlation.png`
  - Output: `data/processed/health/health_pm25_correlation.csv`

#### Afternoon: Azure Functions Skeleton
- [ ] **Setup:** `airsafe-function/` project
  - `function_app.py` with timer trigger (daily at 2 AM WIB = 19:00 UTC)
  - `requirements.txt`: azure-functions, pandas, lightgbm, pykrige, requests
  - `host.json` with default config
  - `local.settings.json` with dev connection strings
  - ETL modules in `etl/` subfolder:
    - `fetch_spku.py` — fetch latest SPKU data
    - `fetch_weather.py` — fetch current weather
    - `run_model.py` — load model + Kriging, predict
    - `upload_results.py` — upload predictions to Blob Storage

- [ ] **Azure Blob Storage structure:**
  ```
  airsafe-data/
  ├── raw/spku/{date}/
  ├── raw/weather/{date}/
  ├── predictions/{date}/pm25_predictions.csv
  └── models/lightgbm_v1.pkl
  ```

#### Evening: Local Testing
- [ ] Test Azure Function locally with `func start`
- [ ] Verify Blob Storage upload/download cycle
- [ ] Test end-to-end: fetch → predict → upload

**Deliverables:**
- ISPA correlation chart (for presentation slide)
- Azure Functions project skeleton
- Local E2E test passing

**Validation:**
- Correlation chart renders correctly
- Azure Function runs locally without errors
- Blob upload/download verified

---

### Day 6 (April 24) — Presentation Charts + Story Polish

**Goal:** Finalize all visualizations and presentation narrative

#### Morning: Prediction Dashboard/Charts
- [ ] **Script:** `scripts/generate_dashboard.py`
  - Jakarta map with school PM2.5 predictions (colored dots)
  - Risk category breakdown pie chart
  - Top 10 most polluted kecamatan bar chart
  - "Coverage gap" overlay: schools >5km from nearest station
  - Use matplotlib + contextily for basemap

#### Afternoon: Presentation Deck Content
- [ ] **Prepare slide content for each chart:**
  1. Problem: Jakarta air pollution + school children impact
  2. Data: 4,215 schools, 21 PM2.5 stations, ISPU 2010-2025
  3. Coverage Gap Map: visual showing 71% schools lack monitoring
  4. Method: LightGBM (temporal) + Kriging (spatial) pipeline
  5. Results: LOSO validation, feature importance
  6. Health Impact: PM2.5 vs ISPA correlation chart
  7. Solution: Azure Functions ETL + risk alerts for parents
  8. Impact: SDG 3 (Good Health), SDG 11 (Sustainable Cities), SDG 4 (Education)

- [ ] **GPT-4o-mini integration:**
  - Generate natural language risk summaries from prediction data
  - Example: "Besok, 12 sekolah di Kecamatan Cilincung berisiko PM2.5 'Tidak Sehat'. Disarankan aktivitas indoor."
  - Script: `scripts/generate_alerts.py`

#### Evening: Documentation + Polish
- [ ] README.md with setup instructions, architecture diagram
- [ ] `docs/methodology.md` with math (Kriging equations, LightGBM formulation)
- [ ] Final model version saved and tagged
- [ ] All scripts have `--help` and proper arg parsing

**Deliverables:**
- 5+ publication-ready charts
- Complete README and methodology docs
- GPT-4o-mini alert generation working

**Validation:**
- All charts render without errors
- README is complete enough for a judge to reproduce
- Alert text is grammatically correct Indonesian

---

## Day 7-10 Preview (April 25-30)

- **Day 7:** Final Azure deployment, end-to-end testing, video demo script
- **Day 8:** Record demo video, polish presentation slides
- **Day 9:** Buffer day — fix bugs, add refinements
- **Day 30 (April 30):** SUBMISSION DEADLINE

---

## Data Source Summary

| Data | Source | API/URL | Format | Frequency |
|------|--------|---------|--------|-----------|
| PM2.5 stations | SPKU DKI Jakarta | `https://spku.jakarta.go.id/` | JSON (scraped) | Hourly |
| ISPU historical | BPLHD DKI | `data/raw/ispu/` (local CSV) | CSV | Daily |
| Weather (historical) | Open-Meteo (ERA5) | `archive-api.open-meteo.com` | JSON | Daily fetch |
| ISPA health | Dinkes Surveilans | `surveilans-dinkes.jakarta.go.id` | Manual export | Monthly |
| School registry | Dinas Pendidikan DKI | `data/processed/schools/` (local) | CSV | Static |
| Kriging interpolation | pykrige | Python library | — | On demand |

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Only 21 active PM2.5 stations | High interpolation uncertainty | Document as limitation; show confidence maps |
| ISPA data not granular enough | Weak correlation chart | Use city-level aggregates + news report numbers |
| Azure deployment issues | Incomplete cloud demo | Focus on local pipeline first; deploy what works |
| Open-Meteo rate limits | Slow data fetch | Batch requests (1 year per call), cache locally |
| LightGBM overfitting | Poor generalization | Temporal split, early stopping, feature selection |

---

*Plan generated by AirSafe Datathon Agent — ready for implementation Day 3*
