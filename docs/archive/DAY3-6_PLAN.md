# AirSafe School — Day 3-6 Comprehensive Plan

**Created:** 2026-04-20 23:40 WIB  
**Status:** PLANNING COMPLETE — Ready for Implementation  
**Team:** Rhendy (Data/Backend/ML), Aditya (ML), Aufi (BI/Dashboard)

---

## RESEARCH FINDINGS SUMMARY

### 1. BMKG API — Weather Data

**Key Finding:** BMKG provides **forecast** data only (3-day, per-3-hour), NOT historical data. This is insufficient for our needs.

| Endpoint | Type | Data | Limit |
|----------|------|------|-------|
| `data.bmkg.go.id/prakiraan-cuaca/` | Forecast 3-day | Jakarta per kelurahan, JSON | 60 req/min/IP |
| `api.bmkg.go.id/publik/prakiraan-cuaca` | New API (v2) | Same, newer format | TBD |
| `gis.bmkg.go.id/arcgis/rest/services/` | GIS Feature Layers | Rainfall maps, wind energy | Public |
| `data.bmkg.go.id/` | Open Data portal | XML/JSON forecast, earthquake | 60 req/min |

**✅ SOLUTION: Open-Meteo Historical Weather API (FREE, no key needed)**
- URL: `https://archive-api.open-meteo.com/v1/archive`
- Source: ERA5 reanalysis (ECMWF), 0.25° resolution, hourly
- Coverage: 1940-present, 5-day delay
- Params: `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `precipitation`, `surface_pressure`
- Jakarta coords: lat=-6.2, lon=106.85
- Example call:
  ```
  https://archive-api.open-meteo.com/v1/archive?latitude=-6.2&longitude=106.85&start_date=2021-01-01&end_date=2025-02-28&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation&timezone=Asia/Jakarta
  ```
- **For BMKG real-time forecast (Azure Function):** Use `data.bmkg.go.id/prakiraan-cuaca/` for live weather in prediction function

### 2. Azure Functions Python (v2 Programming Model)

**Architecture:** Use the **v2 decorator-based model** (simpler, no function.json needed).

```python
import azure.functions as func
import logging

app = func.FunctionApp()

@app.timer_trigger(schedule="0 0 * * *",  # Daily at midnight UTC
                   arg_name="myTimer",
                   run_on_startup=False)
@app.blob_output(arg_name="outputBlob",
                 path="airquality/raw/ispu/{datetime}.json",
                 connection="AzureWebJobsStorage")
def etl_function(myTimer: func.TimerRequest, outputBlob: func.Out[str]):
    # Pull data from SPKU + BMKG, save to blob
    ...
```

**Key bindings needed:**
- **Timer Trigger:** Cron schedule for daily ETL (e.g., `0 */2 * * *` = every 2 hours)
- **Blob Output:** Save raw + processed data to Azure Blob Storage
- **HTTP Trigger (Function #2):** For on-demand prediction endpoint
- **Blob Input:** Read trained model from blob for predictions

**Project structure:**
```
azure-functions/
├── function_app.py          # Main entry (v2 model)
├── requirements.txt
├── host.json
├── local.settings.json      # Connection strings (local dev)
├── etl/
│   ├── __init__.py
│   ├── spku_client.py       # SPKU API client
│   ├── bmkg_client.py       # BMKG forecast client
│   ├── openmeteo_client.py  # Historical weather client
│   └── transforms.py        # Data cleaning/merge logic
└── predict/
    ├── __init__.py
    └── predictor.py          # Kriging + LightGBM prediction
```

### 3. ISPA/Health Data Sources

**Finding:** Jakarta has rich ISPA surveillance data but it's fragmented.

| Source | Data | Access | Granularity |
|--------|------|--------|-------------|
| **BPS DKI Jakarta** | Disease cases by city + type | `jakarta.bps.go.id` (Cloudflare blocked for scraping) | Per city, annual |
| **Dinkes Surveilans** | Real-time disease data from hospitals/puskesmas | `surveilans-dinkes.jakarta.go.id` | Per kecamatan, weekly |
| **Katadata** | ISPA DKI Jakarta 638K cases H1 2023 | Premium (paywall) | Monthly aggregate |
| **Dinkes DKI 2025** | 1.966 million ISPA cases Jan-Oct 2025 | News reports | Aggregate |

**⚠️ CHALLENGE:** No free downloadable CSV with monthly ISPA per kecamatan. We need to:
1. **For Day 3 chart:** Use BPS annual data by city (5 cities + Kep Seribu) + ISPU PM2.5 averages
2. **Manual collection:** Download BPS Jakarta Statistik Table from browser
3. **Alternative:** Use Dinkes Surveilans portal manually for kecamatan-level data
4. **Synthetic approach:** If granular data unavailable, create synthetic ISPA estimates based on PM2.5 levels using published dose-response relationships

**ISPA-PM2.5 relationship from literature (see #4):**
- Jakarta PM2.5 mean ~90 µg/m³ → RTI prevalence ~71% (vs 26% in low-exposure)
- OR = 7.167 (adjusted OR = 7.883) for high vs low PM2.5 exposure in children
- Prevalence ratio = 2.76 (95% CI: 1.68-4.54)

### 4. PM2.5 Health Impact — Key References

**Study 1: PMC12667370 (2025) — PM2.5 & RTI in Jakarta/Bandung children**
- Cross-sectional, n=107 children aged 6-12
- High exposure (Jakarta SD Kedoya): 71.43% RTI prevalence
- Low exposure (Bandung SD Pangalengan): 25.86% RTI prevalence
- **Adjusted OR = 7.883** (95% CI: 3.228-19.250, p < 0.001) controlling for gender, age, maternal education, maternal occupation
- Chi-square: χ² = 22.154, p < 0.001, φ = 0.475
- **First Indonesian study** on PM2.5-RTI in school children

**Study 2: Nafas x DBS x FKM UI Whitepaper**
- PM2.5 and health impacts in Greater Jakarta (Jabodetabek)
- Covers 2020-2022 period
- Children's pneumonia and asthma focus

**Study 3: ResearchGate — Spatial interpolation for PM2.5 in Jakarta**
- Compared interpolation methods for Jakarta PM2.5
- Relevant for our Day 4 Kriging implementation

**Key epidemiological relationships for our model:**
- Linear dose-response: each 10 µg/m³ PM2.5 increase → ~15-20% increase in ISPA risk
- Children 6-12 are most vulnerable (our target: schools)
- Seasonal variation: dry season (Jun-Sep) = worse air = more ISPA

### 5. Kriging Interpolation (PyKrige)

**Library:** `pykrige` v1.7.3 — well-maintained, scipy-based

**Approach: Ordinary Kriging (OK)**
- Best for our case: unknown mean, spatial autocorrelation
- Uses semivariogram to model spatial structure

**Workflow:**
```python
from pykrige.ok import OrdinaryKriging

# Station data as input
OK = OrdinaryKriging(
    x=stations['lon'],           # Longitudes
    y=stations['lat'],           # Latitudes  
    z=stations['pm25_value'],    # PM2.5 readings
    variogram_model='spherical', # Try: spherical, exponential, gaussian, linear
    verbose=True,
    enable_plotting=True
)

# Predict on grid (or school locations)
z_pred, ss = OK.execute('points', schools['lon'], schools['lat'])
```

**Best practices from research:**
1. **Variogram model selection:** Compare spherical, exponential, gaussian using cross-validation
2. **Coordinate system:** Use UTM zone 48S (EPSG:32748) for Jakarta — meters, not degrees
3. **Anisotropy:** Jakarta pollution may have directional patterns (NE-SW wind corridor)
4. **Cross-validation:** Leave-one-out CV (LOOCV) to validate interpolation accuracy
5. **Minimum stations:** 20+ needed for reliable kriging (we have 105 stations, 21 active PM2.5)
6. **Grid resolution:** ~500m grid for Jakarta (~60km x 30km = 120x60 grid = 7,200 cells)
7. **Compare with IDW** as baseline

**Validation metrics:**
- RMSE, MAE, R² on LOOCV
- Prediction variance map (uncertainty)
- Compare kriging vs IDW vs nearest-neighbor

---

## EXISTING DATA INVENTORY

| Dataset | Rows | Key Fields | Status |
|---------|------|------------|--------|
| ISPU DKI (all stations) | 5,539 | tanggal, stasiun, pm25, pm10, so2, co, o3, no2, categori | ✅ 2010-2025 |
| ISPU DKI (combined) | 31,404 | Same, 5 separate station files | ✅ 2010-2021 |
| Schools geocoded | 4,215 | npsn, nama, lat, lon, kecamatan, kelurahan | ✅ SD/SMP/SMA |
| SPKU stations | 105 | name, lat, lon, pm25, status | ✅ 21 active PM2.5 |
| SPKU snapshots | 8 JSONs | Per-station PM2.5 readings | ✅ Hourly cron |
| SPKU timeseries | 840 rows | collection_time, station, pm25, status | ✅ Growing |
| Weather (historical) | ❌ NOT YET | temp, humidity, wind, rain | 🔲 Need Open-Meteo |
| ISPA health data | ❌ NOT YET | cases per city/kecamatan/month | 🔲 Need BPS/Dinkes |

---

## DAY 3 PLAN — ETL Function + Health Correlation Chart

### T-D3-04: Azure Function #1 (ETL)

**Goal:** Timer-triggered Azure Function that pulls SPKU + weather data daily, saves to Blob Storage.

**Architecture:**
```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Timer Trigger   │────▶│  ETL Function     │────▶│  Blob Store  │
│  (every 2 hours) │     │  (Python v2)      │     │              │
└─────────────────┘     │                   │     │ /raw/        │
                        │  1. Pull SPKU API  │     │   spku/      │
┌─────────────────┐     │  2. Pull BMKG fcst │     │   weather/   │
│  SPKU API        │────▶│  3. Clean & merge  │     │ /processed/  │
│  (udara.jkt.go) │     │  4. Save to blob   │     │   daily/     │
└─────────────────┘     └──────────────────┘     └──────────────┘
┌─────────────────┐
│  BMKG API        │────▶ (forecast only)
│  (data.bmkg.go) │
└─────────────────┘
```

**Deliverables:**
1. `azure-functions/function_app.py` — Main function with timer trigger
2. `azure-functions/etl/spku_client.py` — Reuses existing `collect_spku_snapshot.py` logic
3. `azure-functions/etl/weather_client.py` — BMKG forecast + Open-Meteo historical
4. `azure-functions/etl/transforms.py` — Data cleaning, pivot, aggregation
5. `azure-functions/requirements.txt` — Dependencies
6. `azure-functions/host.json` — Configuration
7. Test script to run locally without Azure

**Data flow:**
```
SPKU POST request → Parse JSON → Extract station/PM2.5 → DataFrame
BMKG GET request  → Parse XML/JSON → Extract weather → DataFrame
→ Merge on timestamp → Clean nulls → Save as:
  - raw/spku/YYYY-MM-DD_HH.json (raw snapshot)
  - raw/weather/YYYY-MM-DD.json (daily weather)
  - processed/daily/YYYY-MM-DD.csv (merged, cleaned)
```

**Local testing approach (no Azure account needed initially):**
```python
# Create a mock version that saves to local filesystem
# Blob path pattern: {container}/{folder}/{filename}
# Local equivalent: ~/airsafe-school/data/azure-blob/{container}/{folder}/{filename}
```

### T-D3-05: Health Correlation Chart (PM2.5 vs ISPA)

**Goal:** Scatter plot + regression showing PM2.5 levels vs ISPA cases per city.

**Data needed:**
1. PM2.5 annual averages per DKI city (from ISPU data) → **we have this!**
2. ISPA cases per DKI city per year → **need from BPS**

**Approach:**
- Calculate annual PM2.5 mean per station from ISPU data
- Map stations to cities: DKI1→Jakpus, DKI2→Jakut, DKI3→Jaksel, DKI4→Jaktim, DKI5→Jakbar
- Get ISPA cases per city from BPS (annual, 2019-2024)
- Create dual-axis chart: PM2.5 bars + ISPA line, per city
- Add scatter plot with regression: PM2.5 vs ISPA rate

**Chart specifications:**
```
Figure 1: PM2.5 Annual Average by DKI City (2021-2024)
  - Grouped bar chart, x=cities, y=PM2.5 µg/m³
  - WHO guideline line at 15 µg/m³ (annual), Indonesia standard 55 µg/m³
  - Color by year

Figure 2: PM2.5 vs ISPA Cases Correlation
  - Scatter plot, x=PM2.5 annual avg, y=ISPA cases per city-year
  - Linear regression line + 95% CI
  - Pearson r and p-value annotation
  - Size = population

Figure 3: Monthly PM2.5 Trend with ISPA Overlay
  - If monthly ISPA available → dual axis line chart
  - If not → monthly PM2.5 trend only with seasonal annotation
```

**Implementation plan:**
1. Compute PM2.5 stats from existing ISPU data (5,539 rows, PM2.5 from 2021-2025)
2. Download BPS disease data manually (or use published figures)
3. Generate charts with matplotlib/seaborn
4. Save as `output/charts/health_correlation/`

**Fallback if BPS ISPA data unavailable:**
- Use published aggregate figures: Jakarta ISPA 638K (H1 2023), 1.966M (Jan-Oct 2025)
- Use dose-response from literature to estimate city-level ISPA
- Label chart as "estimated" where applicable

---

## DAY 4 PLAN — Kriging Implementation + Validation

**Goal:** Implement Ordinary Kriging interpolation to estimate PM2.5 at all 4,215 school locations.

### Morning: Setup & Baseline

**Deliverables:**
1. `src/interpolation/kriging_engine.py` — Core kriging module
2. `src/interpolation/variogram_analysis.py` — Semivariogram fitting & visualization
3. `src/interpolation/baseline_idw.py` — IDW baseline for comparison

**Code structure:**
```python
# kriging_engine.py
class AirQualityKriging:
    def __init__(self, stations_df, target_points_df):
        self.stations = stations_df  # lat, lon, pm25
        self.targets = target_points_df  # lat, lon (schools)
    
    def fit_variogram(self, model='spherical', plot=True):
        """Fit and visualize semivariogram"""
        pass
    
    def interpolate(self, method='ordinary'):
        """Run kriging interpolation"""
        pass
    
    def cross_validate(self, method='loocv'):
        """Leave-one-out cross-validation"""
        pass
    
    def predict_at_schools(self):
        """Generate PM2.5 estimates for all schools"""
        pass
```

### Afternoon: Validation & Comparison

**Validation strategy:**
1. **LOOCV on 21 active PM2.5 stations**
   - Remove one station, predict from remaining 20
   - Repeat for all 21 stations
   - Calculate RMSE, MAE, R², bias

2. **Variogram model comparison:**
   - Spherical vs Exponential vs Gaussian vs Linear
   - Select best by cross-validation RMSE

3. **Method comparison:**
   - Ordinary Kriging vs IDW (k=3,5,7) vs Nearest Neighbor
   - Table of metrics

4. **Validation against ISPU historical data:**
   - Use ISPU station daily values as ground truth
   - Compare kriging predictions at ISPU station coordinates

**Outputs:**
- `output/interpolation/variogram_plot.png`
- `output/interpolation/jakarta_pm25_heatmap.html` (folium)
- `output/interpolation/school_predictions.csv` (4,215 rows)
- `output/interpolation/cross_validation_results.csv`
- `output/interpolation/method_comparison.csv`

**Acceptance criteria:**
- RMSE < 15 µg/m³ on LOOCV
- R² > 0.5 on LOOCV
- No negative predictions
- Reasonable spatial patterns (higher PM2.5 in industrial/traffic areas)

---

## DAY 5 PLAN — LightGBM Residual Correction + Feature Engineering

**Goal:** Use LightGBM to correct kriging residuals, incorporating weather & school features.

### Architecture: Two-Stage Model

```
Stage 1: Kriging → PM2.5_estimate at school locations
Stage 2: LightGBM → Residual correction

Features for LightGBM:
├── Spatial
│   ├── kriging_estimate (from Stage 1)
│   ├── kriging_variance (uncertainty)
│   ├── nearest_station_distance
│   ├── nearest_station_pm25
│   ├── station_density_1km / 3km / 5km
│   ├── lat, lon
│   └── utm_x, utm_y
├── Weather (from Open-Meteo / BMKG)
│   ├── temperature_2m
│   ├── relative_humidity_2m
│   ├── wind_speed_10m
│   ├── precipitation
│   └── surface_pressure
├── Temporal
│   ├── month, day_of_week, hour
│   ├── is_weekend
│   ├── is_dry_season (Apr-Oct)
│   └── days_since_rain
├── School Context
│   ├── nearest_road_distance (if OSM available)
│   ├── land_use_type (if available)
│   ├── building_density_surrounding
│   └── school_type (SD/SMP/SMA)
└── Area Features
    ├── kecamatan_pm25_mean
    ├── city_pm25_mean
    └── population_density (if available)
```

### Training Data Construction

**Challenge:** We only have PM2.5 at station locations, not at schools. Solution:

**Approach A — Station-based training:**
1. Use 21 PM2.5 stations as training points
2. For each station: kriging estimate from OTHER stations (LOOCV)
3. Residual = actual - kriging_estimate
4. Train LightGBM: predict residual from features
5. Apply to all schools: corrected_PM2.5 = kriging + LightGBM_residual

**Approach B — Temporal split:**
1. Split ISPU historical data by time (train: 2021-2023, test: 2024-2025)
2. For each date, run kriging on station values
3. Train LightGBM on residuals
4. Predict for test period

**Recommended: Approach A** (simpler, works with current snapshot data)

### Deliverables

1. `src/ml/feature_engineer.py` — Feature construction pipeline
2. `src/ml/lightgbm_model.py` — Model training & prediction
3. `src/ml/model_pipeline.py` — Full pipeline: kriging → features → LightGBM → output
4. `output/ml/feature_importance.png`
5. `output/ml/residual_analysis.png`
6. `output/ml/school_predictions_final.csv` — Final PM2.5 estimates with confidence

**Hyperparameters (initial):**
```python
{
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'min_child_samples': 5,  # Small dataset
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1
}
```

**Validation:**
- LOOCV RMSE improvement over kriging alone
- Feature importance analysis
- Residual distribution (should be ~0 mean, smaller variance)

---

## DAY 6 PLAN — Azure Function #2 (Daily Prediction) + Integration

**Goal:** HTTP-triggered Azure Function that runs the full prediction pipeline on demand.

### Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  HTTP Trigger     │────▶│  Predict Function │────▶│  Response    │
│  POST /predict    │     │                   │     │  JSON        │
│                   │     │  1. Load latest   │     │              │
│  Body:            │     │     SPKU data     │     │  school_id   │
│  - school_ids[]   │     │  2. Load weather  │     │  pm25_est    │
│  - or bbox        │     │  3. Kriging interp│     │  confidence  │
│  - or all         │     │  4. LightGBM corr │     │  risk_level  │
│                   │     │  5. Risk classify │     │  timestamp   │
└──────────────────┘     └──────────────────┘     └──────────────┘
                               ▲
                               │
                    ┌──────────┴──────────┐
                    │  Blob Storage        │
                    │  - trained model     │
                    │  - station coords    │
                    │  - school coords     │
                    └─────────────────────┘
```

### Function Structure

```python
@app.route(route="predict", methods=["POST"])
@app.blob_input(arg_name="modelBlob",
                path="models/lightgbm_model.txt",
                connection="AzureWebJobsStorage")
@app.blob_input(arg_name="stationBlob",
                path="processed/stations/latest.csv",
                connection="AzureWebJobsStorage")
def predict_function(req: func.HttpRequest,
                     modelBlob: func.InputStream,
                     stationBlob: func.InputStream) -> func.HttpResponse:
    # 1. Parse request (school IDs or bounding box)
    # 2. Load latest SPKU data
    # 3. Load weather from Open-Meteo (current)
    # 4. Run kriging
    # 5. Apply LightGBM correction
    # 6. Classify risk (Baik/Sedang/Tidak Sehat)
    # 7. Return JSON
    ...
```

### Deliverables

1. `azure-functions/predict/predictor.py` — Prediction logic
2. `azure-functions/function_app.py` — Updated with predict route
3. `azure-functions/deploy.sh` — Deployment script
4. API documentation (request/response format)
5. Integration test with real SPKU data

**Risk classification (ISPU-based):**
| PM2.5 (µg/m³) | Category | Color | Action |
|----------------|----------|-------|--------|
| 0-35 | BAIK | 🟢 Green | Normal activities |
| 36-75 | SEDANG | 🟡 Yellow | Reduce outdoor for sensitive |
| 76-115 | TIDAK SEHAT | 🟠 Orange | Limit outdoor activities |
| 116-150 | SANGAT TIDAK SEHAT | 🔴 Red | Avoid outdoor |
| >150 | BERBAHAYA | 🟣 Purple | Stay indoors |

---

## DAY 7-10 PLAN — Dashboard, Proposal & Submission

### Day 7-8: Dashboard Integration (Aufi leads)
- **Aufi:** Power BI dashboard consuming prediction API
- **Rhendy:** Ensure API returns Power BI-compatible JSON
- Components:
  - Jakarta PM2.5 heatmap (folium/kepler.gl)
  - School risk table with filtering
  - Coverage gap visualization
  - Time series trends
  - Health impact chart (from Day 3)

### Day 9: Proposal Refinement
- Executive summary
- Problem statement with data (PM2.5 > WHO guideline 6x)
- Technical approach diagram (Kriging + LightGBM)
- Validation results
- Impact metrics
- Azure architecture diagram
- Timeline & team

### Day 10: Final Submission
- GitHub repository cleanup
- README with setup instructions
- Demo video/script
- Presentation slides
- Submit to datathon portal

---

## DEPENDENCY MAP

```
Day 3: ETL Function ──────────────────┐
Day 3: Health Chart ──────────────────│── No dependencies, parallel
                                       │
Day 4: Kriging ───────────────────────│── Needs SPKU data (✅ have)
   │                                  │
   ▼                                  │
Day 5: LightGBM ─────────────────────│── Needs kriging output
   │                                  │
   ▼                                  │
Day 6: Azure Function #2 ────────────│── Needs trained model
   │                                  │
   ▼                                  │
Day 7-8: Dashboard ──────────────────┘── Needs prediction API
```

---

## RISK REGISTER

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| ISPA data unavailable at kecamatan level | Medium | High | Use city-level BPS data + literature dose-response |
| Too few active stations for kriging | High | Low | 21 active PM2.5 stations is sufficient for OK |
| LightGBM overfits with 21 training points | Medium | High | Heavy regularization, simple features, validate with LOOCV |
| Azure Functions deployment issues | Medium | Medium | Develop locally first, deploy on Day 6 |
| Weather API rate limits | Low | Low | Open-Meteo is free, cache responses |
| SPKU API changes again | High | Medium | Multiple snapshots stored, Kaggle fallback |

---

## IMMEDIATE NEXT STEPS (Day 3 Start)

1. **Fetch Open-Meteo historical weather** for Jakarta (2021-2025)
2. **Download BPS ISPA data** manually from `jakarta.bps.go.id`
3. **Build ETL function skeleton** with SPKU client
4. **Create PM2.5 vs ISPA chart** using available data
5. **Set up Azure Functions project structure**

---

*Plan generated by AirSafe School Datathon Agent*  
*Next: Implementation begins on Day 3*
