
========================================
AirSafe School - Day 1 Task Report
========================================
Date: 2026-04-20
Tasks: T-D1-03 (SPKU API), T-D1-04 (School Registries)
========================================

TASK T-D1-03: Explore SPKU API and Pull 1 Week Sample
Status: BLOCKED - API Structure Changed
Expected: spku_sample.json

FINDINGS:
-----------
1. SPKU API Structure:
   - Previously documented endpoints (udara.jakarta.go.id/api/sensor/data, etc.)
     all return 404
   - Current site uses JavaScript-heavy rendering
   - No clear JSON API endpoints found in HTML/JS analysis

2. Available Pages:
   - Main site: https://udara.jakarta.go.id ✓ (200)
   - Station locations: https://udara.jakarta.go.id/lokasi_stasiun ✓ (200)
   - Both return static HTML, require JS for data loading

3. Alternative Data Sources Found:
   a) Kaggle Dataset: ISPU DKI Jakarta (derryderajat/indeks-pencemaran-udara-dki)
      - Contains ISPU data from 5 SPKU stations
      - Access requires Kaggle account/reCAPTCHA
      - May need manual download

   b) Satu Data Jakarta:
      - https://satudata.jakarta.go.id (portal exists)
      - API structure changed (returns HTML, not JSON)
      - Datasets require manual browser access

RECOMMENDATIONS:
----------------
Option 1: Use Kaggle Dataset (Recommended)
- Manual download from Kaggle: derryderajat/indeks-pencemaran-udara-dki
- Contains historical ISPU data from 5 stations
- Covers 2010-2021 timeframe
- Good for EDA and baseline model training

Option 2: Scrape Current Site Data
- Build Selenium-based scraper for udara.jakarta.go.id
- Extract real-time data for prototype
- More complex but provides current data

Option 3: Use ISPU CSV from Satu Data Jakarta
- Search for ISPU datasets on satudata.jakarta.go.id
- Manual download of historical data
- May have 2022-2025 data needed for recent trends


TASK T-D1-04: Download School Registries from BPS/Kemendikbud
Status: IN PROGRESS - Automated Access Blocked
Expected: sd_dki.csv, smp_dki.csv, sma_smk_dki.csv

FINDINGS:
-----------
1. School Data Sources Attempted:
   a) BPS Dukcapil Coordinates:
      - Dataset referenced in proposal
      - Download links tested (returned empty files)

   b) Kemendikbud Dapodik:
      - https://dapo.kemendikdasmen.go.id/sp
      - Returns 403 Forbidden (blocks automated access)
      - Requires authentication/captcha

   c) Satu Data Jakarta:
      - Portal exists but datasets not accessible via API
      - Requires manual browser interaction

2. Current Status:
   - No automated school data downloads successful
   - All major sources require authentication or manual access

RECOMMENDATIONS:
----------------
Option 1: Manual Download (Fastest)
- Access Kemendikbud Dapodik: https://dapo.kemendikdasmen.go.id/sp
- Navigate to Prov. DKI Jakarta (code: 010000)
- Download school data for SD, SMP, SMA, SMK
- Filter for DKI Jakarta schools

Option 2: Use Public School Lists
- GitHub repositories may have scraped school data
- Search for "sekolah dki jakarta csv"
- May have lat/lon coordinates included

Option 3: Create Manual Dataset
- Start with small sample of known schools
- Manually geocode using Nominatim (with rate limiting)
- Expand dataset gradually


NEXT ACTIONS PRIORITY:
------------------------
1. IMMEDIATE (Today):
   - Download ISPU dataset from Kaggle (manual)
   - Download school data from Kemendikbud (manual)
   - Create placeholder spku_sample.json with structure documentation

2. DAY 2 (Tomorrow):
   - Merge downloaded datasets
   - Begin geocoding process
   - Create EDA visualizations

3. DAY 3:
   - Deploy ETL function (can work with downloaded data)
   - Build baseline model with available data


RISK MITIGATION:
----------------
Current blockers align with Risk R-01 (SPKU API unreliable) and data access challenges.
Fallback plan: Use Kaggle ISPU dataset for model training and EDA.
For real-time prototype: Implement simplified scraping or use historical data with "live" simulation.


RECOMMENDATION FOR TEAM:
---------------------------
Focus on getting baseline model working with Kaggle ISPU data.
For Day 2 tasks: Use manually downloaded school data and focus on geocoding pipeline.
Real-time API integration can be implemented in Phase 2 once MVP is established.

========================================
Report Generated: 2026-04-20
========================================
