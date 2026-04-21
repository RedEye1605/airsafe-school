#!/usr/bin/env python3
"""
SPKU Daily Snapshot Collector (with active station filter).
Run via cron hourly.
"""
import os, sys, json, time, requests, pandas as pd
from datetime import datetime, timedelta

OUTPUT_DIR = os.path.expanduser("~/airsafe-school/data/raw/spku")
TIMESERIES_PATH = os.path.join(OUTPUT_DIR, "spku_timeseries.csv")
SNAP_DIR = os.path.join(OUTPUT_DIR, "snapshots")
STALE_DAYS = 7

os.makedirs(SNAP_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'X-Requested-With': 'XMLHttpRequest',
    'Content-Type': 'application/x-www-form-urlencoded'
}

API_URL = "https://udara.jakarta.go.id/api/lokasi_stasiun_udara"

def fetch_snapshot():
    data = {
        'draw': '1', 'start': '0', 'length': '500',
        'columns[0][data]': 'tgl', 'columns[0][name]': 'tgl',
        'columns[0][searchable]': 'true', 'columns[0][search][value]': '',
        'columns[0][search][regex]': 'false',
        'order[0][column]': '0', 'order[0][dir]': 'desc',
        'search[value]': '', 'search[regex]': 'false',
    }
    r = requests.post(API_URL, headers=HEADERS, data=data, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    now = datetime.now()
    cutoff = now - timedelta(days=STALE_DAYS)
    
    result = fetch_snapshot()
    stations = result.get('data', [])
    
    # Filter to active stations (reported within STALE_DAYS)
    active = []
    stale = []
    for s in stations:
        try:
            ts = datetime.strptime(s.get('tgl', ''), '%m/%d/%Y %H:%M:%S')
            if ts >= cutoff:
                s['_is_active'] = True
                active.append(s)
            else:
                s['_is_active'] = False
                stale.append(s)
        except:
            s['_is_active'] = False
            stale.append(s)
    
    print(f"[{now.isoformat()}] Fetched {len(stations)} stations, {len(active)} active (within {STALE_DAYS}d), {len(stale)} stale")
    
    # Save raw snapshot
    ts_str = now.strftime('%Y%m%d_%H%M%S')
    snap_path = os.path.join(SNAP_DIR, f"spku_{ts_str}.json")
    with open(snap_path, 'w') as f:
        json.dump({'collected_at': now.isoformat(), 'active': len(active), 'stale': len(stale), 'data': stations}, f, indent=2)
    
    # Append to time-series CSV (active only)
    rows = []
    for s in active:
        rows.append({
            'collection_time': now.strftime('%Y-%m-%d %H:%M:%S'),
            'station_name': s.get('dataSourceName', ''),
            'station_id': s.get('dataSourceID', ''),
            'latitude': float(s.get('latitude', 0)),
            'longitude': float(s.get('longitude', 0)),
            'parameter': s.get('matricName', ''),
            'value': s.get('value'),
            'status': s.get('status', ''),
            'station_timestamp': s.get('tgl', ''),
            'is_active': True,
        })
    for s in stale:
        rows.append({
            'collection_time': now.strftime('%Y-%m-%d %H:%M:%S'),
            'station_name': s.get('dataSourceName', ''),
            'station_id': s.get('dataSourceID', ''),
            'latitude': float(s.get('latitude', 0)),
            'longitude': float(s.get('longitude', 0)),
            'parameter': s.get('matricName', ''),
            'value': s.get('value'),
            'status': s.get('status', ''),
            'station_timestamp': s.get('tgl', ''),
            'is_active': False,
        })
    
    df = pd.DataFrame(rows)
    
    if os.path.exists(TIMESERIES_PATH):
        existing = pd.read_csv(TIMESERIES_PATH)
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(TIMESERIES_PATH, index=False)
    
    active_pm25 = len([s for s in active if s.get('matricName') == 'PM25'])
    print(f"  Active PM25: {active_pm25} | Stale: {len(stale)} | Total rows: {len(df)}")

if __name__ == '__main__':
    main()
