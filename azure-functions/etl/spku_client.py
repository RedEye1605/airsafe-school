"""
SPKU API Client — Pulls real-time air quality data from Jakarta's SPKU network.

API: https://udara.jakarta.go.id/api/lokasi_stasiun_udara
Method: POST (DataTables-style request)
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

API_URL = "https://udara.jakarta.go.id/api/lokasi_stasiun_udara"
HEADERS = {
    'User-Agent': 'AirSafeSchool/1.0',
    'X-Requested-With': 'XMLHttpRequest',
    'Content-Type': 'application/x-www-form-urlencoded'
}

# Jakarta city mapping for ISPU stations
ISPU_STATION_CITY = {
    'DKI1': 'Jakarta Pusat',
    'DKI2': 'Jakarta Utara',
    'DKI3': 'Jakarta Selatan',
    'DKI4': 'Jakarta Timur',
    'DKI5': 'Jakarta Barat',
}

# Approximate coordinates for ISPU stations
ISPU_STATION_COORDS = {
    'DKI1': (-6.1753, 106.8272),   # Bunderan HI
    'DKI2': (-6.1578, 106.9067),   # Kelapa Gading
    'DKI3': (-6.3692, 106.8186),   # Jagakarsa
    'DKI4': (-6.2942, 106.8897),   # Lubang Buaya
    'DKI5': (-6.1909, 106.7369),   # Kebon Jeruk
}


def fetch_all_stations(stale_days: int = 7) -> dict:
    """
    Fetch all SPKU station readings.
    
    Returns dict with:
        - collected_at: ISO timestamp
        - total: total station count
        - active: stations reporting within stale_days
        - stations: list of station dicts
    """
    data = {
        'draw': '1',
        'start': '0',
        'length': '500',
        'columns[0][data]': 'tgl',
        'columns[0][name]': 'tgl',
        'columns[0][searchable]': 'true',
        'columns[0][search][value]': '',
        'columns[0][search][regex]': 'false',
        'order[0][column]': '0',
        'order[0][dir]': 'desc',
        'search[value]': '',
        'search[regex]': 'false',
    }
    
    resp = requests.post(API_URL, headers=HEADERS, data=data, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    
    stations_raw = result.get('data', [])
    now = datetime.now()
    cutoff = now - timedelta(days=stale_days)
    
    stations = []
    for s in stations_raw:
        try:
            ts = datetime.strptime(s.get('tgl', ''), '%m/%d/%Y %H:%M:%S')
            is_active = ts >= cutoff
        except (ValueError, TypeError):
            ts = None
            is_active = False
        
        stations.append({
            'station_id': s.get('dataSourceID', ''),
            'station_name': s.get('dataSourceName', ''),
            'latitude': float(s.get('latitude', 0)),
            'longitude': float(s.get('longitude', 0)),
            'parameter': s.get('matricName', ''),
            'value': _parse_float(s.get('value')),
            'status': s.get('status', ''),
            'timestamp': s.get('tgl', ''),
            'timestamp_parsed': ts.isoformat() if ts else None,
            'is_active': is_active,
        })
    
    active_count = sum(1 for s in stations if s['is_active'])
    pm25_active = [s for s in stations if s['is_active'] and s['parameter'] == 'PM25']
    
    logger.info(f"SPKU: {len(stations)} total, {active_count} active, {len(pm25_active)} active PM2.5")
    
    return {
        'collected_at': now.isoformat(),
        'total': len(stations),
        'active': active_count,
        'active_pm25': len(pm25_active),
        'stations': stations,
    }


def extract_pm25_readings(spku_data: dict) -> list[dict]:
    """
    Extract PM2.5 readings from SPKU snapshot into flat records.
    """
    records = []
    collected_at = spku_data['collected_at']
    
    for s in spku_data['stations']:
        if s['parameter'] != 'PM25':
            continue
        records.append({
            'collection_time': collected_at,
            'station_name': s['station_name'],
            'station_id': s['station_id'],
            'latitude': s['latitude'],
            'longitude': s['longitude'],
            'pm25': s['value'],
            'status': s['status'],
            'is_active': s['is_active'],
            'station_timestamp': s['timestamp'],
        })
    
    return records


def _parse_float(val) -> Optional[float]:
    """Safely parse a value to float."""
    if val is None or val == '' or val == '-':
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
