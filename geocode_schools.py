#!/usr/bin/env python3
"""
T-D2-04: Geocode all DKI Jakarta schools using Nominatim.
Rate limited to 1 req/s with caching.
"""
import os
import sys
import time
import json
import pandas as pd
import requests
from pathlib import Path
from urllib.parse import quote_plus

# Config
INPUT_DIR = os.path.expanduser("~/airsafe-school/data/raw/schools")
OUTPUT_DIR = os.path.expanduser("~/airsafe-school/data/processed/schools")
CACHE_FILE = os.path.expanduser("~/airsafe-school/data/processed/geocode_cache.json")
USER_AGENT = "AirSafeSchool/1.0 (rhendygio19@gmail.com)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
RATE_LIMIT = 1.1  # seconds between requests (Nominatim requires max 1 req/s)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def geocode_nominatim(query, cache):
    """Geocode using Nominatim with caching."""
    if query in cache:
        return cache[query]
    
    try:
        resp = requests.get(NOMINATIM_URL, params={
            'q': query,
            'format': 'json',
            'limit': 1,
            'countrycodes': 'id',
            'viewbox': '106.4,-6.55,107.15,-5.9',  # DKI Jakarta bounding box
            'bounded': 1,
        }, headers={'User-Agent': USER_AGENT}, timeout=15)
        
        if resp.status_code == 200 and resp.json():
            result = resp.json()[0]
            cache[query] = {
                'lat': float(result['lat']),
                'lon': float(result['lon']),
                'display_name': result.get('display_name', ''),
                'type': result.get('type', ''),
                'source': 'nominatim'
            }
        else:
            cache[query] = {'lat': None, 'lon': None, 'display_name': '', 'type': '', 'source': 'not_found'}
    except Exception as e:
        cache[query] = {'lat': None, 'lon': None, 'display_name': '', 'type': '', 'source': f'error: {str(e)[:50]}'}
    
    return cache[query]

def geocode_schools():
    """Geocode all school CSV files."""
    cache = load_cache()
    
    files = ['sd_dki.csv', 'smp_dki.csv', 'sma_smk_dki.csv']
    all_geocoded = []
    
    for fname in files:
        filepath = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(filepath):
            print(f"Skipping {fname} (not found)")
            continue
        
        df = pd.read_csv(filepath)
        print(f"\nProcessing {fname}: {len(df)} schools")
        
        geocoded_rows = []
        for idx, row in df.iterrows():
            # Try multiple query strategies
            queries = [
                f"{row['nama_sekolah']}, {row['kelurahan']}, {row['kota_kab']}, Jakarta, Indonesia",
                f"{row['nama_sekolah']}, {row['kecamatan']}, {row['kota_kab']}, Jakarta, Indonesia",
                f"{row['alamat']}, {row['kelurahan']}, {row['kota_kab']}, Jakarta, Indonesia",
                f"{row['nama_sekolah']}, Jakarta, Indonesia",
            ]
            
            result = {'lat': None, 'lon': None, 'display_name': '', 'type': '', 'source': 'not_found'}
            for query in queries:
                result = geocode_nominatim(query, cache)
                if result.get('lat') is not None:
                    break
                time.sleep(0.1)  # Small delay between retries
            
            row_dict = row.to_dict()
            row_dict['latitude'] = result.get('lat')
            row_dict['longitude'] = result.get('lon')
            row_dict['geocode_display'] = result.get('display_name', '')
            row_dict['geocode_type'] = result.get('type', '')
            row_dict['geocode_source'] = result.get('source', '')
            geocoded_rows.append(row_dict)
            
            # Progress
            if (idx + 1) % 50 == 0:
                found = sum(1 for r in geocoded_rows if r['latitude'] is not None)
                print(f"  [{idx+1}/{len(df)}] Found: {found}/{idx+1} ({found/(idx+1)*100:.1f}%)")
                save_cache(cache)
            
            time.sleep(RATE_LIMIT)
        
        geocoded_df = pd.DataFrame(geocoded_rows)
        found = geocoded_df['latitude'].notna().sum()
        print(f"  Done: {found}/{len(geocoded_df)} geocoded ({found/len(geocoded_df)*100:.1f}%)")
        all_geocoded.append(geocoded_df)
    
    # Combine all
    combined = pd.concat(all_geocoded, ignore_index=True)
    output_path = os.path.join(OUTPUT_DIR, "schools_geocoded.csv")
    combined.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    found = combined['latitude'].notna().sum()
    print(f"\n{'='*60}")
    print(f"Total: {found}/{len(combined)} schools geocoded ({found/len(combined)*100:.1f}%)")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}")
    
    save_cache(cache)
    return output_path

if __name__ == '__main__':
    geocode_schools()
