#!/usr/bin/env python3
"""
T-D2-04: Geocode DKI Jakarta schools (fixed).
Geocode unique kelurahan centers, then assign to schools.
"""
import os, time, json, pandas as pd, requests

INPUT_DIR = os.path.expanduser("~/airsafe-school/data/raw/schools")
OUTPUT_DIR = os.path.expanduser("~/airsafe-school/data/processed/schools")
CACHE_FILE = os.path.expanduser("~/airsafe-school/data/processed/geocode_cache.json")
USER_AGENT = "AirSafeSchool/1.0 (rhendygio19@gmail.com)"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def geocode(query, cache):
    if query in cache:
        return cache[query]
    try:
        resp = requests.get("https://nominatim.openstreetmap.org/search", params={
            'q': query, 'format': 'json', 'limit': 1, 'countrycodes': 'id',
        }, headers={'User-Agent': USER_AGENT}, timeout=15)
        if resp.status_code == 200 and resp.json():
            r = resp.json()[0]
            cache[query] = {'lat': float(r['lat']), 'lon': float(r['lon']),
                           'display_name': r.get('display_name', ''), 'source': 'nominatim'}
        else:
            cache[query] = {'lat': None, 'lon': None, 'display_name': '', 'source': 'not_found'}
    except Exception as e:
        cache[query] = {'lat': None, 'lon': None, 'display_name': '', 'source': f'error'}
    time.sleep(1.1)
    return cache[query]

def main():
    cache = load_cache()
    
    # Load all school data
    all_dfs = []
    for f in ['sd_dki.csv', 'smp_dki.csv', 'sma_smk_dki.csv']:
        df = pd.read_csv(os.path.join(INPUT_DIR, f))
        all_dfs.append(df)
    schools = pd.concat(all_dfs, ignore_index=True)
    print(f"Total schools: {len(schools)}")
    
    # Get unique kelurahan
    kel_list = schools[['kota_kab', 'kecamatan', 'kelurahan']].drop_duplicates()
    print(f"Unique kelurahan: {len(kel_list)}")
    
    # Clear bad cache entries (all not_found from previous run)
    bad_keys = [k for k, v in cache.items() if v.get('lat') is None]
    for k in bad_keys:
        del cache[k]
    print(f"Cleared {len(bad_keys)} bad cache entries")
    
    kel_coords = {}
    for idx, (_, row) in enumerate(kel_list.iterrows()):
        # Simpler query without bounded constraint
        query = f"{row['kelurahan']}, {row['kecamatan']}, DKI Jakarta, Indonesia"
        result = geocode(query, cache)
        kel_coords[f"{row['kota_kab']}|{row['kecamatan']}|{row['kelurahan']}"] = result
        
        if (idx + 1) % 20 == 0:
            found = sum(1 for v in kel_coords.values() if v['lat'] is not None)
            print(f"  [{idx+1}/{len(kel_list)}] Found: {found}/{idx+1}")
            save_cache(cache)
    
    save_cache(cache)
    
    # Assign coordinates
    def get_coords(row):
        key = f"{row['kota_kab']}|{row['kecamatan']}|{row['kelurahan']}"
        c = kel_coords.get(key, {'lat': None, 'lon': None, 'display_name': '', 'source': 'not_found'})
        return pd.Series({'latitude': c['lat'], 'longitude': c['lon'],
                         'geocode_display': c.get('display_name', ''), 'geocode_source': c.get('source', '')})
    
    coords_df = schools.apply(get_coords, axis=1)
    schools = pd.concat([schools, coords_df], axis=1)
    
    output_path = os.path.join(OUTPUT_DIR, "schools_geocoded.csv")
    schools.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    found = schools['latitude'].notna().sum()
    print(f"\nDone: {found}/{len(schools)} geocoded ({found/len(schools)*100:.1f}%)")
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    main()
