#!/usr/bin/env python3
"""
T-D2-05: Spatial coverage gap map.
Shows SPKU sensor coverage vs school locations with 2km radius.
"""
import os, json, pandas as pd
import folium
from folium import plugins

SPKU_PATH = os.path.expanduser("~/airsafe-school/data/raw/spku/spku_sample.json")
SCHOOL_PATH = os.path.expanduser("~/airsafe-school/data/processed/schools/schools_geocoded.csv")
OUTPUT_PATH = os.path.expanduser("~/airsafe-school/data/eda_03_coverage_gap.html")

def main():
    # Load SPKU stations
    with open(SPKU_PATH) as f:
        spku_data = json.load(f)
    stations = spku_data.get('data', [])
    station_coords = []
    for s in stations:
        lat = float(s.get('latitude', 0))
        lon = float(s.get('longitude', 0))
        if lat != 0 and lon != 0:
            station_coords.append({
                'name': s.get('dataSourceName', ''),
                'lat': lat, 'lon': lon,
                'pm25': s.get('value', 0),
                'status': s.get('status', '')
            })
    df_stations = pd.DataFrame(station_coords).drop_duplicates(subset=['name'])
    print(f"SPKU stations: {len(df_stations)}")
    
    # Load schools
    df_schools = pd.read_csv(SCHOOL_PATH)
    df_schools = df_schools.dropna(subset=['latitude', 'longitude'])
    print(f"Schools with coords: {len(df_schools)}")
    
    # Create map centered on Jakarta
    jakarta_center = [-6.2088, 106.8456]
    m = folium.Map(location=jakarta_center, zoom_start=11, tiles='OpenStreetMap')
    
    # Add SPKU stations with 2km radius circles
    station_group = folium.FeatureGroup(name='SPKU Stations (2km radius)')
    for _, s in df_stations.iterrows():
        # Color based on status
        color = {'Baik': 'green', 'Sedang': 'orange', 'Tidak Sehat': 'red'}.get(s['status'], 'blue')
        
        # Station marker
        folium.Marker(
            location=[s['lat'], s['lon']],
            popup=f"<b>{s['name']}</b><br>PM2.5: {s['pm25']} µg/m³<br>Status: {s['status']}",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(station_group)
        
        # 2km radius circle
        folium.Circle(
            location=[s['lat'], s['lon']],
            radius=2000,  # 2km in meters
            color='blue', fill=True, fill_opacity=0.1, weight=1
        ).add_to(station_group)
    
    station_group.add_to(m)
    
    # Add schools
    school_group = folium.FeatureGroup(name='Schools')
    for _, s in df_schools.iterrows():
        folium.CircleMarker(
            location=[s['latitude'], s['longitude']],
            radius=2, color='red', fill=True, fill_opacity=0.5,
            popup=f"<b>{s['nama_sekolah']}</b><br>{s['jenjang']}<br>{s['kelurahan']}"
        ).add_to(school_group)
    school_group.add_to(m)
    
    # Layer control
    folium.LayerControl().add_to(m)
    
    # Add title
    title_html = '''
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
    z-index:9999;font-size:16px;font-weight:bold;background:white;padding:8px 15px;
    border-radius:5px;border:2px solid #333">
    AirSafe School — SPKU Coverage Gap Map<br>
    <span style="font-size:12px;font-weight:normal">
    🔵 SPKU Stations (2km radius) | 🔴 Schools (n={schools}) | Stations: {stations}
    </span></div>
    '''.format(schools=len(df_schools), stations=len(df_stations))
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save
    m.save(OUTPUT_PATH)
    print(f"\nCoverage map saved: {OUTPUT_PATH}")
    print(f"Stations: {len(df_stations)}")
    print(f"Schools: {len(df_schools)}")
    print(f"Open in browser: file://{OUTPUT_PATH}")

if __name__ == '__main__':
    main()
