#!/usr/bin/env python3
"""
T-D2-05 (Rebuild): Spatial coverage gap map with PM25-only sensors.
Shows schools within 2km of PM2.5 monitoring stations.
"""
import os
import pandas as pd
import folium
from folium import plugins

SCHOOL_PATH = os.path.expanduser("~/airsafe-school/data/processed/schools/schools_geocoded.csv")
SPKU_PATH = os.path.expanduser("~/airsafe-school/data/raw/spku/spku_stations_catalog.csv")
OUTPUT_PATH = os.path.expanduser("~/airsafe-school/data/eda_04_pm25_coverage_gap.html")

# Load schools with coordinates
schools = pd.read_csv(SCHOOL_PATH)
schools = schools.dropna(subset=['latitude', 'longitude'])
print(f"Schools with coords: {len(schools)}")

# Load SPKU stations - filter to PM25 only
spku = pd.read_csv(SPKU_PATH)
pm25_stations = spku[spku['parameter'] == 'PM25'].copy()
print(f"PM25 stations: {len(pm25_stations)}")

# Create map centered on Jakarta
jakarta_center = [-6.2088, 106.8456]
m = folium.Map(location=jakarta_center, zoom_start=11, tiles='OpenStreetMap')

# Add PM25 stations with 2km circles
station_group = folium.FeatureGroup(name='PM25 Stations (2km radius)')
for _, s in pm25_stations.iterrows():
    lat = float(s['latitude'])
    lon = float(s['longitude'])
    name = s.get('station_name', '')
    value = s.get('latest_value', 0)
    
    # Color by status (from status field if available)
    color = {'Baik': 'green', 'Sedang': 'orange', 'Tidak Sehat': 'red'}.get(s.get('status', ''), 'blue')
    
    folium.Marker(
        location=[lat, lon],
        popup=f"<b>{name}</b><br>PM2.5: {value} µg/m³<br>Status: {s.get('status', 'N/A')}",
        icon=folium.Icon(color=color, icon='info-sign')
    ).add_to(station_group)
    
    # 2km buffer
    folium.Circle(
        location=[lat, lon],
        radius=2000,  # 2km in meters
        color='blue', fill=True, fill_opacity=0.1, weight=1
    ).add_to(station_group)

station_group.add_to(m)

# Add schools
school_group = folium.FeatureGroup(name='Schools')
for _, s in schools.iterrows():
    folium.CircleMarker(
        location=[s['latitude'], s['longitude']],
        radius=3,
        color='red', fill=True, fill_opacity=0.5, weight=0.5,
        popup=f"<b>{s['nama_sekolah']}</b><br>{s['jenjang']}<br>{s['kota_kab']} - {s['kecamatan']}<br>{s['kelurahan']}"
    ).add_to(school_group)

school_group.add_to(m)

# Layer control
folium.LayerControl().add_to(m)

# Add title
total_stations = len(pm25_stations)
total_schools = len(schools)
title_html = f'''
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:9999;
            font-size:16px;font-weight:bold;background:white;padding:10px 15px;border-radius:8px;border:2px solid #333;">
  AirSafe School — PM2.5 Coverage Gap Map<br>
  <span style="font-size:12px;font-weight:normal;">
    🔵 {total_stations} PM2.5 Stations (2km radius) | 🔴 {total_schools} Schools
  </span>
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# Save
m.save(OUTPUT_PATH)
print(f"\nCoverage map saved: {OUTPUT_PATH}")
print(f"PM25 stations: {total_stations}")
print(f"Schools: {total_schools}")
print(f"Open in browser: file://{OUTPUT_PATH}")
