#!/usr/bin/env python3
"""
T-D2-05 (Rebuild): Spatial coverage gap map with 3 categories.
Shows schools within 2km of PM2.5 monitoring stations.
Categories: covered (green), outside coverage (orange), failed geocode (red).
"""
import os
import pandas as pd
import folium
from folium import plugins

SCHOOL_PATH = os.path.expanduser("~/airsafe-school/data/processed/schools/schools_geocoded.csv")
SPKU_PATH = os.path.expanduser("~/airsafe-school/data/raw/spku/spku_stations_catalog.csv")
OUTPUT_PATH = os.path.expanduser("~/airsafe-school/data/eda_04_pm25_coverage_gap.html")

# Load schools
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
    
    # Color by status
    color = {'Baik': 'green', 'Sedang': 'orange', 'Tidak Sehat': 'red'}.get(s.get('status', ''), 'blue')
    
    folium.Marker(
        location=[lat, lon],
        popup=f"<b>{name}</b><br>PM2.5: {value} µg/m³<br>Status: {s.get('status', 'N/A')}",
        icon=folium.Icon(color=color, icon='info-sign')
    ).add_to(station_group)
    
    # 2km buffer
    folium.Circle(
        location=[lat, lon],
        radius=2000,
        color='blue', fill=True, fill_opacity=0.1, weight=1
    ).add_to(station_group)

station_group.add_to(m)

# Add schools with 3 categories
covered_group = folium.FeatureGroup(name='Schools (Covered)')
outside_group = folium.FeatureGroup(name='Schools (Outside Coverage)')
failed_group = folium.FeatureGroup(name='Schools (Failed Geocode)')

for _, s in schools.iterrows():
    lat = s['latitude']
    lon = s['longitude']
    name = s['nama_sekolah']
    
    # Determine category
    is_failed = False
    is_covered = False
    is_outside_coverage = False
    
    # Check if covered by any PM25 station (2km radius)
    for _, st in pm25_stations.iterrows():
        st_lat = float(st['latitude'])
        st_lon = float(st['longitude'])
        dist = ((lat - st_lat)**2 + (lon - st_lon)**2)**0.5 * 111.19  # Approximate degrees to km
        if dist <= 2.0:
            is_covered = True
            break
    
    # Set group based on geocode status
    if pd.isna(s['latitude']) or pd.isna(s['longitude']):
        group = failed_group
        color = 'red'
        radius = 3
    elif is_covered:
        group = covered_group
        color = 'green'
        radius = 3
    else:
        group = outside_group
        color = 'orange'
        radius = 3
    
    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        color=color, fill=True, fill_opacity=0.5, weight=0.5,
        popup=f"<b>{name}</b><br>{s['jenjang']} - {s['kota_kab']}<br>{s['kecamatan']}<br>{s['kelurahan']}"
    ).add_to(group)

covered_group.add_to(m)
outside_group.add_to(m)
failed_group.add_to(m)

# Layer control
folium.LayerControl().add_to(m)

# Add title
total_stations = len(pm25_stations)
total_schools = len(schools)
covered = len(schools[schools['latitude'].notna()])

# Count schools within coverage
covered_count = 0
for _, s in schools.iterrows():
    lat = s['latitude']
    lon = s['longitude']
    for _, st in pm25_stations.iterrows():
        st_lat = float(st['latitude'])
        st_lon = float(st['longitude'])
        dist = ((lat - st_lat)**2 + (lon - st_lon)**2)**0.5 * 111.19
        if dist <= 2.0:
            covered_count += 1
            break

outside_count = len(schools) - covered_count - failed
failed_count = schools['latitude'].isna().sum()

title_html = f'''
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:9999;
            font-size:16px;font-weight:bold;background:white;padding:10px 15px;border-radius:8px;border:2px solid #333;">
  AirSafe School — PM2.5 Coverage Gap Map<br>
  <span style="font-size:12px;font-weight:normal;">
    🔵 {total_stations} PM2.5 Stations (2km radius)<br>
    🟢 Covered: {covered:,} schools<br>
    🟠 Outside: {outside_count:,} schools<br>
    🔴 Failed: {failed_count:,} schools<br>
    <small>(Total: {total_schools:,} schools)</small>
  </span>
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# Save
m.save(OUTPUT_PATH)
print(f"Coverage map saved: {OUTPUT_PATH}")
print(f"\n" + "="*70)
print(f"PM25 stations: {total_stations}")
print(f"Total schools: {total_schools}")
print(f"Covered: {covered}")
print(f"Outside coverage: {outside_count}")
print(f"Failed geocode: {failed_count}")
print(f"\nOpen in browser: file://{OUTPUT_PATH}")
