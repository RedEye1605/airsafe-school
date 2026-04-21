#!/usr/bin/env python3
"""
Update progress tracker using Excel COM automation.
Requires: Microsoft Excel with Python pywin32 (Windows) OR xlwings (cross-platform).
"""
import sys

try:
    import openpyxl
    import os
except ImportError as e:
    print(f"Error: {e}")
    print("Please run: pip install openpyxl")
    sys.exit(1)

TRACKER_PATH = "/tmp/datathon-elevate/Datathon Elevate/Airsafe Team Progress Tracker.xlsm"

# Load to read structure
wb_read = openpyxl.load_workbook(TRACKER_PATH)

# Check what sheets exist
print("Available sheets in tracker:")
for sheet in wb_read.sheetnames:
    print(f"  - {sheet}")

print("\n" + "="*60)
print("Manual Update Instructions")
print("="*60)

print("\nTo update T-D1-03 and T-D1-04, open the .xlsm file in Excel and change:\n")

print("1. Go to 'Rhendy - Data' sheet")
print("2. Find these rows (Task ID in Column A):")
print("\n   Row ~11 (T-D1-03):")
print("     - Status (Column G): Change to 'Done'")
print("     - % Complete (Column H): Change to 100% or 1.0")
print("     - Evidence/Link (Column K): data/raw/spku/spku_sample.json + GeoJSON files")
print("     - Next Action (Column I): Move to Day 2 tasks")
print("\n   Row ~12 (T-D1-04):")
print("     - Status (Column G): Change to 'Done'")
print("     - % Complete (Column H): Change to 100% or 1.0")
print("     - Evidence/Link (Column K): data/raw/schools/ (3 CSV files: 3,974 schools total)")
print("     - Next Action (Column I): Start Day 2 geocoding pipeline")

print("\n3. Go to 'Shared & Gates' sheet")
print("4. Find Row ~13 (M-D1-GATE):")
print("     - Status (Column G): Change to 'Done'")
print("     - % Complete (Column H): Change to 100%")

print("\n" + "="*60)
print("Files created for your reference:")
print("="*60)
print("\n1. Tracker summary:")
print("   ~/airsafe-school/TRACKER_UPDATE_SUMMARY.txt")
print("\n2. Detailed completion report:")
print("   ~/airsafe-school/DATA_ACQUISITION_FINAL.md")
print("\n3. Data files:")
print("   ~/airsafe-school/data/raw/schools/ (3 CSV files)")
print("   ~/airsafe-school/data/raw/spku/ (spku_sample.json + 6 GeoJSON files)")
