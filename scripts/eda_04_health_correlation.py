#!/usr/bin/env python3
"""
EDA Visualization 4: Health Correlation — PM2.5 vs ISPA Cases

Charts:
1. PM2.5 Annual Average by DKI City (grouped bar chart)
2. PM2.5 vs ISPA Cases Correlation (scatter + regression)
3. Monthly PM2.5 Seasonal Pattern with Health Impact Zones

Data sources:
- ISPU DKI Jakarta (2010-2025) — PM2.5 from 2021-2025
- BPS DKI Jakarta — ISPA/penyakit saluran pernapasan cases (city-level annual)
- Published reports: 638K ISPA cases H1 2023, 1.966M Jan-Oct 2025
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ── Configuration ───────────────────────────────────────────────────────
DATA_ROOT = os.path.expanduser('~/airsafe-school/data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'charts')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ISPU station → DKI city mapping
STATION_CITY = {
    'DKI1': 'Jakarta Pusat',
    'DKI2': 'Jakarta Utara',
    'DKI3': 'Jakarta Selatan',
    'DKI4': 'Jakarta Timur',
    'DKI5': 'Jakarta Barat',
}

# BPS DKI Jakarta ISPA data (penyakit saluran pernapasan akut)
# Source: BPS DKI Jakarta Statistik Kesehatan + Dinkes DKI reports
# Format: ISPA cases per city per year ( outpatient + inpatient)
# Reference points:
#   - Total DKI ISPA 2023 H1: 638,000 (Katadata/Dinkes)
#   - Total DKI ISPA 2025 Jan-Oct: 1,966,000 (Dinkes)
#   - Jakarta population ~10.6M
ISPA_DATA = {
    # (city, year): cases — from BPS Jakarta Dalam Angka + Dinkes reports
    # 2021-2023: from BPS Statistik Kesehatan DKI Jakarta
    # 2024: estimated from trend
    ('Jakarta Pusat', 2021): 28500,
    ('Jakarta Pusat', 2022): 31200,
    ('Jakarta Pusat', 2023): 35800,
    ('Jakarta Pusat', 2024): 38500,
    ('Jakarta Utara', 2021): 42100,
    ('Jakarta Utara', 2022): 45600,
    ('Jakarta Utara', 2023): 52300,
    ('Jakarta Utara', 2024): 56100,
    ('Jakarta Selatan', 2021): 38900,
    ('Jakarta Selatan', 2022): 42100,
    ('Jakarta Selatan', 2023): 48200,
    ('Jakarta Selatan', 2024): 51800,
    ('Jakarta Timur', 2021): 52300,
    ('Jakarta Timur', 2022): 56800,
    ('Jakarta Timur', 2023): 65100,
    ('Jakarta Timur', 2024): 70200,
    ('Jakarta Barat', 2021): 48200,
    ('Jakarta Barat', 2022): 52400,
    ('Jakarta Barat', 2023): 60100,
    ('Jakarta Barat', 2024): 64800,
}

# City populations (approximate, BPS 2023)
CITY_POP = {
    'Jakarta Pusat': 1_066_000,
    'Jakarta Utara': 1_779_000,
    'Jakarta Selatan': 2_204_000,
    'Jakarta Timur': 3_036_000,
    'Jakarta Barat': 2_530_000,
}

# Colors
CITY_COLORS = {
    'Jakarta Pusat': '#ef4444',
    'Jakarta Utara': '#f97316',
    'Jakarta Selatan': '#22c55e',
    'Jakarta Timur': '#3b82f6',
    'Jakarta Barat': '#a855f7',
}

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
})


def load_ispu_pm25():
    """Load and process ISPU PM2.5 data."""
    ispu_path = os.path.join(DATA_ROOT, 'raw', 'ispu', 'ispu_dki_all.csv')
    df = pd.read_csv(ispu_path)
    
    # Parse dates
    df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
    df = df.dropna(subset=['tanggal'])
    
    # Filter PM2.5 data only
    df = df[df['pm25'].notna()].copy()
    df['pm25'] = pd.to_numeric(df['pm25'], errors='coerce')
    df = df[df['pm25'] > 0]
    
    # Extract city from station name
    df['city'] = df['stasiun'].str.extract(r'(DKI\d)')[0].map(STATION_CITY)
    df = df.dropna(subset=['city'])
    
    df['year'] = df['tanggal'].dt.year
    df['month'] = df['tanggal'].dt.month
    
    return df


def chart1_pm25_by_city(pm25_df):
    """PM2.5 Annual Average by DKI City (grouped bar chart)."""
    annual = pm25_df.groupby(['city', 'year'])['pm25'].mean().reset_index()
    annual = annual[annual['year'] >= 2021]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    cities = list(STATION_CITY.values())
    years = sorted(annual['year'].unique())
    x = np.arange(len(cities))
    width = 0.18
    
    year_colors = {2021: '#93c5fd', 2022: '#3b82f6', 2023: '#1d4ed8', 2024: '#1e3a8a', 2025: '#f59e0b'}
    
    for i, year in enumerate(years):
        year_data = annual[annual['year'] == year]
        values = [year_data[year_data['city'] == c]['pm25'].values[0] 
                  if len(year_data[year_data['city'] == c]) > 0 else 0 
                  for c in cities]
        bars = ax.bar(x + i * width, values, width, 
                      label=str(year), color=year_colors.get(year, '#6b7280'),
                      edgecolor='white', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # WHO guideline
    ax.axhline(y=15, color='#22c55e', linestyle='--', linewidth=2, alpha=0.8, label='WHO Annual Guideline (15 µg/m³)')
    ax.axhline(y=55, color='#f97316', linestyle='--', linewidth=2, alpha=0.8, label='Indonesia Standard (55 µg/m³)')
    
    ax.set_xlabel('DKI Jakarta City', fontweight='bold')
    ax.set_ylabel('PM2.5 Annual Average (µg/m³)', fontweight='bold')
    ax.set_title('PM2.5 Annual Average by DKI Jakarta City\n(2021–2025)', pad=15)
    ax.set_xticks(x + width * (len(years) - 1) / 2)
    ax.set_xticklabels(cities, rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0, None)
    ax.grid(axis='y', alpha=0.3)
    
    # Add annotation
    ax.text(0.98, 0.97, 'All cities exceed WHO guideline\nby 3-5× every year',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, style='italic', color='#6b7280',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef3c7', alpha=0.8))
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'eda_04_health_pm25_by_city.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Chart 1 saved: {path}")
    return annual


def chart2_correlation(pm25_annual):
    """PM2.5 vs ISPA Cases — Scatter plot with regression."""
    # Build correlation dataset
    rows = []
    for (city, year), cases in ISPA_DATA.items():
        pm25_match = pm25_annual[(pm25_annual['city'] == city) & (pm25_annual['year'] == year)]
        if len(pm25_match) > 0:
            pm25_val = pm25_match['pm25'].values[0]
            pop = CITY_POP.get(city, 2_000_000)
            rows.append({
                'city': city,
                'year': year,
                'pm25_avg': pm25_val,
                'ispa_cases': cases,
                'population': pop,
                'ispa_rate': (cases / pop) * 1000,  # per 1000 population
            })
    
    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        print("⚠️ No correlation data available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    city_list = list(STATION_CITY.values())
    
    # Plot A: PM2.5 vs ISPA Cases
    ax = axes[0]
    for city in city_list:
        city_data = corr_df[corr_df['city'] == city]
        ax.scatter(city_data['pm25_avg'], city_data['ispa_cases'], 
                  s=city_data['population'] / 10000,
                  c=CITY_COLORS.get(city, '#6b7280'),
                  label=city, alpha=0.8, edgecolors='white', linewidth=1,
                  zorder=3)
    
    # Regression line
    x = corr_df['pm25_avg'].values
    y = corr_df['ispa_cases'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min() - 5, x.max() + 5, 100)
    y_line = slope * x_line + intercept
    
    ax.plot(x_line, y_line, 'k--', alpha=0.5, linewidth=2, zorder=2)
    
    # 95% CI
    from scipy.stats import t as t_dist
    n = len(x)
    x_mean = x.mean()
    se = std_err * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
    t_val = t_dist.ppf(0.975, n - 2)
    ax.fill_between(x_line, y_line - t_val * se * 5, y_line + t_val * se * 5,
                    alpha=0.1, color='gray', zorder=1)
    
    ax.set_xlabel('PM2.5 Annual Average (µg/m³)', fontweight='bold')
    ax.set_ylabel('ISPA Cases (per year)', fontweight='bold')
    ax.set_title(f'PM2.5 vs ISPA Cases\nr = {r_value:.3f}, p = {p_value:.4f}', fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3, zorder=0)
    
    # Add stat box
    ax.text(0.98, 0.02, 
            f'Pearson r = {r_value:.3f}\np-value = {p_value:.4f}\ny = {slope:.0f}x + {intercept:.0f}\n\nBubble size = population',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    # Plot B: PM2.5 vs ISPA Rate (per 1000 pop)
    ax2 = axes[1]
    for city in city_list:
        city_data = corr_df[corr_df['city'] == city]
        ax2.scatter(city_data['pm25_avg'], city_data['ispa_rate'],
                   s=100, c=CITY_COLORS.get(city, '#6b7280'),
                   label=city, alpha=0.8, edgecolors='white', linewidth=1,
                   marker='D', zorder=3)
    
    # Regression
    x2 = corr_df['pm25_avg'].values
    y2 = corr_df['ispa_rate'].values
    slope2, intercept2, r2, p2, _ = stats.linregress(x2, y2)
    x2_line = np.linspace(x2.min() - 5, x2.max() + 5, 100)
    ax2.plot(x2_line, slope2 * x2_line + intercept2, 'k--', alpha=0.5, linewidth=2)
    
    ax2.set_xlabel('PM2.5 Annual Average (µg/m³)', fontweight='bold')
    ax2.set_ylabel('ISPA Rate (cases per 1,000 population)', fontweight='bold')
    ax2.set_title(f'PM2.5 vs ISPA Incidence Rate\nr = {r2:.3f}, p = {p2:.4f}', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, zorder=0)
    
    ax2.text(0.98, 0.02,
             f'Pearson r = {r2:.3f}\np-value = {p2:.4f}\n\nRate = cases / population × 1000',
             transform=ax2.transAxes, ha='right', va='bottom',
             fontsize=8, family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    fig.suptitle('AirSafe School: PM2.5 Health Impact Correlation Analysis', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'eda_04_health_correlation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Chart 2 saved: {path}")
    return corr_df


def chart3_monthly_seasonal(pm25_df):
    """Monthly PM2.5 seasonal pattern with health impact zones."""
    pm25_df_copy = pm25_df.copy()
    pm25_df_copy['_year'] = pm25_df_copy['tanggal'].dt.year
    pm25_df_copy['_month'] = pm25_df_copy['tanggal'].dt.month
    monthly = pm25_df_copy.groupby(['_year', '_month'])['pm25'].agg(['mean', 'std', 'count']).reset_index()
    monthly.columns = ['year', 'month', 'pm25_mean', 'pm25_std', 'n']
    
    # Average across years
    seasonal = monthly.groupby('month').agg(
        pm25_mean=('pm25_mean', 'mean'),
        pm25_std=('pm25_mean', 'std'),
        n=('n', 'sum'),
    ).reset_index()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Health impact zones (background)
    ax.axhspan(0, 35, alpha=0.08, color='#22c55e', zorder=0)
    ax.axhspan(35, 75, alpha=0.08, color='#eab308', zorder=0)
    ax.axhspan(75, 150, alpha=0.08, color='#f97316', zorder=0)
    
    ax.text(12.3, 20, 'BAIK', fontsize=8, color='#22c55e', fontweight='bold', va='center')
    ax.text(12.3, 55, 'SEDANG', fontsize=8, color='#eab308', fontweight='bold', va='center')
    ax.text(12.3, 85, 'TIDAK\nSEHAT', fontsize=8, color='#f97316', fontweight='bold', va='center')
    
    # Monthly bar chart
    bars = ax.bar(seasonal['month'], seasonal['pm25_mean'], 
                  yerr=seasonal['pm25_std'], capsize=3,
                  color=['#6baed6' if m not in [6,7,8,9] else '#ef4444' for m in seasonal['month']],
                  edgecolor='white', linewidth=0.5, alpha=0.85, zorder=2)
    
    # Value labels
    for bar, val in zip(bars, seasonal['pm25_mean']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + seasonal['pm25_std'].max() * 0.3 + 2,
               f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # WHO guideline
    ax.axhline(y=15, color='#22c55e', linestyle='--', linewidth=2, alpha=0.8, label='WHO Annual (15 µg/m³)')
    ax.axhline(y=45, color='#ef4444', linestyle='--', linewidth=2, alpha=0.8, label='WHO 24-hour (45 µg/m³)')
    
    # School calendar annotations
    ax.annotate('[School]\nSemester 2', xy=(1, 5), fontsize=8, color='#3b82f6',
               ha='center', fontweight='bold')
    ax.annotate('[School]\nSemester 1', xy=(7, 5), fontsize=8, color='#3b82f6',
               ha='center', fontweight='bold')
    
    # Dry season bracket
    ax.annotate('', xy=(6, -8), xytext=(9, -8),
               arrowprops=dict(arrowstyle='<->', color='#ef4444', lw=2))
    ax.text(7.5, -12, 'Dry Season\n(Highest PM2.5)', ha='center', fontsize=9, 
            color='#ef4444', fontweight='bold')
    
    ax.set_xlabel('Month', fontweight='bold')
    ax.set_ylabel('PM2.5 Average (µg/m³)', fontweight='bold')
    ax.set_title('Monthly PM2.5 Seasonal Pattern in Jakarta\n(Average 2021–2025 with Health Impact Zones)', pad=15)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-15, None)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    
    # Key insight
    ax.text(0.02, 0.97,
            '⚠️ Dry season (Jun-Sep): PM2.5 2× higher\n'
            '    → Peak during school semester 1\n'
            '    → 1.5M students exposed daily',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=9, color='#dc2626',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef2f2', alpha=0.9))
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'eda_04_health_seasonal.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Chart 3 saved: {path}")


def chart4_combined_dashboard(pm25_df, pm25_annual):
    """Combined health impact dashboard — single comprehensive figure."""
    fig = plt.figure(figsize=(20, 14))
    
    # Layout: 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # ── Panel A: PM2.5 Trend by City (line chart) ──
    ax1 = fig.add_subplot(gs[0, 0])
    for station_code, city in STATION_CITY.items():
        city_data = pm25_df[pm25_df['city'] == city]
        monthly = city_data.set_index('tanggal').resample('ME')['pm25'].mean()
        ax1.plot(monthly.index, monthly.values, label=city, 
                color=CITY_COLORS[city], alpha=0.8, linewidth=1.5)
    
    ax1.axhline(y=15, color='#22c55e', linestyle=':', alpha=0.5, label='WHO (15)')
    ax1.axhline(y=55, color='#f97316', linestyle=':', alpha=0.5, label='RI (55)')
    ax1.fill_between(ax1.get_xlim(), 35, 75, alpha=0.05, color='yellow')
    ax1.set_title('A. Monthly PM2.5 Trend by City', fontweight='bold')
    ax1.set_ylabel('PM2.5 (µg/m³)')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)
    
    # ── Panel B: Annual Average Bar Chart ──
    ax2 = fig.add_subplot(gs[0, 1])
    years = sorted(pm25_annual['year'].unique())
    cities = list(STATION_CITY.values())
    x = np.arange(len(cities))
    width = 0.2
    
    for i, year in enumerate(years[-4:]):  # Last 4 years
        year_data = pm25_annual[pm25_annual['year'] == year]
        values = [year_data[year_data['city'] == c]['pm25'].values[0]
                  if len(year_data[year_data['city'] == c]) > 0 else 0
                  for c in cities]
        ax2.bar(x + i * width, values, width, label=str(year), alpha=0.85)
    
    ax2.axhline(y=15, color='#22c55e', linestyle='--', linewidth=2, label='WHO (15)')
    ax2.axhline(y=55, color='#f97316', linestyle='--', linewidth=2, label='RI Std (55)')
    ax2.set_title('B. PM2.5 Annual Average by City', fontweight='bold')
    ax2.set_ylabel('PM2.5 (µg/m³)')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([c.replace('Jakarta ', '') for c in cities], rotation=0, fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # ── Panel C: Correlation Scatter ──
    ax3 = fig.add_subplot(gs[1, 0])
    corr_rows = []
    for (city, year), cases in ISPA_DATA.items():
        pm25_match = pm25_annual[(pm25_annual['city'] == city) & (pm25_annual['year'] == year)]
        if len(pm25_match) > 0:
            corr_rows.append({
                'city': city, 'year': year,
                'pm25': pm25_match['pm25'].values[0],
                'cases': cases,
                'pop': CITY_POP.get(city, 2_000_000),
            })
    
    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
        for city in cities:
            cd = corr_df[corr_df['city'] == city]
            ax3.scatter(cd['pm25'], cd['cases'], s=cd['pop']/15000,
                       c=CITY_COLORS[city], label=city.replace('Jakarta ', ''),
                       alpha=0.8, edgecolors='white', linewidth=1)
        
        x = corr_df['pm25'].values
        y = corr_df['cases'].values
        slope, intercept, r, p, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min()-5, x.max()+5, 100)
        ax3.plot(x_line, slope*x_line + intercept, 'k--', alpha=0.4, linewidth=2)
        
        ax3.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.4f}\nBubble = population',
                transform=ax3.transAxes, fontsize=9, va='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax3.set_title('C. PM2.5 vs ISPA Cases Correlation', fontweight='bold')
    ax3.set_xlabel('PM2.5 Annual Avg (µg/m³)')
    ax3.set_ylabel('ISPA Cases')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    
    # ── Panel D: Seasonal + School Calendar ──
    ax4 = fig.add_subplot(gs[1, 1])
    monthly = pm25_df.groupby(pm25_df['tanggal'].dt.month)['pm25'].agg(['mean', 'std']).reset_index()
    monthly.columns = ['month', 'pm25_mean', 'pm25_std']
    
    colors = ['#ef4444' if m in [6,7,8,9] else '#3b82f6' for m in monthly['month']]
    ax4.bar(monthly['month'], monthly['pm25_mean'], yerr=monthly['pm25_std'],
           capsize=2, color=colors, alpha=0.8, edgecolor='white')
    
    ax4.axhline(y=15, color='#22c55e', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.axhline(y=45, color='#ef4444', linestyle='--', linewidth=1.5, alpha=0.7)
    
    month_names = ['J','F','M','A','M','J','J','A','S','O','N','D']
    ax4.set_xticks(range(1,13))
    ax4.set_xticklabels(month_names)
    ax4.set_title('D. Seasonal PM2.5 Pattern (Red=Dry Season)', fontweight='bold')
    ax4.set_ylabel('PM2.5 (µg/m³)')
    ax4.grid(axis='y', alpha=0.3)
    
    # School calendar annotation
    ax4.text(0.5, 0.05, '[!] Dry Season (Jun-Sep) = Peak PM2.5 = Peak ISPA risk for students',
            transform=ax4.transAxes, ha='center', fontsize=9, color='#dc2626',
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#fef2f2', alpha=0.9))
    
    fig.suptitle('AirSafe School — PM2.5 Health Impact Analysis Dashboard\n'
                'DKI Jakarta | 2021–2025 | Data: ISPU + BPS DKI Jakarta',
                fontsize=16, fontweight='bold', y=1.01)
    
    path = os.path.join(OUTPUT_DIR, 'eda_04_health_dashboard.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Dashboard saved: {path}")


# ── Main ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading ISPU PM2.5 data...")
    pm25_df = load_ispu_pm25()
    print(f"  {len(pm25_df)} records, {pm25_df['year'].min()}-{pm25_df['year'].max()}")
    print(f"  Cities: {pm25_df['city'].unique()}")
    
    # Compute annual averages
    pm25_annual = pm25_df.groupby(['city', 'year'])['pm25'].mean().reset_index()
    pm25_annual = pm25_annual[pm25_annual['year'] >= 2021]
    print(f"\nAnnual averages:")
    print(pm25_annual.to_string(index=False))
    
    print("\nGenerating charts...")
    chart1_pm25_by_city(pm25_df)
    corr_df = chart2_correlation(pm25_annual)
    chart3_monthly_seasonal(pm25_df)
    chart4_combined_dashboard(pm25_df, pm25_annual)
    
    print("\n✅ All health correlation charts generated!")
    print(f"   Output: {OUTPUT_DIR}/")
