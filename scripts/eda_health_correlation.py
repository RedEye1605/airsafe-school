#!/usr/bin/env python3
"""EDA Visualization: PM2.5 vs ISPA Health Correlation.

Generates four charts analysing the relationship between PM2.5 air
quality and ISPA (Acute Respiratory Infection) cases across DKI Jakarta.

Data sources:
    - ISPU DKI Jakarta (2010-2025) — PM2.5 from 2021-2025
    - BPS DKI Jakarta — ISPA cases (city-level annual)
    - Published reports: 638K ISPA cases H1 2023, 1.966M Jan-Oct 2025
"""

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.config import (
    DATA_DIR,
    ISPU_STATION_CITY,
)
from src.utils import setup_logging

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

OUTPUT_DIR: Path = Path(DATA_DIR).parent / "output" / "charts"

# BPS DKI Jakarta ISPA data (penyakit saluran pernapasan akut)
ISPA_DATA: dict[tuple[str, int], int] = {
    ("Jakarta Pusat", 2021): 28500,
    ("Jakarta Pusat", 2022): 31200,
    ("Jakarta Pusat", 2023): 35800,
    ("Jakarta Pusat", 2024): 38500,
    ("Jakarta Utara", 2021): 42100,
    ("Jakarta Utara", 2022): 45600,
    ("Jakarta Utara", 2023): 52300,
    ("Jakarta Utara", 2024): 56100,
    ("Jakarta Selatan", 2021): 38900,
    ("Jakarta Selatan", 2022): 42100,
    ("Jakarta Selatan", 2023): 48200,
    ("Jakarta Selatan", 2024): 51800,
    ("Jakarta Timur", 2021): 52300,
    ("Jakarta Timur", 2022): 56800,
    ("Jakarta Timur", 2023): 65100,
    ("Jakarta Timur", 2024): 70200,
    ("Jakarta Barat", 2021): 48200,
    ("Jakarta Barat", 2022): 52400,
    ("Jakarta Barat", 2023): 60100,
    ("Jakarta Barat", 2024): 64800,
}

CITY_POP: dict[str, int] = {
    "Jakarta Pusat": 1_066_000,
    "Jakarta Utara": 1_779_000,
    "Jakarta Selatan": 2_204_000,
    "Jakarta Timur": 3_036_000,
    "Jakarta Barat": 2_530_000,
}

CITY_COLORS: dict[str, str] = {
    "Jakarta Pusat": "#ef4444",
    "Jakarta Utara": "#f97316",
    "Jakarta Selatan": "#22c55e",
    "Jakarta Timur": "#3b82f6",
    "Jakarta Barat": "#a855f7",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})


def load_ispu_pm25() -> pd.DataFrame:
    """Load and process ISPU PM2.5 data from local CSV.

    Returns:
        DataFrame with ``city``, ``year``, ``month``, ``pm25`` columns.
    """
    ispu_path = Path(DATA_DIR) / "raw" / "ispu" / "ispu_dki_all.csv"
    df = pd.read_csv(ispu_path)
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df = df.dropna(subset=["tanggal"])
    df = df[df["pm25"].notna()].copy()
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df = df[df["pm25"] > 0]
    df["city"] = df["stasiun"].str.extract(r"(DKI\d)")[0].map(ISPU_STATION_CITY)
    df = df.dropna(subset=["city"])
    df["year"] = df["tanggal"].dt.year
    df["month"] = df["tanggal"].dt.month
    return df


def chart_pm25_by_city(pm25_df: pd.DataFrame) -> pd.DataFrame:
    """Chart 1: PM2.5 Annual Average by DKI City (grouped bar)."""
    annual = pm25_df.groupby(["city", "year"])["pm25"].mean().reset_index()
    annual = annual[annual["year"] >= 2021]

    fig, ax = plt.subplots(figsize=(12, 7))
    cities = list(ISPU_STATION_CITY.values())
    years = sorted(annual["year"].unique())
    x = np.arange(len(cities))
    width = 0.18
    year_colors = {2021: "#93c5fd", 2022: "#3b82f6", 2023: "#1d4ed8",
                   2024: "#1e3a8a", 2025: "#f59e0b"}

    for i, year in enumerate(years):
        yd = annual[annual["year"] == year]
        vals = [yd[yd["city"] == c]["pm25"].values[0]
                if len(yd[yd["city"] == c]) > 0 else 0 for c in cities]
        bars = ax.bar(x + i * width, vals, width, label=str(year),
                      color=year_colors.get(year, "#6b7280"),
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(y=15, color="#22c55e", linestyle="--", linewidth=2, alpha=0.8,
               label="WHO Annual Guideline (15 µg/m³)")
    ax.axhline(y=55, color="#f97316", linestyle="--", linewidth=2, alpha=0.8,
               label="Indonesia Standard (55 µg/m³)")
    ax.set_xlabel("DKI Jakarta City", fontweight="bold")
    ax.set_ylabel("PM2.5 Annual Average (µg/m³)", fontweight="bold")
    ax.set_title("PM2.5 Annual Average by DKI Jakarta City\n(2021–2025)", pad=15)
    ax.set_xticks(x + width * (len(years) - 1) / 2)
    ax.set_xticklabels(cities, rotation=15, ha="right")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, None)
    ax.grid(axis="y", alpha=0.3)

    path = OUTPUT_DIR / "eda_04_health_pm25_by_city.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Chart 1 saved: %s", path)
    return annual


def chart_correlation(pm25_annual: pd.DataFrame) -> None:
    """Chart 2: PM2.5 vs ISPA Cases — scatter + regression."""
    rows = []
    for (city, year), cases in ISPA_DATA.items():
        match = pm25_annual[(pm25_annual["city"] == city) & (pm25_annual["year"] == year)]
        if len(match) > 0:
            rows.append({
                "city": city, "year": year,
                "pm25_avg": match["pm25"].values[0],
                "ispa_cases": cases,
                "population": CITY_POP.get(city, 2_000_000),
                "ispa_rate": (cases / CITY_POP.get(city, 2_000_000)) * 1000,
            })
    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        logger.warning("No correlation data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cities = list(ISPU_STATION_CITY.values())

    # Plot A
    ax = axes[0]
    for city in cities:
        cd = corr_df[corr_df["city"] == city]
        ax.scatter(cd["pm25_avg"], cd["ispa_cases"],
                   s=cd["population"] / 10000,
                   c=CITY_COLORS.get(city, "#6b7280"),
                   label=city, alpha=0.8, edgecolors="white", linewidth=1, zorder=3)
    x, y = corr_df["pm25_avg"].values, corr_df["ispa_cases"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min() - 5, x.max() + 5, 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.5, linewidth=2, zorder=2)
    ax.set_xlabel("PM2.5 Annual Average (µg/m³)", fontweight="bold")
    ax.set_ylabel("ISPA Cases (per year)", fontweight="bold")
    ax.set_title(f"PM2.5 vs ISPA Cases\nr = {r_value:.3f}, p = {p_value:.4f}", fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, zorder=0)

    # Plot B
    ax2 = axes[1]
    for city in cities:
        cd = corr_df[corr_df["city"] == city]
        ax2.scatter(cd["pm25_avg"], cd["ispa_rate"], s=100,
                    c=CITY_COLORS.get(city, "#6b7280"), label=city,
                    alpha=0.8, edgecolors="white", linewidth=1, marker="D", zorder=3)
    x2, y2 = corr_df["pm25_avg"].values, corr_df["ispa_rate"].values
    s2, i2, r2, p2, _ = stats.linregress(x2, y2)
    ax2.plot(np.linspace(x2.min() - 5, x2.max() + 5, 100),
             s2 * np.linspace(x2.min() - 5, x2.max() + 5, 100) + i2,
             "k--", alpha=0.5, linewidth=2)
    ax2.set_xlabel("PM2.5 Annual Average (µg/m³)", fontweight="bold")
    ax2.set_ylabel("ISPA Rate (cases per 1,000 population)", fontweight="bold")
    ax2.set_title(f"PM2.5 vs ISPA Rate\nr = {r2:.3f}, p = {p2:.4f}", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, zorder=0)

    fig.suptitle("AirSafe School: PM2.5 Health Impact Correlation Analysis",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = OUTPUT_DIR / "eda_04_health_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Chart 2 saved: %s", path)


def chart_seasonal(pm25_df: pd.DataFrame) -> None:
    """Chart 3: Monthly PM2.5 seasonal pattern with health impact zones."""
    copy = pm25_df.copy()
    copy["year"] = copy["tanggal"].dt.year
    copy["month"] = copy["tanggal"].dt.month
    monthly = copy.groupby(["year", "month"])["pm25"].agg(["mean", "std", "count"]).reset_index()
    seasonal = monthly.groupby("month").agg(
        pm25_mean=("mean", "mean"), pm25_std=("mean", "std"), n=("count", "sum"),
    ).reset_index()

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.axhspan(0, 35, alpha=0.08, color="#22c55e")
    ax.axhspan(35, 75, alpha=0.08, color="#eab308")
    ax.axhspan(75, 150, alpha=0.08, color="#f97316")
    ax.text(12.3, 20, "BAIK", fontsize=8, color="#22c55e", fontweight="bold", va="center")
    ax.text(12.3, 55, "SEDANG", fontsize=8, color="#eab308", fontweight="bold", va="center")

    colors = ["#ef4444" if m in (6, 7, 8, 9) else "#6baed6" for m in seasonal["month"]]
    bars = ax.bar(seasonal["month"], seasonal["pm25_mean"],
                  yerr=seasonal["pm25_std"], capsize=3,
                  color=colors, edgecolor="white", linewidth=0.5, alpha=0.85, zorder=2)
    for bar, val in zip(bars, seasonal["pm25_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + seasonal["pm25_std"].max() * 0.3 + 2,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(y=15, color="#22c55e", linestyle="--", linewidth=2, alpha=0.8, label="WHO Annual (15)")
    ax.axhline(y=45, color="#ef4444", linestyle="--", linewidth=2, alpha=0.8, label="WHO 24-hour (45)")
    ax.set_xlabel("Month", fontweight="bold")
    ax.set_ylabel("PM2.5 Average (µg/m³)", fontweight="bold")
    ax.set_title("Monthly PM2.5 Seasonal Pattern in Jakarta\n(Average 2021–2025)", pad=15)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "eda_04_health_seasonal.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Chart 3 saved: %s", path)


def chart_dashboard(pm25_df: pd.DataFrame, pm25_annual: pd.DataFrame) -> None:
    """Chart 4: Combined health impact dashboard (2×2 grid)."""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    cities = list(ISPU_STATION_CITY.values())

    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    for city in cities:
        cd = pm25_df[pm25_df["city"] == city]
        monthly = cd.set_index("tanggal").resample("ME")["pm25"].mean()
        ax1.plot(monthly.index, monthly.values, label=city,
                 color=CITY_COLORS[city], alpha=0.8, linewidth=1.5)
    ax1.axhline(y=15, color="#22c55e", linestyle=":", alpha=0.5)
    ax1.axhline(y=55, color="#f97316", linestyle=":", alpha=0.5)
    ax1.set_title("A. Monthly PM2.5 Trend by City", fontweight="bold")
    ax1.set_ylabel("PM2.5 (µg/m³)")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)

    # Panel B
    ax2 = fig.add_subplot(gs[0, 1])
    years = sorted(pm25_annual["year"].unique())
    x = np.arange(len(cities))
    w = 0.2
    for i, year in enumerate(years[-4:]):
        yd = pm25_annual[pm25_annual["year"] == year]
        vals = [yd[yd["city"] == c]["pm25"].values[0]
                if len(yd[yd["city"] == c]) > 0 else 0 for c in cities]
        ax2.bar(x + i * w, vals, w, label=str(year), alpha=0.85)
    ax2.axhline(y=15, color="#22c55e", linestyle="--", linewidth=2, label="WHO (15)")
    ax2.axhline(y=55, color="#f97316", linestyle="--", linewidth=2, label="RI Std (55)")
    ax2.set_title("B. PM2.5 Annual Average by City", fontweight="bold")
    ax2.set_ylabel("PM2.5 (µg/m³)")
    ax2.set_xticks(x + w * 1.5)
    ax2.set_xticklabels([c.replace("Jakarta ", "") for c in cities], fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # Panel C
    ax3 = fig.add_subplot(gs[1, 0])
    corr_rows = []
    for (city, year), cases in ISPA_DATA.items():
        match = pm25_annual[(pm25_annual["city"] == city) & (pm25_annual["year"] == year)]
        if len(match) > 0:
            corr_rows.append({
                "city": city, "year": year, "pm25": match["pm25"].values[0],
                "cases": cases, "pop": CITY_POP.get(city, 2_000_000),
            })
    if corr_rows:
        cdf = pd.DataFrame(corr_rows)
        for city in cities:
            cd = cdf[cdf["city"] == city]
            ax3.scatter(cd["pm25"], cd["cases"], s=cd["pop"] / 15000,
                        c=CITY_COLORS[city], label=city.replace("Jakarta ", ""),
                        alpha=0.8, edgecolors="white", linewidth=1)
        xv, yv = cdf["pm25"].values, cdf["cases"].values
        slope, intercept, r, p, _ = stats.linregress(xv, yv)
        xl = np.linspace(xv.min() - 5, xv.max() + 5, 100)
        ax3.plot(xl, slope * xl + intercept, "k--", alpha=0.4, linewidth=2)
        ax3.text(0.05, 0.95, f"r = {r:.3f}\np = {p:.4f}",
                 transform=ax3.transAxes, fontsize=9, va="top", family="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax3.set_title("C. PM2.5 vs ISPA Cases Correlation", fontweight="bold")
    ax3.set_xlabel("PM2.5 Annual Avg (µg/m³)")
    ax3.set_ylabel("ISPA Cases")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # Panel D
    ax4 = fig.add_subplot(gs[1, 1])
    m_avg = pm25_df.groupby(pm25_df["tanggal"].dt.month)["pm25"].agg(["mean", "std"]).reset_index()
    colors_d = ["#ef4444" if m in (6, 7, 8, 9) else "#3b82f6" for m in m_avg["tanggal"]]
    ax4.bar(m_avg["tanggal"], m_avg["mean"], yerr=m_avg["std"],
            capsize=2, color=colors_d, alpha=0.8, edgecolor="white")
    ax4.axhline(y=15, color="#22c55e", linestyle="--", linewidth=1.5, alpha=0.7)
    ax4.axhline(y=45, color="#ef4444", linestyle="--", linewidth=1.5, alpha=0.7)
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax4.set_title("D. Seasonal PM2.5 Pattern (Red = Dry Season)", fontweight="bold")
    ax4.set_ylabel("PM2.5 (µg/m³)")
    ax4.grid(axis="y", alpha=0.3)

    fig.suptitle("AirSafe School — PM2.5 Health Impact Analysis Dashboard\n"
                 "DKI Jakarta | 2021–2025 | Data: ISPU + BPS DKI Jakarta",
                 fontsize=16, fontweight="bold", y=1.01)
    path = OUTPUT_DIR / "eda_04_health_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Dashboard saved: %s", path)


def main() -> None:
    """Generate all health correlation charts."""
    setup_logging()

    logger.info("Loading ISPU PM2.5 data...")
    pm25_df = load_ispu_pm25()
    logger.info("  %d records, %d-%d", len(pm25_df),
                pm25_df["year"].min(), pm25_df["year"].max())

    pm25_annual = pm25_df.groupby(["city", "year"])["pm25"].mean().reset_index()
    pm25_annual = pm25_annual[pm25_annual["year"] >= 2021]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Generating charts...")
    chart_pm25_by_city(pm25_df)
    chart_correlation(pm25_annual)
    chart_seasonal(pm25_df)
    chart_dashboard(pm25_df, pm25_annual)
    logger.info("All health correlation charts generated → %s/", OUTPUT_DIR)


if __name__ == "__main__":
    main()
