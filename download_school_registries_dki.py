#!/usr/bin/env python3
"""
Download school registries from Kemendikdasmen reference portal.
Source: https://referensi.data.kemendikdasmen.go.id
Form codes: SD=5, SMP=6, SMA=13, SMK=15
"""
import re
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://referensi.data.kemendikdasmen.go.id"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
}

def get_soup(url: str) -> BeautifulSoup:
    """Fetch URL and return parsed BeautifulSoup."""
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    time.sleep(1)  # Be respectful
    return BeautifulSoup(r.text, "lxml")

def unique_keep_order(items):
    """Remove duplicates while preserving order."""
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def extract_region_links(url: str, level: int):
    """
    Extract region links from a page.
    level=2 -> city/regency links
    level=3 -> kecamatan links
    """
    soup = get_soup(url)
    pattern = re.compile(rf"/pendidikan/(dikdas|dikmen)/\d{{6}}/{level}$")
    links = []
    for a in soup.select("a[href]"):
        href = urljoin(BASE, a["href"])
        if pattern.search(href):
            links.append(href)
    return unique_keep_order(links)

def extract_breadcrumb_context(soup: BeautifulSoup):
    """Extract city and kecamatan from breadcrumb."""
    text = soup.get_text(" ", strip=True)

    # Pattern: Indonesia >> Prov. D.K.I. Jakarta >> Kota Adm. Jakarta Timur >> Kec. Jatinegara
    city = ""
    kec = ""

    m = re.search(
        r"Prov\.\s*D\.K\.I\.\s*Jakarta\s*>>\s*(.*?)\s*>>\s*Kec\.\s*(.*?)(?:DAFTAR SATUAN PENDIDIKAN|JUMLAH DATA SATUAN PENDIDIKAN)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        city = m.group(1).strip()
        kec = m.group(2).strip()

    return city, kec

def parse_school_rows(url: str, jenjang: str):
    """Parse school rows from a filtered kecamatan page."""
    soup = get_soup(url)
    city, kec = extract_breadcrumb_context(soup)

    rows = []
    for tr in soup.select("tr"):
        cols = [td.get_text(" ", strip=True) for td in tr.select("td")]
        # Expected: No | NPSN | Nama | Alamat | Kelurahan | Status
        if len(cols) >= 6 and cols[1].isdigit():
            rows.append({
                "jenjang": jenjang,
                "npsn": cols[1],
                "nama_sekolah": cols[2],
                "alamat": cols[3],
                "kelurahan": cols[4],
                "status": cols[5],
                "kota_kab": city,
                "kecamatan": kec,
                "source_url": url,
            })
    return rows

def collect_dikdas(form_code: int, jenjang: str) -> pd.DataFrame:
    """Collect SD/SMP schools from Dikdas portal."""
    province_url = f"{BASE}/pendidikan/dikdas/010000/1"
    city_links = extract_region_links(province_url, level=2)

    all_rows = []
    for i, city_url in enumerate(city_links, 1):
        print(f"  [{i}/{len(city_links)}] Processing city: {city_url}")
        kec_links = extract_region_links(city_url, level=3)
        print(f"    Found {len(kec_links)} kecamatan")

        for j, kec_url in enumerate(kec_links, 1):
            print(f"    [{j}/{len(kec_links)}] {kec_url}")
            filtered_url = f"{kec_url}/jf/{form_code}/all"
            all_rows.extend(parse_school_rows(filtered_url, jenjang))
            time.sleep(0.5)  # Be respectful

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["npsn"]).sort_values(["kota_kab", "kecamatan", "nama_sekolah"])
    return df

def collect_dikmen(form_code: int, jenjang: str) -> pd.DataFrame:
    """Collect SMA/SMK schools from Dikmen portal."""
    province_url = f"{BASE}/pendidikan/dikmen/010000/1"
    city_links = extract_region_links(province_url, level=2)

    all_rows = []
    for i, city_url in enumerate(city_links, 1):
        print(f"  [{i}/{len(city_links)}] Processing city: {city_url}")
        kec_links = extract_region_links(city_url, level=3)
        print(f"    Found {len(kec_links)} kecamatan")

        for j, kec_url in enumerate(kec_links, 1):
            print(f"    [{j}/{len(kec_links)}] {kec_url}")
            filtered_url = f"{kec_url}/jf/{form_code}/all"
            all_rows.extend(parse_school_rows(filtered_url, jenjang))
            time.sleep(0.5)  # Be respectful

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["npsn"]).sort_values(["kota_kab", "kecamatan", "nama_sekolah"])
    return df

def main():
    print("="*60)
    print("DKI Jakarta School Registry Download")
    print("="*60)

    # Official form codes confirmed from filtered official pages:
    # Dikdas: 5=SD, 6=SMP
    # Dikmen: 13=SMA, 15=SMK

    print("\n[1/4] Downloading SD schools...")
    sd = collect_dikdas(5, "SD")

    print("\n[2/4] Downloading SMP schools...")
    smp = collect_dikdas(6, "SMP")

    print("\n[3/4] Downloading SMA schools...")
    sma = collect_dikmen(13, "SMA")

    print("\n[4/4] Downloading SMK schools...")
    smk = collect_dikmen(15, "SMK")

    # Combine SMA and SMK
    sma_smk = pd.concat([sma, smk], ignore_index=True)

    # Save CSVs
    sd.to_csv("data/raw/schools/sd_dki.csv", index=False, encoding="utf-8-sig")
    smp.to_csv("data/raw/schools/smp_dki.csv", index=False, encoding="utf-8-sig")
    sma_smk.to_csv("data/raw/schools/sma_smk_dki.csv", index=False, encoding="utf-8-sig")

    print("\n" + "="*60)
    print("Done!")
    print(f"sd_dki.csv   : {len(sd):,} rows")
    print(f"smp_dki.csv  : {len(smp):,} rows")
    print(f"sma_smk_dki.csv : {len(sma_smk):,} rows")
    print("="*60)

if __name__ == "__main__":
    main()
