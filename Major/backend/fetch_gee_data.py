"""
fetch_gee_data.py
-----------------
Fetches real MODIS NDVI, SMAP Soil Moisture, and CHIRPS Rainfall
from Google Earth Engine for all 13 monitored districts.
Covers Jan 2025 → latest available data (~Jan/Feb 2026).
Appends results to processed_drought_data.csv, matching existing schema.

Prerequisites:
    pip install earthengine-api pandas numpy
    earthengine authenticate   (one-time, opens browser)

Usage:
    python fetch_gee_data.py
"""

import ee
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ─── Initialize GEE ─────────────────────────────────────────────────────────
# Your GCP project ID is shown in the Google Cloud Console top-left dropdown,
# or at: https://console.cloud.google.com/earth-engine → look at the URL → ?project=YOUR_ID
# It usually looks like "gemini-api-xxxxxx" or a custom name you set.
GEE_PROJECT = "gen-lang-client-0967205331"

if GEE_PROJECT is None:
    GEE_PROJECT = input("Enter your GCP project ID (from console.cloud.google.com top-left): ").strip()

if not GEE_PROJECT:
    print("❌ No project ID provided. Exiting.")
    import sys; sys.exit(1)

print(f"Initializing GEE with project: {GEE_PROJECT}")
ee.Initialize(project=GEE_PROJECT)

# ─── Config ─────────────────────────────────────────────────────────────────
DISTRICTS = {
    "Anantapur":      {"lat": 14.6819, "lng": 77.6006},
    "Kadapa":         {"lat": 14.4674, "lng": 78.8241},
    "Kurnool":        {"lat": 15.8281, "lng": 78.0373},
    "Mahbubnagar":    {"lat": 16.7488, "lng": 77.9857},
    "Chitradurga":    {"lat": 14.2287, "lng": 76.3986},
    "Koppal":         {"lat": 15.3547, "lng": 76.1548},
    "Raichur":        {"lat": 16.2120, "lng": 77.3439},
    "Vijayapura":     {"lat": 16.8302, "lng": 75.7100},
    "Dharmapuri":     {"lat": 12.1211, "lng": 78.1582},
    "Sivaganga":      {"lat": 10.0173, "lng": 78.4815},
    "Ramanathapuram": {"lat":  9.3639, "lng": 78.8395},
    "Palakkad":       {"lat": 10.7867, "lng": 76.6548},
    "Idukki":         {"lat":  9.8894, "lng": 76.9720},
}

START_DATE = "2025-01-01"
END_DATE   = "2026-03-01"  # GEE will return up to what's available

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "processed_drought_data.csv")

# ─── Load existing data to compute normalization ranges ──────────────────────
print("Loading existing CSV for normalization reference...")
existing = pd.read_csv(CSV_PATH)
existing["Date"] = pd.to_datetime(existing["Date"], format="%m/%d/%y")

ndvi_min  = existing["NDVI"].min()
ndvi_max  = existing["NDVI"].max()
smi_min   = existing["SMI"].min()
smi_max   = existing["SMI"].max()
rain_min  = existing["Avg_rainfall"].min()
rain_max  = existing["Avg_rainfall"].max()

print(f"  NDVI range: [{ndvi_min:.3f}, {ndvi_max:.3f}]")
print(f"  SMI  range: [{smi_min:.3f},  {smi_max:.3f}]")
print(f"  Rain range: [{rain_min:.3f}, {rain_max:.3f}]")

def normalize(val, vmin, vmax):
    """Normalize to match training data scale."""
    if vmax - vmin == 0:
        return 0.5
    return float(np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0))

# ─── GEE Collections ────────────────────────────────────────────────────────
# MODIS Terra Vegetation Indices Monthly (1km)
MODIS_NDVI = ee.ImageCollection("MODIS/061/MOD13A3") \
    .filterDate(START_DATE, END_DATE) \
    .select("NDVI")

# SMAP 10km Soil Moisture (NASA/USDA)
SMAP = ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture") \
    .filterDate(START_DATE, END_DATE) \
    .select("ssm")   # surface soil moisture (m³/m³)

# CHIRPS Daily Rainfall (mm) — aggregated to monthly below
# Note: CHIRPS/MONTHLY may not have recent data; CHIRPS/DAILY is more up-to-date
CHIRPS_DAILY = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
    .filterDate(START_DATE, END_DATE) \
    .select("precipitation")

# ─── Helper: sample one image collection at a point ─────────────────────────
def sample_collection(collection, point, scale, band, scale_factor=1.0):
    """
    Returns a list of (date_str, value) tuples from the collection
    sampled at the given point.
    """
    def get_value(img):
        val = img.select(band).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=scale
        ).get(band)
        # return as feature with a date property
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            "value": val
        })

    fc = collection.map(get_value)
    data = fc.getInfo()["features"]
    results = []
    for feat in data:
        props = feat["properties"]
        v = props.get("value")
        if v is not None and v == v:  # not None and not NaN
            results.append((props["date"], float(v) * scale_factor))
    return results

# ─── Fetch data for all districts ───────────────────────────────────────────
rows = []
total = len(DISTRICTS)

for i, (district, coords) in enumerate(DISTRICTS.items(), 1):
    print(f"[{i}/{total}] Fetching {district}...")
    point = ee.Geometry.Point([coords["lng"], coords["lat"]])

    try:
        # MODIS NDVI: raw values are 0-10000, scale factor 0.0001 → actual NDVI
        ndvi_data = sample_collection(MODIS_NDVI, point, scale=1000, band="NDVI", scale_factor=0.0001)
    except Exception as e:
        print(f"  ⚠ NDVI fetch failed for {district}: {e}")
        ndvi_data = []

    try:
        # SMAP SSM: values in m³/m³, typical range 0.02 - 0.45
        smi_data = sample_collection(SMAP, point, scale=10000, band="ssm", scale_factor=1.0)
    except Exception as e:
        print(f"  ⚠ SMAP fetch failed for {district}: {e}")
        smi_data = []

    try:
        # CHIRPS Daily → aggregate to monthly sums
        def monthly_sum(year, month):
            start = ee.Date.fromYMD(year, month, 1)
            end = start.advance(1, 'month')
            monthly_img = CHIRPS_DAILY.filterDate(start, end).sum()
            val = monthly_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=5000
            ).get('precipitation')
            return ee.Feature(None, {
                'date': start.format('YYYY-MM-dd'),
                'value': val
            })

        # Generate month list: Jan 2025 to Feb 2026
        rain_features = []
        for yr in [2025, 2026]:
            months_range = range(1, 13) if yr == 2025 else range(1, 3)
            for mo in months_range:
                try:
                    feat = monthly_sum(yr, mo).getInfo()
                    v = feat['properties'].get('value')
                    if v is not None:
                        rain_features.append((feat['properties']['date'], float(v)))
                except:
                    pass
        rain_data = rain_features
    except Exception as e:
        print(f"  ⚠ CHIRPS fetch failed for {district}: {e}")
        rain_data = []

    # ── Build date-keyed dicts ──────────────────────────────────────────────
    # Use YYYY-MM as the key, average if multiple readings per month
    def to_monthly(data_list):
        monthly = {}
        for date_str, val in data_list:
            key = date_str[:7]  # YYYY-MM
            if key not in monthly:
                monthly[key] = []
            monthly[key].append(val)
        return {k: np.mean(v) for k, v in monthly.items()}

    ndvi_monthly = to_monthly(ndvi_data)
    smi_monthly  = to_monthly(smi_data)
    rain_monthly = to_monthly(rain_data)

    # ── Compute raw normalization refs from existing CSV for this district ──
    dist_existing = existing[existing["Location"] == district]
    if not dist_existing.empty:
        d_ndvi_max = dist_existing["NDVI"].max()
        d_ndvi_min = dist_existing["NDVI"].min()
        d_smi_max  = dist_existing["SMI"].max()
        d_smi_min  = dist_existing["SMI"].min()
        d_rain_max = dist_existing["Avg_rainfall"].max()
        d_rain_min = dist_existing["Avg_rainfall"].min()
    else:
        d_ndvi_min, d_ndvi_max = ndvi_min, ndvi_max
        d_smi_min,  d_smi_max  = smi_min,  smi_max
        d_rain_min, d_rain_max = rain_min, rain_max

    # ── Combine all months we have any data for ────────────────────────────
    all_months = sorted(set(list(ndvi_monthly.keys()) + list(smi_monthly.keys()) + list(rain_monthly.keys())))

    for month_str in all_months:
        raw_ndvi = ndvi_monthly.get(month_str)
        raw_smi  = smi_monthly.get(month_str)
        raw_rain = rain_monthly.get(month_str)

        # MODIS NDVI is already 0-1 (after × 0.0001) — just clip
        ndvi_norm = float(np.clip(raw_ndvi, 0, 1)) if raw_ndvi is not None else None

        # SMAP SSM: normalize to existing scale (0-1)
        # Typical range 0-0.45 m³/m³ → scale using district historical
        smi_norm = normalize(raw_smi, 0.0, 0.45) if raw_smi is not None else None

        # CHIRPS: normalize to existing scale using district historical range
        # Use 500mm as the practical max for South India monsoon
        rain_norm = normalize(raw_rain, 0.0, 500.0) if raw_rain is not None else None

        # Skip rows with no data at all
        if ndvi_norm is None and smi_norm is None and rain_norm is None:
            continue

        # Fill missing with district median from existing
        if ndvi_norm is None:
            ndvi_norm = float(dist_existing["NDVI"].median()) if not dist_existing.empty else 0.3
        if smi_norm is None:
            smi_norm  = float(dist_existing["SMI"].median())  if not dist_existing.empty else 0.3
        if rain_norm is None:
            rain_norm = float(dist_existing["Avg_rainfall"].median()) if not dist_existing.empty else 0.1

        # Format date as MM/DD/YY to match existing CSV
        dt = datetime.strptime(month_str + "-01", "%Y-%m-%d")
        date_str = f"{dt.month}/1/{dt.strftime('%y')}"  # e.g. "1/1/25"

        rows.append({
            "Satellite":    "MODIS_GEE",   # marks this as real 2025 data
            "Date":         date_str,
            "Location":     district,
            "NDVI":         round(ndvi_norm, 6),
            "SMI":          round(smi_norm,  6),
            "Avg_rainfall": round(rain_norm, 6),
        })
        print(f"    {district} {month_str}: NDVI={ndvi_norm:.3f}, SMI={smi_norm:.3f}, Rain={rain_norm:.3f}")

# ─── Append to CSV ───────────────────────────────────────────────────────────
if not rows:
    print("\n⚠ No new data retrieved. Check GEE authentication and date range.")
else:
    new_df = pd.DataFrame(rows, columns=["Satellite", "Date", "Location", "NDVI", "SMI", "Avg_rainfall"])
    print(f"\n✅ Retrieved {len(new_df)} new rows from GEE.")

    # Deduplicate: remove any existing rows from same date+location (avoid double entry)
    existing["Date_str"] = existing["Date"].apply(lambda d: f"{d.month}/1/{d.strftime('%y')}")
    existing_keys = set(zip(existing["Date_str"], existing["Location"]))
    new_df = new_df[~new_df.apply(lambda r: (r["Date"], r["Location"]) in existing_keys, axis=1)]

    if new_df.empty:
        print("All retrieved dates already exist in the CSV. Nothing appended.")
    else:
        # Append
        new_df.to_csv(CSV_PATH, mode="a", header=False, index=False)
        print(f"✅ Appended {len(new_df)} rows to {CSV_PATH}")
        print(f"   Date range: {new_df['Date'].iloc[0]} → {new_df['Date'].iloc[-1]}")
        print("\nRestart the FastAPI backend to reload the new data.")
