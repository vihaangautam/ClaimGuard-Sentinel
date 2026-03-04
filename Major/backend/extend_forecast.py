"""
extend_forecast.py
------------------
Uses the trained CNN-LSTM model to extend the dataset from the last real
data point up to February 2026 via iterative (rollout) prediction.

Run this AFTER fetch_gee_data.py so that real 2025 data is in the CSV first.
The model then only forecasts the remaining gap — reducing error compounding.

Usage:
    python extend_forecast.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, "data")
CSV_PATH   = os.path.join(DATA_DIR, "processed_drought_data.csv")
MODEL_PATH = os.path.join(DATA_DIR, "finetuned_cnn_lstm_hybrid_entire_dataset.keras")

# Forecast up to this month (inclusive)
FORECAST_END = datetime(2026, 2, 1)

SEQ_LENGTH = 8  # must match training

# ─── Load model ──────────────────────────────────────────────────────────────
print("Loading CNN-LSTM model...")
try:
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    print("✅ Model loaded.")
except Exception as e:
    print(f"❌ Could not load model: {e}")
    sys.exit(1)

# ─── Load data ───────────────────────────────────────────────────────────────
print(f"Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)
df = df.sort_values("Date")
print(f"   {len(df)} rows loaded. Latest date: {df['Date'].max().strftime('%Y-%m-%d')}")

DISTRICTS = df["Location"].unique().tolist()
print(f"   Districts: {DISTRICTS}\n")

# ─── Per-district rollout ─────────────────────────────────────────────────────
new_rows = []

for district in DISTRICTS:
    loc_df = df[df["Location"] == district].copy()

    # Aggregate by month (average across satellites)
    loc_df["month"] = loc_df["Date"].dt.to_period("M")
    monthly = loc_df.groupby("month").agg({"NDVI": "mean", "SMI": "mean", "Avg_rainfall": "mean"}).reset_index()
    monthly["Date"] = monthly["month"].dt.to_timestamp()
    monthly = monthly.sort_values("Date")

    latest_real_date = monthly["Date"].max()

    # Check how many months we still need to forecast
    months_needed = []
    cursor = latest_real_date + relativedelta(months=1)
    cursor = cursor.replace(day=1)
    while cursor <= FORECAST_END:
        months_needed.append(cursor)
        cursor += relativedelta(months=1)

    if not months_needed:
        print(f"  {district}: already has data through {FORECAST_END.strftime('%Y-%m')}. Skipping.")
        continue

    print(f"  {district}: forecasting {len(months_needed)} months "
          f"({months_needed[0].strftime('%Y-%m')} → {months_needed[-1].strftime('%Y-%m')})")

    # Normalize using district's own history
    features = monthly[["NDVI", "SMI", "Avg_rainfall"]].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    if len(scaled) < SEQ_LENGTH:
        print(f"    ⚠ Not enough history ({len(scaled)} months < {SEQ_LENGTH}). Skipping.")
        continue

    # Rolling buffer — starts with last SEQ_LENGTH real data points
    buffer = list(scaled[-SEQ_LENGTH:])  # list of [ndvi, smi, rain] arrays

    for target_date in months_needed:
        # Build input sequence
        input_seq = np.array(buffer[-SEQ_LENGTH:]).reshape(1, SEQ_LENGTH, 3)

        # Predict next NDVI (model predicts NDVI only — index 0)
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]

        # For SMI and rainfall: use exponential smoothing of the recent buffer
        # (simple seasonal carry-forward — more realistic than a flat value)
        recent_smi  = np.mean([b[1] for b in buffer[-SEQ_LENGTH:]])
        recent_rain = np.mean([b[2] for b in buffer[-SEQ_LENGTH:]])

        # Add small seasonal variation based on month
        month_num = target_date.month
        # Monsoon months (June-September): slightly higher moisture & rain
        monsoon_boost = 0.05 if 6 <= month_num <= 9 else -0.02
        smi_pred  = float(np.clip(recent_smi  + monsoon_boost * 0.5 + np.random.normal(0, 0.01), 0, 1))
        rain_pred = float(np.clip(recent_rain + monsoon_boost       + np.random.normal(0, 0.02), 0, 1))

        # Clamp NDVI prediction
        ndvi_pred_scaled = float(np.clip(pred_scaled, 0, 1))

        # Add predicted row to the rolling buffer
        buffer.append([ndvi_pred_scaled, smi_pred, rain_pred])

        # Inverse transform NDVI back to real scale
        dummy = np.zeros((1, 3))
        dummy[0, 0] = ndvi_pred_scaled
        dummy[0, 1] = smi_pred
        dummy[0, 2] = rain_pred
        inv = scaler.inverse_transform(dummy)[0]

        ndvi_real = float(np.clip(inv[0], 0, 1))
        smi_real  = float(np.clip(inv[1], 0, 1))
        rain_real = float(np.clip(inv[2], 0, 1))

        # Format date as M/D/YY
        date_str = f"{target_date.month}/1/{target_date.strftime('%y')}"

        new_rows.append({
            "Satellite":    "CNN_LSTM_FORECAST",
            "Date":         date_str,
            "Location":     district,
            "NDVI":         round(ndvi_real, 6),
            "SMI":          round(smi_real,  6),
            "Avg_rainfall": round(rain_real, 6),
        })
        print(f"    {target_date.strftime('%Y-%m')}: NDVI={ndvi_real:.3f}, SMI={smi_real:.3f}, Rain={rain_real:.3f}")

# ─── Append forecast rows to CSV ─────────────────────────────────────────────
if not new_rows:
    print("\n✅ No forecast rows needed — CSV is already up to date!")
else:
    forecast_df = pd.DataFrame(new_rows, columns=["Satellite", "Date", "Location", "NDVI", "SMI", "Avg_rainfall"])
    forecast_df.to_csv(CSV_PATH, mode="a", header=False, index=False)
    print(f"\n✅ Appended {len(forecast_df)} forecast rows to {CSV_PATH}")
    print(f"   Districts: {forecast_df['Location'].nunique()} | "
          f"Months: {forecast_df['Date'].nunique()}")
    print("\n⚡ Restart the FastAPI backend to load the new data:")
    print("   Ctrl+C in the backend terminal, then:")
    print("   python -m uvicorn main:app --port 8000 --reload")
