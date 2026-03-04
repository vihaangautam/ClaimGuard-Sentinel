"""
ClaimGuard Sentinel — FastAPI Backend
Serves real drought data and CNN-LSTM model predictions.
"""

import os
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# TensorFlow / model loading (deferred so the server still boots if TF is slow)
# ---------------------------------------------------------------------------
MODEL = None
MODEL_LOADED = False

def _load_model():
    global MODEL, MODEL_LOADED
    if MODEL_LOADED:
        return MODEL
    try:
        from tensorflow.keras.models import load_model
        model_path = os.path.join(os.path.dirname(__file__), "data", "finetuned_cnn_lstm_hybrid_entire_dataset.keras")
        MODEL = load_model(model_path)
        MODEL_LOADED = True
        print(f"✅ CNN-LSTM model loaded from {model_path}")
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
        MODEL = None
        MODEL_LOADED = True  # don't retry on every request
    return MODEL

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "processed_drought_data.csv")
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
df = df.sort_values("Date")

# District coordinates (all 13 monitored districts)
DISTRICT_COORDS = {
    "Anantapur":      {"lat": 14.6819, "lng": 77.6006, "state": "Andhra Pradesh"},
    "Kadapa":         {"lat": 14.4674, "lng": 78.8241, "state": "Andhra Pradesh"},
    "Kurnool":        {"lat": 15.8281, "lng": 78.0373, "state": "Andhra Pradesh"},
    "Mahbubnagar":    {"lat": 16.7488, "lng": 77.9857, "state": "Telangana"},
    "Chitradurga":    {"lat": 14.2287, "lng": 76.3986, "state": "Karnataka"},
    "Koppal":         {"lat": 15.3547, "lng": 76.1548, "state": "Karnataka"},
    "Raichur":        {"lat": 16.2120, "lng": 77.3439, "state": "Karnataka"},
    "Vijayapura":     {"lat": 16.8302, "lng": 75.7100, "state": "Karnataka"},
    "Dharmapuri":     {"lat": 12.1211, "lng": 78.1582, "state": "Tamil Nadu"},
    "Sivaganga":      {"lat": 10.0173, "lng": 78.4815, "state": "Tamil Nadu"},
    "Ramanathapuram": {"lat":  9.3639, "lng": 78.8395, "state": "Tamil Nadu"},
    "Palakkad":       {"lat": 10.7867, "lng": 76.6548, "state": "Kerala"},
    "Idukki":         {"lat":  9.8894, "lng": 76.9720, "state": "Kerala"},
}

# Pre-compute per-district latest values (aggregated across satellites)
def _compute_district_summary():
    """For each district, get the latest date's average NDVI/SMI/Rainfall."""
    rows = []
    for loc in df["Location"].unique():
        loc_data = df[df["Location"] == loc]
        latest_date = loc_data["Date"].max()
        latest = loc_data[loc_data["Date"] == latest_date]
        avg_ndvi = latest["NDVI"].mean()
        avg_smi  = latest["SMI"].mean()
        avg_rain = latest["Avg_rainfall"].mean()

        # Risk score: inverse of NDVI, clamped 0-1
        # Low NDVI → high risk
        risk = round(max(0.0, min(1.0, 1.0 - avg_ndvi)), 4)

        coords = DISTRICT_COORDS.get(loc, {"lat": 14.0, "lng": 77.0, "state": "Unknown"})

        if risk > 0.7:
            status = "High Risk"
        elif risk > 0.5:
            status = "Warning"
        else:
            status = "Safe"

        rows.append({
            "name": loc,
            "lat": coords["lat"],
            "lng": coords["lng"],
            "state": coords["state"],
            "ndvi": round(avg_ndvi, 4),
            "smi": round(avg_smi, 4),
            "rainfall": round(avg_rain, 4),
            "risk": risk,
            "status": status,
            "latest_date": latest_date.strftime("%Y-%m-%d"),
        })
    rows.sort(key=lambda r: r["risk"], reverse=True)
    return rows

DISTRICT_SUMMARY = _compute_district_summary()

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="ClaimGuard Sentinel API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all in dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ClaimRequest(BaseModel):
    location: str
    claim_date: str  # YYYY-MM-DD

class PredictRequest(BaseModel):
    location: str

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "online", "service": "ClaimGuard Sentinel API", "model_loaded": MODEL_LOADED and MODEL is not None}


@app.get("/api/districts")
def get_districts():
    """Return summary data for all monitored districts."""
    return DISTRICT_SUMMARY


@app.get("/api/alerts")
def get_alerts():
    """
    Generate drought alerts from the data.
    An alert is raised for any district whose latest NDVI is below the drought threshold.
    """
    alerts = []
    alert_id = 1
    for d in DISTRICT_SUMMARY:
        if d["ndvi"] < 0.3:
            # Risk > 70% — matches map red zone
            alerts.append({
                "id": alert_id,
                "district": d["name"],
                "message": f"NDVI at {d['ndvi']:.2f} — below drought threshold, severe risk",
                "type": "danger",
                "ndvi": d["ndvi"],
            })
            alert_id += 1
        elif d["ndvi"] < 0.5:
            # Risk 50-70% — matches map amber zone
            alerts.append({
                "id": alert_id,
                "district": d["name"],
                "message": f"NDVI at {d['ndvi']:.2f} — moderate drought warning, moisture declining",
                "type": "warning",
                "ndvi": d["ndvi"],
            })
            alert_id += 1
        else:
            alerts.append({
                "id": alert_id,
                "district": d["name"],
                "message": f"Vegetation healthy (NDVI {d['ndvi']:.2f}), risk stable",
                "type": "success",
                "ndvi": d["ndvi"],
            })
            alert_id += 1

    # Sort: danger first, then warning, then success
    order = {"danger": 0, "warning": 1, "info": 2, "success": 3}
    alerts.sort(key=lambda a: order.get(a["type"], 9))
    return alerts


@app.get("/api/district/{name}/history")
def get_district_history(name: str):
    """Return monthly NDVI time-series for a district (averaged across satellites)."""
    loc_data = df[df["Location"] == name]
    if loc_data.empty:
        raise HTTPException(status_code=404, detail=f"District '{name}' not found")

    # Group by month, average across satellites
    monthly = loc_data.groupby(loc_data["Date"].dt.to_period("M")).agg({
        "NDVI": "mean",
        "SMI": "mean",
        "Avg_rainfall": "mean",
    }).reset_index()
    monthly["Date"] = monthly["Date"].dt.to_timestamp()

    result = []
    for _, row in monthly.iterrows():
        result.append({
            "date": row["Date"].strftime("%Y-%m"),
            "ndvi": round(row["NDVI"], 4),
            "smi": round(row["SMI"], 4),
            "rainfall": round(row["Avg_rainfall"], 4),
        })
    return result


@app.post("/api/claims/verify")
def verify_claim(req: ClaimRequest):
    """
    Verify a drought insurance claim using satellite data.
    Adapted from Minor_Project/claim_guard.py.
    """
    t_start = time.time()

    claim_date = pd.to_datetime(req.claim_date)
    geo_data = df[df["Location"] == req.location]

    if geo_data.empty:
        raise HTTPException(status_code=404, detail=f"No data for location: {req.location}")

    # Find closest data point
    record = geo_data[geo_data["Date"] == claim_date]
    if record.empty:
        nearest_idx = (geo_data["Date"] - claim_date).abs().idxmin()
        record = geo_data.loc[[nearest_idx]]
        data_date = record["Date"].values[0]
        note = f"Exact date not found. Using nearest available data: {str(data_date)[:10]}"
    else:
        note = "Exact satellite match found."

    ndvi_val = float(record["NDVI"].mean())
    smi_val  = float(record["SMI"].mean())
    rainfall_val = float(record["Avg_rainfall"].mean())

    # Confidence scoring
    confidence = 0.0
    if ndvi_val < 0.3:
        confidence += 0.5
    if smi_val < 0.3:
        confidence += 0.3
    if rainfall_val < 0.1:
        confidence += 0.2

    decision = "APPROVED" if confidence > 0.6 else "FLAGGED_FOR_REVIEW"
    if ndvi_val > 0.5:
        decision = "REJECTED (Healthy Vegetation)"

    latency = round(time.time() - t_start, 3)

    # Build analysis text
    if decision.startswith("APPROVED"):
        analysis = (
            f"Satellite analysis confirms drought conditions in {req.location}. "
            f"NDVI at {ndvi_val:.4f} is well below the 0.30 threshold, with soil moisture at {smi_val:.4f}. "
            f"The multi-spectral indicators are consistent with severe water deprivation. "
            f"Recommendation: APPROVE this claim with {confidence*100:.0f}% confidence."
        )
    elif decision.startswith("REJECTED"):
        analysis = (
            f"Satellite data for {req.location} shows healthy vegetation with NDVI {ndvi_val:.4f}. "
            f"Soil moisture is adequate at {smi_val:.4f}. "
            f"No evidence of drought stress detected. Recommendation: REJECT — vegetation appears normal."
        )
    else:
        analysis = (
            f"Mixed signals detected for {req.location}. NDVI is {ndvi_val:.4f} "
            f"{'(below threshold)' if ndvi_val < 0.3 else '(above threshold)'}. "
            f"Soil moisture: {smi_val:.4f}, Rainfall index: {rainfall_val:.4f}. "
            f"Confidence: {confidence*100:.0f}%. Recommendation: INVESTIGATE further."
        )

    return {
        "claim_id": f"CLM-{np.random.randint(10000, 99999)}",
        "region": req.location,
        "claim_date": req.claim_date,
        "satellite_analysis": {
            "NDVI": round(ndvi_val, 4),
            "Soil_Moisture": round(smi_val, 4),
            "Rainfall_Index": round(rainfall_val, 4),
        },
        "system_decision": decision,
        "verification_latency": f"{latency}s",
        "confidence_score": f"{confidence * 100:.1f}%",
        "analysis": analysis,
        "note": note,
    }


@app.post("/api/predict")
def predict_ndvi(req: PredictRequest):
    """
    Use the CNN-LSTM model to predict next-month NDVI for a district.
    """
    model = _load_model()

    geo_data = df[df["Location"] == req.location]
    if geo_data.empty:
        raise HTTPException(status_code=404, detail=f"No data for location: {req.location}")

    # Aggregate by date (average across satellites), sorted
    agg = geo_data.groupby("Date").agg({
        "NDVI": "mean",
        "SMI": "mean",
        "Avg_rainfall": "mean",
    }).reset_index().sort_values("Date")

    features = agg[["NDVI", "SMI", "Avg_rainfall"]].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    seq_length = 8
    if len(scaled) < seq_length:
        raise HTTPException(status_code=400, detail="Not enough data for prediction")

    # Take last seq_length rows as input
    input_seq = scaled[-seq_length:].reshape(1, seq_length, 3)

    if model is None:
        # Fallback: simple trend extrapolation
        last_ndvi = float(agg["NDVI"].iloc[-1])
        prev_ndvi = float(agg["NDVI"].iloc[-2]) if len(agg) > 1 else last_ndvi
        trend = last_ndvi - prev_ndvi
        predicted_ndvi = max(0, min(1, last_ndvi + trend))
        method = "trend_extrapolation (model unavailable)"
    else:
        t_start = time.time()
        pred_scaled = model.predict(input_seq, verbose=0)
        inference_time = round(time.time() - t_start, 4)

        # Inverse transform: create dummy 3-col array
        dummy = np.zeros((1, 3))
        dummy[0, 0] = pred_scaled[0, 0]
        predicted_ndvi = float(scaler.inverse_transform(dummy)[0, 0])
        predicted_ndvi = max(0, min(1, predicted_ndvi))
        method = f"cnn_lstm_hybrid (inference: {inference_time}s)"

    latest_date = agg["Date"].max()
    last_ndvi = float(agg["NDVI"].iloc[-1])

    return {
        "location": req.location,
        "current_ndvi": round(last_ndvi, 4),
        "predicted_ndvi": round(predicted_ndvi, 4),
        "prediction_date": (latest_date + pd.DateOffset(months=1)).strftime("%Y-%m"),
        "method": method,
        "drought_risk": "HIGH" if predicted_ndvi < 0.3 else "MODERATE" if predicted_ndvi < 0.5 else "LOW",
    }


# ---------------------------------------------------------------------------
# Forecast cache — computed once at startup, not on every request
# ---------------------------------------------------------------------------
FORECAST_CACHE = []

def _compute_forecast():
    """
    Run CNN-LSTM 3-month rollout for all districts.
    Called once at startup; result cached in FORECAST_CACHE.
    """
    model = _load_model()
    results = []

    for loc in df["Location"].unique():
        geo_data = df[df["Location"] == loc]
        agg = geo_data.groupby("Date").agg({
            "NDVI": "mean", "SMI": "mean", "Avg_rainfall": "mean",
        }).reset_index().sort_values("Date")

        features = agg[["NDVI", "SMI", "Avg_rainfall"]].values
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(features)

        seq_length = 8
        if len(scaled) < seq_length:
            continue

        current_ndvi = float(agg["NDVI"].iloc[-1])
        current_smi = float(agg["SMI"].iloc[-1])
        current_rain = float(agg["Avg_rainfall"].iloc[-1])
        latest_date = agg["Date"].max()

        # Predict next 3 months iteratively
        buffer = list(scaled[-seq_length:])
        predictions = []

        for month_offset in range(1, 4):
            input_seq = np.array(buffer[-seq_length:]).reshape(1, seq_length, 3)

            if model is not None:
                pred_scaled = float(model.predict(input_seq, verbose=0)[0, 0])
            else:
                pred_scaled = float(buffer[-1][0])

            pred_scaled = max(0, min(1, pred_scaled))

            dummy = np.zeros((1, 3))
            dummy[0, 0] = pred_scaled
            dummy[0, 1] = float(buffer[-1][1])
            dummy[0, 2] = float(buffer[-1][2])
            inv = scaler.inverse_transform(dummy)[0]
            pred_ndvi = max(0, min(1, float(inv[0])))

            pred_date = (latest_date + pd.DateOffset(months=month_offset)).strftime("%Y-%m")

            predictions.append({
                "month": pred_date,
                "ndvi": round(pred_ndvi, 4),
                "risk": round(max(0.0, min(1.0, 1.0 - pred_ndvi)), 4),
                "level": "HIGH" if pred_ndvi < 0.3 else "MODERATE" if pred_ndvi < 0.5 else "LOW",
            })

            buffer.append([pred_scaled, float(buffer[-1][1]), float(buffer[-1][2])])

        avg_pred_ndvi = np.mean([p["ndvi"] for p in predictions])
        if avg_pred_ndvi < current_ndvi - 0.05:
            trend = "WORSENING"
        elif avg_pred_ndvi > current_ndvi + 0.05:
            trend = "IMPROVING"
        else:
            trend = "STABLE"

        coords = DISTRICT_COORDS.get(loc, {"lat": 14.0, "lng": 77.0, "state": "Unknown"})

        results.append({
            "name": loc,
            "state": coords["state"],
            "lat": coords["lat"],
            "lng": coords["lng"],
            "current_ndvi": round(current_ndvi, 4),
            "current_smi": round(current_smi, 4),
            "current_rainfall": round(current_rain, 4),
            "current_risk": round(max(0.0, min(1.0, 1.0 - current_ndvi)), 4),
            "predictions": predictions,
            "trend": trend,
            "latest_date": latest_date.strftime("%Y-%m-%d"),
        })

    results.sort(key=lambda r: np.mean([p["risk"] for p in r["predictions"]]), reverse=True)
    return results


@app.get("/api/forecast")
def get_forecast():
    """Return cached 3-month forecast (computed at startup, not per-request)."""
    return FORECAST_CACHE


# ---------------------------------------------------------------------------
# Startup event — preload model + compute forecast cache
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    global FORECAST_CACHE
    print("🚀 ClaimGuard Sentinel API starting...")
    print(f"📊 Loaded {len(df)} data records across {df['Location'].nunique()} districts")
    print(f"📈 Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    _load_model()
    print("🔮 Computing 3-month forecast for all districts...")
    FORECAST_CACHE = _compute_forecast()
    print(f"✅ Forecast cached for {len(FORECAST_CACHE)} districts. Server ready!")

