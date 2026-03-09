# 🛡️ ClaimGuard Sentinel

> **Spatiotemporal Drought Forecasting & Automated Claim Verification using Hybrid CNN-LSTM Architecture**

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-green)

**ClaimGuard Sentinel** is a full-stack AI-powered platform that uses **real satellite data** and a **CNN-LSTM deep learning model** to detect fraudulent crop insurance claims and forecast drought conditions 3 months in advance.

<p align="center">
  <img src="screenshots/dashboard.png" alt="ClaimGuard Dashboard" width="90%" />
</p>

---

## 📋 Table of Contents

- [The Problem](#-the-problem)
- [The Solution](#-the-solution)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Data Pipeline](#-data-pipeline)
- [Model Architecture](#-model-architecture)
- [API Endpoints](#-api-endpoints)
- [Performance](#-performance)
- [Team](#-team)

---

## 🧐 The Problem

India's crop insurance program (PMFBY) faces critical challenges:

| Problem | Impact |
|---|---|
| **Fraudulent Claims** | Farmers/intermediaries file drought claims for unaffected areas — manual verification is slow and prone to manipulation |
| **Reactive Risk Management** | Insurance officers only learn about drought *after* claims pour in, depleting reserves |
| **No Data-Driven Verification** | Risk assessment relies on subjective field reports, not real-time satellite data |
| **Payout Delays** | Manual Crop Cutting Experiments (CCE) cause 12–18 month delays in genuine claim settlements |

---

## 💡 The Solution

ClaimGuard Sentinel acts as a **Command Center** for insurance investigators, providing:

1. **Instant claim verification** — Cross-references farmer's claim (location + date) against satellite data → APPROVED / FLAGGED / REJECTED in **< 1 second**
2. **3-month drought forecasting** — CNN-LSTM model predicts NDVI for future months, enabling proactive reserve allocation
3. **Portfolio risk analysis** — K-means clustering identifies correlated drought zones for diversification

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     GOOGLE EARTH ENGINE                          │
│    MODIS (NDVI)  ·  SMAP (Soil Moisture)  ·  CHIRPS (Rainfall)  │
└────────────────────────────┬─────────────────────────────────────┘
                             │ fetch_gee_data.py
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI + Python)                     │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ CSV Dataset  │  │  CNN-LSTM    │  │  K-Means Clustering     │ │
│  │ 13 Districts │  │  .keras      │  │  3 Drought Clusters     │ │
│  │ 2019 → 2026  │  │  model       │  │  6 Features             │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬──────────────┘ │
│         │                 │                      │                │
│         ▼                 ▼                      ▼                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    REST API Endpoints                      │  │
│  │  /api/districts  /api/alerts  /api/claims/verify           │  │
│  │  /api/predict    /api/forecast  /api/clusters              │  │
│  └─────────────────────────┬──────────────────────────────────┘  │
└────────────────────────────┼─────────────────────────────────────┘
                             │ HTTP/JSON
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FRONTEND (React + Vite)                         │
│                                                                  │
│  ┌──────────────┐ ┌─────────────┐ ┌───────────┐ ┌────────────┐ │
│  │  Risk Map    │ │ Investigation│ │ Readiness │ │  Cluster   │ │
│  │  (Leaflet)   │ │ Queue       │ │ Forecast  │ │  Analytics │ │
│  └──────────────┘ └─────────────┘ └───────────┘ └────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 1. 🌍 Situation Forecast (Dashboard)
Real-time risk assessment map of 13 drought-prone districts across South India.
- Interactive Leaflet map with color-coded risk markers (🔴 High > 70% · 🟠 Warning 50-70% · 🟢 Safe < 50%)
- Live alerts feed with severity classification
- **Time-travel**: Date picker loads historical or forecast data on the map
- Accessibility: distinct shape indicators for color-blind users

### 2. 🔍 Investigation Queue
Automated claim verification engine for insurance investigators.
- Takes farmer's **location + claim date** as input
- Cross-references against satellite data (NDVI, SMI, Rainfall) for that region and period
- Returns **APPROVED** / **FLAGGED FOR REVIEW** / **REJECTED** with confidence score and evidence
- Processes claims in **< 1 second**

### 3. 📈 Readiness Forecast
3-month NDVI prediction using the CNN-LSTM model for all 13 districts.
- Iterative rollout: predicts month 1, appends to buffer, predicts month 2, etc.
- Shows trend direction (WORSENING / STABLE / IMPROVING)
- NDVI trajectory chart for visual analysis
- Enables proactive reserve allocation before drought hits

### 4. 📊 Cluster Analytics
K-means clustering on district drought profiles for portfolio diversification.
- 6 features: avg NDVI, avg SMI, avg rainfall, NDVI std dev, latest NDVI, NDVI trend
- NDVI vs SMI scatter plot with cluster coloring
- Radar chart for cluster profiling
- Portfolio diversification insights (avoid concentrating insurance exposure in correlated drought zones)

### 5. ⏱️ Functional Date Picker
Scrub through any date from **Jan 2019 → May 2026**:
- **Historical dates**: loads actual satellite data from that month
- **Current date**: shows latest real data
- **Future dates**: displays CNN-LSTM predicted values (with forecast flag)

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | React 19, Vite 7, Tailwind CSS | SPA with dark-theme "Command Center" UI |
| **Charts** | Recharts, Leaflet + react-leaflet | Data visualization and geospatial mapping |
| **UI Components** | shadcn/ui, Lucide Icons, Sonner | Premium component library with toast notifications |
| **Backend** | FastAPI, Uvicorn | REST API with hot-reload |
| **ML/AI** | TensorFlow/Keras (CNN-LSTM) | NDVI time-series prediction |
| **Data Science** | Pandas, NumPy, scikit-learn | Data processing, MinMaxScaler, K-Means |
| **Data Source** | Google Earth Engine (MODIS, SMAP, CHIRPS) | Real satellite imagery and meteorological data |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** (tested on 3.12)
- **Node.js 18+** and npm
- **TensorFlow 2.x** (CPU is sufficient for inference)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/vihaangautam/ClaimGuard-Sentinel.git
   cd ClaimGuard-Sentinel
   ```

2. **Backend Setup**
   ```bash
   cd Major/backend
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd Major/frontend
   npm install
   ```

### Running the Application

**Terminal 1 — Start the Backend:**
```bash
cd Major/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
> ⏳ First startup takes ~30-60 seconds (loads TensorFlow model, computes 3-month forecast for 13 districts, runs K-means clustering)

**Terminal 2 — Start the Frontend:**
```bash
cd Major/frontend
npm run dev
```

**Open the Dashboard:** Navigate to `http://localhost:5173`

### Fetching Fresh Satellite Data (Optional)

To update the dataset with latest satellite imagery from Google Earth Engine:

```bash
cd Major/backend
# Requires: pip install earthengine-api
# One-time auth: earthengine authenticate
python fetch_gee_data.py
```

---

## 📁 Project Structure

```
MajorProject/
├── Major/
│   ├── backend/
│   │   ├── main.py                  # FastAPI server — all API endpoints
│   │   ├── fetch_gee_data.py        # Google Earth Engine data pipeline
│   │   ├── extend_forecast.py       # Extended forecast utilities
│   │   ├── requirements.txt         # Python dependencies
│   │   └── data/
│   │       ├── processed_drought_data.csv     # NDVI/SMI/Rainfall dataset (2019-2026)
│   │       └── finetuned_cnn_lstm_hybrid_entire_dataset.keras  # Trained model
│   │
│   └── frontend/
│       ├── src/
│       │   ├── App.jsx              # Main app with routing and data fetching
│       │   ├── index.css            # Global styles (dark theme, custom scrollbars)
│       │   ├── lib/
│       │   │   ├── api.js           # API client (fetch helpers)
│       │   │   └── utils.js         # Utility functions (cn helper)
│       │   └── components/
│       │       ├── dashboard/
│       │       │   ├── RiskMap.jsx          # Leaflet map with risk markers
│       │       │   ├── StatsWidget.jsx      # Top-3 risk district cards
│       │       │   ├── LiveAlerts.jsx       # Live alert feed
│       │       │   ├── InvestigationView.jsx # Claim verification UI
│       │       │   ├── ForecastView.jsx     # 3-month forecast dashboard
│       │       │   ├── ClusterView.jsx      # K-means cluster analytics
│       │       │   └── Header.jsx           # Header with date picker
│       │       ├── layout/
│       │       │   └── Sidebar.jsx          # Navigation sidebar
│       │       └── ui/                      # shadcn/ui primitives
│       ├── package.json
│       └── vite.config.js
│
├── MinorProject/                    # Model training (Jupyter notebooks)
│   └── model/
│       ├── finetuned_cnn_lstm_hybrid_entire_dataset.keras  # ← Best model
│       ├── cnn_lstm_hybrid_entire_dataset.keras
│       ├── stacked_lstm_optimized_entire_dataset.keras
│       └── ... (8 model experiments)
│
└── README.md
```

---

## 📡 Data Pipeline

### Data Sources (via Google Earth Engine)

| Source | Dataset | Resolution | Variable |
|---|---|---|---|
| **MODIS** | `MODIS/061/MOD13A3` | 1 km / monthly | NDVI (vegetation health) |
| **SMAP** | `NASA_USDA/HSL/SMAP10KM_soil_moisture` | 10 km / daily | Surface Soil Moisture (m³/m³) |
| **CHIRPS** | `UCSB-CHG/CHIRPS/DAILY` | 5 km / daily | Precipitation (mm) |

### Monitored Districts (13)

| State | Districts |
|---|---|
| **Andhra Pradesh** | Anantapur, Kadapa, Kurnool |
| **Telangana** | Mahbubnagar |
| **Karnataka** | Chitradurga, Koppal, Raichur, Vijayapura |
| **Tamil Nadu** | Dharmapuri, Sivaganga, Ramanathapuram |
| **Kerala** | Palakkad, Idukki |

### Why These Data Sources?

All three sources are **inherently cloud-resistant**:
- **MODIS MOD13A3** → monthly composites with built-in cloud screening (QA flags)
- **SMAP** → microwave L-band radar that **penetrates clouds** by physics
- **CHIRPS** → gauge-calibrated infrared + rain gauge fusion, not raw optical

This eliminates the need for a cloud removal pipeline.

---

## 🧠 Model Architecture

### Hybrid CNN-LSTM

```
Input: (batch, 8 timesteps, 3 features)
           │
    ┌──────▼──────┐
    │  1D-CNN     │  Extracts cross-feature patterns
    │  (spatial)  │  (NDVI ↔ SMI ↔ Rainfall correlations)
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  LSTM       │  Captures temporal dependencies
    │  (temporal) │  (6-8 month drought persistence)
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Dense      │  Regression output
    │  (1 unit)   │  → Predicted NDVI (next month)
    └─────────────┘

Output: Predicted NDVI value (0-1)
```

### Why CNN-LSTM?

| Component | Role | Why Not Alternatives? |
|---|---|---|
| **CNN** | Extracts spatial/cross-feature patterns at each timestep | ARIMA/Prophet are univariate — can't jointly model NDVI + SMI + rainfall |
| **LSTM** | Models temporal dependencies (drought is a time-series phenomenon) | Random Forest/XGBoost can't model sequential dependencies |
| **Hybrid** | Best of both — CNN processes "what" + LSTM processes "when" | Transformers need far more data than 13 districts × 24 months |

### Training

- **Dataset**: 13 districts × 84 months (Jan 2019 – Feb 2026) × 3 features
- **Sequence length**: 8 months lookback
- **Training**: Fine-tuned from base CNN-LSTM on the full dataset
- **8 model variants tested** — fine-tuned CNN-LSTM hybrid performed best

### Prediction Pipeline

```python
# 1. Aggregate district time series (avg across satellites)
# 2. Normalize with MinMaxScaler
# 3. Take last 8 months → shape (1, 8, 3)
# 4. Model.predict → scaled NDVI
# 5. Inverse transform → actual NDVI (0-1)
# 6. Risk = 1 - NDVI
```

For **3-month forecasting**: iterative rollout — predict month 1, append to buffer, predict month 2, etc.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/districts?date=YYYY-MM-DD` | District summary (NDVI, SMI, risk). Optional date for historical/forecast |
| `GET` | `/api/alerts?date=YYYY-MM-DD` | Drought alerts with severity. Optional date filter |
| `GET` | `/api/district/{name}/history` | Monthly NDVI time-series for one district |
| `POST` | `/api/claims/verify` | Verify a claim: `{location, claim_date}` → verdict |
| `POST` | `/api/predict` | Single-district NDVI prediction: `{location}` |
| `GET` | `/api/forecast` | Cached 3-month forecast for all 13 districts |
| `GET` | `/api/clusters` | Cached K-means cluster analysis |

---

## 📊 Performance

| Metric | Value |
|---|---|
| **RMSE** | 0.1423 |
| **MAE** | 0.1099 |
| **Inference Latency** | ~0.5s per prediction |
| **Startup Time** | ~30-60s (model load + forecast + clustering) |
| **Data Coverage** | Jan 2019 – Feb 2026 (84 months) |
| **Forecast Horizon** | 3 months ahead |

---

## 👥 Team

**SRM Institute of Science and Technology**

| Name | Role |
|---|---|
| **S. Kranthi Kumar** | Model Architecture & Deployment |
| **Vihaan Gautam** | Feature Fusion, Data Pipeline & Visualization |
| **Sia Dewan** | Feature Extraction & Documentation |

**Supervisor:** Dr. G. Geetha (Assistant Professor, Dept. of NWC)

---

## 📄 License

This project is developed as part of the Major Project curriculum at SRM Institute of Science and Technology. All rights reserved.
