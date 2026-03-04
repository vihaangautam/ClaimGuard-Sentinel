# 🛡️ ClaimGuard Sentinel

> **Spatiotemporal Drought Forecasting & Automated Claim Verification using Hybrid CNN-LSTM Architecture**

![Status](https://img.shields.io/badge/Status-Prototype%20TRL--6-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Stack](https://img.shields.io/badge/Deep%20Learning-CNN%2B%20LSTM-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**ClaimGuard Sentinel** is a "Cyber-Physical Intelligence" platform designed to revolutionize agricultural insurance. [cite_start]It replaces slow, subjective manual field visits with a serverless, event-driven architecture that fuses satellite imagery and meteorological data to verify drought claims with mathematical certainty[cite: 4, 85, 87].

---

## 🧐 The Problem: The "Blind Spot" in Crop Insurance

In the current agricultural insurance landscape (PMFBY), verifying a reported crop loss is inefficient and prone to error:
* [cite_start]**Operational Bottlenecks:** Verification relies on manual Crop Cutting Experiments (CCE), causing 12–18 month delays in payouts[cite: 81, 82].
* [cite_start]**Subjectivity & Fraud:** Field reports are susceptible to human error and "Ghost Claims" (fraudulent claims for healthy crops)[cite: 87, 90].
* [cite_start]**The Climate Gap:** Existing systems focus on monitoring rather than verification, failing to distinguish between genuine climate distress and false reporting[cite: 85, 91].

## 💡 The Solution

ClaimGuard Sentinel acts as a **Global Command Center** for insurance investigators. [cite_start]It reconstructs the "ground truth" of any farm plot over a 6-month timeline using satellite forensics, enabling verification in **~0.49 seconds**[cite: 373].

### Key Product Capabilities

* **🌍 Global Command Center:** A geospatial dashboard that visualizes risk zones. [cite_start]"Red Zones" indicate AI-confirmed drought (>70% risk), allowing investigators to triage claims instantly[cite: 137].
* [cite_start]**📈 The Truth Chart™ (Anomaly Detection):** Enables temporal comparison between regional vegetation trends and plot-level trends to highlight discrepancies (fraud detection)[cite: 141, 142].
* [cite_start]**🧠 AI Analyst:** Integrates model-generated interpretations to summarize spatio-temporal drought evidence into actionable insights[cite: 143].
* [cite_start]**🌫️ Novel Cloud Removal Pipeline:** Uses HSV-based masking and TELEA in-painting to reconstruct missing vegetation information from cloudy satellite images[cite: 114, 117].

---

## ⚙️ Technical Architecture

[cite_start]The system utilizes a **Serverless, Event-Driven Architecture** for real-time processing and scalability[cite: 3, 4].

### 1. Data Pipeline
* [cite_start]**Ingestion:** Automates collection of **MODIS (NDVI)**, **SMAP (Soil Moisture)**, and **IMD (Rainfall)** data[cite: 5].
* [cite_start]**Preprocessing:** Applies a custom algorithm to remove cloud artifacts and normalize data for consistency[cite: 7, 8].

### 2. The AI Core (Hybrid CNN-LSTM)
We utilize a fused deep learning approach to capture both spatial patterns and temporal history:
* [cite_start]**CNN (Spatial Analysis):** Extracts spatial features to ensure consistency across regions[cite: 135].
* [cite_start]**LSTM (Temporal Analysis):** Models temporal persistence of drought conditions using a 100-unit network with a 6-month sliding window[cite: 130, 132].
* [cite_start]**Fusion Strategy:** Concatenates NDVI, SMI, and Rainfall vectors to create a robust "Drought Signature"[cite: 130].

### 3. Performance Metrics
* [cite_start]**RMSE:** 0.1423 (7.9% improvement over baseline models)[cite: 188, 319].
* [cite_start]**MAE:** 0.1099[cite: 188].
* [cite_start]**Latency:** ~0.49 seconds per prediction (Real-time capable)[cite: 373].

---

## 🛠️ Tech Stack

* [cite_start]**Frontend:** React.js, Leaflet (Geospatial Visualization)[cite: 137].
* [cite_start]**Backend:** Python, Event-Driven Serverless Functions (AWS Lambda / Google Cloud Functions)[cite: 6].
* [cite_start]**Machine Learning:** TensorFlow/Keras (CNN-LSTM implementation)[cite: 132].
* [cite_start]**Database:** Firestore / DynamoDB[cite: 18].
* [cite_start]**Data Sources:** NASA Earthdata (MODIS/SMAP), IMD APIs[cite: 61, 62].

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Node.js & npm
* [cite_start]NVIDIA Drivers (Tested on GTX 1080 for training [cite: 205])

### Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/ClaimGuard-Sentinel.git](https://github.com/your-username/ClaimGuard-Sentinel.git)
    cd ClaimGuard-Sentinel
    ```

2.  **Backend Setup (Model & API)**
    ```bash
    cd backend
    pip install -r requirements.txt
    python app.py
    ```

3.  **Frontend Setup (Dashboard)**
    ```bash
    cd frontend
    npm install
    npm start
    ```

4.  **Access the Command Center**
    Open `http://localhost:3000` to view the Risk Assessment Map.

---

## 📊 Project Roadmap

* [cite_start]**Sprint 1:** Data Acquisition (MODIS/SMAP/IMD) & Preprocessing Pipeline[cite: 448].
* [cite_start]**Sprint 2:** CNN-LSTM Model Architecture Design & Feature Fusion[cite: 450].
* [cite_start]**Sprint 3:** Model Testing, Optimization, and Error Analysis[cite: 452].
* [cite_start]**Sprint 4 (Current):** Deployment, Visualization Dashboard, and Documentation[cite: 454].

---

## 👥 The Team

[cite_start]**SRM Institute of Science and Technology** [cite: 157]
* **S. [cite_start]Kranthi Kumar** - *Model Architecture & Deployment* [cite: 73]
* [cite_start]**Vihaan Gautam** - *Feature Fusion, Data Cleaning & Visualization* [cite: 74]
* [cite_start]**Sia Dewan** - *Feature Extraction & Documentation* [cite: 75]

[cite_start]**Supervisor:** Dr. G. Geetha (Assistant Professor, Dept. of NWC)[cite: 71].

---

## 📄 License

This project is developed for the Major Project curriculum at SRM IST. All rights reserved.
