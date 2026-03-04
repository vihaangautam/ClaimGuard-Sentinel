/**
 * ClaimGuard Sentinel — API Client
 * Centralized fetch helpers for the FastAPI backend.
 */

const API_BASE = "http://localhost:8000";

async function request(path, options = {}) {
    try {
        const res = await fetch(`${API_BASE}${path}`, {
            headers: { "Content-Type": "application/json" },
            ...options,
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `API error: ${res.status}`);
        }
        return await res.json();
    } catch (err) {
        console.error(`[API] ${path} failed:`, err);
        throw err;
    }
}

/** GET /api/districts — all 13 districts with latest NDVI, risk, etc. */
export function fetchDistricts() {
    return request("/api/districts");
}

/** GET /api/alerts — drought alerts derived from real data */
export function fetchAlerts() {
    return request("/api/alerts");
}

/** GET /api/district/:name/history — NDVI time series for charts */
export function fetchDistrictHistory(name) {
    return request(`/api/district/${encodeURIComponent(name)}/history`);
}

/** POST /api/claims/verify — verify a drought claim */
export function verifyClaim(location, claimDate) {
    return request("/api/claims/verify", {
        method: "POST",
        body: JSON.stringify({ location, claim_date: claimDate }),
    });
}

/** POST /api/predict — predict next NDVI using CNN-LSTM */
export function predictNDVI(location) {
    return request("/api/predict", {
        method: "POST",
        body: JSON.stringify({ location }),
    });
}

/** GET /api/forecast — 3-month CNN-LSTM forecast for all districts */
export function fetchForecast() {
    return request("/api/forecast");
}
