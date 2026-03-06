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

/** GET /api/districts — all 13 districts (optionally filtered by date) */
export function fetchDistricts(date) {
    const params = date ? `?date=${date}` : "";
    return request(`/api/districts${params}`);
}

/** GET /api/alerts — drought alerts (optionally filtered by date) */
export function fetchAlerts(date) {
    const params = date ? `?date=${date}` : "";
    return request(`/api/alerts${params}`);
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
