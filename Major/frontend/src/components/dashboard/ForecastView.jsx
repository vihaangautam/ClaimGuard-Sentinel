import React, { useState, useEffect } from 'react';
import { fetchForecast } from "@/lib/api";
import { TrendingUp, TrendingDown, Minus, AlertTriangle, Shield, Activity, MapPin, Droplets, Cloud, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
    AreaChart, Area, Legend, ReferenceLine
} from "recharts";

const RISK_COLORS = {
    HIGH: { bg: "bg-red-500/10", border: "border-red-500/30", text: "text-red-400", fill: "#ef4444" },
    MODERATE: { bg: "bg-amber-500/10", border: "border-amber-500/30", text: "text-amber-400", fill: "#f59e0b" },
    LOW: { bg: "bg-emerald-500/10", border: "border-emerald-500/30", text: "text-emerald-400", fill: "#10b981" },
};

const TREND_CONFIG = {
    WORSENING: { icon: TrendingUp, text: "Risk Increasing", color: "text-red-400", bg: "bg-red-500/10" },
    IMPROVING: { icon: TrendingDown, text: "Risk Decreasing", color: "text-emerald-400", bg: "bg-emerald-500/10" },
    STABLE: { icon: Minus, text: "Stable", color: "text-amber-400", bg: "bg-amber-500/10" },
};

export function ForecastView() {
    const [forecast, setForecast] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedDistrict, setSelectedDistrict] = useState(null);

    useEffect(() => {
        async function load() {
            setLoading(true);
            try {
                const data = await fetchForecast();
                setForecast(data);
                if (data.length > 0) setSelectedDistrict(data[0]);
            } catch (err) {
                console.error("Forecast fetch failed:", err);
            } finally {
                setLoading(false);
            }
        }
        load();
    }, []);

    if (loading) {
        return (
            <div className="h-full flex items-center justify-center flex-col gap-3">
                <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />
                <p className="text-sm text-muted-foreground">Running CNN-LSTM forecasts for all districts...</p>
                <p className="text-xs text-muted-foreground/60">This may take 15-30 seconds</p>
            </div>
        );
    }

    // Summary stats
    const highRiskCount = forecast.filter(d => d.predictions.some(p => p.level === "HIGH")).length;
    const worseningCount = forecast.filter(d => d.trend === "WORSENING").length;
    const improvingCount = forecast.filter(d => d.trend === "IMPROVING").length;

    // Chart data for the selected district
    const chartData = selectedDistrict ? [
        { month: "Current", ndvi: selectedDistrict.current_ndvi, type: "actual" },
        ...selectedDistrict.predictions.map(p => ({ month: p.month.slice(5), ndvi: p.ndvi, type: "forecast" })),
    ] : [];

    // Bar chart data: all districts' avg predicted risk
    const barData = forecast.map(d => ({
        name: d.name.length > 10 ? d.name.slice(0, 9) + "…" : d.name,
        fullName: d.name,
        risk: Math.round(
            (d.predictions.reduce((s, p) => s + p.risk, 0) / d.predictions.length) * 100
        ),
        level: d.predictions[0].level,
    }));

    return (
        <div className="h-full overflow-auto p-1 space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-bold flex items-center gap-2">
                        <Activity className="h-5 w-5 text-blue-400" />
                        Drought Readiness Forecast
                    </h2>
                    <p className="text-sm text-muted-foreground mt-1">
                        CNN-LSTM 3-month predictions for insurance portfolio risk assessment
                    </p>
                </div>
                <div className="flex gap-2">
                    <div className="px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs font-semibold flex items-center gap-1.5">
                        <AlertTriangle className="h-3 w-3" /> {highRiskCount} High Risk
                    </div>
                    <div className="px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs font-semibold flex items-center gap-1.5">
                        <TrendingUp className="h-3 w-3" /> {worseningCount} Worsening
                    </div>
                    <div className="px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold flex items-center gap-1.5">
                        <TrendingDown className="h-3 w-3" /> {improvingCount} Improving
                    </div>
                </div>
            </div>

            <div className="flex gap-4" style={{ height: '520px' }}>
                {/* Left: District Risk Table */}
                <div className="w-[480px] shrink-0 rounded-xl border border-border/50 bg-card shadow-lg overflow-hidden flex flex-col">
                    <div className="px-4 py-3 border-b border-border/50 bg-muted/20">
                        <h3 className="text-sm font-semibold">District Risk Outlook</h3>
                        <p className="text-xs text-muted-foreground">Next 3 months • Click to inspect</p>
                    </div>
                    <div className="overflow-auto flex-1">
                        <table className="w-full text-sm">
                            <thead className="sticky top-0 bg-card z-10">
                                <tr className="border-b border-border/30 text-xs text-muted-foreground">
                                    <th className="text-left py-2 px-3 font-medium">District</th>
                                    <th className="text-center py-2 px-1 font-medium">Now</th>
                                    {forecast[0]?.predictions.map((p, i) => (
                                        <th key={i} className="text-center py-2 px-1 font-medium">{p.month.slice(5)}</th>
                                    ))}
                                    <th className="text-center py-2 px-2 font-medium">Trend</th>
                                </tr>
                            </thead>
                            <tbody>
                                {forecast.map((d) => {
                                    const TrendIcon = TREND_CONFIG[d.trend].icon;
                                    const isSelected = selectedDistrict?.name === d.name;
                                    return (
                                        <tr
                                            key={d.name}
                                            onClick={() => setSelectedDistrict(d)}
                                            className={cn(
                                                "border-b border-border/10 cursor-pointer transition-colors hover:bg-muted/30",
                                                isSelected && "bg-primary/5 border-l-2 border-l-blue-500"
                                            )}
                                        >
                                            <td className="py-2 px-3">
                                                <div className="font-medium text-foreground">{d.name}</div>
                                                <div className="text-[10px] text-muted-foreground">{d.state}</div>
                                            </td>
                                            <td className="text-center py-2 px-1">
                                                <span className={cn(
                                                    "inline-block px-1.5 py-0.5 rounded text-[10px] font-bold",
                                                    d.current_ndvi < 0.3 ? "bg-red-500/20 text-red-400" :
                                                        d.current_ndvi < 0.5 ? "bg-amber-500/20 text-amber-400" :
                                                            "bg-emerald-500/20 text-emerald-400"
                                                )}>
                                                    {d.current_ndvi.toFixed(2)}
                                                </span>
                                            </td>
                                            {d.predictions.map((p, pi) => (
                                                <td key={pi} className="text-center py-2 px-1">
                                                    <span className={cn(
                                                        "inline-block px-1.5 py-0.5 rounded text-[10px] font-bold",
                                                        RISK_COLORS[p.level].bg, RISK_COLORS[p.level].text
                                                    )}>
                                                        {p.ndvi.toFixed(2)}
                                                    </span>
                                                </td>
                                            ))}
                                            <td className="text-center py-2 px-2">
                                                <span className={cn(
                                                    "inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold",
                                                    TREND_CONFIG[d.trend].bg, TREND_CONFIG[d.trend].color
                                                )}>
                                                    <TrendIcon className="h-3 w-3" />
                                                    {d.trend}
                                                </span>
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Right: Detail Panel */}
                <div className="flex-1 overflow-auto h-full">
                    {/* District Detail Card */}
                    {selectedDistrict && (
                        <div className="rounded-xl border border-border/50 bg-card shadow-lg p-4 flex flex-col gap-3 h-fit">
                            <div className="flex items-center justify-between">
                                <div>
                                    <h3 className="text-lg font-bold">{selectedDistrict.name}</h3>
                                    <p className="text-xs text-muted-foreground flex items-center gap-1">
                                        <MapPin className="h-3 w-3" /> {selectedDistrict.state} • Data through {selectedDistrict.latest_date}
                                    </p>
                                </div>
                                <div className={cn(
                                    "px-2 py-1 rounded-lg text-xs font-bold",
                                    TREND_CONFIG[selectedDistrict.trend].bg, TREND_CONFIG[selectedDistrict.trend].color
                                )}>
                                    {TREND_CONFIG[selectedDistrict.trend].text}
                                </div>
                            </div>

                            {/* Current Stats */}
                            <div className="grid grid-cols-3 gap-2">
                                <div className="rounded-lg bg-muted/30 p-2 text-center">
                                    <div className="text-[10px] text-muted-foreground uppercase">NDVI</div>
                                    <div className={cn("text-lg font-bold font-mono",
                                        selectedDistrict.current_ndvi < 0.3 ? "text-red-400" : "text-emerald-400"
                                    )}>
                                        {selectedDistrict.current_ndvi.toFixed(3)}
                                    </div>
                                </div>
                                <div className="rounded-lg bg-muted/30 p-2 text-center">
                                    <div className="text-[10px] text-muted-foreground uppercase flex items-center justify-center gap-1"><Droplets className="h-3 w-3" /> SMI</div>
                                    <div className="text-lg font-bold font-mono text-blue-400">
                                        {selectedDistrict.current_smi.toFixed(3)}
                                    </div>
                                </div>
                                <div className="rounded-lg bg-muted/30 p-2 text-center">
                                    <div className="text-[10px] text-muted-foreground uppercase flex items-center justify-center gap-1"><Cloud className="h-3 w-3" /> Rain</div>
                                    <div className="text-lg font-bold font-mono text-cyan-400">
                                        {selectedDistrict.current_rainfall.toFixed(3)}
                                    </div>
                                </div>
                            </div>

                            {/* NDVI Forecast Mini Chart */}
                            <div className="rounded-lg border border-border/30 bg-muted/10 p-3">
                                <h4 className="text-xs font-semibold mb-2 text-muted-foreground">NDVI Forecast Trajectory</h4>
                                <ResponsiveContainer width="100%" height={220}>
                                    <AreaChart data={chartData}>
                                        <defs>
                                            <linearGradient id="forecastGrad" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                        <XAxis dataKey="month" tick={{ fontSize: 10, fill: "#888" }} />
                                        <YAxis domain={[0, 0.8]} tick={{ fontSize: 10, fill: "#888" }} />
                                        <ReferenceLine y={0.3} stroke="#ef4444" strokeDasharray="4 4" label={{ value: "Drought", fill: "#ef4444", fontSize: 9 }} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: "#1a1a2e", border: "1px solid #333", borderRadius: 8, fontSize: 12 }}
                                            formatter={(v) => [v.toFixed(4), "NDVI"]}
                                        />
                                        <Area type="monotone" dataKey="ndvi" stroke="#3b82f6" fill="url(#forecastGrad)" strokeWidth={2} dot={{ r: 4, fill: "#3b82f6" }} />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Monthly Predictions Detail */}
                            <div className="space-y-2">
                                <h4 className="text-xs font-semibold text-muted-foreground">Monthly Risk Breakdown</h4>
                                {selectedDistrict.predictions.map((p, i) => (
                                    <div key={i} className={cn(
                                        "flex items-center justify-between p-2 rounded-lg border",
                                        RISK_COLORS[p.level].bg, RISK_COLORS[p.level].border
                                    )}>
                                        <div>
                                            <span className="text-sm font-semibold">{p.month}</span>
                                            <span className="text-xs text-muted-foreground ml-2">NDVI: {p.ndvi.toFixed(4)}</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <span className={cn("text-xs font-bold", RISK_COLORS[p.level].text)}>
                                                {Math.round(p.risk * 100)}% Risk
                                            </span>
                                            <span className={cn(
                                                "px-2 py-0.5 rounded text-[10px] font-bold",
                                                RISK_COLORS[p.level].bg, RISK_COLORS[p.level].text
                                            )}>
                                                {p.level}
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>

                            {/* Readiness Recommendation */}
                            <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 p-3">
                                <div className="flex items-center gap-2 mb-1">
                                    <Shield className="h-4 w-4 text-blue-400" />
                                    <span className="text-xs font-bold text-blue-400 uppercase">Portfolio Recommendation</span>
                                </div>
                                <p className="text-xs text-muted-foreground leading-relaxed">
                                    {selectedDistrict.trend === "WORSENING"
                                        ? `${selectedDistrict.name} shows deteriorating conditions. Recommend increasing reserve allocation by ${Math.round(selectedDistrict.predictions[2].risk * 100 - selectedDistrict.current_risk * 100)}% and pre-approving fast-track claim processing for affected farmers.`
                                        : selectedDistrict.trend === "IMPROVING"
                                            ? `${selectedDistrict.name} vegetation recovery detected. Standard reserves are adequate. Monitor monthly to confirm sustained improvement.`
                                            : `${selectedDistrict.name} conditions stable. Maintain current reserve levels. No immediate action required.`
                                    }
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Bottom: All-District Risk Comparison Bar Chart */}
            <div className="rounded-xl border border-border/50 bg-card shadow-lg p-4 shrink-0">
                <h3 className="text-sm font-semibold mb-1">Predicted Average Risk — All Districts</h3>
                <p className="text-xs text-muted-foreground mb-3">3-month forecast average • Higher = more likely to need claim payouts</p>
                <ResponsiveContainer width="100%" height={160}>
                    <BarChart data={barData} barSize={28}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                        <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#888" }} />
                        <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#888" }} unit="%" />
                        <Tooltip
                            contentStyle={{ backgroundColor: "#1a1a2e", border: "1px solid #333", borderRadius: 8, fontSize: 12 }}
                            formatter={(v, name, props) => [`${v}%`, props.payload.fullName]}
                        />
                        <Bar dataKey="risk" radius={[4, 4, 0, 0]}>
                            {barData.map((entry, index) => (
                                <Cell key={index} fill={
                                    entry.risk > 70 ? "#ef4444" : entry.risk > 50 ? "#f59e0b" : "#10b981"
                                } />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}
