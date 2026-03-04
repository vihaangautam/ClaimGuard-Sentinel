import React, { useState, useEffect } from 'react';
import { Loader2, BarChart3, MapPin, Droplets, Cloud, Layers } from "lucide-react";
import { cn } from "@/lib/utils";
import {
    ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend
} from "recharts";

const API_BASE = "http://localhost:8000";

export function ClusterView() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [selectedCluster, setSelectedCluster] = useState(null);

    useEffect(() => {
        async function load() {
            setLoading(true);
            try {
                const res = await fetch(`${API_BASE}/api/clusters`);
                const json = await res.json();
                setData(json);
                if (json.clusters?.length > 0) setSelectedCluster(json.clusters[0]);
            } catch (err) {
                console.error("Cluster fetch failed:", err);
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
                <p className="text-sm text-muted-foreground">Running K-means cluster analysis...</p>
            </div>
        );
    }

    if (!data) return <p className="text-muted-foreground p-4">No data available.</p>;

    const { clusters } = data;

    // Scatter chart data: NDVI vs SMI, colored by cluster
    const scatterData = clusters.flatMap(c =>
        c.members.map(m => ({
            ...m,
            cluster: c.id,
            clusterLabel: c.label,
            color: c.color,
        }))
    );

    // Radar chart data for selected cluster
    const radarData = selectedCluster ? [
        { metric: "Avg NDVI", value: selectedCluster.avg_ndvi * 100, fullMark: 100 },
        { metric: "Avg SMI", value: selectedCluster.avg_smi * 100, fullMark: 100 },
        { metric: "Avg Rainfall", value: Math.min(selectedCluster.avg_rainfall * 100, 100), fullMark: 100 },
        { metric: "Resilience", value: selectedCluster.avg_ndvi > 0.4 ? 80 : selectedCluster.avg_ndvi > 0.25 ? 40 : 15, fullMark: 100 },
        { metric: "Stability", value: Math.max(0, 100 - selectedCluster.members.reduce((s, m) => s + m.std_ndvi, 0) / selectedCluster.members.length * 500), fullMark: 100 },
    ] : [];

    return (
        <div className="h-full overflow-auto p-1 space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-bold flex items-center gap-2">
                        <Layers className="h-5 w-5 text-purple-400" />
                        Cluster Analytics
                    </h2>
                    <p className="text-sm text-muted-foreground mt-1">
                        K-means clustering on district drought profiles — portfolio diversification analysis
                    </p>
                </div>
                <div className="flex gap-2">
                    {clusters.map(c => (
                        <div
                            key={c.id}
                            onClick={() => setSelectedCluster(c)}
                            className={cn(
                                "px-3 py-1.5 rounded-lg text-xs font-semibold cursor-pointer border transition-all",
                                selectedCluster?.id === c.id ? "ring-2 ring-offset-1 ring-offset-background" : "opacity-70 hover:opacity-100"
                            )}
                            style={{
                                backgroundColor: `${c.color}15`,
                                borderColor: `${c.color}40`,
                                color: c.color,
                                ...(selectedCluster?.id === c.id ? { ringColor: c.color } : {}),
                            }}
                        >
                            {c.label} ({c.size})
                        </div>
                    ))}
                </div>
            </div>

            {/* Main content */}
            <div className="flex gap-4" style={{ height: '480px' }}>
                {/* Left: Scatter chart */}
                <div className="flex-1 rounded-xl border border-border/50 bg-card shadow-lg p-4 flex flex-col">
                    <h3 className="text-sm font-semibold mb-1">District Feature Space</h3>
                    <p className="text-xs text-muted-foreground mb-3">NDVI vs Soil Moisture — each dot is a district, color = cluster</p>
                    <div className="flex-1">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ bottom: 10, left: 0, right: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                <XAxis
                                    dataKey="avg_ndvi"
                                    name="Avg NDVI"
                                    tick={{ fontSize: 10, fill: "#888" }}
                                    label={{ value: "Average NDVI", position: "bottom", fontSize: 11, fill: "#888" }}
                                    domain={[0, 0.8]}
                                />
                                <YAxis
                                    dataKey="avg_smi"
                                    name="Avg SMI"
                                    tick={{ fontSize: 10, fill: "#888" }}
                                    label={{ value: "Avg SMI", angle: -90, position: "insideLeft", fontSize: 11, fill: "#888" }}
                                    domain={[0, 0.8]}
                                />
                                <Tooltip
                                    contentStyle={{ backgroundColor: "#1a1a2e", border: "1px solid #333", borderRadius: 8, fontSize: 12 }}
                                    formatter={(value, name) => [value.toFixed(4), name]}
                                    labelFormatter={() => ""}
                                    content={({ payload }) => {
                                        if (!payload?.[0]) return null;
                                        const d = payload[0].payload;
                                        return (
                                            <div className="bg-[#1a1a2e] border border-[#333] rounded-lg p-2 text-xs">
                                                <p className="font-bold text-white">{d.name}</p>
                                                <p className="text-muted-foreground">{d.state} • {d.clusterLabel}</p>
                                                <div className="mt-1 space-y-0.5">
                                                    <p>NDVI: <span className="font-mono">{d.avg_ndvi}</span></p>
                                                    <p>SMI: <span className="font-mono">{d.avg_smi}</span></p>
                                                    <p>Rain: <span className="font-mono">{d.avg_rainfall}</span></p>
                                                </div>
                                            </div>
                                        );
                                    }}
                                />
                                <Scatter data={scatterData} shape="circle">
                                    {scatterData.map((entry, index) => (
                                        <Cell
                                            key={index}
                                            fill={entry.color}
                                            stroke={entry.color}
                                            strokeWidth={selectedCluster?.id === entry.cluster ? 3 : 1}
                                            opacity={selectedCluster?.id === entry.cluster ? 1 : 0.4}
                                            r={selectedCluster?.id === entry.cluster ? 8 : 5}
                                        />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Right: Cluster detail */}
                <div className="w-[420px] shrink-0 flex flex-col gap-4 overflow-auto">
                    {selectedCluster && (
                        <>
                            {/* Cluster Summary */}
                            <div className="rounded-xl border border-border/50 bg-card shadow-lg p-4">
                                <div className="flex items-center justify-between mb-3">
                                    <div className="flex items-center gap-2">
                                        <div className="h-3 w-3 rounded-full" style={{ backgroundColor: selectedCluster.color }} />
                                        <h3 className="text-lg font-bold">{selectedCluster.label}</h3>
                                    </div>
                                    <span className="text-xs text-muted-foreground">{selectedCluster.size} districts</span>
                                </div>

                                {/* Cluster stats */}
                                <div className="grid grid-cols-3 gap-2 mb-3">
                                    <div className="rounded-lg bg-muted/30 p-2 text-center">
                                        <div className="text-[10px] text-muted-foreground uppercase">Avg NDVI</div>
                                        <div className="text-lg font-bold font-mono" style={{ color: selectedCluster.color }}>
                                            {selectedCluster.avg_ndvi.toFixed(3)}
                                        </div>
                                    </div>
                                    <div className="rounded-lg bg-muted/30 p-2 text-center">
                                        <div className="text-[10px] text-muted-foreground uppercase flex items-center justify-center gap-1"><Droplets className="h-3 w-3" /> SMI</div>
                                        <div className="text-lg font-bold font-mono text-blue-400">
                                            {selectedCluster.avg_smi.toFixed(3)}
                                        </div>
                                    </div>
                                    <div className="rounded-lg bg-muted/30 p-2 text-center">
                                        <div className="text-[10px] text-muted-foreground uppercase flex items-center justify-center gap-1"><Cloud className="h-3 w-3" /> Rain</div>
                                        <div className="text-lg font-bold font-mono text-cyan-400">
                                            {selectedCluster.avg_rainfall.toFixed(3)}
                                        </div>
                                    </div>
                                </div>

                                {/* Radar chart */}
                                <div className="rounded-lg border border-border/30 bg-muted/10 p-2">
                                    <h4 className="text-xs font-semibold mb-1 text-muted-foreground">Cluster Profile</h4>
                                    <ResponsiveContainer width="100%" height={180}>
                                        <RadarChart data={radarData}>
                                            <PolarGrid stroke="#333" />
                                            <PolarAngleAxis dataKey="metric" tick={{ fontSize: 9, fill: "#888" }} />
                                            <PolarRadiusAxis tick={false} domain={[0, 100]} />
                                            <Radar
                                                dataKey="value"
                                                stroke={selectedCluster.color}
                                                fill={selectedCluster.color}
                                                fillOpacity={0.2}
                                                strokeWidth={2}
                                            />
                                        </RadarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            {/* Member districts */}
                            <div className="rounded-xl border border-border/50 bg-card shadow-lg p-4 flex-1">
                                <h4 className="text-sm font-semibold mb-2">Districts in this Cluster</h4>
                                <div className="space-y-2">
                                    {selectedCluster.members.map(m => (
                                        <div key={m.name} className="flex items-center justify-between p-2 rounded-lg bg-muted/20 border border-border/20">
                                            <div className="flex items-center gap-2">
                                                <MapPin className="h-3 w-3 text-muted-foreground" />
                                                <div>
                                                    <span className="text-sm font-medium">{m.name}</span>
                                                    <span className="text-[10px] text-muted-foreground ml-2">{m.state}</span>
                                                </div>
                                            </div>
                                            <div className="flex gap-3 text-xs font-mono">
                                                <span className={m.latest_ndvi < 0.3 ? "text-red-400" : m.latest_ndvi < 0.5 ? "text-amber-400" : "text-emerald-400"}>
                                                    {m.latest_ndvi.toFixed(2)}
                                                </span>
                                                <span className="text-muted-foreground">
                                                    {m.ndvi_trend > 0 ? "↑" : m.ndvi_trend < -0.02 ? "↓" : "→"} {m.ndvi_trend.toFixed(3)}
                                                </span>
                                            </div>
                                        </div>
                                    ))}
                                </div>

                                {/* Portfolio insight */}
                                <div className="mt-3 rounded-lg border border-purple-500/20 bg-purple-500/5 p-3">
                                    <div className="flex items-center gap-2 mb-1">
                                        <BarChart3 className="h-4 w-4 text-purple-400" />
                                        <span className="text-xs font-bold text-purple-400 uppercase">Portfolio Insight</span>
                                    </div>
                                    <p className="text-xs text-muted-foreground leading-relaxed">
                                        {selectedCluster.avg_ndvi < 0.3
                                            ? `These ${selectedCluster.size} districts share similar severe drought characteristics. Avoid concentrating >40% of insurance exposure in this cluster. Consider hedging with districts from resilient zones.`
                                            : selectedCluster.avg_ndvi < 0.45
                                                ? `Moderate risk cluster — these districts are transitional. Monitor closely for migration into the severe zone. Maintain balanced exposure.`
                                                : `Resilient districts with healthy vegetation. Low claim probability — suitable for standard premium rates with minimal reserves required.`
                                        }
                                    </p>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>

            {/* Bottom: All clusters comparison */}
            <div className="rounded-xl border border-border/50 bg-card shadow-lg p-4">
                <h3 className="text-sm font-semibold mb-1">Cluster Comparison</h3>
                <p className="text-xs text-muted-foreground mb-3">Side-by-side drought profile of each cluster</p>
                <div className="grid grid-cols-3 gap-3">
                    {clusters.map(c => (
                        <div
                            key={c.id}
                            onClick={() => setSelectedCluster(c)}
                            className={cn(
                                "rounded-lg border p-3 cursor-pointer transition-all",
                                selectedCluster?.id === c.id ? "ring-1" : "opacity-70 hover:opacity-100"
                            )}
                            style={{
                                borderColor: `${c.color}40`,
                                backgroundColor: `${c.color}08`,
                                ...(selectedCluster?.id === c.id ? { boxShadow: `0 0 12px ${c.color}20` } : {}),
                            }}
                        >
                            <div className="flex items-center gap-2 mb-2">
                                <div className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: c.color }} />
                                <span className="text-sm font-bold" style={{ color: c.color }}>{c.label}</span>
                            </div>
                            <div className="grid grid-cols-3 gap-1 text-center">
                                <div>
                                    <div className="text-[9px] text-muted-foreground uppercase">NDVI</div>
                                    <div className="text-sm font-bold font-mono">{c.avg_ndvi.toFixed(3)}</div>
                                </div>
                                <div>
                                    <div className="text-[9px] text-muted-foreground uppercase">SMI</div>
                                    <div className="text-sm font-bold font-mono">{c.avg_smi.toFixed(3)}</div>
                                </div>
                                <div>
                                    <div className="text-[9px] text-muted-foreground uppercase">Rain</div>
                                    <div className="text-sm font-bold font-mono">{c.avg_rainfall.toFixed(3)}</div>
                                </div>
                            </div>
                            <div className="mt-2 text-[10px] text-muted-foreground">
                                {c.members.map(m => m.name).join(", ")}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
