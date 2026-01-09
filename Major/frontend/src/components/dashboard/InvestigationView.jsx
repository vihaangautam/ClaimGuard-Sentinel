import React, { useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent, CardDescription, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Sparkles, FileText, CheckCircle, XCircle, ArrowLeft, ArrowRight } from "lucide-react";

// Mock Data Generators for varied charts
const generateChartData = (type) => {
    const base = type === 'Drought' ? 0.2 : 0.45;
    return [
        { month: 'Aug', farm: base + 0.1, district: 0.48, threshold: 0.3 },
        { month: 'Sep', farm: base + 0.08, district: 0.46, threshold: 0.3 },
        { month: 'Oct', farm: base + 0.05, district: 0.45, threshold: 0.3 },
        { month: 'Nov', farm: base - 0.05, district: 0.44, threshold: 0.3 },
        { month: 'Dec', farm: base - 0.15, district: 0.42, threshold: 0.3 },
        { month: 'Jan', farm: base - 0.18, district: 0.41, threshold: 0.3 },
        { month: 'Feb (FC)', farm: base - 0.20, district: 0.40, threshold: 0.3 },
    ];
}

export function InvestigationView({ claim, onBack, onProcessClaim, queueLength, currentIndex }) {
    const [aiAnalysis, setAiAnalysis] = useState(null)

    // Reset AI analysis when claim changes
    useEffect(() => {
        setAiAnalysis(null)
    }, [claim])

    if (!claim) return <div className="p-8 text-center text-muted-foreground">No active claim selected.</div>;

    const chartData = generateChartData(claim.type);

    const handleAnalze = () => {
        // Dynamic AI Analysis based on Claim Type
        let analysis = "";
        if (claim.type === 'Drought') {
            analysis = `I've analyzed the spectral data for ${claim.district}. While the district average is effectively normal (NDVI 0.42), ${claim.name}'s plot shows a severe deviation (NDVI 0.15). This -64% anomaly confirms localized crop stress consistent with water deprivation. Recommendation: APPROVE.`;
        } else {
            analysis = `Analysis for ${claim.district} indicates high moisture levels. However, ${claim.name}'s plot does not show the expected spectral signature for flood damage. The vegetation index resembles healthy crop cover masked by cloud cover. Recommendation: INVESTIGATE ON-SITE.`;
        }
        setAiAnalysis(analysis);
    }

    return (
        <div className="h-full flex flex-col gap-4">
            {/* Navigation / Header for Page */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <Button variant="ghost" className="gap-2 text-muted-foreground hover:text-white pl-0" onClick={onBack}>
                        <ArrowLeft className="h-4 w-4" /> Back to Map
                    </Button>
                    <div className="h-4 w-px bg-border" />
                    <span className="text-sm text-muted-foreground">
                        Queue: <span className="text-foreground font-mono">{currentIndex + 1} / {queueLength}</span>
                    </span>
                </div>

                <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm" onClick={onProcessClaim} className="gap-2">
                        Skip / Next <ArrowRight className="h-4 w-4" />
                    </Button>
                </div>
            </div>

            <div className="flex flex-1 gap-4 overflow-hidden">
                {/* Left Panel: Farmer Claim Card */}
                <Card className="w-1/3 flex flex-col bg-card/50 backdrop-blur-sm border-border/50 animate-in slide-in-from-left-4 duration-500">
                    <CardHeader>
                        <div className="flex justify-between items-start">
                            <div className="h-12 w-12 rounded-full bg-indigo-900/50 border border-indigo-500/30 flex items-center justify-center text-lg font-bold text-indigo-300">
                                {claim.name.split(' ').map(n => n[0]).join('')}
                            </div>
                            <Badge variant="outline" className="text-amber-500 border-amber-500/50 bg-amber-500/10 animate-pulse">
                                {claim.status}
                            </Badge>
                        </div>
                        <CardTitle className="mt-4 text-2xl">{claim.name}</CardTitle>
                        <CardDescription className="font-mono text-xs opacity-70">ID: #{claim.id} • {claim.date}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-6 flex-1">
                        <div className="grid grid-cols-2 gap-y-6 gap-x-2 text-sm">
                            <div className="flex flex-col gap-1">
                                <span className="text-muted-foreground text-xs uppercase tracking-wider">District</span>
                                <span className="font-medium text-lg text-white">{claim.district}</span>
                            </div>
                            <div className="flex flex-col gap-1">
                                <span className="text-muted-foreground text-xs uppercase tracking-wider">Risk Level</span>
                                <span className={`font-medium text-lg ${claim.risk === 'High' ? 'text-red-500' : 'text-green-500'}`}>
                                    {claim.risk} Risk
                                </span>
                            </div>
                            <div className="flex flex-col gap-1">
                                <span className="text-muted-foreground text-xs uppercase tracking-wider">Crop</span>
                                <span className="font-medium">Groundnut (Kharif)</span>
                            </div>
                            <div className="flex flex-col gap-1">
                                <span className="text-muted-foreground text-xs uppercase tracking-wider">Claim Cause</span>
                                <span className="font-medium">{claim.type}</span>
                            </div>
                        </div>

                        <div className="mt-4 p-4 rounded-lg bg-blue-950/20 border border-blue-900/30 text-xs text-blue-200/70 leading-relaxed">
                            <p><strong>System Flag:</strong> {claim.risk === 'High' ? 'CRITICAL DISCREPANCY DETECTED.' : 'Routine validation required.'} Satellite imagery suggests vegetation index does not correlate with reported loss magnitude.</p>
                        </div>
                    </CardContent>
                    <CardFooter className="flex flex-col gap-3 pt-2 pb-6">
                        <Button
                            className="w-full bg-emerald-600 hover:bg-emerald-500 text-white gap-2 font-bold h-12 shadow-[0_0_20px_rgba(16,185,129,0.3)] transition-all hover:scale-[1.02]"
                            onClick={() => {
                                // In a real app, this would verify first.
                                onProcessClaim();
                            }}
                        >
                            <CheckCircle className="h-5 w-5" /> APPROVE CLAIM (PAY)
                        </Button>
                        <Button
                            variant="destructive"
                            className="w-full gap-2 font-bold h-12 bg-red-900/50 border border-red-500/20 hover:bg-red-900/80 hover:scale-[1.02] transition-all"
                            onClick={onProcessClaim}
                        >
                            <XCircle className="h-5 w-5" /> REJECT (FLAG FRAUD)
                        </Button>
                    </CardFooter>
                </Card>

                {/* Right Panel: The Truth Chart */}
                <Card className="flex-1 flex flex-col bg-card/50 backdrop-blur-sm border-border/50 relative overflow-hidden transition-all duration-500">
                    <CardHeader className="flex flex-row items-center justify-between pb-2">
                        <div>
                            <CardTitle className="text-lg">Vegetation Truth Chart™</CardTitle>
                            <p className="text-sm text-muted-foreground">Multi-spectral Analysis vs. Cluster Baseline</p>
                        </div>
                        <Button size="sm" variant="outline" onClick={handleAnalze} className="gap-2 text-indigo-400 border-indigo-500/30 hover:bg-indigo-950/30 shadow-[0_0_15px_rgba(99,102,241,0.1)]">
                            <Sparkles className="h-4 w-4" />
                            Ask AI Analyst
                        </Button>
                    </CardHeader>

                    <CardContent className="flex-1 w-full min-h-[300px] p-0 pb-4 pr-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                                <defs>
                                    <linearGradient id="farmGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                                    </linearGradient>
                                    <linearGradient id="districtGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#94a3b8" stopOpacity={0.1} />
                                        <stop offset="95%" stopColor="#94a3b8" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} vertical={false} />
                                <XAxis dataKey="month" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} dy={10} />
                                <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} domain={[0, 0.8]} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#020617', borderColor: '#1e293b', color: '#f8fafc', borderRadius: '12px', boxShadow: '0 10px 15px -3px 0 rgba(0, 0, 0, 0.5)' }}
                                    itemStyle={{ color: '#f8fafc' }}
                                    cursor={{ stroke: '#64748b', strokeWidth: 1, strokeDasharray: '4 4' }}
                                />

                                {/* Threshold Line */}
                                <ReferenceLine y={0.3} stroke="#ef4444" strokeDasharray="3 3" label={{ position: 'right', value: 'Drought Threshold (0.3)', fill: '#ef4444', fontSize: 10 }} />

                                {/* District Average (Baseline) */}
                                <Area type="monotone" dataKey="district" stroke="#94a3b8" fillOpacity={1} fill="url(#districtGradient)" strokeWidth={2} name="District Avg (Healthy)" />

                                {/* Farm Data (The Suspect) */}
                                <Area type="monotone" dataKey="farm" stroke="#3b82f6" fillOpacity={1} fill="url(#farmGradient)" strokeWidth={3} name="Claimant Farm" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </CardContent>

                    {/* AI Insight Overlay */}
                    {aiAnalysis && (
                        <div className="absolute bottom-4 right-4 left-4 bg-indigo-950/95 backdrop-blur-xl border border-indigo-500/50 p-6 rounded-xl shadow-2xl animate-in slide-in-from-bottom-4 zoom-in-95 duration-300">
                            <div className="flex items-start gap-4">
                                <div className="p-3 bg-indigo-500/20 rounded-lg shrink-0">
                                    <Sparkles className="h-6 w-6 text-indigo-400 animate-pulse" />
                                </div>
                                <div className="flex-1">
                                    <h4 className="font-bold text-indigo-100 text-sm mb-1 flex items-center gap-2">
                                        AI ANALYST INSIGHT
                                        <span className="text-[10px] bg-indigo-500/20 px-2 py-0.5 rounded text-indigo-300">CONFIDENCE: 94%</span>
                                    </h4>
                                    <p className="text-sm text-indigo-100/90 leading-relaxed font-light">
                                        {aiAnalysis}
                                    </p>
                                </div>
                                <button onClick={() => setAiAnalysis(null)} className="text-indigo-400 hover:text-white transition-colors">
                                    <XCircle className="h-5 w-5" />
                                </button>
                            </div>
                        </div>
                    )}
                </Card>
            </div>
        </div>
    );
}
