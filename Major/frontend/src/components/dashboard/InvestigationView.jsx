import React, { useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent, CardDescription, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Sparkles, CheckCircle, XCircle, ArrowLeft, ArrowRight, Loader2 } from "lucide-react";
import { fetchDistrictHistory, verifyClaim, predictNDVI } from "@/lib/api";

export function InvestigationView({ claim, onBack, onProcessClaim, queueLength, currentIndex }) {
    const [aiAnalysis, setAiAnalysis] = useState(null)
    const [chartData, setChartData] = useState([])
    const [prediction, setPrediction] = useState(null)
    const [loadingChart, setLoadingChart] = useState(false)
    const [loadingAI, setLoadingAI] = useState(false)

    // Fetch real chart data when claim/district changes
    useEffect(() => {
        setAiAnalysis(null)
        setPrediction(null)
        setChartData([])

        if (!claim) return;

        async function load() {
            setLoadingChart(true)
            try {
                const history = await fetchDistrictHistory(claim.district)
                // Take last 12 months for the chart
                const recent = history.slice(-12)
                const formatted = recent.map(h => ({
                    month: h.date,
                    farm: h.ndvi,
                    smi: h.smi,
                    rainfall: h.rainfall,
                    threshold: 0.3,
                }))
                setChartData(formatted)

                // Also get prediction
                try {
                    const pred = await predictNDVI(claim.district)
                    setPrediction(pred)
                } catch (e) {
                    console.warn("Prediction failed:", e)
                }
            } catch (err) {
                console.error("Failed to load history:", err)
            } finally {
                setLoadingChart(false)
            }
        }
        load()
    }, [claim])

    if (!claim) return <div className="p-8 text-center text-muted-foreground">No active claim selected.</div>;

    const handleAnalyze = async () => {
        setLoadingAI(true)
        try {
            const result = await verifyClaim(claim.district, claim.date)
            setAiAnalysis(result)
        } catch (err) {
            setAiAnalysis({
                analysis: "Failed to reach verification API. Ensure the backend is running.",
                system_decision: "ERROR",
                confidence_score: "0%",
            })
        } finally {
            setLoadingAI(false)
        }
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
                    {prediction && (
                        <Badge variant="outline" className={`mr-2 ${prediction.drought_risk === 'HIGH' ? 'border-red-500/50 text-red-400 bg-red-950/30' :
                                prediction.drought_risk === 'MODERATE' ? 'border-amber-500/50 text-amber-400 bg-amber-950/30' :
                                    'border-green-500/50 text-green-400 bg-green-950/30'
                            }`}>
                            Predicted: {prediction.predicted_ndvi?.toFixed(3)} ({prediction.drought_risk})
                        </Badge>
                    )}
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
                                {claim.district.slice(0, 2).toUpperCase()}
                            </div>
                            <Badge variant="outline" className="text-amber-500 border-amber-500/50 bg-amber-500/10 animate-pulse">
                                {claim.status}
                            </Badge>
                        </div>
                        <CardTitle className="mt-4 text-2xl">{claim.district}</CardTitle>
                        <CardDescription className="font-mono text-xs opacity-70">ID: #{claim.id} • {claim.date}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-6 flex-1">
                        <div className="grid grid-cols-2 gap-y-6 gap-x-2 text-sm">
                            <div className="flex flex-col gap-1">
                                <span className="text-muted-foreground text-xs uppercase tracking-wider">NDVI</span>
                                <span className={`font-medium text-lg ${(claim.ndvi || 0) < 0.3 ? 'text-red-400' : 'text-green-400'}`}>
                                    {claim.ndvi?.toFixed(4) || 'N/A'}
                                </span>
                            </div>
                            <div className="flex flex-col gap-1">
                                <span className="text-muted-foreground text-xs uppercase tracking-wider">Risk Level</span>
                                <span className={`font-medium text-lg ${claim.risk === 'High' ? 'text-red-500' : 'text-amber-400'}`}>
                                    {claim.risk} Risk
                                </span>
                            </div>
                            <div className="flex flex-col gap-1">
                                <span className="text-muted-foreground text-xs uppercase tracking-wider">Soil Moisture</span>
                                <span className="font-medium">{claim.smi?.toFixed(4) || 'N/A'}</span>
                            </div>
                            <div className="flex flex-col gap-1">
                                <span className="text-muted-foreground text-xs uppercase tracking-wider">Rainfall</span>
                                <span className="font-medium">{claim.rainfall?.toFixed(4) || 'N/A'}</span>
                            </div>
                        </div>

                        {prediction && (
                            <div className={`mt-4 p-4 rounded-lg border text-xs leading-relaxed ${prediction.drought_risk === 'HIGH' ? 'bg-red-950/20 border-red-900/30 text-red-200/70' :
                                    prediction.drought_risk === 'MODERATE' ? 'bg-amber-950/20 border-amber-900/30 text-amber-200/70' :
                                        'bg-green-950/20 border-green-900/30 text-green-200/70'
                                }`}>
                                <p><strong>CNN-LSTM Forecast:</strong> Next month NDVI predicted at <strong>{prediction.predicted_ndvi?.toFixed(4)}</strong> ({prediction.prediction_date}).
                                    Drought risk: <strong>{prediction.drought_risk}</strong>.
                                    Method: {prediction.method}
                                </p>
                            </div>
                        )}
                    </CardContent>
                    <CardFooter className="flex flex-col gap-3 pt-2 pb-6">
                        <Button
                            className="w-full bg-emerald-600 hover:bg-emerald-500 text-white gap-2 font-bold h-12 shadow-[0_0_20px_rgba(16,185,129,0.3)] transition-all hover:scale-[1.02]"
                            onClick={onProcessClaim}
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
                            <p className="text-sm text-muted-foreground">Real Satellite NDVI — {claim.district}</p>
                        </div>
                        <Button
                            size="sm"
                            variant="outline"
                            onClick={handleAnalyze}
                            disabled={loadingAI}
                            className="gap-2 text-indigo-400 border-indigo-500/30 hover:bg-indigo-950/30 shadow-[0_0_15px_rgba(99,102,241,0.1)]"
                        >
                            {loadingAI ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                            {loadingAI ? 'Verifying...' : 'Verify with AI'}
                        </Button>
                    </CardHeader>

                    <CardContent className="flex-1 w-full min-h-[300px] p-0 pb-4 pr-4">
                        {loadingChart ? (
                            <div className="flex items-center justify-center h-full gap-3">
                                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                                <span className="text-sm text-muted-foreground">Loading satellite data...</span>
                            </div>
                        ) : (
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                                    <defs>
                                        <linearGradient id="ndviGradient" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} vertical={false} />
                                    <XAxis dataKey="month" stroke="#94a3b8" fontSize={10} tickLine={false} axisLine={false} dy={10} />
                                    <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} domain={[0, 0.8]} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#020617', borderColor: '#1e293b', color: '#f8fafc', borderRadius: '12px', boxShadow: '0 10px 15px -3px 0 rgba(0, 0, 0, 0.5)' }}
                                        itemStyle={{ color: '#f8fafc' }}
                                        cursor={{ stroke: '#64748b', strokeWidth: 1, strokeDasharray: '4 4' }}
                                    />

                                    {/* Drought Threshold Line */}
                                    <ReferenceLine y={0.3} stroke="#ef4444" strokeDasharray="3 3" label={{ position: 'right', value: 'Drought (0.3)', fill: '#ef4444', fontSize: 10 }} />

                                    {/* Real NDVI Data */}
                                    <Area type="monotone" dataKey="farm" stroke="#3b82f6" fillOpacity={1} fill="url(#ndviGradient)" strokeWidth={3} name="NDVI" />
                                </AreaChart>
                            </ResponsiveContainer>
                        )}
                    </CardContent>

                    {/* AI Verification Overlay */}
                    {aiAnalysis && (
                        <div className="absolute bottom-4 right-4 left-4 bg-indigo-950/95 backdrop-blur-xl border border-indigo-500/50 p-6 rounded-xl shadow-2xl animate-in slide-in-from-bottom-4 zoom-in-95 duration-300">
                            <div className="flex items-start gap-4">
                                <div className="p-3 bg-indigo-500/20 rounded-lg shrink-0">
                                    <Sparkles className="h-6 w-6 text-indigo-400 animate-pulse" />
                                </div>
                                <div className="flex-1">
                                    <h4 className="font-bold text-indigo-100 text-sm mb-1 flex items-center gap-2">
                                        AI VERIFICATION RESULT
                                        <span className={`text-[10px] px-2 py-0.5 rounded ${aiAnalysis.system_decision?.startsWith('APPROVED') ? 'bg-green-500/20 text-green-300' :
                                                aiAnalysis.system_decision?.startsWith('REJECTED') ? 'bg-red-500/20 text-red-300' :
                                                    'bg-amber-500/20 text-amber-300'
                                            }`}>
                                            {aiAnalysis.system_decision} • Confidence: {aiAnalysis.confidence_score}
                                        </span>
                                    </h4>
                                    <p className="text-sm text-indigo-100/90 leading-relaxed font-light">
                                        {aiAnalysis.analysis}
                                    </p>
                                    {aiAnalysis.verification_latency && (
                                        <p className="text-[10px] text-indigo-400 mt-2">
                                            Verified in {aiAnalysis.verification_latency} • Claim ID: {aiAnalysis.claim_id}
                                        </p>
                                    )}
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
