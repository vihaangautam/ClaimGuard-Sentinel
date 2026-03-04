import React from 'react';
import { Card, CardContent } from "@/components/ui/card";

export function StatsWidget({ districts = [] }) {
    // Show top 3 highest-risk districts
    const top3 = districts.slice(0, 3);

    const getCardStyle = (risk) => {
        if (risk > 0.7) return "bg-gradient-to-br from-red-950/30 to-background border-red-900/20";
        if (risk > 0.5) return "bg-gradient-to-br from-amber-950/30 to-background border-amber-900/20";
        return "bg-gradient-to-br from-background to-background border-muted/20";
    };

    const getRiskColor = (risk) => {
        if (risk > 0.7) return "text-red-400";
        if (risk > 0.5) return "text-amber-400";
        return "text-green-400";
    };

    if (top3.length === 0) {
        return (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {[1, 2, 3].map(i => (
                    <Card key={i} className="bg-gradient-to-br from-background to-background border-muted/20">
                        <CardContent className="p-4">
                            <div className="h-12 animate-pulse bg-muted/20 rounded" />
                        </CardContent>
                    </Card>
                ))}
            </div>
        );
    }

    return (
        <div className="grid grid-cols-3 gap-3">
            {top3.map((district) => (
                <Card key={district.name} className={getCardStyle(district.risk)}>
                    <CardContent className="p-3 text-center">
                        <div className="flex items-center justify-center gap-1.5 mb-1">
                            <p className="text-[10px] text-muted-foreground/80 uppercase tracking-wider font-semibold truncate">{district.name}</p>
                            {district.risk > 0.5 && (
                                <div className={`h-2 w-2 rounded-full shrink-0 ${district.risk > 0.7 ? 'bg-red-500 animate-pulse' : 'bg-amber-500'}`} />
                            )}
                        </div>
                        <p className="text-[10px] text-muted-foreground/50 uppercase">NDVI</p>
                        <p className="text-2xl font-bold font-mono">{district.ndvi?.toFixed(2)}</p>
                        <p className={`text-xs font-bold mt-1 ${getRiskColor(district.risk)}`}>Risk: {(district.risk * 100).toFixed(0)}%</p>
                    </CardContent>
                </Card>
            ))}
        </div>
    );
}
