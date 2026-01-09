import React from 'react';
import { Card, CardContent } from "@/components/ui/card";

export function StatsWidget() {
    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="bg-gradient-to-br from-red-950/30 to-background border-red-900/20">
                <CardContent className="p-4">
                    <div className="flex justify-between items-start">
                        <div>
                            <p className="text-xs text-muted-foreground uppercase">Anantapur</p>
                            <p className="text-2xl font-bold font-mono mt-1">0.15</p>
                        </div>
                        <div className="h-2 w-2 rounded-full bg-red-500 animate-pulse" />
                    </div>
                    <div className="mt-2 flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">NDVI</span>
                        <span className="text-red-400 font-bold">Risk: 78%</span>
                    </div>
                </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-background to-background border-muted/20">
                <CardContent className="p-4">
                    <div className="flex justify-between items-start">
                        <div>
                            <p className="text-xs text-muted-foreground uppercase">Chitradurga</p>
                            <p className="text-2xl font-bold font-mono mt-1">0.52</p>
                        </div>
                    </div>
                    <div className="mt-2 flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">NDVI</span>
                        <span className="text-green-400 font-bold">Risk: 42%</span>
                    </div>
                </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-amber-950/30 to-background border-amber-900/20">
                <CardContent className="p-4">
                    <div className="flex justify-between items-start">
                        <div>
                            <p className="text-xs text-muted-foreground uppercase">Ballari</p>
                            <p className="text-2xl font-bold font-mono mt-1">0.28</p>
                        </div>
                        <div className="h-2 w-2 rounded-full bg-amber-500" />
                    </div>
                    <div className="mt-2 flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">NDVI</span>
                        <span className="text-amber-400 font-bold">Risk: 65%</span>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
