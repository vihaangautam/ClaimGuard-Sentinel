import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { AlertTriangle, CheckCircle, Droplets, Bug } from "lucide-react";

const alerts = [
    {
        id: 1,
        district: "Anantapur",
        message: "Forecasted NDVI drop below 0.20 in 15 days",
        type: "danger",
        icon: AlertTriangle,
        time: "2 mins ago"
    },
    {
        id: 2,
        district: "Chitradurga",
        message: "Vegetation healthy, risk stable",
        type: "success",
        icon: CheckCircle,
        time: "5 mins ago"
    },
    {
        id: 3,
        district: "Ballari",
        message: "Moisture level declining, irrigation recommended",
        type: "warning",
        icon: Droplets,
        time: "12 mins ago"
    },
    {
        id: 4,
        district: "Anantapur",
        message: "Pest activity detected in northern zones",
        type: "info",
        icon: Bug,
        time: "28 mins ago"
    }
];

export function LiveAlerts() {
    return (
        <Card className="h-full border-none bg-muted/10 shadow-none">
            <CardHeader className="pb-3">
                <CardTitle className="text-lg font-medium">Live Alerts Feed</CardTitle>
                <p className="text-sm text-muted-foreground">Real-time Risk Notifications</p>
            </CardHeader>
            <CardContent className="flex flex-col gap-3 overflow-y-auto pr-2" style={{ maxHeight: 'calc(100% - 80px)' }}>
                {alerts.map((alert) => {
                    const Icon = alert.icon;
                    let colorClass = "bg-muted text-foreground";
                    if (alert.type === 'danger') colorClass = "bg-red-500/10 text-red-500 border-red-500/20";
                    if (alert.type === 'success') colorClass = "bg-green-500/10 text-green-500 border-green-500/20";
                    if (alert.type === 'warning') colorClass = "bg-amber-500/10 text-amber-500 border-amber-500/20";
                    if (alert.type === 'info') colorClass = "bg-blue-500/10 text-blue-500 border-blue-500/20";

                    return (
                        <div key={alert.id} className={`flex flex-col gap-1 rounded-md border p-3 ${colorClass}`}>
                            <div className="flex items-center gap-2">
                                <Icon className="h-4 w-4" />
                                <span className="font-bold text-sm">{alert.district}</span>
                            </div>
                            <p className="text-xs opacity-90">{alert.message}</p>
                            <span className="text-[10px] opacity-60 mt-1">{alert.time}</span>
                        </div>
                    );
                })}
            </CardContent>
        </Card>
    );
}
