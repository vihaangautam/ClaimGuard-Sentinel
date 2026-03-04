import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { AlertTriangle, CheckCircle, Droplets, Info } from "lucide-react";

const ICONS = {
    danger: AlertTriangle,
    warning: Droplets,
    info: Info,
    success: CheckCircle,
};

export function LiveAlerts({ alerts = [] }) {
    if (alerts.length === 0) {
        return (
            <Card className="h-full border-none bg-muted/10 shadow-none">
                <CardHeader className="pb-3">
                    <CardTitle className="text-lg font-medium">Live Alerts Feed</CardTitle>
                    <p className="text-sm text-muted-foreground">Real-time Risk Notifications</p>
                </CardHeader>
                <CardContent className="flex items-center justify-center h-32">
                    <p className="text-sm text-muted-foreground">Connecting to satellite feed...</p>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className="h-full border-none bg-muted/10 shadow-none">
            <CardHeader className="pb-3">
                <CardTitle className="text-lg font-medium">Live Alerts Feed</CardTitle>
                <p className="text-sm text-muted-foreground">
                    {alerts.filter(a => a.type === 'danger').length} critical •{' '}
                    {alerts.filter(a => a.type === 'warning').length} warnings •{' '}
                    {alerts.filter(a => a.type === 'success').length} healthy
                </p>
            </CardHeader>
            <CardContent className="flex flex-col gap-3 overflow-y-auto pr-2" style={{ maxHeight: 'calc(100% - 80px)' }}>
                {alerts.map((alert) => {
                    const Icon = ICONS[alert.type] || Info;
                    let colorClass = "bg-muted text-foreground";
                    if (alert.type === 'danger') colorClass = "bg-red-500/10 text-red-500 border-red-500/20";
                    if (alert.type === 'success') colorClass = "bg-green-500/10 text-green-500 border-green-500/20";
                    if (alert.type === 'warning') colorClass = "bg-amber-500/10 text-amber-500 border-amber-500/20";
                    if (alert.type === 'info') colorClass = "bg-blue-500/10 text-blue-500 border-blue-500/20";

                    return (
                        <div key={alert.id} className={`flex items-center gap-2 rounded-md border px-3 py-2 ${colorClass}`}>
                            <Icon className="h-4 w-4 shrink-0" />
                            <div className="flex-1 min-w-0">
                                <div className="flex items-center justify-between">
                                    <span className="font-bold text-sm">{alert.district}</span>
                                    {alert.ndvi != null && (
                                        <span className="text-xs font-mono opacity-70">NDVI {alert.ndvi.toFixed(2)}</span>
                                    )}
                                </div>
                                <p className="text-[11px] opacity-80">{alert.message}</p>
                            </div>
                        </div>
                    );
                })}
            </CardContent>
        </Card>
    );
}
