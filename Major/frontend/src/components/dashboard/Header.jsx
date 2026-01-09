import React from 'react';
import { Badge } from "@/components/ui/badge";
import { Calendar, RefreshCw } from "lucide-react";

export function Header({ simulatedDate, setSimulatedDate }) {
    return (
        <header className="flex h-16 w-full items-center justify-between border-b bg-background px-6">
            <div className="flex items-center gap-2">
                {/* Breadcrumb or Page Title (Optional) */}
                <h2 className="text-sm font-medium text-muted-foreground">Command Center / Overview</h2>
            </div>

            <div className="flex items-center gap-4 rounded-md border bg-muted/20 px-4 py-2">
                <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Simulated Date</span>
                <div className="h-6 w-px bg-border" />
                <Calendar className="h-4 w-4 text-blue-500" />
                <input
                    type="date"
                    value={simulatedDate}
                    onChange={(e) => setSimulatedDate(e.target.value)}
                    className="bg-transparent text-sm font-mono font-bold focus:outline-none"
                />
            </div>

            <div className="flex items-center gap-2">
                <Badge variant="outline" className="border-green-800 text-green-500 bg-green-950/30">
                    <div className="mr-1 h-2 w-2 animate-pulse rounded-full bg-green-500" />
                    LIVE
                </Badge>
                <button className="rounded-full p-2 hover:bg-muted">
                    <RefreshCw className="h-4 w-4 text-muted-foreground" />
                </button>
            </div>
        </header>
    );
}
