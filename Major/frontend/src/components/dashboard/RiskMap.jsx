import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { Card } from "@/components/ui/card";

// Mock Data for Districts
const districts = [
    { id: 1, name: "Anantapur", lat: 14.6819, lng: 77.6006, risk: 0.78, ndvi: 0.15, status: "High Risk" },
    { id: 2, name: "Chitradurga", lat: 14.2287, lng: 76.3986, risk: 0.42, ndvi: 0.52, status: "Safe" },
    { id: 3, name: "Ballari", lat: 15.1394, lng: 76.9214, risk: 0.65, ndvi: 0.28, status: "Warning" },
];

export function RiskMap({ onSelectDistrict }) {
    // Leaflet needs to be dynamically loaded in some SSR contexts, but standard Vite React is fine.

    return (
        <Card className="h-full w-full overflow-hidden border-none bg-muted/5 relative">
            <div className="absolute top-4 left-4 z-[400] bg-background/90 p-2 rounded-md shadow-lg border">
                <h3 className="font-bold text-sm">Risk Assessment Map</h3>
                <p className="text-xs text-muted-foreground">3 Districts • Real-time Monitoring</p>
            </div>

            <MapContainer
                center={[14.68, 77.00]}
                zoom={8}
                scrollWheelZoom={false}
                style={{ height: '100%', width: '100%', background: '#0f172a' }}
                className="z-0"
            >
                {/* Dark Mode Map Style */}
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                />

                {districts.map((district) => {
                    let color = "#10b981"; // Safe (Green)
                    if (district.risk > 0.5) color = "#f59e0b"; // Warning (Amber)
                    if (district.risk > 0.7) color = "#ef4444"; // Danger (Red)

                    return (
                        <CircleMarker
                            key={district.id}
                            center={[district.lat, district.lng]}
                            pathOptions={{
                                color: color,
                                fillColor: color,
                                fillOpacity: 0.6,
                                weight: 2
                            }}
                            radius={20}
                            eventHandlers={{
                                click: () => onSelectDistrict(district),
                            }}
                        >
                            <Tooltip direction="top" offset={[0, -20]} opacity={1} permanent={true} className="bg-transparent border-none text-white font-bold shadow-none">
                                {district.status === "High Risk" && (
                                    <span className="bg-red-900/80 text-red-200 px-2 py-1 rounded text-xs border border-red-500/50">
                                        {district.risk * 100}%
                                    </span>
                                )}
                                <div className="mt-1 text-center text-xs drop-shadow-md">{district.name}</div>
                            </Tooltip>
                            <Popup className="text-black">
                                <div className="p-1">
                                    <h4 className="font-bold">{district.name}</h4>
                                    <p>Risk: {(district.risk * 100).toFixed(0)}%</p>
                                    <p>NDVI: {district.ndvi}</p>
                                </div>
                            </Popup>
                        </CircleMarker>
                    );
                })}
            </MapContainer>

            {/* Legend */}
            <div className="absolute bottom-4 right-4 z-[400] flex flex-col gap-2 bg-background/80 p-2 rounded shadow text-xs">
                <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-red-500" />
                    <span>High Risk ({'>'}70%)</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-amber-500" />
                    <span>Warning (50-70%)</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-green-500" />
                    <span>Safe ({'<'}50%)</span>
                </div>
            </div>
        </Card>
    );
}
