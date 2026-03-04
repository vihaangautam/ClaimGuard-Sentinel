import React from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { Card } from "@/components/ui/card";

export function RiskMap({ districts = [], onSelectDistrict }) {
    return (
        <Card className="h-full w-full overflow-hidden border-none bg-muted/5 relative">
            <div className="absolute top-4 left-14 z-[400] bg-background/90 backdrop-blur-sm px-3 py-2 rounded-lg shadow-lg border border-border/50">
                <h3 className="font-bold text-sm">Risk Assessment Map</h3>
                <p className="text-xs text-muted-foreground/80">{districts.length} Districts • Real-time Monitoring</p>
            </div>

            <MapContainer
                center={[13.5, 77.5]}
                zoom={7}
                scrollWheelZoom={false}
                style={{ height: '100%', width: '100%', background: '#0f172a' }}
                className="z-0"
            >
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                />

                {districts.map((district, i) => {
                    let color = "#10b981"; // Safe (Green)
                    if (district.risk > 0.5) color = "#f59e0b"; // Warning (Amber)
                    if (district.risk > 0.7) color = "#ef4444"; // Danger (Red)

                    const radius = 12 + (district.risk * 15); // Bigger = higher risk

                    return (
                        <CircleMarker
                            key={district.name || i}
                            center={[district.lat, district.lng]}
                            pathOptions={{
                                color: color,
                                fillColor: color,
                                fillOpacity: 0.5,
                                weight: 2
                            }}
                            radius={radius}
                            eventHandlers={{
                                click: () => onSelectDistrict(district),
                            }}
                        >
                            <Tooltip direction="top" offset={[0, -15]} opacity={1} permanent={false} className="bg-transparent border-none text-white font-bold shadow-none">
                                <span className={`px-2 py-1 rounded text-xs border ${district.risk > 0.7 ? 'bg-red-900/80 text-red-200 border-red-500/50' :
                                    district.risk > 0.5 ? 'bg-amber-900/80 text-amber-200 border-amber-500/50' :
                                        'bg-green-900/80 text-green-200 border-green-500/50'
                                    }`}>
                                    {district.name} — NDVI: {district.ndvi?.toFixed(2)}
                                </span>
                            </Tooltip>
                            <Popup className="text-black">
                                <div className="p-1 min-w-[180px]">
                                    <h4 className="font-bold text-base">{district.name}</h4>
                                    <p className="text-xs text-gray-500 mb-2">{district.state}</p>
                                    <div className="space-y-1 text-sm">
                                        <p>Risk: <strong>{(district.risk * 100).toFixed(0)}%</strong></p>
                                        <p>NDVI: <strong>{district.ndvi?.toFixed(4)}</strong></p>
                                        <p>Soil Moisture: <strong>{district.smi?.toFixed(4)}</strong></p>
                                        <p>Rainfall: <strong>{district.rainfall?.toFixed(4)}</strong></p>
                                        <p className="text-xs mt-1 text-gray-400">Latest: {district.latest_date}</p>
                                    </div>
                                </div>
                            </Popup>
                        </CircleMarker>
                    );
                })}
            </MapContainer>

            {/* Legend */}
            <div className="absolute bottom-8 right-4 z-[400] flex flex-col gap-1.5 bg-background/90 backdrop-blur-sm p-2.5 rounded-lg shadow-lg border border-border/50 text-xs">
                <div className="flex items-center gap-2">
                    <div className="h-0 w-0 border-l-[5px] border-r-[5px] border-b-[8px] border-l-transparent border-r-transparent border-b-red-500" />
                    <span>High Risk ({'>'}70%)</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-sm bg-amber-500 rotate-45 scale-75" />
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
