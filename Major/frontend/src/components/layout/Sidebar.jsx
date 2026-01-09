import React, { useState } from 'react';
import { cn } from "@/lib/utils";
import { LayoutDashboard, FileSearch, Settings, Users, BarChart3, ChevronLeft, ChevronRight, Menu } from "lucide-react";

export function Sidebar({ activePage, setActivePage, isCollapsed, setIsCollapsed }) {
  const navItems = [
    { id: 'dashboard', label: 'Situation Forecast', icon: LayoutDashboard },
    { id: 'investigation', label: 'Investigation Queue', icon: FileSearch, count: 5 },
    { id: 'analytics', label: 'Cluster Analytics', icon: BarChart3 },
    { id: 'team', label: 'Field Agents', icon: Users },
    { id: 'settings', label: 'System Settings', icon: Settings },
  ];

  return (
    <div
      className={cn(
        "flex flex-col border-r bg-card/30 backdrop-blur-sm transition-all duration-300 ease-in-out relative hidden md:flex",
        isCollapsed ? "w-16" : "w-64"
      )}
    >
      {/* Toggle Button */}
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="absolute -right-3 top-6 z-50 flex h-6 w-6 items-center justify-center rounded-full border bg-background text-muted-foreground shadow-md hover:text-foreground"
      >
        {isCollapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronLeft className="h-3 w-3" />}
      </button>

      {/* Brand Header */}
      <div className={cn("mb-2 px-2 py-4 flex items-center overflow-hidden whitespace-nowrap", isCollapsed ? "justify-center" : "justify-start gap-3")}>
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded bg-blue-600 font-bold text-white shadow-lg shadow-blue-900/20">
          CS
        </div>
        <div className={cn("transition-opacity duration-300", isCollapsed ? "opacity-0 w-0 hidden" : "opacity-100")}>
          <h2 className="text-sm font-bold tracking-tight text-foreground leading-none">
            ClaimGuard
          </h2>
          <p className="text-xs text-muted-foreground">Sentinel</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex flex-col gap-2 px-2 mt-4">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activePage === item.id;

          return (
            <button
              key={item.id}
              onClick={() => setActivePage(item.id)}
              className={cn(
                "flex items-center rounded-lg py-2 transition-all hover:bg-muted group relative",
                isActive ? "bg-primary/10 text-primary hover:bg-primary/15" : "text-muted-foreground",
                isCollapsed ? "justify-center px-0" : "justify-start px-3"
              )}
              title={isCollapsed ? item.label : undefined}
            >
              <Icon className={cn("h-4 w-4 shrink-0", isActive && "text-blue-400")} />

              {!isCollapsed && (
                <span className="ml-3 text-sm font-medium truncate">
                  {item.label}
                </span>
              )}

              {/* Counter Badge */}
              {item.count && (
                <span className={cn(
                  "flex h-4 w-4 items-center justify-center rounded-full bg-blue-600 text-[10px] font-bold text-white",
                  isCollapsed ? "absolute top-1 right-1 h-2 w-2 p-0 border border-background" : "ml-auto"
                )}>
                  {!isCollapsed && item.count}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      <div className="mt-auto p-2">
        <div className={cn("rounded-lg bg-muted/20 p-2 border border-border/50", isCollapsed ? "flex justify-center" : "")}>
          {!isCollapsed ? (
            <>
              <h4 className="text-[10px] uppercase font-semibold text-muted-foreground mb-2">System Status</h4>
              <div className="flex items-center gap-2 text-xs text-emerald-400">
                <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
                <span>Model Online</span>
              </div>
            </>
          ) : (
            <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.5)]" title="System Online" />
          )}
        </div>
      </div>
    </div>
  );
}
