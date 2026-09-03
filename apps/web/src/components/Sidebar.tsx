'use client';

import React from 'react';
import {
  LayoutDashboard,
  Film,
  MapPin,
  Camera,
  Layers,
  FileSpreadsheet,
} from 'lucide-react';

export type ActiveTab = 'command' | 'sessions' | 'map' | 'live';

interface SidebarProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  pendingReviewsCount: number;
}

export function Sidebar({ activeTab, onSelectTab, pendingReviewsCount }: SidebarProps) {
  const navItems: Array<{ id: ActiveTab; label: string; icon: React.ReactNode; badge?: number }> = [
    {
      id: 'command',
      label: 'Command Center',
      icon: <LayoutDashboard className="w-4 h-4" />,
      badge: pendingReviewsCount > 0 ? pendingReviewsCount : undefined,
    },
    {
      id: 'sessions',
      label: 'Drive Sessions',
      icon: <Film className="w-4 h-4" />,
    },
    {
      id: 'map',
      label: 'Spatial Map',
      icon: <MapPin className="w-4 h-4" />,
    },
    {
      id: 'live',
      label: 'Live Dashcam',
      icon: <Camera className="w-4 h-4" />,
    },
  ];

  return (
    <aside className="w-64 border-r border-command-border bg-command-surface flex flex-col justify-between p-4 min-h-[calc(100vh-4rem)]">
      <div className="space-y-6">
        <div>
          <span className="text-[10px] font-mono tracking-wider text-command-muted uppercase px-3">
            Operations Navigation
          </span>
          <nav className="mt-2 space-y-1">
            {navItems.map((item) => {
              const active = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
                    active
                      ? 'bg-command-elevated text-radar-bright border border-command-border shadow-sm'
                      : 'text-command-muted hover:bg-command-elevated/50 hover:text-command-text'
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <span className={active ? 'text-radar-bright' : 'text-command-muted'}>
                      {item.icon}
                    </span>
                    <span>{item.label}</span>
                  </div>
                  {item.badge !== undefined && (
                    <span className="px-1.5 py-0.5 text-[10px] font-mono font-bold rounded-full bg-accent-amber/20 text-accent-amber border border-accent-amber/30">
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Operational Safety Notice */}
        <div className="p-3 rounded-lg bg-command-elevated/70 border border-command-border text-[11px] text-command-muted space-y-1.5 font-mono">
          <div className="flex items-center space-x-1.5 text-accent-amber font-semibold">
            <span className="w-1.5 h-1.5 rounded-full bg-accent-amber animate-pulse" />
            <span>Operational Notice</span>
          </div>
          <p className="leading-relaxed">
            Suggestions are unverified candidate boxes. Human sign-off required for work-orders.
          </p>
        </div>
      </div>

      {/* Footer Info */}
      <div className="text-[10px] font-mono text-command-muted border-t border-command-border pt-4 px-2 flex justify-between items-center">
        <span>RoadSense v1.0.0</span>
        <span className="text-radar-bright">Offline First</span>
      </div>
    </aside>
  );
}
