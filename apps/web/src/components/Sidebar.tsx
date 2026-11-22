'use client';

import React from 'react';
import {
  LayoutDashboard,
  Film,
  MapPin,
  Camera,
  Layers,
  ScanLine,
} from 'lucide-react';

export type ActiveTab = 'command' | 'sessions' | 'map' | 'live' | 'parking' | 'anpr';

interface SidebarProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  pendingReviewsCount: number;
}

export function Sidebar({ activeTab, onSelectTab, pendingReviewsCount }: SidebarProps) {
  const navItems: Array<{ id: ActiveTab; label: string; icon: React.ReactNode; badge?: number }> = [
    {
      id: 'anpr',
      label: 'Plate recognition',
      icon: <ScanLine className="w-4 h-4" />,
    },
    {
      id: 'command',
      label: 'Road inspection',
      icon: <LayoutDashboard className="w-4 h-4" />,
      badge: pendingReviewsCount > 0 ? pendingReviewsCount : undefined,
    },
    {
      id: 'parking',
      label: 'Parking layout',
      icon: <Layers className="w-4 h-4" />,
    },
    {
      id: 'sessions',
      label: 'Drive sessions',
      icon: <Film className="w-4 h-4" />,
    },
    {
      id: 'map',
      label: 'Map',
      icon: <MapPin className="w-4 h-4" />,
    },
    {
      id: 'live',
      label: 'Live camera',
      icon: <Camera className="w-4 h-4" />,
    },
  ];


  return (
    <aside className="w-64 bg-command-bg flex flex-col justify-between px-4 py-6 min-h-[calc(100vh-5rem)] border-r border-command-border">
      <div className="space-y-8">
        <div>
          <span className="text-[12px] text-command-muted px-3">Workspace</span>
          <nav className="mt-3 space-y-1.5">
            {navItems.map((item) => {
              const active = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  aria-current={active ? 'page' : undefined}
                  className={`w-full flex items-center justify-between pl-3.5 pr-3 py-2.5 rounded-full text-[14px] transition-all ${
                    active
                      ? 'bg-brand-soft text-brand-deep font-medium'
                      : 'text-command-muted hover:bg-command-elevated hover:text-command-text'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span className={active ? 'text-brand' : 'text-command-subtle'}>{item.icon}</span>
                    <span>{item.label}</span>
                  </div>
                  {item.badge !== undefined && (
                    <span className="min-w-[22px] px-1.5 py-0.5 text-[11px] font-medium rounded-full bg-brand text-white text-center">
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        <div className="p-4 rounded-3xl bg-command-surface border border-command-border text-[12px] text-command-muted space-y-2">
          <div className="flex items-center gap-2 text-command-text font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-accent-amber" />
            <span>Human review first</span>
          </div>
          <p className="leading-relaxed">
            Model outputs are unverified suggestions. A person confirms readings and work orders.
          </p>
        </div>
      </div>

      <div className="text-[12px] text-command-muted px-3 flex justify-between items-center">
        <span>RoadSense v1.0</span>
        <span>Runs locally</span>
      </div>
    </aside>
  );
}
