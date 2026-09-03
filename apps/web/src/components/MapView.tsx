'use client';

import React from 'react';
import { MapPin, Navigation, Info } from 'lucide-react';
import { DriveSession, RoadEvent } from '../lib/types';

interface MapViewProps {
  session: DriveSession | null;
  roadEvents: RoadEvent[];
}

export function MapView({ session, roadEvents }: MapViewProps) {
  const hasGps = session?.route_telemetry && session.route_telemetry.length > 0;

  return (
    <div className="flex flex-col h-full rounded-xl glass-panel-elevated border border-command-border overflow-hidden">
      {/* Map Header */}
      <div className="px-4 py-3 bg-command-surface border-b border-command-border flex items-center justify-between text-xs font-mono">
        <div className="flex items-center space-x-2">
          <MapPin className="w-4 h-4 text-radar-bright" />
          <span className="font-bold text-command-text uppercase tracking-wider">
            Spatial Route & Geolocation View
          </span>
        </div>
        <div className="flex items-center space-x-2 text-[11px] text-command-muted">
          <span className="w-2 h-2 rounded-full bg-accent-cyan" />
          <span>Telemetry Status: {hasGps ? 'Embedded GPS Present' : 'No GPS in Video'}</span>
        </div>
      </div>

      {/* Map Area / Empty Boundary State */}
      <div className="flex-1 bg-command-bg flex flex-col items-center justify-center p-8 text-center">
        {hasGps ? (
          <div className="space-y-4 max-w-md">
            <Navigation className="w-12 h-12 text-radar-bright mx-auto animate-bounce" />
            <h4 className="text-sm font-bold text-command-text">
              GPS Route Trace Available
            </h4>
            <p className="text-xs font-mono text-command-muted">
              {session?.route_telemetry?.length} synchronized GPS coordinate points loaded from dashcam NMEA stream.
            </p>
          </div>
        ) : (
          <div className="space-y-4 max-w-md p-6 rounded-xl bg-command-surface/50 border border-command-border">
            <div className="w-12 h-12 rounded-full bg-command-elevated border border-command-border flex items-center justify-center mx-auto text-command-muted">
              <Navigation className="w-6 h-6" />
            </div>
            <h4 className="text-sm font-bold text-command-text">
              No Sensor GPS Telemetry in Upload
            </h4>
            <div className="p-3 rounded bg-command-elevated/70 border border-command-border text-left space-y-1.5 text-[11px] font-mono text-command-muted">
              <div className="flex items-center space-x-1.5 text-accent-amber font-semibold">
                <Info className="w-3.5 h-3.5 shrink-0" />
                <span>Zero-Fabrication Policy</span>
              </div>
              <p>
                RoadSense India never fabricates or simulates artificial GPS coordinates. Geolocation pins activate strictly when verified GPS NMEA metadata accompanies the dashcam ingest.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
