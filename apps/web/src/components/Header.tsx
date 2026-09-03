'use client';

import React from 'react';
import {
  ShieldCheck,
  Cpu,
  Radio,
  Upload,
  Activity,
  AlertCircle,
} from 'lucide-react';
import { SystemHealth } from '../lib/types';

interface HeaderProps {
  health: SystemHealth | null;
  onOpenUpload: () => void;
  onOpenHealth: () => void;
}

export function Header({ health, onOpenUpload, onOpenHealth }: HeaderProps) {
  return (
    <header className="h-16 border-b border-command-border bg-command-surface/90 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-40">
      {/* Brand Identity */}
      <div className="flex items-center space-x-3">
        <div className="relative flex items-center justify-center w-9 h-9 rounded-lg bg-radar-dim border border-radar-green/40 shadow-radar">
          <Radio className="w-5 h-5 text-radar-bright animate-pulse" />
          <div className="absolute inset-0 rounded-lg bg-radar-green/20 animate-radar-pulse pointer-events-none" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <span className="font-bold text-lg tracking-wider text-command-text">
              ROADSENSE<span className="text-radar-bright">.INDIA</span>
            </span>
            <span className="px-1.5 py-0.5 text-[10px] font-mono tracking-widest bg-command-elevated border border-command-border rounded text-command-muted uppercase">
              Ops Suite
            </span>
          </div>
          <p className="text-[11px] text-command-muted tracking-tight">
            Local Dashcam Inspection & Verification Engine
          </p>
        </div>
      </div>

      {/* Center Provenance & Hardware Status */}
      <div className="hidden lg:flex items-center space-x-4 text-xs font-mono">
        {/* Model Provenance Verification Badge */}
        <div className="flex items-center space-x-2 bg-command-elevated px-3 py-1.5 rounded-md border border-command-border">
          <ShieldCheck className="w-4 h-4 text-radar-bright" />
          <span className="text-command-muted">Model:</span>
          <span className="text-command-text font-semibold">D40 Pothole YOLOv8n</span>
          <span className="text-radar-bright bg-radar-dim/40 px-1.5 py-0.5 rounded text-[10px] border border-radar-green/30">
            SHA: {health?.model_hash_prefix || 'bdf07ad8'}...
          </span>
        </div>

        {/* Hardware Acceleration */}
        <div className="flex items-center space-x-2 bg-command-elevated px-3 py-1.5 rounded-md border border-command-border">
          <Cpu className="w-4 h-4 text-accent-cyan" />
          <span className="text-command-muted">Compute:</span>
          <span className="text-accent-cyan font-semibold">
            {health?.mps_available ? 'Apple MPS (Metal)' : health?.cuda_available ? 'CUDA GPU' : 'CPU (Local)'}
          </span>
        </div>
      </div>

      {/* Right Action Controls */}
      <div className="flex items-center space-x-3">
        {/* System Health Status Button */}
        <button
          onClick={onOpenHealth}
          className="flex items-center space-x-2 px-3 py-1.5 rounded-md bg-command-elevated hover:bg-command-border text-command-text text-xs border border-command-border transition-colors"
          title="Inspect Local System Health & Provenance"
        >
          <Activity className="w-4 h-4 text-radar-bright" />
          <span className="hidden sm:inline font-mono">
            {health?.status === 'operational' ? 'System Ready' : 'Connecting...'}
          </span>
          <div
            className={`w-2 h-2 rounded-full ${
              health?.database_connected ? 'bg-radar-bright animate-ping' : 'bg-accent-amber'
            }`}
          />
        </button>

        {/* Upload Action Button */}
        <button
          onClick={onOpenUpload}
          className="flex items-center space-x-2 px-4 py-1.5 rounded-md bg-radar-bright hover:bg-radar-green text-command-bg font-semibold text-xs transition-all shadow-radar hover:shadow-lg"
        >
          <Upload className="w-4 h-4" />
          <span>New Upload</span>
        </button>
      </div>
    </header>
  );
}
