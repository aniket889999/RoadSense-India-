'use client';

import React from 'react';
import { Activity, ArrowUpRight, Cpu, ShieldCheck, Upload } from 'lucide-react';
import { SystemHealth } from '../lib/types';

interface HeaderProps {
  health: SystemHealth | null;
  onOpenUpload: () => void;
  onOpenHealth: () => void;
  context?: 'operations' | 'anpr';
}

export function Header({ health, onOpenUpload, onOpenHealth, context = 'operations' }: HeaderProps) {
  const ready = health?.status === 'operational';
  return (
    <header className="h-20 px-6 lg:px-8 flex items-center justify-between sticky top-0 z-40 bg-command-bg/85 backdrop-blur-md border-b border-command-border">
      {/* Wordmark */}
      <div className="flex items-baseline gap-3">
        <span className="text-[26px] font-semibold tracking-[-0.03em] text-command-text leading-none">
          RoadSense
        </span>
        <span className="hidden sm:inline text-sm text-command-muted leading-none">
          {context === 'anpr' ? 'Plate recognition, reviewed' : 'Road inspection, reviewed'}
        </span>
      </div>

      {/* Context strip */}
      <div className="hidden lg:flex items-center gap-2 text-[13px]">
        {context === 'anpr' ? (
          ['Detect', 'Read', 'Check', 'Review'].map((step, index) => (
            <span key={step} className="flex items-center gap-2">
              <span className="px-3.5 py-1.5 rounded-full bg-command-elevated text-command-text border border-command-border">
                {step}
              </span>
              {index < 3 && <span className="text-command-subtle">→</span>}
            </span>
          ))
        ) : (
          <>
            <span className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-command-elevated border border-command-border">
              <ShieldCheck className="w-4 h-4 text-radar-bright" />
              <span className="text-command-muted">Model</span>
              <span className="text-command-text">D40 Pothole YOLOv8n</span>
              <span className="font-mono text-[11px] text-command-muted">{health?.model_hash_prefix || 'bdf07ad8'}…</span>
            </span>
            <span className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-command-elevated border border-command-border">
              <Cpu className="w-4 h-4 text-accent-cyan" />
              <span className="text-command-text">
                {health?.mps_available ? 'Apple MPS' : health?.cuda_available ? 'CUDA GPU' : 'CPU'}
              </span>
            </span>
          </>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenHealth}
          className="flex items-center gap-2 pl-3 pr-4 py-2 rounded-full bg-command-elevated hover:bg-command-border text-command-text text-[13px] border border-command-border transition-colors"
          title={context === 'anpr' ? 'General operations health. ANPR availability is checked when an image is recognized.' : 'Inspect local system health and provenance'}
        >
          <span className={`w-2 h-2 rounded-full ${health?.database_connected ? 'bg-radar-bright' : 'bg-accent-amber'}`} />
          <Activity className="w-4 h-4 text-command-muted" />
          <span className="hidden sm:inline">
            {context === 'anpr' ? 'Health' : ready ? 'System ready' : 'Connecting…'}
          </span>
        </button>

        {context !== 'anpr' ? (
          <button
            onClick={onOpenUpload}
            className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-brand hover:bg-brand-bright text-white font-medium text-[13px] shadow-brand transition-colors"
          >
            <Upload className="w-4 h-4" />
            <span>New upload</span>
          </button>
        ) : (
          <a
            href="#anpr-workspace"
            className="hidden md:flex items-center gap-1.5 px-5 py-2.5 rounded-full bg-command-text text-white font-medium text-[13px] hover:bg-slate-700 transition-colors"
          >
            <span>Inspect an image</span>
            <ArrowUpRight className="w-4 h-4" />
          </a>
        )}
      </div>
    </header>
  );
}
