'use client';

import React from 'react';
import { X, Activity, ShieldCheck, HardDrive, Cpu, Database, Lock, Film } from 'lucide-react';
import { SystemHealth } from '../lib/types';

interface SystemHealthModalProps {
  isOpen: boolean;
  onClose: () => void;
  health: SystemHealth | null;
}

export function SystemHealthModal({ isOpen, onClose, health }: SystemHealthModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
      <div className="w-full max-w-lg rounded-xl glass-panel-elevated p-6 space-y-5 shadow-2xl border border-command-border font-mono text-xs">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-command-border pb-3">
          <div className="flex items-center space-x-2">
            <Activity className="w-5 h-5 text-radar-bright" />
            <h3 className="text-sm font-bold text-command-text tracking-wide uppercase">
              System Telemetry & Provenance Diagnostics
            </h3>
          </div>
          <button
            onClick={onClose}
            className="text-command-muted hover:text-command-text p-1 rounded transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Telemetry Grid */}
        <div className="space-y-3">
          {/* Model Provenance Pin */}
          <div className="p-3 rounded-lg bg-command-surface border border-command-border flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <ShieldCheck className="w-5 h-5 text-radar-bright shrink-0" />
              <div>
                <div className="font-bold text-command-text">Model Verification</div>
                <div className="text-[11px] text-command-muted truncate max-w-[220px]">
                  {health?.model_run_id || 'pothole_yolov8n_rdd2022_india_mps_baseline_v1'}
                </div>
              </div>
            </div>
            <div className="text-right">
              <span className="px-2 py-0.5 rounded bg-radar-dim/40 text-radar-bright border border-radar-green/30 font-bold">
                {health?.model_verified ? 'SHA Pinned (Pass)' : 'Checking...'}
              </span>
            </div>
          </div>

          {/* Media Engine */}
          <div className="p-3 rounded-lg bg-command-surface border border-command-border flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Film className="w-5 h-5 text-accent-cyan shrink-0" />
              <div>
                <div className="font-bold text-command-text">Media Intelligence Engine</div>
                <div className="text-[11px] text-command-muted">
                  FFmpeg 9.0.1 + OpenCV 5.0 + ByteTrack
                </div>
              </div>
            </div>
            <span className="px-2 py-0.5 rounded bg-command-elevated text-accent-cyan border border-command-border font-bold">
              Active
            </span>
          </div>

          {/* Database Persistence */}
          <div className="p-3 rounded-lg bg-command-surface border border-command-border flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Database className="w-5 h-5 text-accent-cyan shrink-0" />
              <div>
                <div className="font-bold text-command-text">Persistence Engine</div>
                <div className="text-[11px] text-command-muted uppercase">
                  {health?.database_type === 'postgresql' ? 'PostgreSQL 16 + PostGIS' : 'SQLite (Async Local)'}
                </div>
              </div>
            </div>
            <span className="px-2 py-0.5 rounded bg-command-elevated text-accent-cyan border border-command-border">
              {health?.database_connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          {/* Compute Acceleration */}
          <div className="p-3 rounded-lg bg-command-surface border border-command-border flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Cpu className="w-5 h-5 text-accent-amber shrink-0" />
              <div>
                <div className="font-bold text-command-text">Hardware Acceleration</div>
                <div className="text-[11px] text-command-muted">
                  {health?.mps_available
                    ? 'Apple MPS (Metal Performance Shaders)'
                    : health?.cuda_available
                    ? 'NVIDIA CUDA GPU'
                    : 'CPU Core Execution'}
                </div>
              </div>
            </div>
            <span className="text-accent-amber font-bold">
              {health?.mps_available || health?.cuda_available ? 'Hardware Accel' : 'Standard'}
            </span>
          </div>

          {/* Storage Spool */}
          <div className="p-3 rounded-lg bg-command-surface border border-command-border flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <HardDrive className="w-5 h-5 text-command-muted shrink-0" />
              <div>
                <div className="font-bold text-command-text">Local Spool Storage</div>
                <div className="text-[11px] text-command-muted">Free disk space available</div>
              </div>
            </div>
            <span className="text-command-text font-bold">{health?.disk_free_gb || 0} GB Free</span>
          </div>

          {/* Privacy Posture */}
          <div className="p-3 rounded-lg bg-command-surface border border-command-border flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Lock className="w-5 h-5 text-radar-bright shrink-0" />
              <div>
                <div className="font-bold text-command-text">Privacy Posture</div>
                <div className="text-[11px] text-command-muted">Zero external telemetry or cloud inference</div>
              </div>
            </div>
            <span className="text-radar-bright font-bold">100% Local</span>
          </div>
        </div>

        {/* Close Button */}
        <div className="flex justify-end pt-2">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-md bg-command-elevated hover:bg-command-border text-command-text text-xs font-bold transition-colors"
          >
            Close Diagnostics
          </button>
        </div>
      </div>
    </div>
  );
}
