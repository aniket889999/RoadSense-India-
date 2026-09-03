'use client';

import React from 'react';
import { Camera, AlertTriangle, Play, Square, Radio, ShieldAlert } from 'lucide-react';
import { useCamera } from '../hooks/useCamera';

export function LiveCameraView() {
  const { isActive, error, videoRef, startCamera, stopCamera } = useCamera();

  return (
    <div className="flex flex-col h-full rounded-xl glass-panel-elevated border border-command-border overflow-hidden">
      {/* Live Header */}
      <div className="px-4 py-3 bg-command-surface border-b border-command-border flex items-center justify-between text-xs font-mono">
        <div className="flex items-center space-x-2">
          <Camera className="w-4 h-4 text-radar-bright" />
          <span className="font-bold text-command-text uppercase tracking-wider">
            Live Connected Dashcam / Camera Ingest
          </span>
        </div>
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1.5">
            <span
              className={`w-2 h-2 rounded-full ${
                isActive ? 'bg-radar-bright animate-ping' : 'bg-command-muted'
              }`}
            />
            <span className="text-[11px] text-command-muted">
              {isActive ? 'Local Stream Active' : 'Camera Disconnected'}
            </span>
          </div>
          {isActive ? (
            <button
              onClick={stopCamera}
              className="px-3 py-1 rounded bg-accent-red/20 hover:bg-accent-red text-white text-[11px] font-bold border border-accent-red/30 flex items-center space-x-1 transition-colors"
            >
              <Square className="w-3 h-3" />
              <span>Stop Camera</span>
            </button>
          ) : (
            <button
              onClick={startCamera}
              className="px-3 py-1 rounded bg-radar-bright hover:bg-radar-green text-command-bg text-[11px] font-bold flex items-center space-x-1 transition-all shadow-radar"
            >
              <Play className="w-3 h-3" />
              <span>Start Camera</span>
            </button>
          )}
        </div>
      </div>

      {/* Main Viewport */}
      <div className="relative flex-1 bg-black flex items-center justify-center overflow-hidden min-h-[400px]">
        {isActive ? (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-contain max-h-[560px]"
            />

            {/* Radar scan line overlay */}
            <div className="absolute inset-0 pointer-events-none overflow-hidden">
              <div className="w-full h-0.5 bg-gradient-to-r from-transparent via-radar-bright to-transparent opacity-60 animate-scan-line" />
            </div>
          </>
        ) : (
          <div className="text-center p-8 space-y-4 max-w-md">
            <div className="w-14 h-14 rounded-full bg-command-elevated border border-command-border flex items-center justify-center mx-auto text-command-muted">
              <Radio className="w-7 h-7" />
            </div>
            <h4 className="text-sm font-bold text-command-text">
              Local Camera Stream Inactive
            </h4>
            <p className="text-xs font-mono text-command-muted">
              Click &ldquo;Start Camera&rdquo; to connect your local USB dashcam or laptop camera. Direct loopback only; no video leaves this machine.
            </p>
          </div>
        )}

        {/* Continuous Safety Warning Banner */}
        <div className="absolute top-3 left-3 bg-black/85 backdrop-blur-md px-3.5 py-2 rounded-lg border border-accent-amber/40 text-[11px] font-mono text-accent-amber flex items-center space-x-2 shadow-lg">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>Experimental operator review — not a driver alert or ADAS warning</span>
        </div>

        {/* Inference Connection Status Pill */}
        <div className="absolute bottom-3 right-3 bg-black/85 backdrop-blur-md px-3 py-1.5 rounded border border-command-border text-[10px] font-mono text-command-muted flex items-center space-x-2">
          <span className="w-1.5 h-1.5 rounded-full bg-accent-amber" />
          <span>Server WebRTC Inference: Offline (Boundary Active)</span>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="p-3 bg-accent-red/10 border-t border-accent-red/30 text-accent-red text-xs font-mono flex items-center space-x-2">
          <ShieldAlert className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
