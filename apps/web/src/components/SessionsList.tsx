'use client';

import React from 'react';
import { Film, CheckCircle2, Clock, AlertTriangle, Download, ArrowRight } from 'lucide-react';
import { DriveSession } from '../lib/types';
import { getArtifactDownloadUrl } from '../lib/api';

interface SessionsListProps {
  sessions: DriveSession[];
  activeSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
}

export function SessionsList({
  sessions,
  activeSessionId,
  onSelectSession,
}: SessionsListProps) {
  return (
    <div className="flex flex-col h-full rounded-xl glass-panel-elevated border border-command-border overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-command-surface border-b border-command-border flex items-center justify-between text-xs font-mono">
        <div className="flex items-center space-x-2">
          <Film className="w-4 h-4 text-radar-bright" />
          <span className="font-bold text-command-text uppercase tracking-wider">
            Recorded Drive Sessions Ledger
          </span>
        </div>
        <span className="text-[11px] text-command-muted">
          {sessions.length} Session{sessions.length !== 1 ? 's' : ''} Stored Locally
        </span>
      </div>

      {/* Sessions Table / List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {sessions.length === 0 ? (
          <div className="text-center py-16 text-xs font-mono text-command-muted">
            No drive sessions recorded yet. Click &ldquo;New Upload&rdquo; in the top bar to ingest a dashcam video.
          </div>
        ) : (
          sessions.map((sess) => {
            const isSelected = activeSessionId === sess.id;
            const isComplete = sess.processing_state === 'complete';
            const isFailed = sess.processing_state === 'failed';
            const isRunning = !isComplete && !isFailed;

            return (
              <div
                key={sess.id}
                className={`p-4 rounded-lg border transition-all ${
                  isSelected
                    ? 'bg-command-elevated border-radar-bright shadow-radar'
                    : 'bg-command-surface/70 border-command-border hover:border-command-subtle'
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-mono font-bold text-command-text">
                        {sess.source_filename}
                      </span>
                      {isComplete && (
                        <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-radar-dim/30 text-radar-bright border border-radar-green/30 flex items-center space-x-1">
                          <CheckCircle2 className="w-3 h-3" />
                          <span>PROCESSED</span>
                        </span>
                      )}
                      {isRunning && (
                        <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-accent-amber/20 text-accent-amber border border-accent-amber/30 flex items-center space-x-1 animate-pulse">
                          <Clock className="w-3 h-3" />
                          <span className="uppercase">{sess.processing_state}...</span>
                        </span>
                      )}
                      {isFailed && (
                        <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-accent-red/20 text-accent-red border border-accent-red/30 flex items-center space-x-1">
                          <AlertTriangle className="w-3 h-3" />
                          <span>FAILED</span>
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] font-mono text-command-muted space-x-3">
                      <span>ID: {sess.id.slice(0, 8)}</span>
                      <span>•</span>
                      <span>SHA: {sess.source_hash.slice(0, 10)}...</span>
                      <span>•</span>
                      <span>{new Date(sess.started_at).toLocaleString()}</span>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center space-x-2 font-mono text-xs">
                    {isComplete && (
                      <a
                        href={getArtifactDownloadUrl(sess.id, 'report_zip')}
                        download
                        className="p-2 rounded bg-command-elevated hover:bg-command-border text-command-text border border-command-border transition-colors flex items-center space-x-1.5"
                        title="Download Field Dossier (.zip)"
                      >
                        <Download className="w-3.5 h-3.5 text-radar-bright" />
                        <span className="hidden sm:inline">Dossier</span>
                      </a>
                    )}
                    <button
                      onClick={() => onSelectSession(sess.id)}
                      className={`px-3 py-1.5 rounded flex items-center space-x-1 font-bold transition-all ${
                        isSelected
                          ? 'bg-radar-bright text-command-bg shadow-radar'
                          : 'bg-command-elevated hover:bg-command-border text-command-text border border-command-border'
                      }`}
                    >
                      <span>Inspect</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                {/* Processing Metrics Row */}
                {isComplete && (
                  <div className="mt-3 pt-3 border-t border-command-border grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono">
                    <div>
                      <span className="text-command-muted">Duration:</span>{' '}
                      <span className="text-command-text font-bold">
                        {(sess.source_duration_seconds || 0).toFixed(1)}s
                      </span>
                    </div>
                    <div>
                      <span className="text-command-muted">Sampled:</span>{' '}
                      <span className="text-command-text font-bold">
                        {sess.sampled_frames_count || 0} frames
                      </span>
                    </div>
                    <div>
                      <span className="text-command-muted">Pothole Detections:</span>{' '}
                      <span className="text-radar-bright font-bold">
                        {sess.total_detections_count || 0}
                      </span>
                    </div>
                    <div>
                      <span className="text-command-muted">Process Time:</span>{' '}
                      <span className="text-accent-cyan font-bold">
                        {(sess.processing_duration_seconds || 0).toFixed(2)}s
                      </span>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
