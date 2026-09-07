'use client';

import React, { useState } from 'react';
import { ParkingValidationEvidenceReport } from '../lib/types';
import { runSyntheticValidation } from '../lib/parkingApi';
import {
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Download,
  FileCheck,
  FileJson,
  Fingerprint,
  Layers,
  Loader2,
  Play,
  ShieldCheck,
  X,
  XCircle,
  Video,
  Activity,
} from 'lucide-react';

interface SyntheticValidationModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SyntheticValidationModal({ isOpen, onClose }: SyntheticValidationModalProps) {
  const [report, setReport] = useState<ParkingValidationEvidenceReport | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleRunValidation = async () => {
    setIsRunning(true);
    setErrorMessage(null);
    try {
      const res = await runSyntheticValidation();
      setReport(res);
    } catch (err: any) {
      console.error('Synthetic validation failed:', err);
      setErrorMessage(err.message || 'Validation suite failed to complete.');
    } finally {
      setIsRunning(false);
    }
  };

  const downloadReportJson = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `validation_evidence_${report.run_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-command-bg border border-command-border rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-command-border flex items-center justify-between bg-command-surface/50">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-bold text-sm text-command-text">
                Stationary Camera Synthetic Validation Harness
              </h3>
              <p className="text-[11px] font-mono text-command-muted">
                Phase 2C Deterministic End-to-End Operator Workflow
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-command-muted hover:text-white hover:bg-command-elevated transition-all"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-5 overflow-y-auto space-y-4 flex-1">
          {/* Top Synthetic Warning Notice */}
          <div className="p-3 rounded-xl bg-amber-950/40 border border-amber-500/40 text-amber-300 text-xs font-mono flex items-start space-x-2.5">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-bold uppercase tracking-wide">
                SYNTHETIC TEST EVIDENCE — NOT REAL-WORLD PERFORMANCE EVIDENCE
              </span>
              <p className="text-[11px] text-amber-200/80 leading-relaxed">
                This automated validation harness uses a locally generated 100% deterministic zero-motion scene
                fixture and injectable detector test doubles. Real-world validation remains gated until field stationary footage is supplied.
              </p>
            </div>
          </div>

          {/* Action Card */}
          <div className="p-4 rounded-xl glass-panel border border-command-border flex items-center justify-between">
            <div>
              <h4 className="font-bold text-xs text-command-text">Execute Automated Validation Suite</h4>
              <p className="text-[11px] font-mono text-command-muted mt-0.5">
                Generates stationary fixture → assesses stability → evaluates gate → executes occupancy → verifies artifacts.
              </p>
            </div>
            <button
              onClick={handleRunValidation}
              disabled={isRunning}
              className={`py-2 px-4 rounded-lg font-bold text-xs font-mono flex items-center space-x-2 transition-all shadow-radar ${
                isRunning
                  ? 'bg-command-elevated text-command-muted border border-command-border cursor-not-allowed'
                  : 'bg-radar-bright hover:bg-radar-green text-command-bg'
              }`}
            >
              {isRunning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Executing Pipeline...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  <span>Run Validation Suite</span>
                </>
              )}
            </button>
          </div>

          {errorMessage && (
            <div className="p-3 rounded-xl bg-rose-950/60 border border-rose-500/40 text-rose-300 text-xs font-mono flex items-center space-x-2">
              <XCircle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{errorMessage}</span>
            </div>
          )}

          {/* Report Display */}
          {report && (
            <div className="space-y-4">
              {/* Outcome Banner */}
              <div
                className={`p-4 rounded-xl border flex items-center justify-between ${
                  report.passed
                    ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
                    : 'bg-rose-950/40 border-rose-500/40 text-rose-300'
                }`}
              >
                <div className="flex items-center space-x-3">
                  {report.passed ? (
                    <ShieldCheck className="w-7 h-7 text-emerald-400" />
                  ) : (
                    <XCircle className="w-7 h-7 text-rose-400" />
                  )}
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-bold text-sm uppercase">
                        {report.passed ? 'All Validation Checks Passed' : 'Validation Failures Detected'}
                      </span>
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                          report.passed ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'
                        }`}
                      >
                        Gate: {report.operational_gate}
                      </span>
                    </div>
                    <p className="text-[11px] font-mono text-command-muted mt-0.5">
                      Run ID: {report.run_id} · Git SHA: {(report.git_commit_sha || report.git_sha || 'N/A').slice(0, 8)} · Duration: {report.duration_seconds}s
                    </p>
                  </div>
                </div>

                <button
                  onClick={downloadReportJson}
                  className="py-1.5 px-3 rounded-lg bg-command-elevated hover:bg-command-elevated/80 border border-command-border text-cyan-300 text-xs font-mono font-bold flex items-center space-x-1.5 shadow-sm"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download Evidence JSON</span>
                </button>
              </div>

              {/* KPI Summary */}
              <div className="grid grid-cols-4 gap-2 text-center font-mono">
                <div className="p-3 rounded-xl bg-command-surface border border-command-border">
                  <span className="text-[10px] text-command-muted uppercase block">Total Bays</span>
                  <span className="text-base font-bold text-white">{report.total_bays}</span>
                </div>
                <div className="p-3 rounded-xl bg-rose-950/30 border border-rose-500/30">
                  <span className="text-[10px] text-rose-400 uppercase block">Occupied</span>
                  <span className="text-base font-bold text-rose-300">{report.final_occupied_count}</span>
                </div>
                <div className="p-3 rounded-xl bg-emerald-950/30 border border-emerald-500/30">
                  <span className="text-[10px] text-emerald-400 uppercase block">Vacant</span>
                  <span className="text-base font-bold text-emerald-300">{report.final_vacant_count}</span>
                </div>
                <div className="p-3 rounded-xl bg-cyan-950/30 border border-cyan-500/30">
                  <span className="text-[10px] text-cyan-400 uppercase block">Transitions</span>
                  <span className="text-base font-bold text-cyan-300">{report.total_state_transitions}</span>
                </div>
              </div>

              {/* Validation Checkpoints Table */}
              <div className="p-4 rounded-xl glass-panel border border-command-border space-y-2.5">
                <div className="flex items-center justify-between border-b border-command-border pb-2">
                  <span className="font-bold text-xs text-command-text">Automated Assertion Checkpoints ({report.checks.length})</span>
                  <span className="text-[10px] font-mono text-command-muted">Deterministic Double Mode</span>
                </div>

                <div className="space-y-1.5 font-mono text-xs">
                  {report.checks.map((chk, idx) => (
                    <div
                      key={idx}
                      className={`p-2.5 rounded-lg border flex items-start justify-between ${
                        chk.passed
                          ? 'bg-command-surface/50 border-command-border'
                          : 'bg-rose-950/40 border-rose-500/40'
                      }`}
                    >
                      <div className="flex items-start space-x-2">
                        {chk.passed ? (
                          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                        ) : (
                          <XCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                        )}
                        <div>
                          <div className="flex items-center space-x-2">
                            <span className="font-bold text-command-text">{chk.check_name}</span>
                            <span className="px-1.5 py-0.2 rounded text-[9px] bg-command-elevated text-command-muted">
                              {chk.category}
                            </span>
                          </div>
                          {chk.details && (
                            <p className="text-[11px] text-command-muted mt-0.5">{chk.details}</p>
                          )}
                        </div>
                      </div>
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          chk.passed
                            ? 'bg-emerald-500/20 text-emerald-400'
                            : 'bg-rose-500/20 text-rose-400'
                        }`}
                      >
                        {chk.passed ? 'PASSED' : 'FAILED'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Provenance and Cryptographic Fingerprints */}
              <div className="p-4 rounded-xl bg-command-surface/40 border border-command-border space-y-2 font-mono text-[11px] text-command-muted">
                <div className="flex items-center space-x-1.5 text-command-text font-bold uppercase text-xs">
                  <Fingerprint className="w-4 h-4 text-radar-bright" />
                  <span>Validation Run Provenance Hashes</span>
                </div>
                <div className="grid grid-cols-2 gap-2 pt-1">
                  <div className="truncate">
                    <span className="text-command-muted">Input Video: </span>
                    <span className="text-white">{report.input_video_sha256.slice(0, 16)}...</span>
                  </div>
                  <div className="truncate">
                    <span className="text-command-muted">Reference Image: </span>
                    <span className="text-white">{report.reference_image_sha256.slice(0, 16)}...</span>
                  </div>
                  <div className="truncate">
                    <span className="text-command-muted">Verified Layout: </span>
                    <span className="text-radar-bright">{report.layout_canonical_sha256.slice(0, 16)}...</span>
                  </div>
                  <div className="truncate">
                    <span className="text-command-muted">Annotated Video: </span>
                    <span className="text-cyan-300">{report.output_video_sha256 ? `${report.output_video_sha256.slice(0, 16)}...` : 'N/A'}</span>
                  </div>
                  <div className="truncate">
                    <span className="text-command-muted">Timeline JSONL: </span>
                    <span className="text-amber-300">{report.timeline_sha256 ? `${report.timeline_sha256.slice(0, 16)}...` : 'N/A'}</span>
                  </div>
                  <div className="truncate">
                    <span className="text-command-muted">Summary JSON: </span>
                    <span className="text-emerald-300">{report.summary_sha256 ? `${report.summary_sha256.slice(0, 16)}...` : 'N/A'}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-command-border bg-command-surface/50 flex items-center justify-end">
          <button
            onClick={onClose}
            className="py-1.5 px-4 rounded-lg bg-command-elevated hover:bg-command-elevated/80 border border-command-border text-command-text text-xs font-mono font-bold transition-all"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
