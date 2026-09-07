'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  CameraOperationalGate,
  ParkingOccupancyJob,
  BayOccupancyState,
  BaySummaryItem,
  SpaceType,
} from '../lib/types';
import {
  getCameraOperationalGate,
  listOccupancyJobs,
  getOccupancyJob,
  submitOccupancyJob,
  cancelOccupancyJob,
  getOccupancyVideoUrl,
  getOccupancyManifestUrl,
  getOccupancyTimelineUrl,
} from '../lib/parkingApi';
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Download,
  FileCheck,
  FileJson,
  FileSpreadsheet,
  FileText,
  Fingerprint,
  History,
  Layers,
  Loader2,
  Maximize2,
  Play,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  UploadCloud,
  Video,
  XCircle,
  Ban,
  Activity,
  Car,
  Filter,
  Eye,
  Info,
} from 'lucide-react';

interface ParkingOccupancyPanelProps {
  cameraId: string | null;
  cameraName?: string;
  activeLayoutId?: string | null;
}

export function ParkingOccupancyPanel({
  cameraId,
  cameraName,
  activeLayoutId,
}: ParkingOccupancyPanelProps) {
  const [gate, setGate] = useState<CameraOperationalGate | null>(null);
  const [jobs, setJobs] = useState<ParkingOccupancyJob[]>([]);
  const [selectedJob, setSelectedJob] = useState<ParkingOccupancyJob | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);

  // Video Upload State
  const [selectedVideoFile, setSelectedVideoFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Filter and Inspection State
  const [bayFilter, setBayFilter] = useState<'ALL' | 'OCCUPIED' | 'VACANT' | 'UNKNOWN'>('ALL');
  const [inspectedBayLabel, setInspectedBayLabel] = useState<string | null>(null);

  // Load Gate and Jobs
  const loadData = useCallback(async () => {
    if (!cameraId) {
      setGate(null);
      setJobs([]);
      setSelectedJob(null);
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);
    try {
      const [gateRes, jobsRes] = await Promise.all([
        getCameraOperationalGate(cameraId).catch(() => null),
        listOccupancyJobs(cameraId).catch(() => []),
      ]);
      setGate(gateRes);
      setJobs(jobsRes);
      if (jobsRes.length > 0) {
        // Keep currently selected job if exists, else first
        setSelectedJob((prev) => {
          if (!prev) return jobsRes[0];
          const updated = jobsRes.find((j) => j.id === prev.id);
          return updated || jobsRes[0];
        });
      }
    } catch (err: any) {
      console.error('Failed to load occupancy data:', err);
      setErrorMessage(err.message || 'Failed to load occupancy data');
    } finally {
      setIsLoading(false);
    }
  }, [cameraId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Polling for in-progress jobs
  useEffect(() => {
    if (!selectedJob) return;
    const inProgressStatuses = [
      'QUEUED',
      'VALIDATING',
      'DETECTING',
      'TRACKING',
      'CLASSIFYING_OCCUPANCY',
      'RENDERING',
      'ENCODING',
    ];

    if (inProgressStatuses.includes(selectedJob.status)) {
      const interval = setInterval(async () => {
        try {
          const fresh = await getOccupancyJob(selectedJob.id);
          setSelectedJob(fresh);
          setJobs((prev) => prev.map((j) => (j.id === fresh.id ? fresh : j)));
        } catch (err) {
          console.error('Error polling occupancy job:', err);
        }
      }, 1500);
      return () => clearInterval(interval);
    }
  }, [selectedJob]);

  // Handle Video Selection
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedVideoFile(e.target.files[0]);
      setErrorMessage(null);
    }
  };

  // Handle Submit Job
  const handleSubmitJob = async () => {
    if (!cameraId || !selectedVideoFile) return;

    setIsSubmitting(true);
    setErrorMessage(null);
    try {
      const job = await submitOccupancyJob(cameraId, selectedVideoFile);
      setJobs((prev) => [job, ...prev]);
      setSelectedJob(job);
      setSelectedVideoFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err: any) {
      console.error('Failed to submit occupancy job:', err);
      setErrorMessage(err.message || 'Occupancy job submission rejected by system.');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle Cancel Job
  const handleCancelJob = async () => {
    if (!selectedJob) return;
    setIsCancelling(true);
    try {
      const res = await cancelOccupancyJob(selectedJob.id);
      if (res.cancelled) {
        const fresh = await getOccupancyJob(selectedJob.id);
        setSelectedJob(fresh);
        setJobs((prev) => prev.map((j) => (j.id === fresh.id ? fresh : j)));
      }
    } catch (err: any) {
      console.error('Failed to cancel occupancy job:', err);
      setErrorMessage(err.message || 'Failed to cancel job');
    } finally {
      setIsCancelling(false);
    }
  };

  const isGateAllowed = gate?.operational_gate === 'ALLOWED';
  const isGateBlocked = gate?.operational_gate === 'BLOCKED' || !gate;

  const baySummaryEntries: [string, BaySummaryItem][] = selectedJob?.bay_summary
    ? Object.entries(selectedJob.bay_summary)
    : [];

  const filteredBays = baySummaryEntries.filter(([_, item]) => {
    if (bayFilter === 'ALL') return true;
    return item.final_state === bayFilter;
  });

  const inspectedBay = inspectedBayLabel && selectedJob?.bay_summary
    ? selectedJob.bay_summary[inspectedBayLabel]
    : null;

  return (
    <div className="space-y-4">
      {/* 1. Stability Gate Status Banner */}
      <div
        id="parking-occupancy-gate-banner"
        className={`p-3.5 rounded-xl border transition-all ${
          isGateAllowed
            ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
            : 'bg-rose-950/40 border-rose-500/40 text-rose-300'
        }`}
      >
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-2.5">
            {isGateAllowed ? (
              <ShieldCheck className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
            ) : (
              <ShieldAlert className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            )}
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-xs uppercase tracking-wider">
                  Operational Gate: {isGateAllowed ? 'ALLOWED' : 'BLOCKED'}
                </span>
                <span
                  className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold ${
                    isGateAllowed
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                  }`}
                >
                  {gate?.aggregate_decision || (isGateAllowed ? 'STABLE' : 'UNSTABLE / UNVERIFIED')}
                </span>
              </div>
              <p className="text-[11px] text-command-muted mt-1 leading-snug">
                {isGateAllowed
                  ? 'Active layout and camera calibration verified stable. Occupancy video pipeline is enabled.'
                  : 'Fail-Closed Gate active: Occupancy inference is prohibited until a fresh STABLE camera assessment is confirmed.'}
              </p>
              {gate?.gate_reasons && gate.gate_reasons.length > 0 && (
                <div className="mt-2 space-y-1">
                  <span className="text-[10px] font-mono font-bold text-rose-400 uppercase">
                    Gate Enforcement Reasons:
                  </span>
                  <ul className="list-disc list-inside text-[10px] font-mono text-rose-200/80 space-y-0.5">
                    {gate.gate_reasons.map((r, idx) => (
                      <li key={idx}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>

          <button
            onClick={loadData}
            disabled={isLoading}
            className="p-1.5 rounded-lg bg-command-elevated border border-command-border text-command-muted hover:text-white transition-all text-xs"
            title="Refresh Gate Status"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* 2. Video Upload and Job Submission */}
      <div className="p-3.5 rounded-xl glass-panel border border-command-border space-y-3">
        <div className="flex items-center justify-between border-b border-command-border pb-2">
          <div className="flex items-center space-x-2">
            <Video className="w-4 h-4 text-radar-bright" />
            <span className="font-bold text-xs text-command-text">
              Run Parking Occupancy Pipeline
            </span>
          </div>
          <span className="text-[10px] font-mono text-command-muted">Phase 2B Pipeline</span>
        </div>

        <div className="space-y-2.5">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            accept="video/mp4,video/quicktime,video/x-msvideo"
            className="hidden"
            id="occupancy-video-upload-input"
          />

          <div
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-all ${
              selectedVideoFile
                ? 'border-radar-green/60 bg-radar-green/5 text-command-text'
                : 'border-command-border hover:border-radar-green/40 hover:bg-command-elevated/30 text-command-muted'
            }`}
          >
            <UploadCloud className="w-7 h-7 mx-auto mb-1.5 text-radar-bright opacity-80" />
            {selectedVideoFile ? (
              <div>
                <p className="text-xs font-bold text-radar-bright truncate max-w-xs mx-auto">
                  {selectedVideoFile.name}
                </p>
                <p className="text-[10px] font-mono text-command-muted">
                  {(selectedVideoFile.size / (1024 * 1024)).toFixed(2)} MB · Ready for processing
                </p>
              </div>
            ) : (
              <div>
                <p className="text-xs font-semibold">Select parking observation video</p>
                <p className="text-[10px] font-mono text-command-muted">MP4, MOV, or AVI</p>
              </div>
            )}
          </div>

          {errorMessage && (
            <div
              id="occupancy-error-banner"
              className="p-2.5 rounded-lg bg-rose-950/60 border border-rose-500/40 text-rose-300 text-[11px] font-mono flex items-start space-x-2"
            >
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <span className="font-bold">Execution Blocked / Error:</span>
                <p className="leading-snug">{errorMessage}</p>
              </div>
            </div>
          )}

          <div className="flex items-center space-x-2">
            <button
              id="start-occupancy-pipeline-btn"
              onClick={handleSubmitJob}
              disabled={isSubmitting || !selectedVideoFile || !cameraId}
              className={`flex-1 py-2 rounded-lg font-bold text-xs font-mono flex items-center justify-center space-x-2 transition-all shadow-radar ${
                selectedVideoFile && cameraId && !isSubmitting
                  ? 'bg-radar-bright hover:bg-radar-green text-command-bg'
                  : 'bg-command-elevated text-command-muted border border-command-border cursor-not-allowed'
              }`}
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Submitting Video...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5" />
                  <span>Process Gated Occupancy</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* 3. Active Job Execution & Progress */}
      {selectedJob && (
        <div className="p-3.5 rounded-xl glass-panel border border-command-border space-y-3">
          <div className="flex items-center justify-between border-b border-command-border pb-2">
            <div className="flex items-center space-x-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span className="font-bold text-xs text-command-text">
                Job Details: {selectedJob.id.slice(0, 8)}...
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <span
                className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase ${
                  selectedJob.status === 'COMPLETE'
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    : selectedJob.status === 'BLOCKED_BY_STABILITY_GATE' || selectedJob.status === 'FAILED'
                    ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                    : selectedJob.status === 'CANCELLED'
                    ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                    : 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 animate-pulse'
                }`}
              >
                {selectedJob.status.replace(/_/g, ' ')}
              </span>
            </div>
          </div>

          {/* Progress Bar for Active Jobs */}
          {selectedJob.status !== 'COMPLETE' &&
            selectedJob.status !== 'FAILED' &&
            selectedJob.status !== 'CANCELLED' &&
            selectedJob.status !== 'BLOCKED_BY_STABILITY_GATE' && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-[11px] font-mono">
                  <div className="flex items-center space-x-1.5 text-cyan-300">
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>{selectedJob.stage_message || selectedJob.status}</span>
                  </div>
                  <span className="font-bold text-command-text">
                    {selectedJob.progress_pct.toFixed(0)}%
                  </span>
                </div>
                <div className="w-full bg-command-surface h-2 rounded-full overflow-hidden border border-command-border">
                  <div
                    className="bg-cyan-400 h-full transition-all duration-300"
                    style={{ width: `${Math.max(5, selectedJob.progress_pct)}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-[10px] font-mono text-command-muted">
                  <span>
                    Frames: {selectedJob.processed_frames} / {selectedJob.total_frames || '...'}
                  </span>
                  <button
                    onClick={handleCancelJob}
                    disabled={isCancelling}
                    className="text-rose-400 hover:text-rose-300 underline font-bold"
                  >
                    {isCancelling ? 'Cancelling...' : 'Cancel Job'}
                  </button>
                </div>
              </div>
            )}

          {/* Blocked by gate message */}
          {selectedJob.status === 'BLOCKED_BY_STABILITY_GATE' && (
            <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-500/30 text-rose-300 text-xs space-y-2">
              <div className="flex items-center space-x-2 font-bold text-rose-400">
                <Ban className="w-4 h-4" />
                <span>Job Terminated by Stability Gate</span>
              </div>
              <p className="text-[11px] leading-relaxed">
                {selectedJob.failure_message ||
                  'The uploaded video failed stability verification. Recalibration or a fixed-camera recording is required before occupancy can be computed.'}
              </p>
              {selectedJob.gate_reasons && selectedJob.gate_reasons.length > 0 && (
                <ul className="list-disc list-inside text-[10px] font-mono text-rose-200/80">
                  {selectedJob.gate_reasons.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {/* Completed Job KPIs and Actions */}
          {selectedJob.status === 'COMPLETE' && (
            <div className="space-y-3.5">
              {/* AI Notice Banner */}
              <div className="p-2.5 rounded-lg bg-amber-950/40 border border-amber-500/40 text-amber-300 text-[11px] font-mono flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                <span className="font-bold">
                  AI ESTIMATE — NOT HUMAN VERIFIED. Work orders require field inspection.
                </span>
              </div>

              {/* KPI Counters Banner */}
              <div className="grid grid-cols-4 gap-2 text-center font-mono">
                <div className="p-2.5 rounded-lg bg-command-elevated border border-command-border">
                  <span className="text-[10px] text-command-muted uppercase block">Total Bays</span>
                  <span className="text-base font-bold text-white">{selectedJob.total_bays}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-rose-950/40 border border-rose-500/40">
                  <span className="text-[10px] text-rose-400 uppercase block">Occupied</span>
                  <span className="text-base font-bold text-rose-300">
                    {selectedJob.final_occupied_count}
                  </span>
                </div>
                <div className="p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-500/40">
                  <span className="text-[10px] text-emerald-400 uppercase block">Vacant</span>
                  <span className="text-base font-bold text-emerald-300">
                    {selectedJob.final_vacant_count}
                  </span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-700/60">
                  <span className="text-[10px] text-slate-400 uppercase block">Unknown</span>
                  <span className="text-base font-bold text-slate-300">
                    {selectedJob.final_unknown_count}
                  </span>
                </div>
              </div>

              {/* Video Player */}
              {selectedJob.has_annotated_video && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-command-text flex items-center space-x-1.5">
                      <Play className="w-3.5 h-3.5 text-radar-bright" />
                      <span>Annotated Pipeline Playback</span>
                    </span>
                    <span className="text-[10px] font-mono text-command-muted">
                      {selectedJob.total_frames} frames @ {selectedJob.fps.toFixed(1)} fps
                    </span>
                  </div>
                  <div className="rounded-lg overflow-hidden border border-command-border bg-black aspect-video flex items-center justify-center">
                    <video
                      id="annotated-occupancy-video-player"
                      controls
                      playsInline
                      className="w-full h-full object-contain"
                      src={getOccupancyVideoUrl(selectedJob.id)}
                    >
                      Your browser does not support HTML5 video streaming.
                    </video>
                  </div>
                </div>
              )}

              {/* Download Action Buttons */}
              <div className="flex flex-wrap gap-2 pt-1">
                {selectedJob.has_annotated_video && (
                  <a
                    href={getOccupancyVideoUrl(selectedJob.id)}
                    download={`annotated_${selectedJob.id}.mp4`}
                    className="flex-1 py-1.5 px-3 rounded-lg bg-command-elevated hover:bg-command-elevated/80 border border-command-border text-radar-bright text-xs font-mono font-bold flex items-center justify-center space-x-1.5 transition-all shadow-sm"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Annotated MP4</span>
                  </a>
                )}
                {selectedJob.has_manifest && (
                  <a
                    href={getOccupancyManifestUrl(selectedJob.id)}
                    download={`manifest_${selectedJob.id}.json`}
                    className="flex-1 py-1.5 px-3 rounded-lg bg-command-elevated hover:bg-command-elevated/80 border border-command-border text-cyan-300 text-xs font-mono font-bold flex items-center justify-center space-x-1.5 transition-all shadow-sm"
                  >
                    <FileJson className="w-3.5 h-3.5" />
                    <span>Manifest JSON</span>
                  </a>
                )}
                {selectedJob.has_timeline && (
                  <a
                    href={getOccupancyTimelineUrl(selectedJob.id)}
                    download={`timeline_${selectedJob.id}.jsonl`}
                    className="flex-1 py-1.5 px-3 rounded-lg bg-command-elevated hover:bg-command-elevated/80 border border-command-border text-amber-300 text-xs font-mono font-bold flex items-center justify-center space-x-1.5 transition-all shadow-sm"
                  >
                    <FileSpreadsheet className="w-3.5 h-3.5" />
                    <span>Timeline JSONL</span>
                  </a>
                )}
              </div>

              {/* Filterable Bay List & Evidence */}
              <div className="space-y-2 border-t border-command-border pt-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-1.5">
                    <Filter className="w-3.5 h-3.5 text-command-muted" />
                    <span className="font-bold text-xs text-command-text">Bay Occupancy Ledger</span>
                  </div>
                  {/* Bay Filter Tabs */}
                  <div className="flex bg-command-surface rounded p-0.5 text-[10px] font-mono border border-command-border">
                    {(['ALL', 'OCCUPIED', 'VACANT', 'UNKNOWN'] as const).map((mode) => (
                      <button
                        key={mode}
                        onClick={() => setBayFilter(mode)}
                        className={`px-2 py-0.5 rounded transition-all ${
                          bayFilter === mode
                            ? 'bg-command-elevated text-white font-bold'
                            : 'text-command-muted hover:text-white'
                        }`}
                      >
                        {mode}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="max-h-56 overflow-y-auto space-y-1.5 pr-1 text-xs">
                  {filteredBays.length === 0 ? (
                    <div className="p-3 text-center text-command-muted text-xs font-mono">
                      No bays match current filter.
                    </div>
                  ) : (
                    filteredBays.map(([label, item]) => {
                      const isInspected = inspectedBayLabel === label;
                      return (
                        <div
                          key={label}
                          onClick={() => setInspectedBayLabel(isInspected ? null : label)}
                          className={`p-2 rounded-lg border transition-all cursor-pointer ${
                            isInspected
                              ? 'bg-command-elevated border-radar-bright shadow-sm'
                              : 'bg-command-surface border-command-border hover:border-command-muted'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              <span className="font-bold text-command-text">{label}</span>
                              <span className="text-[10px] font-mono text-command-muted">
                                {item.space_type}
                              </span>
                            </div>
                            <div className="flex items-center space-x-2">
                              <span
                                className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold ${
                                  item.final_state === 'OCCUPIED'
                                    ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                                    : item.final_state === 'VACANT'
                                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                                    : 'bg-slate-700/30 text-slate-400 border border-slate-600/30'
                                }`}
                              >
                                {item.final_state} · {item.final_confidence.toFixed(2)}
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>

                {/* Selected Bay Evidence Drawer */}
                {inspectedBay && (
                  <div className="p-3 rounded-lg bg-command-elevated border border-command-border text-xs space-y-2">
                    <div className="flex items-center justify-between border-b border-command-border pb-1.5">
                      <span className="font-bold text-radar-bright">
                        Bay Evidence: {inspectedBay.operator_label}
                      </span>
                      <button
                        onClick={() => setInspectedBayLabel(null)}
                        className="text-command-muted hover:text-white text-[11px]"
                      >
                        Close
                      </button>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-[10px] font-mono text-command-muted">
                      <div>
                        <span>Space Type:</span>{' '}
                        <strong className="text-white">{inspectedBay.space_type}</strong>
                      </div>
                      <div>
                        <span>Final State:</span>{' '}
                        <strong
                          className={
                            inspectedBay.final_state === 'OCCUPIED'
                              ? 'text-rose-400'
                              : inspectedBay.final_state === 'VACANT'
                              ? 'text-emerald-400'
                              : 'text-slate-400'
                          }
                        >
                          {inspectedBay.final_state}
                        </strong>
                      </div>
                      <div>
                        <span>Confidence:</span>{' '}
                        <strong className="text-white">
                          {(inspectedBay.final_confidence * 100).toFixed(1)}%
                        </strong>
                      </div>
                      <div>
                        <span>State Transitions:</span>{' '}
                        <strong className="text-white">{inspectedBay.transitions_count}</strong>
                      </div>
                      {inspectedBay.last_vehicle_track_id && (
                        <div className="col-span-2">
                          <span>Last Contributing Vehicle:</span>{' '}
                          <strong className="text-cyan-400">
                            Track #{inspectedBay.last_vehicle_track_id}
                          </strong>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Provenance Audit Hashes */}
              <div className="p-2.5 rounded-lg bg-command-surface/50 border border-command-border space-y-1 font-mono text-[10px] text-command-muted">
                <div className="flex items-center space-x-1 text-command-text font-bold uppercase">
                  <Fingerprint className="w-3.5 h-3.5 text-radar-bright" />
                  <span>Immutable Pipeline Provenance</span>
                </div>
                <div className="truncate">
                  <span>Detector SHA: </span>
                  <span className="text-cyan-300">{selectedJob.detector_checkpoint_sha256?.slice(0, 16)}...</span>
                </div>
                <div className="truncate">
                  <span>Layout SHA: </span>
                  <span className="text-radar-bright">{selectedJob.layout_canonical_sha256?.slice(0, 16)}...</span>
                </div>
                <div className="truncate">
                  <span>Config SHA: </span>
                  <span className="text-amber-300">{selectedJob.occupancy_config_sha256?.slice(0, 16)}...</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 4. Previous Occupancy Jobs History */}
      {jobs.length > 1 && (
        <div className="p-3.5 rounded-xl glass-panel border border-command-border space-y-2.5 text-xs">
          <div className="flex items-center space-x-1.5 border-b border-command-border pb-2 text-command-text font-bold">
            <History className="w-4 h-4 text-command-muted" />
            <span>Past Occupancy Pipeline Executions ({jobs.length})</span>
          </div>

          <div className="max-h-40 overflow-y-auto space-y-1 pr-1 font-mono text-[11px]">
            {jobs.map((job) => {
              const isSelected = selectedJob?.id === job.id;
              return (
                <div
                  key={job.id}
                  onClick={() => setSelectedJob(job)}
                  className={`p-2 rounded-lg border transition-all cursor-pointer flex items-center justify-between ${
                    isSelected
                      ? 'bg-command-elevated border-radar-bright text-white'
                      : 'bg-command-surface border-command-border text-command-muted hover:text-white'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <span className="font-bold">{job.id.slice(0, 8)}...</span>
                    <span className="text-[10px] text-command-muted">
                      {new Date(job.created_at).toLocaleTimeString()}
                    </span>
                  </div>
                  <span
                    className={`px-1.5 py-0.2 rounded text-[9px] font-bold ${
                      job.status === 'COMPLETE'
                        ? 'bg-emerald-500/20 text-emerald-400'
                        : job.status === 'BLOCKED_BY_STABILITY_GATE'
                        ? 'bg-rose-500/20 text-rose-400'
                        : 'bg-cyan-500/20 text-cyan-400'
                    }`}
                  >
                    {job.status}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
