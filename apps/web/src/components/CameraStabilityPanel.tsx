'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  CameraOperationalGate,
  CameraStabilityAuditEvent,
  StabilityAssessment,
  StabilityDecision,
} from '../lib/types';
import {
  getCameraOperationalGate,
  listCameraStabilityAssessments,
  getStabilityAssessmentDetail,
  assessCameraStability,
  cancelStabilityAssessment,
  acknowledgeStabilityAssessment,
  listCameraStabilityAuditEvents,
} from '../lib/parkingApi';
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  FileCheck,
  Fingerprint,
  History,
  Layers,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  UploadCloud,
  Video,
  XCircle,
  Ban,
  Activity,
} from 'lucide-react';

interface CameraStabilityPanelProps {
  cameraId: string | null;
  cameraName?: string;
  hasReferenceImage: boolean;
  activeLayoutId?: string | null;
  onCalibrationInvalidated?: () => void;
}

export function CameraStabilityPanel({
  cameraId,
  cameraName,
  hasReferenceImage,
  activeLayoutId,
  onCalibrationInvalidated,
}: CameraStabilityPanelProps) {
  const [gate, setGate] = useState<CameraOperationalGate | null>(null);
  const [assessments, setAssessments] = useState<StabilityAssessment[]>([]);
  const [auditEvents, setAuditEvents] = useState<CameraStabilityAuditEvent[]>([]);
  const [selectedAssessment, setSelectedAssessment] = useState<StabilityAssessment | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isAssessing, setIsAssessing] = useState(false);
  const [assessmentError, setAssessmentError] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);

  // File upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Acknowledgement modal state
  const [isAckModalOpen, setIsAckModalOpen] = useState(false);
  const [ackOperatorLabel, setAckOperatorLabel] = useState('Senior SiteOps Operator');
  const [ackNote, setAckNote] = useState('');
  const [ackTriggerInvalidation, setAckTriggerInvalidation] = useState(false);
  const [ackConfirmInvalidation, setAckConfirmInvalidation] = useState(false);
  const [ackInvalidationReason, setAckInvalidationReason] = useState('Camera drift exceeds tolerance thresholds');
  const [isAcknowledging, setIsAcknowledging] = useState(false);
  const [ackError, setAckError] = useState<string | null>(null);

  // Load Gate, Assessments, and Audit Events
  const loadStabilityData = useCallback(async () => {
    if (!cameraId) {
      setGate(null);
      setAssessments([]);
      setAuditEvents([]);
      setSelectedAssessment(null);
      return;
    }

    setIsLoading(true);
    setAssessmentError(null);
    try {
      const [gateRes, assessmentsRes, auditsRes] = await Promise.all([
        getCameraOperationalGate(cameraId),
        listCameraStabilityAssessments(cameraId),
        listCameraStabilityAuditEvents(cameraId).catch(() => []),
      ]);
      setGate(gateRes);
      setAssessments(assessmentsRes);
      setAuditEvents(auditsRes);
      if (assessmentsRes.length > 0) {
        setSelectedAssessment(assessmentsRes[0]);
      } else {
        setSelectedAssessment(null);
      }
    } catch (err: any) {
      console.error('Failed to load stability data:', err);
      setAssessmentError(err.message || 'Failed to load stability data');
    } finally {
      setIsLoading(false);
    }
  }, [cameraId]);

  useEffect(() => {
    loadStabilityData();
  }, [loadStabilityData]);

  // Polling for in-progress assessment
  useEffect(() => {
    if (!selectedAssessment || !['QUEUED', 'VALIDATING', 'ANALYZING'].includes(selectedAssessment.status)) {
      return;
    }

    const intervalId = setInterval(async () => {
      try {
        const updated = await getStabilityAssessmentDetail(selectedAssessment.id);
        setSelectedAssessment(updated);
        setAssessments((prev) => prev.map((a) => (a.id === updated.id ? updated : a)));

        if (['COMPLETE', 'FAILED', 'CANCELLED'].includes(updated.status)) {
          if (cameraId) {
            const [gateRes, auditsRes] = await Promise.all([
              getCameraOperationalGate(cameraId),
              listCameraStabilityAuditEvents(cameraId).catch(() => []),
            ]);
            setGate(gateRes);
            setAuditEvents(auditsRes);
          }
        }
      } catch (err) {
        console.error('Failed to poll assessment progress:', err);
      }
    }, 1500);

    return () => clearInterval(intervalId);
  }, [selectedAssessment?.id, selectedAssessment?.status, cameraId]);

  // Handle run assessment
  const handleRunAssessment = async () => {
    if (!cameraId) return;
    if (!selectedFile) {
      setAssessmentError('Please select a video file to upload for stability analysis.');
      return;
    }

    setIsAssessing(true);
    setAssessmentError(null);
    try {
      const result = await assessCameraStability(cameraId, selectedFile);
      setAssessments((prev) => [result, ...prev]);
      setSelectedAssessment(result);
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err: any) {
      console.error('Stability assessment submission failed:', err);
      setAssessmentError(err.message || 'Assessment submission failed');
    } finally {
      setIsAssessing(false);
    }
  };

  // Handle cancel assessment
  const handleCancelAssessment = async () => {
    if (!selectedAssessment) return;
    setIsCancelling(true);
    try {
      await cancelStabilityAssessment(selectedAssessment.id);
      const updated = await getStabilityAssessmentDetail(selectedAssessment.id);
      setSelectedAssessment(updated);
      setAssessments((prev) => prev.map((a) => (a.id === updated.id ? updated : a)));
    } catch (err: any) {
      console.error('Failed to cancel assessment:', err);
      setAssessmentError(err.message || 'Failed to cancel assessment');
    } finally {
      setIsCancelling(false);
    }
  };

  // Handle acknowledge submit
  const handleAcknowledgeConfirm = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedAssessment) return;

    setIsAcknowledging(true);
    setAckError(null);
    try {
      await acknowledgeStabilityAssessment(
        selectedAssessment.id,
        ackOperatorLabel,
        ackNote,
        ackTriggerInvalidation,
        ackTriggerInvalidation ? ackConfirmInvalidation : false,
        ackTriggerInvalidation ? ackInvalidationReason : undefined
      );

      setIsAckModalOpen(false);
      await loadStabilityData();

      if (ackTriggerInvalidation && onCalibrationInvalidated) {
        onCalibrationInvalidated();
      }
    } catch (err: any) {
      setAckError(err.message || 'Failed to submit acknowledgement');
    } finally {
      setIsAcknowledging(false);
    }
  };

  const isGateAllowed = gate?.operational_gate === 'ALLOWED';
  const isJobRunning = selectedAssessment && ['QUEUED', 'VALIDATING', 'ANALYZING'].includes(selectedAssessment.status);

  if (!cameraId) {
    return (
      <div className="p-4 rounded-xl glass-panel border border-command-border text-center text-xs text-command-muted">
        Select a camera to inspect stability and operational gate.
      </div>
    );
  }

  return (
    <div className="space-y-3 font-mono text-xs">
      {/* 1. Operational Gate Banner (Fail-Closed) */}
      <div
        id="camera-operational-gate-banner"
        className={`p-3.5 rounded-xl border flex flex-col space-y-2 transition-all ${
          isGateAllowed
            ? 'bg-emerald-950/30 border-emerald-600/50 text-emerald-300'
            : 'bg-rose-950/40 border-rose-600/60 text-rose-300'
        }`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {isGateAllowed ? (
              <ShieldCheck className="w-5 h-5 text-emerald-400 shrink-0" />
            ) : (
              <ShieldAlert className="w-5 h-5 text-rose-400 shrink-0 animate-pulse" />
            )}
            <div>
              <span className="font-bold text-xs tracking-wider uppercase">
                OPERATIONAL GATE: {gate?.operational_gate || 'BLOCKED'}
              </span>
              <div className="text-[10px] opacity-80">
                {isGateAllowed
                  ? 'Active layout geometry verified stable. Ready for downstream operations.'
                  : 'FAIL-CLOSED: Downstream occupancy inference is strictly blocked.'}
              </div>
            </div>
          </div>
          <button
            onClick={loadStabilityData}
            disabled={isLoading}
            className="p-1 rounded bg-black/40 hover:bg-black/60 border border-command-border text-command-muted hover:text-white"
            title="Refresh Gate Status"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {/* Prominent Fail-Closed Notice */}
        <div className={`p-2 rounded text-[11px] font-sans font-medium border ${
          isGateAllowed
            ? 'bg-emerald-900/30 border-emerald-700/40 text-emerald-200'
            : 'bg-rose-900/40 border-rose-700/60 text-rose-100'
        }`}>
          {isGateAllowed ? (
            <span>Camera stability verified. Geometry matches reference and verified layout.</span>
          ) : (
            <span className="flex items-center space-x-1.5 font-bold">
              <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400" />
              <span>Parking occupancy is blocked until camera geometry is verified stable</span>
            </span>
          )}
        </div>

        {/* Gate Reasons */}
        {gate && gate.gate_reasons && gate.gate_reasons.length > 0 && (
          <div className="pt-1 text-[10px] space-y-1">
            <span className="font-bold uppercase tracking-wider text-command-muted">Gate Evaluation Reasons:</span>
            <ul className="list-disc list-inside space-y-0.5 opacity-90">
              {gate.gate_reasons.map((r, idx) => (
                <li key={idx} className="break-words">{r}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* 2. Run Stability Assessment Card */}
      <div className="p-3.5 rounded-xl glass-panel border border-command-border space-y-3">
        <div className="flex items-center justify-between border-b border-command-border pb-2">
          <div className="flex items-center space-x-2 text-command-text font-bold uppercase tracking-wider">
            <Video className="w-4 h-4 text-radar-bright" />
            <span>Assess Camera Stability</span>
          </div>
          <span className="text-[10px] text-command-muted">Asynchronous ORB + RANSAC Engine</span>
        </div>

        {!hasReferenceImage ? (
          <div className="p-2.5 rounded bg-amber-950/30 border border-amber-600/40 text-amber-300 text-[11px]">
            A reference frame must be uploaded to this camera before running stability checks.
          </div>
        ) : (
          <div className="space-y-2.5">
            <div className="space-y-1.5">
              <label className="text-[10px] text-command-muted uppercase">Select Assessment Video File (Max 200 MB)</label>
              <div className="flex items-center space-x-2">
                <input
                  type="file"
                  ref={fileInputRef}
                  accept="video/mp4,video/avi,video/mov,video/mkv"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      setSelectedFile(e.target.files[0]);
                    }
                  }}
                  className="hidden"
                />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className={`flex-1 flex items-center justify-center space-x-1.5 px-3 py-2 rounded border text-xs transition-all ${
                    selectedFile
                      ? 'bg-blue-900/40 border-blue-500 text-blue-200'
                      : 'bg-command-elevated border-command-border hover:border-radar-bright text-command-text'
                  }`}
                >
                  <UploadCloud className="w-4 h-4" />
                  <span className="truncate">{selectedFile ? selectedFile.name : 'Choose Video File from Computer...'}</span>
                </button>
              </div>
            </div>

            {assessmentError && (
              <div className="p-2 rounded bg-rose-950/40 border border-rose-600/50 text-rose-300 text-[11px]">
                {assessmentError}
              </div>
            )}

            <button
              onClick={handleRunAssessment}
              disabled={isAssessing || !selectedFile}
              className="w-full flex items-center justify-center space-x-2 px-3 py-2 rounded bg-radar-bright hover:bg-radar-green text-command-bg font-bold shadow-radar transition-all disabled:opacity-40"
            >
              {isAssessing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Staging & Submitting Job...</span>
                </>
              ) : (
                <>
                  <Sliders className="w-4 h-4" />
                  <span>Submit Stability Assessment</span>
                </>
              )}
            </button>
          </div>
        )}
      </div>

      {/* 3. In-Progress Job Progress Banner */}
      {isJobRunning && selectedAssessment && (
        <div className="p-3.5 rounded-xl bg-blue-950/30 border border-blue-600/50 space-y-2">
          <div className="flex items-center justify-between text-blue-300">
            <div className="flex items-center space-x-2 font-bold">
              <Activity className="w-4 h-4 animate-pulse text-blue-400" />
              <span>Assessment Job In Progress ({selectedAssessment.status})</span>
            </div>
            <button
              onClick={handleCancelAssessment}
              disabled={isCancelling}
              className="flex items-center space-x-1 px-2 py-0.5 rounded bg-rose-900/50 border border-rose-600 text-rose-200 text-[10px] hover:bg-rose-900"
            >
              <Ban className="w-3 h-3" />
              <span>{isCancelling ? 'Cancelling...' : 'Cancel Job'}</span>
            </button>
          </div>

          <div className="w-full bg-black/50 rounded-full h-2 overflow-hidden border border-blue-800">
            <div
              className="bg-blue-500 h-2 transition-all duration-300 rounded-full"
              style={{ width: `${Math.max(5, selectedAssessment.progress_pct)}%` }}
            />
          </div>

          <div className="flex items-center justify-between text-[10px] text-blue-200/80">
            <span>{selectedAssessment.stage_message || 'Processing frames...'}</span>
            <span className="font-bold">{selectedAssessment.progress_pct.toFixed(0)}%</span>
          </div>
        </div>
      )}

      {/* 4. Active Assessment Details */}
      {selectedAssessment && selectedAssessment.status === 'COMPLETE' && (
        <div className="p-3.5 rounded-xl glass-panel border border-command-border space-y-3">
          <div className="flex items-center justify-between border-b border-command-border pb-2">
            <div className="flex items-center space-x-2">
              <FileCheck className="w-4 h-4 text-command-muted" />
              <span className="font-bold text-command-text uppercase tracking-wider">Assessment Metrics</span>
            </div>
            <span
              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                selectedAssessment.aggregate_decision === 'STABLE'
                  ? 'bg-emerald-900/60 text-emerald-300 border border-emerald-600'
                  : 'bg-rose-900/60 text-rose-300 border border-rose-600'
              }`}
            >
              {selectedAssessment.aggregate_decision}
            </span>
          </div>

          {/* Metric Summary Cards */}
          <div className="grid grid-cols-2 gap-2">
            <div className="p-2 rounded bg-command-bg border border-command-border space-y-0.5">
              <span className="text-[10px] text-command-muted uppercase">Max Translation</span>
              <div className="text-sm font-bold text-command-text">
                {selectedAssessment.summary_metrics?.max_translation_magnitude_px?.toFixed(1) ?? 'N/A'} px
              </div>
              <span className="text-[9px] text-command-muted">
                Norm: {((selectedAssessment.summary_metrics?.max_translation_normalized || 0) * 100).toFixed(2)}% (limit 1.0%)
              </span>
            </div>

            <div className="p-2 rounded bg-command-bg border border-command-border space-y-0.5">
              <span className="text-[10px] text-command-muted uppercase">Max Rotation</span>
              <div className="text-sm font-bold text-command-text">
                {selectedAssessment.summary_metrics?.max_rotation_degrees?.toFixed(2) ?? 'N/A'}°
              </div>
              <span className="text-[9px] text-command-muted">Limit: 1.00°</span>
            </div>

            <div className="p-2 rounded bg-command-bg border border-command-border space-y-0.5">
              <span className="text-[10px] text-command-muted uppercase">Max Scale Drift</span>
              <div className="text-sm font-bold text-command-text">
                {((selectedAssessment.summary_metrics?.max_scale_change || 0) * 100).toFixed(2)}%
              </div>
              <span className="text-[9px] text-command-muted">Limit: 3.00%</span>
            </div>

            <div className="p-2 rounded bg-command-bg border border-command-border space-y-0.5">
              <span className="text-[10px] text-command-muted uppercase">Min Inlier Ratio</span>
              <div className="text-sm font-bold text-command-text">
                {((selectedAssessment.summary_metrics?.min_inlier_ratio || 0) * 100).toFixed(1)}%
              </div>
              <span className="text-[9px] text-command-muted">Min required: 25.0%</span>
            </div>
          </div>

          {/* Provenance Details */}
          <div className="p-2 rounded bg-command-bg border border-command-border space-y-1 text-[10px]">
            <div className="flex items-center space-x-1 text-radar-bright font-bold">
              <Fingerprint className="w-3 h-3" />
              <span>Provenance & Audit Hash</span>
            </div>
            <div className="space-y-0.5 text-command-muted font-mono">
              <div>Ref Image SHA: <span className="text-command-text text-[9px] break-all">{selectedAssessment.reference_image_sha256.slice(0, 16)}...</span></div>
              {selectedAssessment.video_sha256 && (
                <div>Video SHA: <span className="text-command-text text-[9px] break-all">{selectedAssessment.video_sha256.slice(0, 16)}...</span></div>
              )}
              <div>Config: <span className="text-white">v{selectedAssessment.config_version || '1.0.0'}</span> ({selectedAssessment.config_sha256 ? selectedAssessment.config_sha256.slice(0, 8) : 'default'})</div>
              <div>Engine: <span className="text-white">{selectedAssessment.algorithm_version}</span> (OpenCV {selectedAssessment.opencv_version})</div>
              <div>Completed: <span className="text-white">{new Date(selectedAssessment.created_at).toLocaleString()}</span></div>
            </div>
          </div>

          {/* Per-Sample Measurements Table */}
          {selectedAssessment.sample_measurements && selectedAssessment.sample_measurements.length > 0 && (
            <div className="space-y-1.5">
              <span className="text-[10px] text-command-muted uppercase tracking-wider block">Sample Evidence Breakdown</span>
              <div className="max-h-36 overflow-y-auto border border-command-border rounded">
                <table className="w-full text-left text-[10px] border-collapse">
                  <thead className="bg-command-elevated text-command-muted uppercase text-[9px] sticky top-0">
                    <tr>
                      <th className="p-1 border-b border-command-border">#</th>
                      <th className="p-1 border-b border-command-border">Time (s)</th>
                      <th className="p-1 border-b border-command-border">Trans</th>
                      <th className="p-1 border-b border-command-border">Rot</th>
                      <th className="p-1 border-b border-command-border">Inliers</th>
                      <th className="p-1 border-b border-command-border">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-command-border text-command-text">
                    {selectedAssessment.sample_measurements.map((s, idx) => (
                      <tr key={idx} className="hover:bg-command-elevated/40">
                        <td className="p-1 font-bold">{s.sample_index + 1}</td>
                        <td className="p-1">{s.timestamp_seconds.toFixed(1)}s</td>
                        <td className="p-1">{s.translation_magnitude_px.toFixed(1)}px</td>
                        <td className="p-1">{s.rotation_degrees.toFixed(2)}°</td>
                        <td className="p-1">{(s.inlier_ratio * 100).toFixed(0)}%</td>
                        <td className="p-1">
                          <span
                            className={`px-1 py-0.2 rounded text-[9px] font-bold ${
                              s.decision === 'STABLE' ? 'text-emerald-400' : 'text-rose-400'
                            }`}
                          >
                            {s.decision}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Acknowledgement Action Button */}
          <div className="pt-1">
            <button
              id="open-ack-modal-btn"
              onClick={() => {
                setAckError(null);
                setAckConfirmInvalidation(false);
                setAckTriggerInvalidation(false);
                setIsAckModalOpen(true);
              }}
              className="w-full flex items-center justify-center space-x-1.5 px-3 py-2 rounded bg-amber-600/20 border border-amber-600/50 hover:bg-amber-600/30 text-amber-300 font-bold transition-all"
            >
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>Record Operator Acknowledgement</span>
            </button>
          </div>
        </div>
      )}

      {/* 5. Audit Events Ledger */}
      {auditEvents.length > 0 && (
        <div className="p-3 rounded-xl glass-panel border border-command-border space-y-2">
          <div className="flex items-center space-x-1.5 text-command-muted uppercase text-[10px]">
            <Fingerprint className="w-3.5 h-3.5 text-radar-bright" />
            <span>Immutable Stability Audit Log ({auditEvents.length})</span>
          </div>
          <div className="space-y-1.5 max-h-36 overflow-y-auto">
            {auditEvents.map((evt) => (
              <div
                key={evt.id}
                className="p-2 rounded bg-command-bg border border-command-border text-[10px] space-y-0.5"
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-white uppercase">{evt.event_type}</span>
                  <span className="text-[9px] text-command-muted">{new Date(evt.created_at).toLocaleTimeString()}</span>
                </div>
                <div className="text-command-muted">
                  Operator: <span className="text-command-text">{evt.operator_identity}</span>
                </div>
                <div className="text-command-muted">
                  Reason: <span className="text-amber-300 italic">{evt.explicit_reason}</span>
                </div>
                {evt.note && <div className="text-[9px] text-command-muted italic">"{evt.note}"</div>}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 6. Assessment History List */}
      {assessments.length > 1 && (
        <div className="p-3 rounded-xl glass-panel border border-command-border space-y-2">
          <div className="flex items-center space-x-1.5 text-command-muted uppercase text-[10px]">
            <History className="w-3.5 h-3.5" />
            <span>Assessment History ({assessments.length})</span>
          </div>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {assessments.map((a) => (
              <button
                key={a.id}
                onClick={() => setSelectedAssessment(a)}
                className={`w-full text-left p-1.5 rounded flex items-center justify-between border text-[10px] transition-all ${
                  selectedAssessment?.id === a.id
                    ? 'bg-command-elevated border-radar-bright text-white'
                    : 'bg-command-bg border-command-border text-command-muted hover:text-command-text'
                }`}
              >
                <div className="flex items-center space-x-2 truncate">
                  <span
                    className={`w-2 h-2 rounded-full ${
                      a.aggregate_decision === 'STABLE' ? 'bg-emerald-400' : 'bg-rose-400'
                    }`}
                  />
                  <span>{new Date(a.created_at).toLocaleTimeString()}</span>
                  <span className="text-[9px] text-command-muted font-mono">{a.video_sha256 ? a.video_sha256.slice(0, 8) : a.status}</span>
                </div>
                <span className="font-bold text-[9px]">{a.aggregate_decision || a.status}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 7. Acknowledge Assessment Modal */}
      {isAckModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-xl glass-panel-elevated border border-command-border p-5 space-y-4 font-mono text-xs">
            <div className="flex items-center space-x-2 text-command-text font-bold text-sm">
              <AlertTriangle className="w-4 h-4 text-amber-400" />
              <span>Operator Stability Review & Acknowledgement</span>
            </div>

            <form onSubmit={handleAcknowledgeConfirm} className="space-y-3">
              {ackError && (
                <div className="p-2 rounded bg-rose-950/40 border border-rose-600/50 text-rose-300 text-[11px]">
                  {ackError}
                </div>
              )}
              <div>
                <label className="block text-[10px] text-command-muted uppercase mb-1">Operator Identifier</label>
                <input
                  type="text"
                  value={ackOperatorLabel}
                  onChange={(e) => setAckOperatorLabel(e.target.value)}
                  required
                  className="w-full px-2.5 py-1.5 rounded bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-radar-bright"
                />
              </div>

              <div>
                <label className="block text-[10px] text-command-muted uppercase mb-1">Operator Note (Optional)</label>
                <textarea
                  value={ackNote}
                  onChange={(e) => setAckNote(e.target.value)}
                  rows={2}
                  placeholder="Record operational observations regarding camera vibration or shift..."
                  className="w-full px-2.5 py-1.5 rounded bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-radar-bright"
                />
              </div>

              {selectedAssessment?.aggregate_decision === 'UNSTABLE' ? (
                <div className="p-3 rounded bg-rose-950/30 border border-rose-600/40 space-y-2">
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={ackTriggerInvalidation}
                      onChange={(e) => setAckTriggerInvalidation(e.target.checked)}
                      className="rounded border-command-border bg-command-bg text-radar-bright focus:ring-0"
                    />
                    <span className="font-bold text-rose-300 text-xs">
                      Trigger Audited Layout Invalidation
                    </span>
                  </label>

                  {ackTriggerInvalidation && (
                    <div className="space-y-2 pt-1 border-t border-rose-800/40">
                      <div>
                        <label className="block text-[10px] text-command-muted uppercase mb-1">
                          Invalidation Reason (Required)
                        </label>
                        <input
                          type="text"
                          value={ackInvalidationReason}
                          onChange={(e) => setAckInvalidationReason(e.target.value)}
                          required
                          className="w-full px-2.5 py-1.5 rounded bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-radar-bright"
                        />
                      </div>

                      <label className="flex items-center space-x-2 cursor-pointer pt-1">
                        <input
                          type="checkbox"
                          checked={ackConfirmInvalidation}
                          onChange={(e) => setAckConfirmInvalidation(e.target.checked)}
                          required
                          className="rounded border-command-border bg-command-bg text-radar-bright focus:ring-0"
                        />
                        <span className="text-[10px] text-rose-200 font-bold">
                          I explicitly confirm camera geometry is compromised and must be invalidated.
                        </span>
                      </label>
                    </div>
                  )}
                </div>
              ) : (
                <div className="p-2 rounded bg-command-bg border border-command-border text-[10px] text-command-muted">
                  Note: Automated layout invalidation is restricted to UNSTABLE decisions. Assessment status is {selectedAssessment?.aggregate_decision || 'N/A'}.
                </div>
              )}

              <div className="flex justify-end space-x-2 pt-2">
                <button
                  type="button"
                  onClick={() => setIsAckModalOpen(false)}
                  className="px-3 py-1.5 rounded bg-command-elevated text-command-muted hover:text-white"
                >
                  Cancel
                </button>
                <button
                  id="confirm-ack-submit-btn"
                  type="submit"
                  disabled={isAcknowledging || !ackOperatorLabel.trim() || (ackTriggerInvalidation && !ackConfirmInvalidation)}
                  className="px-3 py-1.5 rounded bg-amber-600 hover:bg-amber-500 text-white font-bold disabled:opacity-50"
                >
                  {isAcknowledging ? 'Submitting...' : 'Confirm Acknowledgement'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
