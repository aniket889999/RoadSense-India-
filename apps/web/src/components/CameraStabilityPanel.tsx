'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  CameraOperationalGate,
  StabilityAssessment,
  StabilityDecision,
} from '../lib/types';
import {
  getCameraOperationalGate,
  listCameraStabilityAssessments,
  assessCameraStability,
  acknowledgeStabilityAssessment,
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
  const [selectedAssessment, setSelectedAssessment] = useState<StabilityAssessment | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isAssessing, setIsAssessing] = useState(false);
  const [assessmentError, setAssessmentError] = useState<string | null>(null);

  // File upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [useLocalFile, setUseLocalFile] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Acknowledgement modal state
  const [isAckModalOpen, setIsAckModalOpen] = useState(false);
  const [ackOperatorLabel, setAckOperatorLabel] = useState('Senior SiteOps Operator');
  const [ackNote, setAckNote] = useState('');
  const [ackTriggerInvalidation, setAckTriggerInvalidation] = useState(false);
  const [ackInvalidationReason, setAckInvalidationReason] = useState('Camera drift exceeds tolerance thresholds');
  const [isAcknowledging, setIsAcknowledging] = useState(false);
  const [ackError, setAckError] = useState<string | null>(null);

  // Load Gate and Assessments
  const loadStabilityData = useCallback(async () => {
    if (!cameraId) {
      setGate(null);
      setAssessments([]);
      setSelectedAssessment(null);
      return;
    }

    setIsLoading(true);
    setAssessmentError(null);
    try {
      const [gateRes, assessmentsRes] = await Promise.all([
        getCameraOperationalGate(cameraId),
        listCameraStabilityAssessments(cameraId),
      ]);
      setGate(gateRes);
      setAssessments(assessmentsRes);
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

  // Handle run assessment
  const handleRunAssessment = async () => {
    if (!cameraId) return;
    if (!useLocalFile && !selectedFile) {
      setAssessmentError('Please select a video file or choose the local negative test video.');
      return;
    }

    setIsAssessing(true);
    setAssessmentError(null);
    try {
      let result: StabilityAssessment;
      if (useLocalFile) {
        result = await assessCameraStability(cameraId, undefined, 'PARKING LOT TEST.mp4');
      } else if (selectedFile) {
        result = await assessCameraStability(cameraId, selectedFile);
      } else {
        throw new Error('No video selected');
      }

      // Refresh gate and assessments list
      await loadStabilityData();
      setSelectedAssessment(result);
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err: any) {
      console.error('Stability assessment failed:', err);
      setAssessmentError(err.message || 'Assessment execution failed');
    } finally {
      setIsAssessing(false);
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
          <span className="text-[10px] text-command-muted">OpenCV ORB + RANSAC</span>
        </div>

        {!hasReferenceImage ? (
          <div className="p-2.5 rounded bg-amber-950/30 border border-amber-600/40 text-amber-300 text-[11px]">
            A reference frame must be uploaded to this camera before running stability checks.
          </div>
        ) : (
          <div className="space-y-2.5">
            {/* Input Selection: Local video or file upload */}
            <div className="space-y-1.5">
              <label className="text-[10px] text-command-muted uppercase">Select Assessment Video</label>
              <div className="flex items-center space-x-2">
                <input
                  type="file"
                  ref={fileInputRef}
                  accept="video/mp4,video/avi,video/mov,video/mkv"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      setSelectedFile(e.target.files[0]);
                      setUseLocalFile(false);
                    }
                  }}
                  className="hidden"
                />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className={`flex-1 flex items-center justify-center space-x-1.5 px-3 py-1.5 rounded border text-xs transition-all ${
                    selectedFile && !useLocalFile
                      ? 'bg-blue-900/40 border-blue-500 text-blue-200'
                      : 'bg-command-elevated border-command-border hover:border-radar-bright text-command-text'
                  }`}
                >
                  <UploadCloud className="w-3.5 h-3.5" />
                  <span className="truncate">{selectedFile ? selectedFile.name : 'Choose Video File...'}</span>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setUseLocalFile(!useLocalFile);
                    setSelectedFile(null);
                  }}
                  className={`px-2.5 py-1.5 rounded border text-[11px] whitespace-nowrap transition-all ${
                    useLocalFile
                      ? 'bg-amber-900/40 border-amber-500 text-amber-200 font-bold'
                      : 'bg-command-elevated border-command-border hover:text-white text-command-muted'
                  }`}
                  title="Use local negative test video (PARKING LOT TEST.mp4)"
                >
                  {useLocalFile ? '✓ Local Video' : 'Use Local Test Video'}
                </button>
              </div>

              {useLocalFile && (
                <div className="p-2 rounded bg-command-bg border border-command-border text-[10px] text-command-muted">
                  Using server negative test video: <span className="text-white font-bold">PARKING LOT TEST.mp4</span> (known camera pan/tilt drift).
                </div>
              )}
            </div>

            {assessmentError && (
              <div className="p-2 rounded bg-rose-950/40 border border-rose-600/50 text-rose-300 text-[11px]">
                {assessmentError}
              </div>
            )}

            <button
              onClick={handleRunAssessment}
              disabled={isAssessing || (!selectedFile && !useLocalFile)}
              className="w-full flex items-center justify-center space-x-2 px-3 py-2 rounded bg-radar-bright hover:bg-radar-green text-command-bg font-bold shadow-radar transition-all disabled:opacity-40"
            >
              {isAssessing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Computing Homography & Drift...</span>
                </>
              ) : (
                <>
                  <Sliders className="w-4 h-4" />
                  <span>Run Stability Assessment</span>
                </>
              )}
            </button>
          </div>
        )}
      </div>

      {/* 3. Active Assessment Inspection */}
      {selectedAssessment && (
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
                {selectedAssessment.summary_metrics?.max_translation_px?.toFixed(1) ?? 'N/A'} px
              </div>
              <span className="text-[9px] text-command-muted">
                Norm: {((selectedAssessment.summary_metrics?.max_translation_normalized || 0) * 100).toFixed(2)}% (limit 2%)
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
              <span className="text-[9px] text-command-muted">Limit: 2.00%</span>
            </div>

            <div className="p-2 rounded bg-command-bg border border-command-border space-y-0.5">
              <span className="text-[10px] text-command-muted uppercase">Min Inlier Ratio</span>
              <div className="text-sm font-bold text-command-text">
                {((selectedAssessment.summary_metrics?.min_inlier_ratio || 0) * 100).toFixed(1)}%
              </div>
              <span className="text-[9px] text-command-muted">Min required: 30.0%</span>
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
              <div>Video SHA: <span className="text-command-text text-[9px] break-all">{selectedAssessment.video_sha256.slice(0, 16)}...</span></div>
              <div>Algorithm: <span className="text-white">{selectedAssessment.algorithm_version}</span> ({selectedAssessment.opencv_version})</div>
              <div>Timestamp: <span className="text-white">{new Date(selectedAssessment.created_at).toLocaleString()}</span></div>
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

          {/* Acknowledgement Status & Action */}
          <div className="pt-1">
            {selectedAssessment.operator_acknowledged_at ? (
              <div className="p-2 rounded bg-blue-950/30 border border-blue-600/40 text-[10px] space-y-0.5 text-blue-300">
                <div className="font-bold flex items-center space-x-1">
                  <CheckCircle2 className="w-3 h-3 text-blue-400" />
                  <span>Acknowledged by {selectedAssessment.operator_label}</span>
                </div>
                <div>At: {new Date(selectedAssessment.operator_acknowledged_at).toLocaleString()}</div>
                {selectedAssessment.operator_note && (
                  <div className="text-white italic">"{selectedAssessment.operator_note}"</div>
                )}
              </div>
            ) : (
              <button
                id="open-ack-modal-btn"
                onClick={() => {
                  setAckError(null);
                  setIsAckModalOpen(true);
                }}
                className="w-full flex items-center justify-center space-x-1.5 px-3 py-2 rounded bg-amber-600/20 border border-amber-600/50 hover:bg-amber-600/30 text-amber-300 font-bold transition-all"
              >
                <AlertTriangle className="w-3.5 h-3.5" />
                <span>Acknowledge Assessment / Review Drift</span>
              </button>
            )}
          </div>
        </div>
      )}

      {/* 4. Assessment History */}
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
                  <span className="text-[9px] text-command-muted font-mono">{a.video_sha256.slice(0, 8)}</span>
                </div>
                <span className="font-bold text-[9px]">{a.aggregate_decision}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 5. Acknowledge Assessment Modal */}
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
                <label className="block text-[10px] text-command-muted uppercase mb-1">Operator Label</label>
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

              {selectedAssessment?.aggregate_decision !== 'STABLE' && (
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
                  <p className="text-[10px] text-rose-200/80">
                    If enabled, this will transition the camera's verified layout revision to INVALIDATED and lock it in the immutable audit ledger.
                  </p>

                  {ackTriggerInvalidation && (
                    <div className="pt-1">
                      <label className="block text-[10px] text-command-muted uppercase mb-1">
                        Invalidation Reason
                      </label>
                      <input
                        type="text"
                        value={ackInvalidationReason}
                        onChange={(e) => setAckInvalidationReason(e.target.value)}
                        required
                        className="w-full px-2.5 py-1.5 rounded bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-radar-bright"
                      />
                    </div>
                  )}
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
                  disabled={isAcknowledging || !ackOperatorLabel.trim()}
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
