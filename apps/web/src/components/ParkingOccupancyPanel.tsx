'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  CameraOperationalGate,
  ParkingOccupancyJob,
  BayOccupancyState,
  BaySummaryItem,
  SpaceType,
  CapacitySnapshot,
  CapacityState,
  HazardAssociation,
  PavementInspectionRecord,
} from '../lib/types';
import {
  getCameraOperationalGate,
  listOccupancyJobs,
  getOccupancyJob,
  submitOccupancyJob,
  cancelOccupancyJob,
  deleteOccupancyJob,
  getOccupancyVideoUrl,
  getOccupancyManifestUrl,
  getOccupancyTimelineUrl,
  getOccupancySummaryUrl,
  getCameraCapacitySnapshot,
  listHazardAssociations,
  createHazardAssociation,
  reviewHazardAssociation,
  transitionHazardLifecycle,
  recordPavementInspection,
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
  Trash2,
  Cpu,
  Lock,
  CheckSquare,
  Square,
  HelpCircle,
  Sparkles,
  Plus,
  X,
  Check,
} from 'lucide-react';

interface ParkingOccupancyPanelProps {
  cameraId: string | null;
  cameraName?: string;
  activeLayoutId?: string | null;
  onNavigateToLayout?: () => void;
  onNavigateToStability?: () => void;
}

export function ParkingOccupancyPanel({
  cameraId,
  cameraName,
  activeLayoutId,
  onNavigateToLayout,
  onNavigateToStability,
}: ParkingOccupancyPanelProps) {
  const [gate, setGate] = useState<CameraOperationalGate | null>(null);
  const [jobs, setJobs] = useState<ParkingOccupancyJob[]>([]);
  const [selectedJob, setSelectedJob] = useState<ParkingOccupancyJob | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [deleteOperatorLabel, setDeleteOperatorLabel] = useState('site_operator');
  const [deleteReason, setDeleteReason] = useState('Operator initiated evidence and job purge');

  // Phase 3A Safe Usable Capacity & Hazard Association State
  const [capacitySnapshot, setCapacitySnapshot] = useState<CapacitySnapshot | null>(null);
  const [hazards, setHazards] = useState<HazardAssociation[]>([]);
  const [showAddHazardModal, setShowAddHazardModal] = useState(false);
  const [showInspectionModal, setShowInspectionModal] = useState(false);

  // Add Hazard Form
  const [newHazardBayId, setNewHazardBayId] = useState('');
  const [newHazardTargetType, setNewHazardTargetType] = useState<'BAY' | 'APPROACH_ZONE'>('BAY');
  const [newHazardLabel, setNewHazardLabel] = useState('');
  const [newHazardSeverity, setNewHazardSeverity] = useState('MEDIUM');
  const [newHazardNotes, setNewHazardNotes] = useState('');
  const [isSubmittingHazard, setIsSubmittingHazard] = useState(false);

  // Record Inspection Form
  const [newInspectorLabel, setNewInspectorLabel] = useState('site_inspector');
  const [newInspectionSessionId, setNewInspectionSessionId] = useState('');
  const [newInspectionNotes, setNewInspectionNotes] = useState('');
  const [isSubmittingInspection, setIsSubmittingInspection] = useState(false);

  // Video Upload & Safety Check State
  const [selectedVideoFile, setSelectedVideoFile] = useState<File | null>(null);
  const [operatorConfirmationAcknowledged, setOperatorConfirmationAcknowledged] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Filter and Inspection State
  const [bayFilter, setBayFilter] = useState<
    'ALL' | 'USABLE_AVAILABLE' | 'OCCUPIED' | 'HAZARD_BLOCKED' | 'APPROACH_BLOCKED' | 'VACANT_UNASSESSED' | 'UNKNOWN'
  >('ALL');
  const [inspectedBayLabel, setInspectedBayLabel] = useState<string | null>(null);

  // Load Gate, Jobs, Capacity Snapshot, and Hazard Associations
  const loadData = useCallback(async () => {
    if (!cameraId) {
      setGate(null);
      setJobs([]);
      setSelectedJob(null);
      setCapacitySnapshot(null);
      setHazards([]);
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);
    try {
      const [gateData, jobsData, snapshotData, hazardsData] = await Promise.all([
        getCameraOperationalGate(cameraId).catch(() => null),
        listOccupancyJobs(cameraId).catch(() => []),
        getCameraCapacitySnapshot(cameraId).catch(() => null),
        listHazardAssociations(cameraId).catch(() => []),
      ]);
      setGate(gateData);
      setJobs(jobsData);
      setCapacitySnapshot(snapshotData);
      setHazards(hazardsData);
      if (jobsData.length > 0 && !selectedJob) {
        setSelectedJob(jobsData[0]);
      }
    } catch (err: any) {
      console.error('Failed to load parking occupancy data:', err);
      setErrorMessage(err.message || 'Failed to load occupancy data');
    } finally {
      setIsLoading(false);
    }
  }, [cameraId, selectedJob]);

  useEffect(() => {
    loadData();
  }, [loadData]);

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

  // Handle Safe Delete Job with operator label and reason
  const handleDeleteJob = async () => {
    if (!selectedJob) return;
    setIsDeleting(true);
    try {
      await deleteOccupancyJob(selectedJob.id, {
        confirmation_acknowledged: true,
        local_operator_label: deleteOperatorLabel.trim() || 'site_operator',
        deletion_reason: deleteReason.trim() || 'Operator initiated evidence and job purge',
        expected_status: selectedJob.status,
      });
      setJobs((prev) => prev.filter((j) => j.id !== selectedJob.id));
      setSelectedJob(jobs.find((j) => j.id !== selectedJob.id) || null);
      setDeleteConfirmOpen(false);
    } catch (err: any) {
      console.error('Failed to delete occupancy job:', err);
      setErrorMessage(err.message || 'Failed to delete job');
    } finally {
      setIsDeleting(false);
    }
  };

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
      'PUBLISHING',
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
      setOperatorConfirmationAcknowledged(false);
    }
  };

  // Handle Submit Job
  const handleSubmitJob = async () => {
    if (!cameraId || !selectedVideoFile || !operatorConfirmationAcknowledged) return;

    setIsSubmitting(true);
    setErrorMessage(null);
    try {
      const job = await submitOccupancyJob(cameraId, selectedVideoFile);
      setJobs((prev) => [job, ...prev]);
      setSelectedJob(job);
      setSelectedVideoFile(null);
      setOperatorConfirmationAcknowledged(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err: any) {
      console.error('Failed to submit occupancy job:', err);
      setErrorMessage(err.detail || err.message || 'Occupancy job submission rejected by system.');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Reviewer actions
  const handleReviewHazard = async (assocId: string, reviewState: 'CONFIRMED' | 'REJECTED' | 'NEEDS_REVIEW', expectedVersion: number) => {
    try {
      await reviewHazardAssociation(assocId, {
        review_state: reviewState,
        reviewer_identity: 'operator_reviewer',
        explicit_reason: `Operator audit set review state to ${reviewState}`,
        expected_version: expectedVersion,
      });
      await loadData();
    } catch (err: any) {
      console.error('Failed to review hazard:', err);
      setErrorMessage(err.message || 'Failed to review hazard');
    }
  };

  const handleLifecycleHazard = async (assocId: string, lifecycleState: 'ACTIVE' | 'MITIGATED' | 'RESOLVED', expectedVersion: number) => {
    try {
      await transitionHazardLifecycle(assocId, {
        lifecycle_state: lifecycleState,
        operator_identity: 'operator_reviewer',
        explicit_reason: `Operator transition lifecycle to ${lifecycleState}`,
        expected_version: expectedVersion,
      });
      await loadData();
    } catch (err: any) {
      console.error('Failed to update hazard lifecycle:', err);
      setErrorMessage(err.message || 'Failed to update hazard lifecycle');
    }
  };

  const handleCreateHazard = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!cameraId || !newHazardBayId || !newHazardLabel.trim()) return;
    setIsSubmittingHazard(true);
    try {
      await createHazardAssociation(cameraId, {
        parking_space_id: newHazardBayId,
        target_type: newHazardTargetType,
        hazard_label: newHazardLabel.trim(),
        severity_label: newHazardSeverity,
        notes: newHazardNotes.trim() || undefined,
        created_by: 'site_operator',
      });
      setShowAddHazardModal(false);
      setNewHazardLabel('');
      setNewHazardNotes('');
      await loadData();
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to create hazard association');
    } finally {
      setIsSubmittingHazard(false);
    }
  };

  const handleRecordInspection = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!cameraId) return;
    setIsSubmittingInspection(true);
    try {
      await recordPavementInspection(cameraId, {
        session_id: newInspectionSessionId.trim() || undefined,
        inspector_label: newInspectorLabel.trim() || 'site_inspector',
        notes: newInspectionNotes.trim() || undefined,
      });
      setShowInspectionModal(false);
      setNewInspectionSessionId('');
      setNewInspectionNotes('');
      await loadData();
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to record pavement inspection');
    } finally {
      setIsSubmittingInspection(false);
    }
  };

  const getCapacityBadgeStyle = (state?: CapacityState | string) => {
    switch (state) {
      case 'USABLE_AVAILABLE':
        return 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
      case 'OCCUPIED':
        return 'bg-rose-500/20 text-rose-400 border border-rose-500/30';
      case 'HAZARD_BLOCKED':
        return 'bg-fuchsia-500/20 text-fuchsia-400 border border-fuchsia-500/30';
      case 'APPROACH_BLOCKED':
        return 'bg-orange-500/20 text-orange-400 border border-orange-500/30';
      case 'VACANT_UNASSESSED':
        return 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30';
      case 'INFERENCE_BLOCKED':
        return 'bg-rose-950/60 text-rose-300 border border-rose-600/50';
      case 'UNKNOWN':
      case 'OCCLUDED':
      default:
        return 'bg-slate-700/30 text-slate-400 border border-slate-600/30';
    }
  };

  const isGateAllowed = gate?.operational_gate === 'ALLOWED';
  const isGateBlocked = gate?.operational_gate === 'BLOCKED' || !gate;

  const hasReferenceImage = Boolean(gate?.reference_image_sha256);
  const hasVerifiedLayout = Boolean(gate?.layout_canonical_sha256);
  const hasCompletedAssessment = Boolean(gate?.assessment_id);
  const isDetectorReady = true; // Local YOLOv8 checkpoint verified locally

  const isReadyForOccupancy = isGateAllowed && hasReferenceImage && hasVerifiedLayout;

  const baySummaryEntries: [string, BaySummaryItem][] = selectedJob?.bay_summary
    ? Object.entries(selectedJob.bay_summary)
    : [];

  const filteredBays = baySummaryEntries.filter(([label, item]) => {
    if (bayFilter === 'ALL') return true;
    const dec = capacitySnapshot?.decisions[label];
    if (dec) {
      if (bayFilter === 'UNKNOWN') {
        return dec.capacity_state === 'UNKNOWN' || dec.capacity_state === 'OCCLUDED';
      }
      return dec.capacity_state === bayFilter;
    }
    return item.final_state === bayFilter;
  });

  const inspectedBay = inspectedBayLabel && selectedJob?.bay_summary
    ? selectedJob.bay_summary[inspectedBayLabel]
    : null;

  const inspectedDecision = inspectedBayLabel && capacitySnapshot?.decisions
    ? capacitySnapshot.decisions[inspectedBayLabel]
    : null;

  const inspectedBayHazards = inspectedBayLabel
    ? hazards.filter((h) => {
        const dec = inspectedDecision;
        return h.parking_space_id === dec?.bay_id || h.parking_space_id === inspectedBayLabel;
      })
    : [];

  return (
    <div className="space-y-4">
      {/* 1. Readiness Checklist */}
      <div className="p-3.5 rounded-xl glass-panel border border-command-border space-y-2.5">
        <div className="flex items-center justify-between border-b border-command-border pb-2">
          <div className="flex items-center space-x-2">
            <CheckSquare className="w-4 h-4 text-radar-bright" />
            <span className="font-bold text-xs text-command-text">
              Parking Operator Readiness Checklist
            </span>
          </div>
          <span className="text-[10px] font-mono text-command-muted">
            Camera: {cameraName || (cameraId ? cameraId.slice(0, 8) : 'None')}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs font-mono">
          {/* Item 1: Reference image */}
          <div className="p-2 rounded-lg bg-command-surface border border-command-border flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {hasReferenceImage ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
              ) : (
                <XCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
              )}
              <span className="text-command-text">Reference Image Ingested</span>
            </div>
            {!hasReferenceImage && onNavigateToLayout && (
              <button
                onClick={onNavigateToLayout}
                className="text-[10px] text-radar-bright hover:underline font-bold"
              >
                Upload Ref
              </button>
            )}
          </div>

          {/* Item 2: Verified Layout */}
          <div className="p-2 rounded-lg bg-command-surface border border-command-border flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {hasVerifiedLayout ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
              ) : (
                <XCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
              )}
              <span className="text-command-text">Layout Human-Verified</span>
            </div>
            {!hasVerifiedLayout && onNavigateToLayout && (
              <button
                onClick={onNavigateToLayout}
                className="text-[10px] text-radar-bright hover:underline font-bold"
              >
                Verify Layout
              </button>
            )}
          </div>

          {/* Item 3: Stability Assessment */}
          <div className="p-2 rounded-lg bg-command-surface border border-command-border flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {hasCompletedAssessment ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
              ) : (
                <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
              )}
              <span className="text-command-text">Stability Assessed</span>
            </div>
            {!hasCompletedAssessment && onNavigateToStability && (
              <button
                onClick={onNavigateToStability}
                className="text-[10px] text-radar-bright hover:underline font-bold"
              >
                Run Stability
              </button>
            )}
          </div>

          {/* Item 4: Operational Gate */}
          <div className="p-2 rounded-lg bg-command-surface border border-command-border flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {isGateAllowed ? (
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
              ) : (
                <ShieldAlert className="w-3.5 h-3.5 text-rose-400 shrink-0" />
              )}
              <span className="text-command-text">
                Gate: <strong className={isGateAllowed ? 'text-emerald-400' : 'text-rose-400'}>{isGateAllowed ? 'ALLOWED' : 'BLOCKED'}</strong>
              </span>
            </div>
            {isGateBlocked && onNavigateToStability && (
              <button
                onClick={onNavigateToStability}
                className="text-[10px] text-rose-400 hover:underline font-bold"
              >
                Inspect Gate
              </button>
            )}
          </div>
        </div>
      </div>

      {/* 2. Stability Gate Status Banner */}
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

      {/* 2.5 Safe Usable Capacity Operational Scoreboard (Phase 3A) */}
      <div className="p-3.5 rounded-xl glass-panel border border-command-border space-y-3">
        <div className="flex items-center justify-between border-b border-command-border pb-2">
          <div className="flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-emerald-400" />
            <span className="font-bold text-xs text-command-text">
              Safe Usable Parking Capacity (Phase 3A Fusion)
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowAddHazardModal(true)}
              className="px-2 py-1 rounded bg-command-elevated hover:bg-command-elevated/80 border border-command-border text-fuchsia-300 font-bold text-[10px] flex items-center space-x-1 transition-all"
            >
              <Plus className="w-3 h-3" />
              <span>Associate Hazard</span>
            </button>
            <button
              onClick={() => setShowInspectionModal(true)}
              className="px-2 py-1 rounded bg-command-elevated hover:bg-command-elevated/80 border border-command-border text-cyan-300 font-bold text-[10px] flex items-center space-x-1 transition-all"
            >
              <Layers className="w-3 h-3" />
              <span>Record Inspection</span>
            </button>
            <button
              onClick={loadData}
              disabled={isLoading}
              className="p-1 rounded bg-command-elevated border border-command-border text-command-muted hover:text-white text-[10px] transition-all"
              title="Refresh Capacity Snapshot"
            >
              <RefreshCw className={`w-3 h-3 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* System Banner if Gate is Blocked */}
        {isGateBlocked && (
          <div
            id="capacity-inference-blocked-banner"
            className="p-3 rounded-lg bg-rose-950/80 border border-rose-500 text-rose-300 text-xs font-mono font-bold flex items-center space-x-2"
          >
            <Ban className="w-4 h-4 text-rose-400 shrink-0" />
            <span>INFERENCE_BLOCKED: Camera Operational Gate is BLOCKED. Official safe usable capacity claims are prohibited until camera stability is confirmed.</span>
          </div>
        )}

        {/* 7 Reconciled KPI Counters */}
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 text-center font-mono text-xs">
          <div className="p-2 rounded-lg bg-emerald-950/40 border border-emerald-500/40">
            <span className="text-[10px] text-emerald-400 uppercase block font-bold">Usable Available (SUC)</span>
            <span className="text-lg font-bold text-emerald-300">
              {capacitySnapshot?.usable_available_count ?? 0}
            </span>
          </div>
          <div className="p-2 rounded-lg bg-cyan-950/40 border border-cyan-500/40">
            <span className="text-[10px] text-cyan-400 uppercase block font-bold">Physical Vacant</span>
            <span className="text-lg font-bold text-cyan-300">
              {capacitySnapshot?.physical_vacant_count ?? selectedJob?.final_vacant_count ?? 0}
            </span>
          </div>
          <div className="p-2 rounded-lg bg-rose-950/40 border border-rose-500/40">
            <span className="text-[10px] text-rose-400 uppercase block font-bold">Occupied</span>
            <span className="text-lg font-bold text-rose-300">
              {capacitySnapshot?.occupied_count ?? selectedJob?.final_occupied_count ?? 0}
            </span>
          </div>
          <div className="p-2 rounded-lg bg-fuchsia-950/40 border border-fuchsia-500/40">
            <span className="text-[10px] text-fuchsia-400 uppercase block font-bold">Hazard Blocked</span>
            <span className="text-lg font-bold text-fuchsia-300">
              {capacitySnapshot?.hazard_blocked_count ?? 0}
            </span>
          </div>
          <div className="p-2 rounded-lg bg-orange-950/40 border border-orange-500/40">
            <span className="text-[10px] text-orange-400 uppercase block font-bold">Approach Blocked</span>
            <span className="text-lg font-bold text-orange-300">
              {capacitySnapshot?.approach_blocked_count ?? 0}
            </span>
          </div>
          <div className="p-2 rounded-lg bg-yellow-950/40 border border-yellow-500/40">
            <span className="text-[10px] text-yellow-400 uppercase block font-bold">Vacant Unassessed</span>
            <span className="text-lg font-bold text-yellow-300">
              {capacitySnapshot?.vacant_unassessed_count ?? 0}
            </span>
          </div>
          <div className="p-2 rounded-lg bg-slate-900/60 border border-slate-700/60">
            <span className="text-[10px] text-slate-400 uppercase block font-bold">Unknown / Occluded</span>
            <span className="text-lg font-bold text-slate-300">
              {(capacitySnapshot?.unknown_count ?? 0) + (capacitySnapshot?.occluded_count ?? 0)}
            </span>
          </div>
        </div>

        {/* Visual Language Legend */}
        <div className="p-2 rounded-lg bg-command-surface border border-command-border space-y-1.5 font-mono text-[10px]">
          <span className="text-command-text font-bold uppercase tracking-wider block">
            Safe Usable Capacity Policy Color Language:
          </span>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-1.5 text-center">
            <div className="p-1 rounded bg-emerald-950/60 border border-emerald-500/40 text-emerald-300 flex items-center space-x-1 justify-center">
              <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />
              <span>GREEN = USABLE</span>
            </div>
            <div className="p-1 rounded bg-rose-950/60 border border-rose-500/40 text-rose-300 flex items-center space-x-1 justify-center">
              <span className="w-2 h-2 rounded-full bg-rose-500 inline-block" />
              <span>RED = OCCUPIED</span>
            </div>
            <div className="p-1 rounded bg-fuchsia-950/60 border border-fuchsia-500/40 text-fuchsia-300 flex items-center space-x-1 justify-center">
              <span className="w-2 h-2 rounded-full bg-fuchsia-500 inline-block" />
              <span>MAGENTA = HAZARD</span>
            </div>
            <div className="p-1 rounded bg-orange-950/60 border border-orange-500/40 text-orange-300 flex items-center space-x-1 justify-center">
              <span className="w-2 h-2 rounded-full bg-orange-500 inline-block" />
              <span>ORANGE = APPROACH</span>
            </div>
            <div className="p-1 rounded bg-yellow-950/60 border border-yellow-500/40 text-yellow-300 flex items-center space-x-1 justify-center">
              <span className="w-2 h-2 rounded-full bg-yellow-400 inline-block" />
              <span>YELLOW = UNASSESSED</span>
            </div>
            <div className="p-1 rounded bg-slate-900/80 border border-slate-700/60 text-slate-300 flex items-center space-x-1 justify-center">
              <span className="w-2 h-2 rounded-full bg-slate-400 inline-block" />
              <span>GREY = UNKNOWN</span>
            </div>
          </div>
        </div>

        {/* Audit Provenance & Snapshot Hash Details */}
        {capacitySnapshot && (
          <div className="p-2 rounded bg-command-surface/60 border border-command-border flex flex-wrap items-center justify-between text-[10px] font-mono text-command-muted gap-1">
            <div className="flex items-center space-x-2">
              <Fingerprint className="w-3.5 h-3.5 text-radar-bright" />
              <span>Snapshot SHA: <strong className="text-white">{capacitySnapshot.snapshot_sha256.slice(0, 16)}...</strong></span>
            </div>
            <div>
              <span>Config SHA: <strong className="text-white">{capacitySnapshot.policy_config_sha256.slice(0, 16)}...</strong></span>
            </div>
            <div>
              <span>Evaluated: <strong className="text-white">{new Date(capacitySnapshot.evaluated_at).toLocaleTimeString()}</strong></span>
            </div>
          </div>
        )}
      </div>

      {/* 3. Video Upload and Safe Job Submission */}
      <div className="p-3.5 rounded-xl glass-panel border border-command-border space-y-3">
        <div className="flex items-center justify-between border-b border-command-border pb-2">
          <div className="flex items-center space-x-2">
            <Video className="w-4 h-4 text-radar-bright" />
            <span className="font-bold text-xs text-command-text">
              Run Parking Occupancy Pipeline
            </span>
          </div>
          <span className="text-[10px] font-mono text-command-muted">Phase 2C Pipeline</span>
        </div>

        {/* Local Processing and Privacy Notice */}
        <div className="p-2 rounded-lg bg-command-surface/50 border border-command-border text-[11px] font-mono text-command-muted flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Lock className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
            <span>100% On-Premise Execution · Media remains confined to local disk</span>
          </div>
          <span className="text-[10px] text-command-muted">Max: 200MB (MP4/MOV/AVI)</span>
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
                <p className="text-[10px] font-mono text-command-muted">H.264 MP4, QuickTime MOV, or AVI</p>
              </div>
            )}
          </div>

          {/* Explicit Operator Confirmation Checkbox */}
          {selectedVideoFile && (
            <div
              onClick={() => setOperatorConfirmationAcknowledged(!operatorConfirmationAcknowledged)}
              className={`p-2.5 rounded-lg border flex items-start space-x-2 cursor-pointer transition-all ${
                operatorConfirmationAcknowledged
                  ? 'bg-command-elevated border-radar-bright text-command-text'
                  : 'bg-command-surface border-command-border text-command-muted hover:text-white'
              }`}
            >
              {operatorConfirmationAcknowledged ? (
                <CheckSquare className="w-4 h-4 text-radar-bright shrink-0 mt-0.5" />
              ) : (
                <Square className="w-4 h-4 text-command-muted shrink-0 mt-0.5" />
              )}
              <div className="text-[11px] font-mono leading-tight">
                <span>
                  I confirm this footage corresponds to stationary camera{' '}
                  <strong className="text-white">{cameraName || cameraId}</strong> with active verified layout.
                </span>
              </div>
            </div>
          )}

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
              disabled={isSubmitting || !selectedVideoFile || !cameraId || !operatorConfirmationAcknowledged || isGateBlocked}
              className={`flex-1 py-2 rounded-lg font-bold text-xs font-mono flex items-center justify-center space-x-2 transition-all shadow-radar ${
                selectedVideoFile && cameraId && operatorConfirmationAcknowledged && !isGateBlocked && !isSubmitting
                  ? 'bg-radar-bright hover:bg-radar-green text-command-bg'
                  : 'bg-command-elevated text-command-muted border border-command-border cursor-not-allowed opacity-60'
              }`}
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Submitting Video...</span>
                </>
              ) : isGateBlocked ? (
                <>
                  <Ban className="w-3.5 h-3.5 text-rose-400" />
                  <span>Processing Prohibited (Gate BLOCKED)</span>
                </>
              ) : !operatorConfirmationAcknowledged && selectedVideoFile ? (
                <>
                  <Lock className="w-3.5 h-3.5" />
                  <span>Acknowledge Confirmation to Proceed</span>
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

      {/* 4. Active Job Execution & Progress */}
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
                  'The uploaded video failed stability verification. Recalibration or a stationary camera recording is required before occupancy can be computed.'}
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
                  AI-generated operational evidence — human review required. Work orders require physical site inspection.
                </span>
              </div>

              {/* Synchronized Visual Legend */}
              <div className="p-2.5 rounded-lg bg-command-surface border border-command-border space-y-1.5 font-mono text-[10px]">
                <span className="text-command-text font-bold uppercase tracking-wider block">
                  Synchronized Visual Overlay Legend:
                </span>
                <div className="grid grid-cols-4 gap-1.5">
                  <div className="p-1 rounded bg-rose-950/60 border border-rose-500/40 text-rose-300 flex items-center space-x-1 justify-center">
                    <span className="w-2 h-2 rounded-full bg-rose-500 inline-block" />
                    <span>RED = OCCUPIED</span>
                  </div>
                  <div className="p-1 rounded bg-emerald-950/60 border border-emerald-500/40 text-emerald-300 flex items-center space-x-1 justify-center">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />
                    <span>GREEN = VACANT</span>
                  </div>
                  <div className="p-1 rounded bg-amber-950/60 border border-amber-500/40 text-amber-300 flex items-center space-x-1 justify-center">
                    <span className="w-2 h-2 rounded-full bg-amber-500 inline-block" />
                    <span>AMBER = OCCLUDED</span>
                  </div>
                  <div className="p-1 rounded bg-slate-900/80 border border-slate-700/60 text-slate-300 flex items-center space-x-1 justify-center">
                    <span className="w-2 h-2 rounded-full bg-slate-400 inline-block" />
                    <span>GREY = UNKNOWN</span>
                  </div>
                </div>
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

              {/* 4 Download Action Buttons */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 font-mono text-xs">
                {selectedJob.has_annotated_video && (
                  <a
                    href={getOccupancyVideoUrl(selectedJob.id)}
                    download={`annotated_${selectedJob.id}.mp4`}
                    className="py-2 px-2.5 rounded-lg bg-command-elevated hover:bg-command-elevated/80 border border-command-border text-radar-bright font-bold flex items-center justify-center space-x-1 transition-all shadow-sm truncate"
                  >
                    <Download className="w-3.5 h-3.5 shrink-0" />
                    <span>Annotated MP4</span>
                  </a>
                )}
                {selectedJob.has_manifest && (
                  <a
                    href={getOccupancyManifestUrl(selectedJob.id)}
                    download={`manifest_${selectedJob.id}.json`}
                    className="py-2 px-2.5 rounded-lg bg-command-elevated hover:bg-command-elevated/80 border border-command-border text-cyan-300 font-bold flex items-center justify-center space-x-1 transition-all shadow-sm truncate"
                  >
                    <FileJson className="w-3.5 h-3.5 shrink-0" />
                    <span>Manifest JSON</span>
                  </a>
                )}
                {selectedJob.has_timeline && (
                  <a
                    href={getOccupancyTimelineUrl(selectedJob.id)}
                    download={`timeline_${selectedJob.id}.jsonl`}
                    className="py-2 px-2.5 rounded-lg bg-command-elevated hover:bg-command-elevated/80 border border-command-border text-amber-300 font-bold flex items-center justify-center space-x-1 transition-all shadow-sm truncate"
                  >
                    <FileSpreadsheet className="w-3.5 h-3.5 shrink-0" />
                    <span>Timeline JSONL</span>
                  </a>
                )}
                <a
                  href={getOccupancySummaryUrl(selectedJob.id)}
                  download={`summary_${selectedJob.id}.json`}
                  className="py-2 px-2.5 rounded-lg bg-command-elevated hover:bg-command-elevated/80 border border-command-border text-emerald-300 font-bold flex items-center justify-center space-x-1 transition-all shadow-sm truncate"
                >
                  <FileText className="w-3.5 h-3.5 shrink-0" />
                  <span>Summary JSON</span>
                </a>
              </div>

              {/* Filterable Bay List & Evidence */}
              <div className="space-y-2 border-t border-command-border pt-3">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <div className="flex items-center space-x-1.5">
                    <Filter className="w-3.5 h-3.5 text-command-muted" />
                    <span className="font-bold text-xs text-command-text">Bay Safe Usable Capacity Ledger</span>
                  </div>
                  {/* Bay Filter Tabs */}
                  <div className="flex flex-wrap bg-command-surface rounded p-0.5 text-[10px] font-mono border border-command-border gap-0.5">
                    {(
                      [
                        { key: 'ALL', label: 'ALL' },
                        { key: 'USABLE_AVAILABLE', label: 'USABLE' },
                        { key: 'OCCUPIED', label: 'OCCUPIED' },
                        { key: 'HAZARD_BLOCKED', label: 'HAZARD' },
                        { key: 'APPROACH_BLOCKED', label: 'APPROACH' },
                        { key: 'VACANT_UNASSESSED', label: 'UNASSESSED' },
                        { key: 'UNKNOWN', label: 'UNKNOWN' },
                      ] as const
                    ).map((mode) => (
                      <button
                        key={mode.key}
                        onClick={() => setBayFilter(mode.key)}
                        className={`px-1.5 py-0.5 rounded transition-all ${
                          bayFilter === mode.key
                            ? 'bg-command-elevated text-white font-bold'
                            : 'text-command-muted hover:text-white'
                        }`}
                      >
                        {mode.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="max-h-64 overflow-y-auto space-y-1.5 pr-1 text-xs">
                  {filteredBays.length === 0 ? (
                    <div className="p-3 text-center text-command-muted text-xs font-mono">
                      No bays match current filter.
                    </div>
                  ) : (
                    filteredBays.map(([label, item]) => {
                      const isInspected = inspectedBayLabel === label;
                      const dec = capacitySnapshot?.decisions[label];
                      const bayHazards = hazards.filter(
                        (h) => h.parking_space_id === dec?.bay_id || h.parking_space_id === label
                      );
                      const pendingSuggestions = bayHazards.filter(
                        (h) => h.review_state === 'UNREVIEWED' || h.review_state === 'NEEDS_REVIEW'
                      );

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
                              <span className="text-[9px] font-mono px-1 py-0.2 rounded bg-slate-800 text-slate-400 border border-slate-700">
                                Physical: {item.final_state}
                              </span>
                            </div>
                            <div className="flex items-center space-x-1.5">
                              {dec ? (
                                <span
                                  className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold ${getCapacityBadgeStyle(
                                    dec.capacity_state
                                  )}`}
                                >
                                  {dec.capacity_state}
                                </span>
                              ) : (
                                <span
                                  className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold ${
                                    item.final_state === 'OCCUPIED'
                                      ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                                      : 'bg-slate-700/30 text-slate-400 border border-slate-600/30'
                                  }`}
                                >
                                  {item.final_state}
                                </span>
                              )}
                            </div>
                          </div>

                          {/* Inspection, reason codes, and AI suggestion notices */}
                          <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[9px] font-mono">
                            {dec && (
                              <>
                                {dec.pavement_inspection_fresh && dec.inspection_age_seconds != null ? (
                                  <span className="text-emerald-400 bg-emerald-950/40 px-1 py-0.5 rounded border border-emerald-800/40">
                                    Pavement Fresh ({Math.round(dec.inspection_age_seconds / 60)}m ago)
                                  </span>
                                ) : (
                                  <span className="text-amber-400 bg-amber-950/40 px-1 py-0.5 rounded border border-amber-800/40 font-bold">
                                    No current pavement inspection
                                  </span>
                                )}

                                {dec.reason_codes.map((rc) => (
                                  <span
                                    key={rc}
                                    className="text-command-muted bg-command-bg px-1 py-0.5 rounded border border-command-border"
                                  >
                                    {rc}
                                  </span>
                                ))}
                              </>
                            )}

                            {pendingSuggestions.length > 0 && (
                              <span className="text-cyan-300 bg-cyan-950/40 px-1 py-0.5 rounded border border-cyan-800/40 flex items-center space-x-0.5">
                                <Info className="w-2.5 h-2.5" />
                                <span>{pendingSuggestions.length} AI Suggestion(s) Pending Review</span>
                              </span>
                            )}
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>

                {/* Selected Bay Evidence & Hazard Management Drawer */}
                {inspectedBay && (
                  <div className="p-3 rounded-lg bg-command-elevated border border-command-border text-xs space-y-3">
                    <div className="flex items-center justify-between border-b border-command-border pb-1.5">
                      <div className="flex items-center space-x-2">
                        <span className="font-bold text-radar-bright">
                          Bay Inspection: {inspectedBay.operator_label}
                        </span>
                        {inspectedDecision && (
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold ${getCapacityBadgeStyle(
                              inspectedDecision.capacity_state
                            )}`}
                          >
                            {inspectedDecision.capacity_state}
                          </span>
                        )}
                      </div>
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
                        <span>Physical Occupancy:</span>{' '}
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
                        <span>Detection Confidence:</span>{' '}
                        <strong className="text-white">
                          {(inspectedBay.final_confidence * 100).toFixed(1)}%
                        </strong>
                      </div>
                      <div>
                        <span>Transitions Count:</span>{' '}
                        <strong className="text-white">{inspectedBay.transitions_count}</strong>
                      </div>
                      {inspectedBay.last_vehicle_track_id && (
                        <div className="col-span-2">
                          <span>Last Contributing Vehicle Track:</span>{' '}
                          <strong className="text-cyan-400">
                            Track #{inspectedBay.last_vehicle_track_id}
                          </strong>
                        </div>
                      )}
                      {inspectedDecision && (
                        <div className="col-span-2 space-y-1 pt-1 border-t border-command-border">
                          <div>
                            <span>Inspection Freshness:</span>{' '}
                            <strong
                              className={
                                inspectedDecision.pavement_inspection_fresh
                                  ? 'text-emerald-400'
                                  : 'text-amber-400'
                              }
                            >
                              {inspectedDecision.pavement_inspection_fresh
                                ? `Valid (${Math.round(
                                    (inspectedDecision.inspection_age_seconds || 0) / 60
                                  )} min old)`
                                : 'No Current Pavement Inspection'}
                            </strong>
                          </div>
                          <div>
                            <span>Policy Reason Codes:</span>{' '}
                            <span className="text-command-text">
                              {inspectedDecision.reason_codes.join(', ') || 'NONE'}
                            </span>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Associated Hazards & Reviewer Actions */}
                    <div className="space-y-2 pt-1 border-t border-command-border">
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] font-bold text-command-text flex items-center space-x-1">
                          <ShieldAlert className="w-3 h-3 text-fuchsia-400" />
                          <span>Pavement Hazard Associations ({inspectedBayHazards.length})</span>
                        </span>
                        <button
                          onClick={() => {
                            setNewHazardBayId(
                              inspectedDecision?.bay_id || inspectedBay.operator_label
                            );
                            setNewHazardTargetType('BAY');
                            setShowAddHazardModal(true);
                          }}
                          className="px-2 py-0.5 rounded bg-command-surface hover:bg-command-surface/80 border border-command-border text-fuchsia-300 font-bold text-[10px] flex items-center space-x-1"
                        >
                          <Plus className="w-2.5 h-2.5" />
                          <span>Link Hazard</span>
                        </button>
                      </div>

                      {inspectedBayHazards.length === 0 ? (
                        <div className="text-[10px] font-mono text-command-muted p-2 rounded bg-command-surface/50 border border-command-border">
                          No pavement hazards linked to this bay.
                        </div>
                      ) : (
                        <div className="space-y-1.5">
                          {inspectedBayHazards.map((hazard) => (
                            <div
                              key={hazard.id}
                              className="p-2 rounded bg-command-surface border border-command-border space-y-1 text-[10px] font-mono"
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex items-center space-x-1.5">
                                  <span className="font-bold text-white">{hazard.hazard_label}</span>
                                  <span className="px-1 py-0.2 rounded bg-slate-800 text-slate-300 border border-slate-700">
                                    {hazard.severity_label}
                                  </span>
                                  <span className="px-1 py-0.2 rounded bg-slate-800 text-slate-400 border border-slate-700">
                                    {hazard.target_type}
                                  </span>
                                </div>
                                <div className="flex items-center space-x-1">
                                  <span
                                    className={`px-1.5 py-0.2 rounded font-bold ${
                                      hazard.review_state === 'CONFIRMED'
                                        ? 'bg-emerald-500/20 text-emerald-400'
                                        : hazard.review_state === 'UNREVIEWED' || hazard.review_state === 'NEEDS_REVIEW'
                                        ? 'bg-cyan-500/20 text-cyan-300'
                                        : 'bg-slate-700/40 text-slate-400'
                                    }`}
                                  >
                                    {hazard.review_state}
                                  </span>
                                  <span
                                    className={`px-1.5 py-0.2 rounded font-bold ${
                                      hazard.lifecycle_state === 'ACTIVE'
                                        ? 'bg-rose-500/20 text-rose-400'
                                        : hazard.lifecycle_state === 'MITIGATED'
                                        ? 'bg-amber-500/20 text-amber-300'
                                        : 'bg-slate-700/40 text-slate-400'
                                    }`}
                                  >
                                    {hazard.lifecycle_state}
                                  </span>
                                </div>
                              </div>

                              {(hazard.review_state === 'UNREVIEWED' || hazard.review_state === 'NEEDS_REVIEW') && (
                                <div className="p-1 rounded bg-cyan-950/30 border border-cyan-800/40 text-cyan-300 text-[9px] flex items-center justify-between">
                                  <span>Unverified AI Suggestion (does not affect capacity until verified)</span>
                                  <div className="flex space-x-1">
                                    <button
                                      onClick={() => handleReviewHazard(hazard.id, 'CONFIRMED', hazard.version)}
                                      className="px-1.5 py-0.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white font-bold"
                                    >
                                      Confirm
                                    </button>
                                    <button
                                      onClick={() => handleReviewHazard(hazard.id, 'REJECTED', hazard.version)}
                                      className="px-1.5 py-0.5 rounded bg-slate-700 hover:bg-slate-600 text-white"
                                    >
                                      Reject
                                    </button>
                                  </div>
                                </div>
                              )}

                              {hazard.review_state === 'CONFIRMED' && (
                                <div className="flex items-center justify-between pt-0.5">
                                  <span className="text-[9px] text-command-muted">
                                    v{hazard.version} · Reviewer: {hazard.reviewed_by || hazard.created_by}
                                  </span>
                                  <div className="flex space-x-1">
                                    {hazard.lifecycle_state === 'ACTIVE' ? (
                                      <>
                                        <button
                                          onClick={() => handleLifecycleHazard(hazard.id, 'MITIGATED', hazard.version)}
                                          className="px-1.5 py-0.5 rounded bg-amber-600 hover:bg-amber-500 text-white font-bold"
                                        >
                                          Mitigate
                                        </button>
                                        <button
                                          onClick={() => handleLifecycleHazard(hazard.id, 'RESOLVED', hazard.version)}
                                          className="px-1.5 py-0.5 rounded bg-slate-700 hover:bg-slate-600 text-white"
                                        >
                                          Resolve
                                        </button>
                                      </>
                                    ) : (
                                      <button
                                        onClick={() => handleLifecycleHazard(hazard.id, 'ACTIVE', hazard.version)}
                                        className="px-1.5 py-0.5 rounded bg-rose-600 hover:bg-rose-500 text-white font-bold"
                                      >
                                        Reactivate
                                      </button>
                                    )}
                                  </div>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Provenance and Retention Controls */}
              <div className="p-2.5 rounded-lg bg-command-surface/50 border border-command-border space-y-2 font-mono text-[10px] text-command-muted">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-1 text-command-text font-bold uppercase">
                    <Fingerprint className="w-3.5 h-3.5 text-radar-bright" />
                    <span>Immutable Pipeline Provenance</span>
                  </div>
                  <button
                    onClick={() => setDeleteConfirmOpen(true)}
                    className="text-rose-400 hover:text-rose-300 font-bold flex items-center space-x-1"
                  >
                    <Trash2 className="w-3 h-3" />
                    <span>Purge Artifacts</span>
                  </button>
                </div>

                <div className="space-y-0.5 truncate">
                  <div>
                    <span>Detector SHA: </span>
                    <span className="text-cyan-300">{selectedJob.detector_checkpoint_sha256?.slice(0, 16)}...</span>
                  </div>
                  <div>
                    <span>Layout SHA: </span>
                    <span className="text-radar-bright">{selectedJob.layout_canonical_sha256?.slice(0, 16)}...</span>
                  </div>
                  <div>
                    <span>Config SHA: </span>
                    <span className="text-amber-300">{selectedJob.occupancy_config_sha256?.slice(0, 16)}...</span>
                  </div>
                  <div className="text-[9px] text-command-muted/70 pt-0.5">
                    Storage: /media/parking_jobs/{selectedJob.id}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 5. Previous Occupancy Jobs History */}
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

      {/* Safe Delete Confirmation Modal */}
      {deleteConfirmOpen && selectedJob && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-command-bg border border-command-border rounded-xl p-5 max-w-md w-full space-y-4 shadow-2xl">
            <div className="flex items-center space-x-2 text-rose-400 font-bold text-sm">
              <AlertTriangle className="w-5 h-5" />
              <span>Confirm Fail-Closed Artifact Deletion</span>
            </div>
            <p className="text-xs text-command-muted font-mono leading-relaxed">
              Are you sure you want to delete job <strong className="text-white">{selectedJob.id.slice(0, 8)}</strong> and its associated annotated video and timeline artifacts?
              This operation purges local disk artifacts and records an immutable tombstone audit event.
            </p>

            <div className="space-y-3 pt-1">
              <div>
                <label className="block text-[11px] font-mono text-command-muted mb-1">
                  Operator Identity Assertion:
                </label>
                <input
                  type="text"
                  value={deleteOperatorLabel}
                  onChange={(e) => setDeleteOperatorLabel(e.target.value)}
                  placeholder="e.g. operator_name"
                  className="w-full bg-command-surface border border-command-border rounded-lg px-2.5 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-radar-bright"
                />
              </div>

              <div>
                <label className="block text-[11px] font-mono text-command-muted mb-1">
                  Deletion & Purge Reason:
                </label>
                <input
                  type="text"
                  value={deleteReason}
                  onChange={(e) => setDeleteReason(e.target.value)}
                  placeholder="e.g. Routine retention policy purge"
                  className="w-full bg-command-surface border border-command-border rounded-lg px-2.5 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-radar-bright"
                />
              </div>
            </div>

            <div className="flex items-center justify-end space-x-2 pt-2">
              <button
                onClick={() => setDeleteConfirmOpen(false)}
                disabled={isDeleting}
                className="py-1.5 px-3 rounded-lg bg-command-surface border border-command-border text-command-text text-xs font-mono"
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteJob}
                disabled={isDeleting || !deleteOperatorLabel.trim() || deleteReason.trim().length < 5}
                className="py-1.5 px-4 rounded-lg bg-rose-600 hover:bg-rose-700 disabled:opacity-50 text-white font-bold text-xs font-mono flex items-center space-x-1.5 shadow-sm"
              >
                {isDeleting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
                <span>{isDeleting ? 'Deleting...' : 'Confirm Purge'}</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Add Hazard Association Modal */}
      {showAddHazardModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-command-bg border border-command-border rounded-xl p-5 max-w-md w-full space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-command-border pb-2">
              <div className="flex items-center space-x-2 text-fuchsia-400 font-bold text-sm">
                <ShieldAlert className="w-5 h-5" />
                <span>Associate Pavement Hazard</span>
              </div>
              <button
                onClick={() => setShowAddHazardModal(false)}
                className="text-command-muted hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleCreateHazard} className="space-y-3 font-mono text-xs">
              <div>
                <label className="block text-command-muted mb-1">Target Type</label>
                <div className="flex space-x-2">
                  <button
                    type="button"
                    onClick={() => setNewHazardTargetType('BAY')}
                    className={`flex-1 py-1.5 rounded border text-xs font-bold ${
                      newHazardTargetType === 'BAY'
                        ? 'bg-fuchsia-600 border-fuchsia-500 text-white'
                        : 'bg-command-surface border-command-border text-command-muted'
                    }`}
                  >
                    Bay Hazard
                  </button>
                  <button
                    type="button"
                    onClick={() => setNewHazardTargetType('APPROACH_ZONE')}
                    className={`flex-1 py-1.5 rounded border text-xs font-bold ${
                      newHazardTargetType === 'APPROACH_ZONE'
                        ? 'bg-orange-600 border-orange-500 text-white'
                        : 'bg-command-surface border-command-border text-command-muted'
                    }`}
                  >
                    Approach Zone Hazard
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-command-muted mb-1">Target Identifier (Bay Label or Space ID)</label>
                <input
                  type="text"
                  required
                  value={newHazardBayId}
                  onChange={(e) => setNewHazardBayId(e.target.value)}
                  placeholder="e.g. Bay 1 or space UUID"
                  className="w-full bg-command-surface border border-command-border rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-radar-bright"
                />
              </div>

              <div>
                <label className="block text-command-muted mb-1">Hazard Label / Description</label>
                <input
                  type="text"
                  required
                  value={newHazardLabel}
                  onChange={(e) => setNewHazardLabel(e.target.value)}
                  placeholder="e.g. Pothole / rut in bay ingress"
                  className="w-full bg-command-surface border border-command-border rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-radar-bright"
                />
              </div>

              <div>
                <label className="block text-command-muted mb-1">Severity Category</label>
                <select
                  value={newHazardSeverity}
                  onChange={(e) => setNewHazardSeverity(e.target.value)}
                  className="w-full bg-command-surface border border-command-border rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-radar-bright"
                >
                  <option value="LOW">LOW</option>
                  <option value="MEDIUM">MEDIUM</option>
                  <option value="HIGH">HIGH</option>
                  <option value="CRITICAL">CRITICAL</option>
                </select>
                <span className="text-[10px] text-command-muted block mt-0.5">
                  Severity classification based on operator review (never invented depth/structural values).
                </span>
              </div>

              <div>
                <label className="block text-command-muted mb-1">Operator Audit Notes (Optional)</label>
                <textarea
                  rows={2}
                  value={newHazardNotes}
                  onChange={(e) => setNewHazardNotes(e.target.value)}
                  placeholder="Inspection notes or visual assessment reference"
                  className="w-full bg-command-surface border border-command-border rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-radar-bright"
                />
              </div>

              <div className="flex items-center justify-end space-x-2 pt-2 border-t border-command-border">
                <button
                  type="button"
                  onClick={() => setShowAddHazardModal(false)}
                  disabled={isSubmittingHazard}
                  className="py-1.5 px-3 rounded-lg bg-command-surface border border-command-border text-command-text"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSubmittingHazard || !newHazardBayId.trim() || !newHazardLabel.trim()}
                  className="py-1.5 px-4 rounded-lg bg-fuchsia-600 hover:bg-fuchsia-700 disabled:opacity-50 text-white font-bold flex items-center space-x-1.5 shadow-sm"
                >
                  {isSubmittingHazard ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Plus className="w-3.5 h-3.5" />
                  )}
                  <span>Create Association</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Record Pavement Inspection Modal */}
      {showInspectionModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-command-bg border border-command-border rounded-xl p-5 max-w-md w-full space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-command-border pb-2">
              <div className="flex items-center space-x-2 text-cyan-400 font-bold text-sm">
                <Layers className="w-5 h-5" />
                <span>Record Pavement Inspection</span>
              </div>
              <button
                onClick={() => setShowInspectionModal(false)}
                className="text-command-muted hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleRecordInspection} className="space-y-3 font-mono text-xs">
              <p className="text-[11px] text-command-muted leading-relaxed">
                Recording an inspection certifies that the parking pavement was assessed by human or mobile inspection. Safe Usable Capacity policy requires fresh inspection within the configured limit (3600s) to declare empty bays as USABLE_AVAILABLE.
              </p>

              <div>
                <label className="block text-command-muted mb-1">Inspector Identity Label</label>
                <input
                  type="text"
                  required
                  value={newInspectorLabel}
                  onChange={(e) => setNewInspectorLabel(e.target.value)}
                  placeholder="e.g. site_inspector_john"
                  className="w-full bg-command-surface border border-command-border rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-radar-bright"
                />
              </div>

              <div>
                <label className="block text-command-muted mb-1">Inspection Session ID (Optional)</label>
                <input
                  type="text"
                  value={newInspectionSessionId}
                  onChange={(e) => setNewInspectionSessionId(e.target.value)}
                  placeholder="e.g. survey_session_2026_09_13"
                  className="w-full bg-command-surface border border-command-border rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-radar-bright"
                />
              </div>

              <div>
                <label className="block text-command-muted mb-1">Inspection Notes (Optional)</label>
                <textarea
                  rows={2}
                  value={newInspectionNotes}
                  onChange={(e) => setNewInspectionNotes(e.target.value)}
                  placeholder="Ground survey notes, weather conditions, inspection tool details"
                  className="w-full bg-command-surface border border-command-border rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-radar-bright"
                />
              </div>

              <div className="flex items-center justify-end space-x-2 pt-2 border-t border-command-border">
                <button
                  type="button"
                  onClick={() => setShowInspectionModal(false)}
                  disabled={isSubmittingInspection}
                  className="py-1.5 px-3 rounded-lg bg-command-surface border border-command-border text-command-text"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSubmittingInspection || !newInspectorLabel.trim()}
                  className="py-1.5 px-4 rounded-lg bg-cyan-600 hover:bg-cyan-700 disabled:opacity-50 text-white font-bold flex items-center space-x-1.5 shadow-sm"
                >
                  {isSubmittingInspection ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Layers className="w-3.5 h-3.5" />
                  )}
                  <span>Record Inspection</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
