'use client';

import React, { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2, ChevronRight, Circle, Loader2, RefreshCw } from 'lucide-react';
import {
  Camera,
  CameraOperationalGate,
  CapacitySnapshot,
  ParkingLayoutRevision,
  ParkingOccupancyJob,
  Site,
} from '../lib/types';
import { getCameraCapacitySnapshot, getCameraOperationalGate, listOccupancyJobs } from '../lib/parkingApi';

type RightTab = 'editor' | 'stability' | 'occupancy';
type StepState = 'done' | 'current' | 'blocked' | 'todo';

interface ParkingSetupGuideProps {
  site: Site | null;
  camera: Camera | null;
  layout: ParkingLayoutRevision | null;
  layouts: ParkingLayoutRevision[];
  refreshKey: number;
  onGoTo: (tab: RightTab) => void;
  onCreateSite: () => void;
  onCreateCamera: () => void;
  onUploadReference: () => void;
}

interface Step {
  key: string;
  title: string;
  state: StepState;
  detail: string;
  action?: { label: string; run: () => void };
}

const ACTIVE_JOB = new Set(['QUEUED', 'VALIDATING', 'DETECTING', 'TRACKING', 'CLASSIFYING_OCCUPANCY', 'RENDERING', 'ENCODING']);

function humanReason(code: string): string {
  return code
    .replace(/_/g, ' ')
    .toLowerCase()
    .replace(/^\w/, (c) => c.toUpperCase());
}

export function ParkingSetupGuide({
  site, camera, layout, layouts, refreshKey, onGoTo, onCreateSite, onCreateCamera, onUploadReference,
}: ParkingSetupGuideProps) {
  const [gate, setGate] = useState<CameraOperationalGate | null>(null);
  const [jobs, setJobs] = useState<ParkingOccupancyJob[]>([]);
  const [capacity, setCapacity] = useState<CapacitySnapshot | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    if (!camera) {
      setGate(null); setJobs([]); setCapacity(null);
      return;
    }
    setLoading(true);
    const [g, j, c] = await Promise.allSettled([
      getCameraOperationalGate(camera.id),
      listOccupancyJobs(camera.id),
      getCameraCapacitySnapshot(camera.id),
    ]);
    setGate(g.status === 'fulfilled' ? g.value : null);
    setJobs(j.status === 'fulfilled' ? j.value : []);
    setCapacity(c.status === 'fulfilled' ? c.value : null);
    setLoading(false);
  }, [camera]);

  useEffect(() => { load(); }, [load, refreshKey, layout?.id, layout?.status]);

  // Poll while an occupancy job is running so the strip reflects progress.
  const running = jobs.find((j) => ACTIVE_JOB.has(j.status));
  useEffect(() => {
    if (!running) return;
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, [running, load]);

  const activeBays = (layout?.parking_spaces ?? []).filter((s) => s.active !== false).length;
  const verifiedLayout = layouts.find((l) => l.status === 'VERIFIED') ?? null;
  const latestDone = jobs
    .filter((j) => j.status === 'COMPLETE')
    .sort((a, b) => (b.completed_at ?? '').localeCompare(a.completed_at ?? ''))[0];
  const gateAllowed = gate?.operational_gate === 'ALLOWED' && gate.is_fresh;

  const steps: Step[] = [
    {
      key: 'site',
      title: 'Site & camera',
      state: site && camera ? 'done' : 'current',
      detail: site && camera ? `${site.name} · ${camera.name}` : !site ? 'Create the parking site first.' : 'Add the fixed camera that watches the lot.',
      action: !site ? { label: 'New site', run: onCreateSite } : !camera ? { label: 'New camera', run: onCreateCamera } : undefined,
    },
    {
      key: 'reference',
      title: 'Reference frame',
      state: !camera ? 'todo' : camera.reference_image_sha256 ? 'done' : 'current',
      detail: camera?.reference_image_sha256
        ? `${camera.reference_width ?? '?'}×${camera.reference_height ?? '?'} px still from this camera`
        : 'Upload a still frame from the camera; bays are drawn on it.',
      action: camera && !camera.reference_image_sha256 ? { label: 'Upload frame', run: onUploadReference } : undefined,
    },
    {
      key: 'bays',
      title: 'Draw bays',
      state: !camera?.reference_image_sha256 ? 'todo' : activeBays > 0 || verifiedLayout ? 'done' : 'current',
      detail: activeBays > 0 ? `${activeBays} bay${activeBays === 1 ? '' : 's'} in revision ${layout?.revision_number}` : 'Outline every bay (and its approach zone) on the frame.',
      action: camera?.reference_image_sha256 && !activeBays ? { label: 'Open editor', run: () => onGoTo('editor') } : undefined,
    },
    {
      key: 'verify',
      title: 'Verify layout',
      state: verifiedLayout ? 'done' : activeBays > 0 ? 'current' : 'todo',
      detail: verifiedLayout
        ? `Revision ${verifiedLayout.revision_number} verified by a reviewer`
        : layout?.status === 'PENDING_REVIEW'
        ? 'Submitted: a reviewer must verify it.'
        : 'Validate, submit, then have a second person verify.',
      action: !verifiedLayout && activeBays > 0 ? { label: 'Review layout', run: () => onGoTo('editor') } : undefined,
    },
    {
      key: 'stability',
      title: 'Camera stability',
      state: !verifiedLayout ? 'todo' : gateAllowed ? 'done' : gate ? 'blocked' : 'current',
      detail: gateAllowed
        ? 'Camera has not moved since calibration.'
        : gate?.gate_reasons?.length
        ? humanReason(gate.gate_reasons[0])
        : 'Run a stability check with a short clip before counting.',
      action: verifiedLayout && !gateAllowed ? { label: 'Check stability', run: () => onGoTo('stability') } : undefined,
    },
    {
      key: 'occupancy',
      title: 'Count occupancy',
      state: running ? 'current' : latestDone ? 'done' : gateAllowed ? 'current' : 'todo',
      detail: running
        ? `${humanReason(running.status)} · ${Math.round(running.progress_pct)}%`
        : latestDone
        ? `Last run: ${latestDone.final_occupied_count} occupied / ${latestDone.total_bays} bays`
        : 'Upload a recorded clip to count each bay over time.',
      action: gateAllowed && !running ? { label: latestDone ? 'Run again' : 'Run occupancy', run: () => onGoTo('occupancy') } : undefined,
    },
    {
      key: 'capacity',
      title: 'Safe capacity',
      state: capacity && capacity.operational_gate === 'ALLOWED' ? 'done' : latestDone ? 'current' : 'todo',
      detail: capacity
        ? `${capacity.usable_available_count} usable of ${capacity.total_bays} bays`
        : 'Combines occupancy with reviewed hazards.',
      action: latestDone ? { label: 'Open', run: () => onGoTo('occupancy') } : undefined,
    },
  ];

  const current = steps.find((s) => s.state === 'current' || s.state === 'blocked');
  const doneCount = steps.filter((s) => s.state === 'done').length;

  const kpis = capacity
    ? [
        { label: 'Bays', value: capacity.total_bays },
        { label: 'Usable now', value: capacity.usable_available_count, tone: 'text-emerald-700' },
        { label: 'Occupied', value: capacity.occupied_count, tone: 'text-rose-700' },
        { label: 'Hazard / approach blocked', value: capacity.hazard_blocked_count + capacity.approach_blocked_count, tone: 'text-amber-700' },
        { label: 'Unknown / occluded', value: capacity.unknown_count + capacity.occluded_count + capacity.inference_blocked_count },
      ]
    : latestDone
    ? [
        { label: 'Bays', value: latestDone.total_bays },
        { label: 'Vacant (last run)', value: latestDone.final_vacant_count, tone: 'text-emerald-700' },
        { label: 'Occupied (last run)', value: latestDone.final_occupied_count, tone: 'text-rose-700' },
        { label: 'Unknown / occluded', value: latestDone.final_unknown_count + latestDone.final_occluded_count },
      ]
    : null;

  return (
    <section aria-label="Parking setup progress" className="rounded-3xl bg-command-surface border border-command-border shadow-subtle p-4 lg:p-5 font-sans space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-[12px] font-medium text-brand-deep">Parking occupancy · {doneCount} of {steps.length} steps done</p>
          <h2 className="text-[20px] leading-tight font-semibold text-command-text">
            {current ? `Next: ${current.title.toLowerCase()}` : 'Lot is set up and counting'}
          </h2>
          <p className="text-[13px] text-command-muted mt-0.5">
            {current ? current.detail : 'Counts come from recorded clips; unknown or blocked evidence never counts as free.'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {current?.action && (
            <button
              onClick={current.action.run}
              className="flex items-center gap-1.5 px-4 py-2 rounded-full bg-brand hover:bg-brand-bright text-white text-[13px] font-medium shadow-brand"
            >
              {current.action.label} <ChevronRight className="w-4 h-4" />
            </button>
          )}
          <button
            onClick={load}
            disabled={!camera || loading}
            className="p-2 rounded-full bg-command-elevated hover:bg-command-border text-command-muted disabled:opacity-40"
            title="Refresh status"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
          </button>
        </div>
      </div>

      <ol className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
        {steps.map((s, i) => (
          <li key={s.key}>
            <button
              type="button"
              onClick={s.action?.run}
              disabled={!s.action}
              className={`w-full h-full text-left rounded-2xl border px-3 py-2.5 transition-colors ${
                s.state === 'done'
                  ? 'border-emerald-200 bg-emerald-50'
                  : s.state === 'blocked'
                  ? 'border-amber-200 bg-amber-50'
                  : s.state === 'current'
                  ? 'border-brand/40 bg-brand-soft'
                  : 'border-command-border bg-command-bg'
              } ${s.action ? 'hover:border-brand/60 cursor-pointer' : 'cursor-default'}`}
            >
              <div className="flex items-center gap-1.5 text-[12px] font-medium text-command-text">
                {s.state === 'done' ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                ) : s.state === 'blocked' ? (
                  <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
                ) : (
                  <Circle className={`w-4 h-4 shrink-0 ${s.state === 'current' ? 'text-brand' : 'text-command-subtle'}`} />
                )}
                <span className="text-command-muted">{i + 1}.</span> {s.title}
              </div>
              <p className="mt-1 text-[11px] leading-snug text-command-muted line-clamp-2">{s.detail}</p>
            </button>
          </li>
        ))}
      </ol>

      {kpis && (
        <dl className="grid grid-cols-2 sm:grid-cols-5 gap-2">
          {kpis.map((k) => (
            <div key={k.label} className="flex flex-col-reverse rounded-2xl bg-command-bg border border-command-border px-4 py-3">
              <dt className="text-[11px] text-command-muted mt-0.5">{k.label}</dt>
              <dd className={`text-[22px] font-semibold leading-none ${k.tone ?? 'text-command-text'}`}>{k.value}</dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}
