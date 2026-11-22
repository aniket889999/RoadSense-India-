'use client';

import React, { useState } from 'react';
import { Building2, FlaskConical, MapPin } from 'lucide-react';
import { PlateResult } from '../../lib/anprTypes';

/**
 * Registration card for one detected plate.
 *
 * Top: issuing state/UT and RTO office decoded by the API from a pinned public RTO directory.
 * Bottom (optional demo): a SIMULATED owner section generated in the browser from the plate text.
 * The simulated values are fictional placeholders, never sent to the API and never included in the
 * JSON review report or annotated PNG export. Nothing here is a registry lookup.
 */

const SIMULATED_CLASSES = ['Goods carrier (simulated)', 'Motor car (simulated)', 'Bus (simulated)', 'Tractor-trailer (simulated)'];

function stableHash(text: string): number {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function simulatedRecord(plateText: string, bhYear: number | null) {
  const hash = stableHash(plateText || 'unknown');
  return {
    owner: `Demo Owner #${String(hash % 10000).padStart(4, '0')}`,
    year: bhYear ?? 2008 + (hash % 17),
    yearSource: bhYear ? 'from the BH code on the plate' : 'simulated',
    vehicleClass: SIMULATED_CLASSES[hash % SIMULATED_CLASSES.length],
  };
}

function Row({ label, value, muted }: { label: string; value: React.ReactNode; muted?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-2.5 border-b border-white/5 last:border-0">
      <span className="text-[12px] text-command-muted shrink-0">{label}</span>
      <span className={`text-right text-[14px] ${muted ? 'text-command-muted' : 'text-white'}`}>{value}</span>
    </div>
  );
}

export function RegistrationCard({ plate }: { plate: PlateResult }) {
  const [showSimulated, setShowSimulated] = useState(true);
  const text = plate.candidate_text || plate.normalized_text || '';
  const format = plate.plate_format ?? null;
  const origin = format?.registration_origin ?? null;
  const bhYear = format?.pattern === 'BH_SERIES' && /^\d{2}/.test(text) ? 2000 + Number(text.slice(0, 2)) : null;
  const sim = simulatedRecord(text, bhYear);

  return (
    <section aria-label="Registration card" className="rounded-4xl bg-command-surface border border-white/5 overflow-hidden">
      <div className="grid grid-cols-1 lg:grid-cols-12">
        {/* Plate + issuing authority (real public directory data) */}
        <div className="lg:col-span-7 p-6 lg:p-8 space-y-6">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <p className="text-[12px] text-command-muted">Registration card · {plate.plate_id}</p>
              <p className="mt-2 font-mono text-[34px] lg:text-[44px] leading-none tracking-[0.06em] text-white break-all">
                {text || '—'}
              </p>
              <p className="mt-2 text-[12px] text-command-muted">
                OCR reading, not human verified{plate.candidate_text ? ' · cleaned candidate' : ''}
              </p>
            </div>
            {origin && (
              <span className="px-4 py-2 rounded-full bg-brand text-white text-[13px] font-medium">
                {origin.state_name?.split(' (')[0] ?? origin.state_code} · {origin.rto_code}
              </span>
            )}
          </div>

          <div>
            <div className="flex items-center gap-2 text-white text-[15px] font-medium mb-1">
              <MapPin className="w-4 h-4 text-brand" />
              <span>Issuing authority</span>
            </div>
            {format?.pattern === 'STATE_SERIES' && origin ? (
              <div>
                <Row label="State / UT" value={origin.state_name ?? origin.state_code} />
                <Row label="RTO code" value={origin.rto_code} />
                <Row
                  label="RTO office"
                  value={origin.rto_in_table ? origin.rto_office ?? 'Not listed in source' : 'RTO not in table'}
                  muted={!origin.rto_in_table || !origin.rto_office}
                />
                {origin.rto_jurisdiction && <Row label="Jurisdiction" value={origin.rto_jurisdiction} />}
                {origin.rto_notes && <Row label="Notes" value={origin.rto_notes} muted />}
              </div>
            ) : format?.pattern === 'BH_SERIES' ? (
              <Row label="Series" value="Bharat (BH) series — not tied to a state RTO" />
            ) : (
              <p className="text-[13px] text-command-muted py-2">
                The reading does not fit a supported plate pattern, so no state or RTO can be decoded. Check the crop.
              </p>
            )}
            <p className="mt-3 flex items-center gap-1.5 text-[11px] text-command-muted">
              <Building2 className="w-3.5 h-3.5" />
              Public RTO directory (Wikipedia rev {origin?.source_revision ?? 'n/a'}, CC BY-SA 4.0). Says where a code is
              issued; decoded from the unverified reading.
            </p>
          </div>
        </div>

        {/* Simulated owner section (browser-only demo) */}
        <div className="lg:col-span-5 relative p-6 lg:p-8 bg-[repeating-linear-gradient(135deg,rgba(237,68,57,0.10)_0_12px,transparent_12px_24px)] border-t lg:border-t-0 lg:border-l border-white/5">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-white text-[15px] font-medium">
              <FlaskConical className="w-4 h-4 text-brand" />
              <span>Owner record</span>
            </div>
            <label className="flex items-center gap-2 text-[12px] text-command-muted cursor-pointer select-none">
              <input
                type="checkbox"
                checked={showSimulated}
                onChange={(e) => setShowSimulated(e.target.checked)}
                className="accent-[#ED4439]"
              />
              Demo mode
            </label>
          </div>

          <div className="mt-4 inline-flex px-3 py-1 rounded-full bg-brand text-white text-[11px] font-semibold tracking-wide">
            SIMULATED — NOT A REAL RECORD
          </div>

          {showSimulated ? (
            <div className="relative mt-4">
              <span
                aria-hidden="true"
                className="pointer-events-none absolute inset-0 flex items-center justify-center text-[44px] font-semibold text-brand/15 -rotate-12 select-none"
              >
                SIMULATED
              </span>
              <Row label="Owner" value={sim.owner} />
              <Row label="Registration year" value={`${sim.year} (${sim.yearSource})`} />
              <Row label="Vehicle class" value={sim.vehicleClass} />
              <Row label="Source" value="Generated in this browser for the demo" muted />
            </div>
          ) : (
            <p className="mt-4 text-[13px] text-command-muted">Demo owner section hidden.</p>
          )}

          <p className="mt-5 text-[11px] leading-relaxed text-command-muted">
            Placeholder values derived from the reading so the demo is repeatable. RoadSense performs no
            registry (VAHAN) lookup and holds no owner data. These values are not sent anywhere and are not part of the
            JSON report or annotated PNG.
          </p>
        </div>
      </div>
    </section>
  );
}
