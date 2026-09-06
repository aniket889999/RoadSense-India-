'use client';

import React, { useState } from 'react';
import { AlertTriangle, CheckCircle2, ShieldCheck, X } from 'lucide-react';

interface LayoutVerificationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onVerify: (operatorLabel: string, note?: string) => Promise<void>;
  revisionNumber: number;
  spacesCount: number;
}

export function LayoutVerificationModal({
  isOpen,
  onClose,
  onVerify,
  revisionNumber,
  spacesCount,
}: LayoutVerificationModalProps) {
  const [operatorLabel, setOperatorLabel] = useState('');
  const [confirmed, setConfirmed] = useState(false);
  const [note, setNote] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!operatorLabel.trim()) {
      setError('Operator label is required.');
      return;
    }
    if (!confirmed) {
      setError('You must explicitly confirm ground-truth verification.');
      return;
    }

    setIsSubmitting(true);
    setError(null);
    try {
      await onVerify(operatorLabel.trim(), note.trim() || undefined);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to verify layout.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="w-full max-w-lg rounded-xl glass-panel-elevated border border-command-border bg-command-surface p-6 shadow-2xl space-y-5">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-command-border pb-3">
          <div className="flex items-center space-x-2 text-radar-bright">
            <ShieldCheck className="w-5 h-5" />
            <h2 className="text-sm font-bold uppercase tracking-wider font-mono">
              Verify Parking Layout (Rev #{revisionNumber})
            </h2>
          </div>
          <button onClick={onClose} className="text-command-muted hover:text-command-text">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Warning Notice */}
        <div className="p-3.5 rounded-lg bg-accent-amber/10 border border-accent-amber/30 text-accent-amber text-xs space-y-1.5 font-mono">
          <div className="flex items-center space-x-1.5 font-bold">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>Operational Integrity Assertion</span>
          </div>
          <p className="leading-relaxed text-[11px] text-accent-amber/90">
            Verifying this layout promotes it to <strong>VERIFIED</strong> status and computes its canonical SHA-256 fingerprint. Once verified, geometry cannot be mutated in place. Any active prior verified layout for this camera will be superseded.
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4 text-xs font-mono">
          {error && (
            <div className="p-2.5 rounded bg-accent-red/20 border border-accent-red/40 text-accent-red text-xs">
              {error}
            </div>
          )}

          <div className="space-y-1.5">
            <label className="text-command-text font-bold">
              Local Operator Name / Identifier <span className="text-accent-red">*</span>
            </label>
            <input
              type="text"
              value={operatorLabel}
              onChange={(e) => setOperatorLabel(e.target.value)}
              placeholder="e.g., Alice (Site Operations Lead)"
              className="w-full px-3 py-2 rounded bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-radar-bright"
            />
            <span className="text-[10px] text-command-muted">
              Note: This is a local operator assertion stamped in the immutable audit ledger.
            </span>
          </div>

          <div className="space-y-1.5">
            <label className="text-command-text font-bold">Verification Notes (Optional)</label>
            <textarea
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="e.g., Verified 12 bays against ground-truth markings on Level 1 North."
              rows={2}
              className="w-full px-3 py-2 rounded bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-radar-bright"
            />
          </div>

          <div className="pt-2">
            <label className="flex items-start space-x-2.5 cursor-pointer">
              <input
                type="checkbox"
                checked={confirmed}
                onChange={(e) => setConfirmed(e.target.checked)}
                className="mt-0.5 rounded border-command-border bg-command-bg text-radar-bright focus:ring-0"
              />
              <span className="text-[11px] text-command-text leading-tight">
                I explicitly acknowledge that I have manually verified the {spacesCount} parking space boundaries and approach zones against real ground truth.
              </span>
            </label>
          </div>

          <div className="flex justify-end space-x-3 pt-3 border-t border-command-border">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded bg-command-elevated border border-command-border text-command-muted hover:text-command-text"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="px-5 py-2 rounded bg-radar-bright hover:bg-radar-green text-command-bg font-bold shadow-radar flex items-center space-x-1.5 disabled:opacity-50"
            >
              <CheckCircle2 className="w-4 h-4" />
              <span>{isSubmitting ? 'Verifying...' : 'Confirm & Verify'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
