'use client';

import React, { useState } from 'react';
import { AlertOctagon, X } from 'lucide-react';

interface CameraCalibrationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onInvalidate: (operatorLabel: string, reason: string, note?: string) => Promise<void>;
  cameraName: string;
}

export function CameraCalibrationModal({
  isOpen,
  onClose,
  onInvalidate,
  cameraName,
}: CameraCalibrationModalProps) {
  const [operatorLabel, setOperatorLabel] = useState('');
  const [reason, setReason] = useState('');
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
    if (!reason.trim()) {
      setError('Invalidation reason is required.');
      return;
    }

    setIsSubmitting(true);
    setError(null);
    try {
      await onInvalidate(operatorLabel.trim(), reason.trim(), note.trim() || undefined);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to invalidate calibration.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="w-full max-w-lg rounded-xl glass-panel-elevated border border-accent-red/40 bg-command-surface p-6 shadow-2xl space-y-5">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-command-border pb-3">
          <div className="flex items-center space-x-2 text-accent-red">
            <AlertOctagon className="w-5 h-5" />
            <h2 className="text-sm font-bold uppercase tracking-wider font-mono">
              Invalidate Camera Calibration ({cameraName})
            </h2>
          </div>
          <button onClick={onClose} className="text-command-muted hover:text-command-text">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Warning Notice */}
        <div className="p-3.5 rounded-lg bg-accent-red/10 border border-accent-red/30 text-accent-red text-xs space-y-1.5 font-mono">
          <p className="leading-relaxed text-[11px] text-accent-red/90">
            <strong>Warning:</strong> Invalidating camera calibration immediately marks the camera as <strong>INVALIDATED</strong> and revokes all active verified layout revisions for occupancy inference.
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
              placeholder="e.g., Alice (Site Operations)"
              className="w-full px-3 py-2 rounded bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-accent-red"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-command-text font-bold">
              Invalidation Reason <span className="text-accent-red">*</span>
            </label>
            <input
              type="text"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="e.g., Camera pole shifted after storm / physical maintenance"
              className="w-full px-3 py-2 rounded bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-accent-red"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-command-text font-bold">Additional Notes (Optional)</label>
            <textarea
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Additional context on camera movement..."
              rows={2}
              className="w-full px-3 py-2 rounded bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-accent-red"
            />
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
              className="px-5 py-2 rounded bg-accent-red hover:bg-red-600 text-white font-bold shadow-lg flex items-center space-x-1.5 disabled:opacity-50"
            >
              <span>{isSubmitting ? 'Invalidating...' : 'Confirm Invalidation'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
