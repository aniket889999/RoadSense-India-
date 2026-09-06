'use client';

import React, { useState } from 'react';
import { AlertTriangle, UploadCloud, X } from 'lucide-react';

interface ReferenceImageReplaceModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirmReplace: (file: File, operatorLabel: string, reason: string) => Promise<void>;
  pendingFile: File | null;
  cameraName: string;
}

export function ReferenceImageReplaceModal({
  isOpen,
  onClose,
  onConfirmReplace,
  pendingFile,
  cameraName,
}: ReferenceImageReplaceModalProps) {
  const [operatorLabel, setOperatorLabel] = useState('');
  const [reason, setReason] = useState('');
  const [confirmed, setConfirmed] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen || !pendingFile) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!confirmed) {
      setError('You must explicitly confirm acknowledgement before replacing the reference frame.');
      return;
    }
    const cleanOp = operatorLabel.trim();
    const cleanReason = reason.trim();
    if (!cleanOp) {
      setError('Local operator label is required.');
      return;
    }
    if (!cleanReason || cleanReason.length < 3) {
      setError('Please provide a meaningful reason for replacing the reference image.');
      return;
    }

    setIsUploading(true);
    setError(null);
    try {
      await onConfirmReplace(pendingFile, cleanOp, cleanReason);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to replace reference image.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in font-mono">
      <div className="w-full max-w-lg rounded-xl glass-panel-elevated border border-accent-amber/40 shadow-2xl p-6 space-y-5">
        <div className="flex items-center justify-between border-b border-command-border pb-3">
          <div className="flex items-center space-x-2 text-amber-400 font-bold text-sm">
            <AlertTriangle className="w-5 h-5 shrink-0" />
            <span>Confirm Reference Frame Replacement: {cameraName}</span>
          </div>
          <button
            onClick={onClose}
            disabled={isUploading}
            className="text-command-muted hover:text-command-text transition-colors disabled:opacity-50"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {error && (
          <div className="p-3 rounded-md bg-accent-red/20 border border-accent-red/40 text-accent-red text-xs">
            {error}
          </div>
        )}

        <div className="p-3.5 rounded-lg bg-accent-amber/10 border border-accent-amber/30 text-amber-300 text-xs space-y-2">
          <p className="font-bold uppercase tracking-wide text-[11px]">Lifecycle Consequence Warning:</p>
          <p className="leading-relaxed">
            Replacing the camera reference frame will <span className="font-bold text-white underline">permanently invalidate</span> all existing layout revisions (DRAFT, PENDING_REVIEW, and VERIFIED) for this camera.
            Any polygon geometry calibrated against the old reference frame will become invalid for occupancy inference.
          </p>
          <p className="text-command-muted text-[11px]">
            New File: <span className="text-white font-mono">{pendingFile.name}</span> ({(pendingFile.size / 1024).toFixed(1)} KB)
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4 text-xs">
          <div className="space-y-1.5">
            <label className="block text-command-text font-bold uppercase tracking-wider text-[11px]">
              Local Operator Label <span className="text-accent-red">*</span>
            </label>
            <input
              type="text"
              value={operatorLabel}
              onChange={(e) => setOperatorLabel(e.target.value)}
              placeholder="e.g. op-field-engineer-01"
              required
              disabled={isUploading}
              className="w-full px-3 py-2 rounded-md bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-amber-400 text-xs"
            />
          </div>

          <div className="space-y-1.5">
            <label className="block text-command-text font-bold uppercase tracking-wider text-[11px]">
              Replacement Reason <span className="text-accent-red">*</span>
            </label>
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="e.g. Camera repositioned to cover newly paved section and expanded EV bays"
              rows={2}
              required
              disabled={isUploading}
              className="w-full px-3 py-2 rounded-md bg-command-bg border border-command-border text-command-text focus:outline-none focus:border-amber-400 text-xs resize-none"
            />
          </div>

          <div className="flex items-start space-x-2 pt-1">
            <input
              type="checkbox"
              id="confirm-replacement"
              checked={confirmed}
              onChange={(e) => setConfirmed(e.target.checked)}
              disabled={isUploading}
              className="mt-0.5 rounded bg-command-bg border-command-border text-amber-500 focus:ring-0 focus:ring-offset-0 cursor-pointer"
            />
            <label htmlFor="confirm-replacement" className="text-command-text text-[11px] leading-relaxed cursor-pointer select-none">
              I acknowledge that this action is irreversible and all geometry tied to the old image will be permanently invalidated.
            </label>
          </div>

          <div className="flex items-center justify-end space-x-3 pt-3 border-t border-command-border">
            <button
              type="button"
              onClick={onClose}
              disabled={isUploading}
              className="px-4 py-2 rounded-md bg-command-elevated border border-command-border text-command-muted hover:text-command-text text-xs"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isUploading || !confirmed || !operatorLabel.trim() || !reason.trim()}
              className="px-4 py-2 rounded-md bg-amber-500 hover:bg-amber-400 text-command-bg font-bold text-xs transition-all disabled:opacity-50 flex items-center space-x-1.5"
            >
              <UploadCloud className="w-4 h-4" />
              <span>{isUploading ? 'Uploading & Invalidating...' : 'Confirm & Replace Image'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
