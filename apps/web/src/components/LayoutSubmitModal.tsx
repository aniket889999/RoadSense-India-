'use client';

import React, { useState } from 'react';
import { AlertCircle, Send, X } from 'lucide-react';

interface LayoutSubmitModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmitConfirm: (operatorLabel: string, note?: string) => Promise<void>;
  revisionNumber: number;
}

export function LayoutSubmitModal({
  isOpen,
  onClose,
  onSubmitConfirm,
  revisionNumber,
}: LayoutSubmitModalProps) {
  const [operatorLabel, setOperatorLabel] = useState('');
  const [note, setNote] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const cleanOp = operatorLabel.trim();
    if (!cleanOp || cleanOp.length < 2) {
      setError('Please enter a valid, non-blank operator name / identifier.');
      return;
    }

    setIsSubmitting(true);
    setError(null);
    try {
      await onSubmitConfirm(cleanOp, note.trim() || undefined);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to submit layout for review.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-fade-in font-mono">
      <div className="w-full max-w-md rounded-xl glass-panel-elevated border border-command-border shadow-2xl p-6 space-y-5">
        <div className="flex items-center justify-between border-b border-command-border pb-3">
          <div className="flex items-center space-x-2 text-radar-bright font-bold text-sm">
            <Send className="w-4 h-4" />
            <span>Submit Layout Revision #{revisionNumber}</span>
          </div>
          <button
            onClick={onClose}
            disabled={isSubmitting}
            className="text-command-muted hover:text-command-text transition-colors disabled:opacity-50"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {error && (
          <div className="p-3 rounded-md bg-accent-red/20 border border-accent-red/40 text-accent-red text-xs flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4 text-xs">
          <p className="text-command-muted leading-relaxed">
            Submitting layout revision #{revisionNumber} shifts its status from{' '}
            <span className="text-blue-400 font-bold">DRAFT</span> to{' '}
            <span className="text-amber-400 font-bold">PENDING_REVIEW</span>.
            Enter your operator identity assertion below.
          </p>

          <div className="space-y-1.5">
            <label className="block text-command-text font-bold uppercase tracking-wider text-[11px]">
              Local Operator Label / Identifier <span className="text-accent-red">*</span>
            </label>
            <input
              type="text"
              value={operatorLabel}
              onChange={(e) => setOperatorLabel(e.target.value)}
              placeholder="e.g. op-aniket-north"
              required
              disabled={isSubmitting}
              className="w-full px-3 py-2 rounded-md bg-command-bg border border-command-border text-command-text placeholder:text-command-muted focus:outline-none focus:border-radar-bright text-xs"
            />
          </div>

          <div className="space-y-1.5">
            <label className="block text-command-muted uppercase tracking-wider text-[11px]">
              Submission Note (Optional)
            </label>
            <textarea
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="e.g. Added approach zone polygons for north parking bays"
              rows={2}
              disabled={isSubmitting}
              className="w-full px-3 py-2 rounded-md bg-command-bg border border-command-border text-command-text placeholder:text-command-muted focus:outline-none focus:border-radar-bright text-xs resize-none"
            />
          </div>

          <div className="flex items-center justify-end space-x-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting}
              className="px-4 py-2 rounded-md bg-command-elevated border border-command-border text-command-muted hover:text-command-text text-xs"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting || !operatorLabel.trim()}
              className="px-4 py-2 rounded-md bg-radar-bright hover:bg-radar-green text-command-bg font-bold text-xs shadow-radar transition-all disabled:opacity-50 flex items-center space-x-1.5"
            >
              {isSubmitting ? <span>Submitting...</span> : <span>Submit for Review</span>}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
