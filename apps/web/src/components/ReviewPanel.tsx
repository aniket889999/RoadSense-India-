'use client';

import React, { useState } from 'react';
import {
  CheckCircle2,
  XCircle,
  Clock,
  HelpCircle,
  Download,
  FileCheck,
  History,
  Tag,
  Crosshair,
} from 'lucide-react';
import { ReviewActionType, ReviewStatus, RoadEvent } from '../lib/types';
import { getArtifactDownloadUrl, reviewRoadEvent } from '../lib/api';

interface ReviewPanelProps {
  sessionId: string;
  events: RoadEvent[];
  selectedEvent: RoadEvent | null;
  onSelectEvent: (event: RoadEvent) => void;
  onEventUpdated: (updatedEvent: RoadEvent) => void;
}

export function ReviewPanel({
  sessionId,
  events,
  selectedEvent,
  onSelectEvent,
  onEventUpdated,
}: ReviewPanelProps) {
  const [statusFilter, setStatusFilter] = useState<'ALL' | ReviewStatus>('ALL');
  const [reviewerNote, setReviewerNote] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const filteredEvents = events.filter((ev) => {
    if (statusFilter === 'ALL') return true;
    return ev.review_status === statusFilter;
  });

  const handleReviewAction = async (action: ReviewActionType) => {
    if (!selectedEvent) return;
    setIsSubmitting(true);
    try {
      const updated = await reviewRoadEvent(selectedEvent.id, action, reviewerNote);
      onEventUpdated(updated);
      setReviewerNote('');
    } catch (err) {
      console.error('Failed to submit review action:', err);
    } finally {
      setIsSubmitting(false);
    }
  };

  const getStatusBadge = (status: ReviewStatus) => {
    switch (status) {
      case 'CONFIRMED':
        return (
          <span className="px-2 py-0.5 text-[10px] font-mono font-semibold rounded bg-radar-dim/40 text-radar-bright border border-radar-green/40 flex items-center space-x-1">
            <CheckCircle2 className="w-3 h-3" />
            <span>CONFIRMED</span>
          </span>
        );
      case 'REJECTED':
        return (
          <span className="px-2 py-0.5 text-[10px] font-mono font-semibold rounded bg-accent-red/20 text-accent-red border border-accent-red/30 flex items-center space-x-1">
            <XCircle className="w-3 h-3" />
            <span>REJECTED</span>
          </span>
        );
      case 'NEEDS_REVISIT':
        return (
          <span className="px-2 py-0.5 text-[10px] font-mono font-semibold rounded bg-accent-amber/20 text-accent-amber border border-accent-amber/30 flex items-center space-x-1">
            <Clock className="w-3 h-3" />
            <span>REVISIT</span>
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 text-[10px] font-mono font-semibold rounded bg-command-border text-command-muted border border-command-subtle flex items-center space-x-1">
            <HelpCircle className="w-3 h-3" />
            <span>PENDING</span>
          </span>
        );
    }
  };

  return (
    <div className="w-full lg:w-96 flex flex-col h-full rounded-xl glass-panel-elevated border border-command-border overflow-hidden">
      {/* Panel Header */}
      <div className="p-4 bg-command-surface border-b border-command-border space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <FileCheck className="w-4 h-4 text-radar-bright" />
            <h3 className="text-xs font-bold font-mono tracking-wider text-command-text uppercase">
              Inspection Review Queue
            </h3>
          </div>
          <span className="text-[11px] font-mono text-command-muted">
            {events.length} Events ({events.filter((e) => e.review_status === 'CONFIRMED').length} Confirmed)
          </span>
        </div>

        {/* Filter Pills */}
        <div className="flex items-center space-x-1 overflow-x-auto text-[10px] font-mono">
          {(['ALL', 'PENDING_REVIEW', 'CONFIRMED', 'REJECTED', 'NEEDS_REVISIT'] as const).map((st) => (
            <button
              key={st}
              onClick={() => setStatusFilter(st)}
              className={`px-2 py-1 rounded whitespace-nowrap transition-colors ${
                statusFilter === st
                  ? 'bg-command-elevated text-radar-bright border border-command-border font-bold'
                  : 'text-command-muted hover:text-command-text'
              }`}
            >
              {st.replace('_', ' ')}
            </button>
          ))}
        </div>
      </div>

      {/* Events List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {filteredEvents.length === 0 ? (
          <div className="text-center py-12 text-xs font-mono text-command-muted">
            No road events matching filter.
          </div>
        ) : (
          filteredEvents.map((ev) => {
            const isSelected = selectedEvent?.id === ev.id;
            return (
              <div
                key={ev.id}
                onClick={() => onSelectEvent(ev)}
                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                  isSelected
                    ? 'bg-command-elevated border-radar-bright shadow-radar'
                    : 'bg-command-surface/60 border-command-border hover:border-command-subtle'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-mono font-bold text-command-text">
                      @ {ev.first_seen_seconds.toFixed(2)}s - {ev.last_seen_seconds.toFixed(2)}s
                    </span>
                    {ev.track_id !== undefined && ev.track_id !== null && (
                      <span className="px-1.5 py-0.5 text-[9px] font-mono rounded bg-radar-dim/30 text-radar-bright border border-radar-green/30">
                        Track #{ev.track_id}
                      </span>
                    )}
                  </div>
                  {getStatusBadge(ev.review_status)}
                </div>

                <div className="mt-2 flex items-center justify-between text-[11px] font-mono text-command-muted">
                  <span>Confidence: <strong className="text-radar-bright">{(ev.representative_confidence * 100).toFixed(0)}%</strong></span>
                  <span>{ev.support_count} frame{ev.support_count > 1 ? 's' : ''}</span>
                </div>

                {ev.reviewer_note && (
                  <p className="mt-1 text-[10px] text-command-muted italic line-clamp-1">
                    &ldquo;{ev.reviewer_note}&rdquo;
                  </p>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Review Action Drawer for Selected Event */}
      {selectedEvent && (
        <div className="p-4 bg-command-surface border-t border-command-border space-y-3">
          <div className="text-xs font-mono font-bold text-command-text flex items-center justify-between">
            <div className="flex items-center space-x-1.5">
              <Crosshair className="w-3.5 h-3.5 text-radar-bright" />
              <span>Event @ {selectedEvent.first_seen_seconds.toFixed(2)}s</span>
            </div>
            <span className="text-[10px] text-command-muted font-mono">
              {selectedEvent.track_id !== undefined && selectedEvent.track_id !== null ? `Track #${selectedEvent.track_id}` : `ID: ${selectedEvent.id.slice(0, 8)}`}
            </span>
          </div>

          <textarea
            value={reviewerNote}
            onChange={(e) => setReviewerNote(e.target.value)}
            placeholder="Inspector review notes (e.g. valid pothole, shadow, manhole)..."
            className="w-full text-xs font-mono bg-command-elevated border border-command-border rounded p-2 text-command-text placeholder-command-muted focus:outline-none focus:border-radar-bright resize-none h-16"
          />

          {/* Action Buttons */}
          <div className="grid grid-cols-3 gap-2">
            <button
              onClick={() => handleReviewAction('CONFIRM')}
              disabled={isSubmitting}
              className="py-2 px-1 rounded bg-radar-bright hover:bg-radar-green text-command-bg font-bold text-xs flex flex-col items-center justify-center transition-colors shadow-radar disabled:opacity-50"
            >
              <CheckCircle2 className="w-4 h-4 mb-0.5" />
              <span>Confirm</span>
            </button>
            <button
              onClick={() => handleReviewAction('REJECT')}
              disabled={isSubmitting}
              className="py-2 px-1 rounded bg-accent-red hover:bg-red-600 text-white font-bold text-xs flex flex-col items-center justify-center transition-colors disabled:opacity-50"
            >
              <XCircle className="w-4 h-4 mb-0.5" />
              <span>Reject</span>
            </button>
            <button
              onClick={() => handleReviewAction('NEEDS_REVISIT')}
              disabled={isSubmitting}
              className="py-2 px-1 rounded bg-accent-amber hover:bg-amber-600 text-command-bg font-bold text-xs flex flex-col items-center justify-center transition-colors disabled:opacity-50"
            >
              <Clock className="w-4 h-4 mb-0.5" />
              <span>Revisit</span>
            </button>
          </div>

          {/* Audit History Snippet */}
          {selectedEvent.review_actions && selectedEvent.review_actions.length > 0 && (
            <div className="pt-2 border-t border-command-border text-[10px] font-mono text-command-muted space-y-1">
              <div className="flex items-center space-x-1 font-semibold text-command-text">
                <History className="w-3 h-3 text-radar-bright" />
                <span>Audit History</span>
              </div>
              {selectedEvent.review_actions.map((act) => (
                <div key={act.id} className="truncate">
                  • {act.action} ({act.previous_status} → {act.new_status})
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Export Inspection Package Button */}
      <div className="p-3 bg-command-elevated border-t border-command-border">
        <a
          href={getArtifactDownloadUrl(sessionId, 'report_zip')}
          download
          className="w-full py-2 px-3 rounded bg-command-border hover:bg-command-subtle text-command-text text-xs font-mono font-bold flex items-center justify-center space-x-2 transition-colors"
        >
          <Download className="w-4 h-4 text-radar-bright" />
          <span>Export Field Dossier (.zip)</span>
        </a>
      </div>
    </div>
  );
}
