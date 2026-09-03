'use client';

import React, { useEffect, useState, useCallback } from 'react';
import { Header } from '../components/Header';
import { Sidebar, ActiveTab } from '../components/Sidebar';
import { VideoViewport } from '../components/VideoViewport';
import { ReviewPanel } from '../components/ReviewPanel';
import { SessionsList } from '../components/SessionsList';
import { MapView } from '../components/MapView';
import { LiveCameraView } from '../components/LiveCameraView';
import { UploadModal } from '../components/UploadModal';
import { SystemHealthModal } from '../components/SystemHealthModal';
import {
  fetchDetections,
  fetchRoadEvents,
  fetchSessionDetail,
  fetchSessions,
  fetchSystemHealth,
} from '../lib/api';
import { DriveSession, RawDetection, RoadEvent, SystemHealth } from '../lib/types';
import { useSessionWebSocket } from '../hooks/useWebSocket';
import { Activity, AlertCircle, CheckCircle2, Film, Loader2, Radio } from 'lucide-react';

export default function OperationsDashboard() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('command');
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [sessions, setSessions] = useState<DriveSession[]>([]);
  const [activeSession, setActiveSession] = useState<DriveSession | null>(null);
  const [detections, setDetections] = useState<RawDetection[]>([]);
  const [roadEvents, setRoadEvents] = useState<RoadEvent[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<RoadEvent | null>(null);

  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [isHealthOpen, setIsHealthOpen] = useState(false);
  const [confidenceFilter, setConfidenceFilter] = useState(0.25);
  const [isLoadingSession, setIsLoadingSession] = useState(false);

  // WebSocket progress tracking for active session
  const { progress } = useSessionWebSocket(
    activeSession && activeSession.processing_state !== 'complete' && activeSession.processing_state !== 'failed'
      ? activeSession.id
      : null
  );

  // Load initial health and sessions
  const refreshSessions = useCallback(async () => {
    try {
      const sessList = await fetchSessions();
      setSessions(sessList);
      if (sessList.length > 0 && !activeSession) {
        loadSessionData(sessList[0].id);
      }
    } catch (err) {
      console.error('Failed to load sessions:', err);
    }
  }, [activeSession]);

  useEffect(() => {
    fetchSystemHealth()
      .then(setHealth)
      .catch((err) => console.error('Health fetch error:', err));
    refreshSessions();
  }, [refreshSessions]);

  // Load session detections and road events
  const loadSessionData = async (sessionId: string) => {
    setIsLoadingSession(true);
    try {
      const sess = await fetchSessionDetail(sessionId);
      setActiveSession(sess);

      if (sess.processing_state === 'complete') {
        const [dets, events] = await Promise.all([
          fetchDetections(sessionId),
          fetchRoadEvents(sessionId),
        ]);
        setDetections(dets);
        setRoadEvents(events);
        if (events.length > 0) {
          setSelectedEvent(events[0]);
        } else {
          setSelectedEvent(null);
        }
      } else {
        setDetections([]);
        setRoadEvents([]);
        setSelectedEvent(null);
      }
    } catch (err) {
      console.error('Failed to load session details:', err);
    } finally {
      setIsLoadingSession(false);
    }
  };

  // When WebSocket signals completion, refresh session details
  useEffect(() => {
    if (progress?.stage === 'complete' && activeSession) {
      loadSessionData(activeSession.id);
      refreshSessions();
    }
  }, [progress, activeSession, refreshSessions]);

  const handleEventUpdated = (updated: RoadEvent) => {
    setRoadEvents((prev) =>
      prev.map((e) => (e.id === updated.id ? updated : e))
    );
    setSelectedEvent(updated);
  };

  const pendingReviewsCount = roadEvents.filter(
    (e) => e.review_status === 'PENDING_REVIEW'
  ).length;

  return (
    <div className="flex flex-col min-h-screen bg-command-bg text-command-text">
      {/* Top Operations Header */}
      <Header
        health={health}
        onOpenUpload={() => setIsUploadOpen(true)}
        onOpenHealth={() => setIsHealthOpen(true)}
      />

      {/* Main Container */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Navigation Sidebar */}
        <Sidebar
          activeTab={activeTab}
          onSelectTab={setActiveTab}
          pendingReviewsCount={pendingReviewsCount}
        />

        {/* Central Operations Workspace */}
        <main className="flex-1 p-5 overflow-y-auto space-y-4">
          {/* Active Job Progress Banner (Shown during inference) */}
          {activeSession &&
            activeSession.processing_state !== 'complete' &&
            activeSession.processing_state !== 'failed' && (
              <div className="p-4 rounded-xl glass-panel-elevated border border-radar-green/40 shadow-radar space-y-2 font-mono text-xs">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2 text-radar-bright font-bold">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="uppercase">
                      Stage: {progress?.stage || activeSession.processing_state}...
                    </span>
                  </div>
                  <span className="text-command-text font-bold">
                    {progress?.percentage?.toFixed(0) || 0}% Complete
                  </span>
                </div>
                <div className="w-full bg-command-surface h-2 rounded-full overflow-hidden border border-command-border">
                  <div
                    className="bg-radar-bright h-full transition-all duration-300"
                    style={{ width: `${progress?.percentage || 10}%` }}
                  />
                </div>
                <p className="text-[11px] text-command-muted">
                  {progress?.message || 'Processing dashcam frames using verified YOLOv8n detector...'}
                </p>
              </div>
            )}

          {/* Tab Views */}
          {activeTab === 'command' && (
            <>
              {activeSession ? (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 h-[calc(100vh-8.5rem)]">
                  {/* Video Viewport Stage */}
                  <div className="lg:col-span-8 flex flex-col h-full">
                    <VideoViewport
                      session={activeSession}
                      detections={detections}
                      roadEvents={roadEvents}
                      selectedEvent={selectedEvent}
                      onSelectEvent={setSelectedEvent}
                      confidenceFilter={confidenceFilter}
                      onConfidenceFilterChange={setConfidenceFilter}
                    />
                  </div>

                  {/* Right Review Queue Panel */}
                  <div className="lg:col-span-4 flex flex-col h-full">
                    <ReviewPanel
                      sessionId={activeSession.id}
                      events={roadEvents}
                      selectedEvent={selectedEvent}
                      onSelectEvent={setSelectedEvent}
                      onEventUpdated={handleEventUpdated}
                    />
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-[calc(100vh-10rem)] rounded-xl glass-panel border border-command-border p-8 text-center space-y-4">
                  <Film className="w-16 h-16 text-command-muted animate-pulse" />
                  <h3 className="text-base font-bold text-command-text">
                    No Active Drive Session Selected
                  </h3>
                  <p className="text-xs font-mono text-command-muted max-w-md">
                    Upload a dashcam video recording to start hardware-accelerated local inference, temporal event fusion, and human inspection review.
                  </p>
                  <button
                    onClick={() => setIsUploadOpen(true)}
                    className="px-5 py-2 rounded-md bg-radar-bright hover:bg-radar-green text-command-bg font-bold text-xs font-mono transition-all shadow-radar"
                  >
                    Upload Dashcam Video
                  </button>
                </div>
              )}
            </>
          )}

          {activeTab === 'sessions' && (
            <div className="h-[calc(100vh-8.5rem)]">
              <SessionsList
                sessions={sessions}
                activeSessionId={activeSession?.id || null}
                onSelectSession={(id) => {
                  loadSessionData(id);
                  setActiveTab('command');
                }}
              />
            </div>
          )}

          {activeTab === 'map' && (
            <div className="h-[calc(100vh-8.5rem)]">
              <MapView session={activeSession} roadEvents={roadEvents} />
            </div>
          )}

          {activeTab === 'live' && (
            <div className="h-[calc(100vh-8.5rem)]">
              <LiveCameraView />
            </div>
          )}
        </main>
      </div>

      {/* Upload Modal */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onSessionCreated={(newSession) => {
          refreshSessions();
          loadSessionData(newSession.id);
          setActiveTab('command');
        }}
      />

      {/* System Health Diagnostics Modal */}
      <SystemHealthModal
        isOpen={isHealthOpen}
        onClose={() => setIsHealthOpen(false)}
        health={health}
      />
    </div>
  );
}
