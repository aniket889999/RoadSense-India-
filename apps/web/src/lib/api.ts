import {
  Artifact,
  DriveSession,
  RawDetection,
  ReviewActionType,
  RoadEvent,
  SystemHealth,
} from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export async function fetchSystemHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_BASE}/health/system`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch system health: ${res.statusText}`);
  }
  return res.json();
}

export async function uploadSession(file: File): Promise<DriveSession> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/api/v1/sessions/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || 'Upload failed.');
  }

  return res.json();
}

export async function fetchSessions(limit: number = 50): Promise<DriveSession[]> {
  const res = await fetch(`${API_BASE}/api/v1/sessions?limit=${limit}`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch sessions: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchSessionDetail(sessionId: string): Promise<DriveSession> {
  const res = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch session: ${res.statusText}`);
  }
  return res.json();
}

export async function triggerProcessing(
  sessionId: string,
  params: {
    confidence_threshold?: number;
    iou_threshold?: number;
    sampling_fps?: number;
    max_frames?: number;
    window_start_seconds?: number;
    window_duration_seconds?: number;
    apply_privacy_mask?: boolean;
  } = {}
): Promise<{ message: string; session_id: string }> {
  const res = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      confidence_threshold: params.confidence_threshold ?? 0.25,
      iou_threshold: params.iou_threshold ?? 0.45,
      sampling_fps: params.sampling_fps ?? 5.0,
      max_frames: params.max_frames ?? 150,
      window_start_seconds: params.window_start_seconds ?? 0.0,
      window_duration_seconds: params.window_duration_seconds ?? 30.0,
      apply_privacy_mask: params.apply_privacy_mask ?? false,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to trigger processing.');
  }

  return res.json();
}

export async function cancelSession(sessionId: string): Promise<{ message: string }> {
  const res = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/cancel`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to cancel session.');
  }
  return res.json();
}

export async function deleteSession(
  sessionId: string,
  options: { delete_source_media?: boolean; delete_artifacts?: boolean; delete_database_record?: boolean } = {}
): Promise<{ message: string }> {
  const res = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to delete session.');
  }
  return res.json();
}

export async function fetchDetections(sessionId: string): Promise<RawDetection[]> {
  const res = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/detections`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch detections: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchRoadEvents(sessionId: string): Promise<RoadEvent[]> {
  const res = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/road-events`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch road events: ${res.statusText}`);
  }
  return res.json();
}

export async function reviewRoadEvent(
  eventId: string,
  action: ReviewActionType,
  reviewerNote?: string
): Promise<RoadEvent> {
  const res = await fetch(`${API_BASE}/api/v1/road-events/${eventId}/review`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      action,
      reviewer_note: reviewerNote || null,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to submit review.');
  }

  return res.json();
}

export async function fetchArtifacts(sessionId: string): Promise<Artifact[]> {
  const res = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/artifacts`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch artifacts: ${res.statusText}`);
  }
  return res.json();
}

export function getArtifactDownloadUrl(sessionId: string, artifactType: string): string {
  return `${API_BASE}/api/v1/sessions/${sessionId}/artifacts/${artifactType}/download`;
}
