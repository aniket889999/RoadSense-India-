/**
 * API client functions for RoadSense SiteOps Parking Layout & ROI Editor.
 */

import {
  Camera,
  LayoutValidationResult,
  ParkingLayoutRevision,
  ParkingSpace,
  ApproachZone,
  Site,
  StabilityAssessment,
  CameraOperationalGate,
  CameraStabilityAuditEvent,
  ParkingOccupancyJob,
  ParkingOccupancyJobCancelResponse,
  ParkingValidationEvidenceReport,
} from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function fetchSites(): Promise<Site[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/sites`);
  if (!res.ok) throw new Error(`Failed to fetch sites: ${res.statusText}`);
  return res.json();
}

export async function createSite(name: string, description?: string, timezone: string = 'UTC'): Promise<Site> {
  const res = await fetch(`${API_BASE_URL}/api/v1/sites`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description, timezone }),
  });
  if (!res.ok) throw new Error(`Failed to create site: ${res.statusText}`);
  return res.json();
}

export async function fetchSite(siteId: string): Promise<Site> {
  const res = await fetch(`${API_BASE_URL}/api/v1/sites/${siteId}`);
  if (!res.ok) throw new Error(`Failed to fetch site details: ${res.statusText}`);
  return res.json();
}

export async function fetchSiteCameras(siteId: string): Promise<Camera[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/sites/${siteId}/cameras`);
  if (!res.ok) throw new Error(`Failed to fetch cameras for site: ${res.statusText}`);
  return res.json();
}

export async function createCamera(
  siteId: string,
  name: string,
  description?: string,
  camera_position_description?: string
): Promise<Camera> {
  const res = await fetch(`${API_BASE_URL}/api/v1/sites/${siteId}/cameras`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description, camera_position_description }),
  });
  if (!res.ok) throw new Error(`Failed to create camera: ${res.statusText}`);
  return res.json();
}

export async function fetchCamera(cameraId: string): Promise<Camera> {
  const res = await fetch(`${API_BASE_URL}/api/v1/cameras/${cameraId}`);
  if (!res.ok) throw new Error(`Failed to fetch camera: ${res.statusText}`);
  return res.json();
}

export async function uploadCameraReferenceImage(
  cameraId: string,
  file: File,
  confirmReplacement?: boolean,
  operatorLabel?: string,
  reason?: string
): Promise<Camera> {
  const formData = new FormData();
  formData.append('file', file, file.name);
  if (confirmReplacement) {
    formData.append('confirm_replacement', 'true');
    if (operatorLabel) formData.append('operator_label', operatorLabel);
    if (reason) formData.append('reason', reason);
  }

  const res = await fetch(`${API_BASE_URL}/api/v1/cameras/${cameraId}/reference-image`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to upload reference image');
  }
  return res.json();
}

export function getCameraReferenceImageUrl(cameraId: string): string {
  return `${API_BASE_URL}/api/v1/cameras/${cameraId}/reference-image`;
}

export async function invalidateCameraCalibration(
  cameraId: string,
  localOperatorLabel: string,
  invalidationReason: string,
  note?: string
): Promise<Camera> {
  const res = await fetch(`${API_BASE_URL}/api/v1/cameras/${cameraId}/invalidate-calibration`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      local_operator_label: localOperatorLabel,
      invalidation_reason: invalidationReason,
      note,
    }),
  });
  if (!res.ok) throw new Error(`Failed to invalidate camera calibration: ${res.statusText}`);
  return res.json();
}

export async function fetchCameraLayouts(cameraId: string): Promise<ParkingLayoutRevision[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/cameras/${cameraId}/layouts`);
  if (!res.ok) throw new Error(`Failed to fetch layouts: ${res.statusText}`);
  return res.json();
}

export async function createDraftLayout(
  cameraId: string,
  parkingSpaces: ParkingSpace[],
  approachZones?: ApproachZone[]
): Promise<ParkingLayoutRevision> {
  const res = await fetch(`${API_BASE_URL}/api/v1/cameras/${cameraId}/layouts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      parking_spaces: parkingSpaces,
      approach_zones: approachZones || [],
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to create draft layout');
  }
  return res.json();
}

export async function fetchLayout(layoutId: string): Promise<ParkingLayoutRevision> {
  const res = await fetch(`${API_BASE_URL}/api/v1/layouts/${layoutId}`);
  if (!res.ok) throw new Error(`Failed to fetch layout revision: ${res.statusText}`);
  return res.json();
}

export async function updateLayout(
  layoutId: string,
  parkingSpaces: ParkingSpace[],
  approachZones?: ApproachZone[]
): Promise<ParkingLayoutRevision> {
  const res = await fetch(`${API_BASE_URL}/api/v1/layouts/${layoutId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      parking_spaces: parkingSpaces,
      approach_zones: approachZones || [],
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to update layout');
  }
  return res.json();
}

export async function validateLayout(layoutId: string): Promise<LayoutValidationResult> {
  const res = await fetch(`${API_BASE_URL}/api/v1/layouts/${layoutId}/validate`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Failed to validate layout: ${res.statusText}`);
  return res.json();
}

export async function submitLayout(
  layoutId: string,
  localOperatorLabel?: string,
  note?: string
): Promise<ParkingLayoutRevision> {
  const res = await fetch(`${API_BASE_URL}/api/v1/layouts/${layoutId}/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      local_operator_label: localOperatorLabel,
      note,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to submit layout');
  }
  return res.json();
}

export async function verifyLayout(
  layoutId: string,
  localOperatorLabel: string,
  confirmationAcknowledged: boolean,
  note?: string
): Promise<ParkingLayoutRevision> {
  const res = await fetch(`${API_BASE_URL}/api/v1/layouts/${layoutId}/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      local_operator_label: localOperatorLabel,
      confirmation_acknowledged: confirmationAcknowledged,
      note,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to verify layout');
  }
  return res.json();
}

export async function invalidateLayout(
  layoutId: string,
  localOperatorLabel: string,
  invalidationReason: string,
  note?: string
): Promise<ParkingLayoutRevision> {
  const res = await fetch(`${API_BASE_URL}/api/v1/layouts/${layoutId}/invalidate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      local_operator_label: localOperatorLabel,
      invalidation_reason: invalidationReason,
      note,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to invalidate layout');
  }
  return res.json();
}

// ============================================================================
// Camera Stability Assessment & Operational Gate (Phase 2A)
// ============================================================================

export async function getCameraOperationalGate(cameraId: string): Promise<CameraOperationalGate> {
  const res = await fetch(`${API_BASE_URL}/api/v1/cameras/${cameraId}/stability/gate`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Failed to fetch operational gate: ${res.statusText}`);
  }
  return res.json();
}

export async function listCameraStabilityAssessments(cameraId: string): Promise<StabilityAssessment[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/cameras/${cameraId}/stability/assessments`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Failed to list stability assessments: ${res.statusText}`);
  }
  return res.json();
}

export async function getStabilityAssessmentDetail(assessmentId: string): Promise<StabilityAssessment> {
  const res = await fetch(`${API_BASE_URL}/api/v1/stability/assessments/${assessmentId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Failed to fetch assessment detail: ${res.statusText}`);
  }
  return res.json();
}

export async function assessCameraStability(
  cameraId: string,
  videoFile: File
): Promise<StabilityAssessment> {
  const formData = new FormData();
  formData.append('file', videoFile, videoFile.name);

  const res = await fetch(`${API_BASE_URL}/api/v1/cameras/${cameraId}/stability/assess`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to assess camera stability');
  }
  return res.json();
}

export async function cancelStabilityAssessment(assessmentId: string): Promise<{ assessment_id: string; status: string; cancelled: boolean; message: string }> {
  const res = await fetch(`${API_BASE_URL}/api/v1/stability/assessments/${assessmentId}/cancel`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to cancel stability assessment');
  }
  return res.json();
}

export async function acknowledgeStabilityAssessment(
  assessmentId: string,
  operatorLabel: string,
  note?: string,
  triggerInvalidation?: boolean,
  confirmInvalidation?: boolean,
  invalidationReason?: string
): Promise<CameraStabilityAuditEvent> {
  const res = await fetch(`${API_BASE_URL}/api/v1/stability/assessments/${assessmentId}/acknowledge`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      local_operator_label: operatorLabel,
      note,
      trigger_calibration_invalidation: triggerInvalidation || false,
      confirm_invalidation: confirmInvalidation || false,
      invalidation_reason: invalidationReason,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to acknowledge stability assessment');
  }
  return res.json();
}

export async function listCameraStabilityAuditEvents(cameraId: string): Promise<CameraStabilityAuditEvent[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/cameras/${cameraId}/stability/audit-events`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to list stability audit events');
  }
  return res.json();
}

// ============================================================================
// Gate-Controlled Parking Occupancy (Phase 2B)
// ============================================================================

export async function submitOccupancyJob(
  cameraId: string,
  videoFile: File
): Promise<ParkingOccupancyJob> {
  const formData = new FormData();
  formData.append('file', videoFile, videoFile.name);

  const res = await fetch(`${API_BASE_URL}/api/v1/cameras/${cameraId}/occupancy/jobs`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to submit parking occupancy job');
  }
  return res.json();
}

export async function listOccupancyJobs(cameraId: string): Promise<ParkingOccupancyJob[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/cameras/${cameraId}/occupancy/jobs`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to list occupancy jobs');
  }
  return res.json();
}

export async function getOccupancyJob(jobId: string): Promise<ParkingOccupancyJob> {
  const res = await fetch(`${API_BASE_URL}/api/v1/occupancy/jobs/${jobId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to fetch occupancy job details');
  }
  return res.json();
}

export async function cancelOccupancyJob(jobId: string): Promise<ParkingOccupancyJobCancelResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/occupancy/jobs/${jobId}/cancel`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to cancel occupancy job');
  }
  return res.json();
}

export function getOccupancyVideoUrl(jobId: string): string {
  return `${API_BASE_URL}/api/v1/occupancy/jobs/${jobId}/video`;
}

export function getOccupancyManifestUrl(jobId: string): string {
  return `${API_BASE_URL}/api/v1/occupancy/jobs/${jobId}/manifest`;
}

export function getOccupancyTimelineUrl(jobId: string): string {
  return `${API_BASE_URL}/api/v1/occupancy/jobs/${jobId}/timeline`;
}

export function getOccupancySummaryUrl(jobId: string): string {
  return `${API_BASE_URL}/api/v1/parking/jobs/${jobId}/summary`;
}

export async function deleteOccupancyJob(jobId: string): Promise<{ deleted: boolean; job_id: string; message: string }> {
  const res = await fetch(`${API_BASE_URL}/api/v1/parking/jobs/${jobId}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to delete occupancy job');
  }
  return res.json();
}

export async function runSyntheticValidation(): Promise<ParkingValidationEvidenceReport> {
  const res = await fetch(`${API_BASE_URL}/api/v1/parking/validation/run-synthetic`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to execute synthetic validation');
  }
  return res.json();
}
