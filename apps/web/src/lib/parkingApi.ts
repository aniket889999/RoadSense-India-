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
