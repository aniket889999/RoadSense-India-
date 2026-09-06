export type ReviewStatus = 'PENDING_REVIEW' | 'CONFIRMED' | 'REJECTED' | 'NEEDS_REVISIT' | 'SPLITED' | 'MERGED';

export type ReviewActionType = 'CONFIRM' | 'REJECT' | 'NEEDS_REVISIT' | 'SPLIT' | 'MERGE';

export interface ReviewAction {
  id: string;
  event_id: string;
  action: ReviewActionType;
  previous_status: string;
  new_status: string;
  reviewer_note?: string | null;
  created_at: string;
}

export interface RoadEvent {
  id: string;
  session_id: string;
  first_seen_seconds: number;
  last_seen_seconds: number;
  first_frame_index: number;
  last_frame_index: number;
  track_id?: number | null;
  representative_detection_id?: string | null;
  representative_confidence: number;
  representative_bbox: {
    x_min: number;
    y_min: number;
    x_max: number;
    y_max: number;
  };
  support_count: number;
  evidence_crop_path?: string | null;
  review_status: ReviewStatus;
  reviewer_note?: string | null;
  reviewed_at?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  created_at: string;
  review_actions?: ReviewAction[];
}

export interface RawDetection {
  id: string;
  session_id: string;
  frame_index: number;
  timestamp_seconds: number;
  confidence: number;
  class_id: number;
  track_id?: number | null;
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  road_event_id?: string | null;
  created_at: string;
}

export interface MediaMetadata {
  source_filename: string;
  file_size_bytes: number;
  sha256: string;
  container_format: string;
  video_codec: string;
  width: number;
  height: number;
  rotation_degrees: number;
  duration_seconds: number;
  avg_fps: number;
  real_fps: number;
  time_base: string;
  frame_count?: number | null;
  is_variable_frame_rate: boolean;
  has_audio: boolean;
  audio_codec?: string | null;
}

export interface DriveSession {
  id: string;
  mode: 'upload' | 'live';
  source_filename: string;
  source_hash: string;
  processing_state: 'queued' | 'validating' | 'sampling' | 'model_loading' | 'detecting' | 'tracking' | 'fusing_events' | 'encoding' | 'complete' | 'failed' | 'cancelled';
  started_at: string;
  completed_at?: string | null;
  source_duration_seconds?: number | null;
  source_fps?: number | null;
  source_width?: number | null;
  source_height?: number | null;
  total_source_frames?: number | null;
  sampled_frames_count?: number | null;
  frames_with_detections?: number | null;
  total_detections_count?: number | null;
  processing_duration_seconds?: number | null;
  error_message?: string | null;
  media_metadata?: MediaMetadata | null;
  model_provenance?: {
    run_id: string;
    checkpoint_sha256: string;
    tracker_sha256?: string;
    git_sha: string;
    device: string;
    confidence_threshold: number;
    iou_threshold: number;
    privacy_masked?: boolean;
  } | null;
  route_telemetry?: Array<{ lat: number; lon: number; timestamp: number }> | null;
}

export interface Artifact {
  id: string;
  session_id: string;
  artifact_type: 'raw_video' | 'annotated_video' | 'report_zip' | 'detections_csv' | 'metadata_json' | 'evidence_crop';
  relative_path: string;
  sha256: string;
  file_size_bytes: number;
  created_at: string;
}

export interface SystemHealth {
  status: string;
  timestamp: string;
  api_version: string;
  database_connected: boolean;
  database_type: string;
  model_verified: boolean;
  model_hash_prefix?: string | null;
  model_run_id?: string | null;
  mps_available: boolean;
  cuda_available: boolean;
  active_jobs: number;
  disk_free_gb: number;
}

export interface SessionProgressEvent {
  session_id: string;
  stage: string;
  processed_frames: number;
  total_frames: number;
  percentage: number;
  current_fps?: number | null;
  message?: string | null;
  detections_found: number;
}

// ============================================================================
// Parking Domain & ROI Editor Types (Phase 1)
// ============================================================================

export type CalibrationStatus = 'NOT_CONFIGURED' | 'PENDING_REVIEW' | 'VERIFIED' | 'INVALIDATED';

export type LayoutRevisionStatus = 'DRAFT' | 'PENDING_REVIEW' | 'VERIFIED' | 'SUPERSEDED' | 'INVALIDATED';

export type SpaceType = 'STANDARD' | 'ACCESSIBLE' | 'EV_CHARGING' | 'LOADING' | 'EMERGENCY' | 'OTHER';

export interface NormalizedPoint {
  x: number;
  y: number;
}

export interface ParkingSpace {
  id?: string;
  operator_label: string;
  space_type: SpaceType;
  polygon_normalized: NormalizedPoint[];
  active: boolean;
}

export interface ApproachZone {
  id?: string;
  parking_space_id: string;
  polygon_normalized: NormalizedPoint[];
}

export interface Site {
  id: string;
  name: string;
  description?: string | null;
  timezone: string;
  active: boolean;
  created_at: string;
  updated_at: string;
  cameras_count: number;
}

export interface Camera {
  id: string;
  site_id: string;
  name: string;
  description?: string | null;
  reference_image_path?: string | null;
  reference_image_sha256?: string | null;
  reference_width?: number | null;
  reference_height?: number | null;
  calibration_status: CalibrationStatus;
  camera_position_description?: string | null;
  created_at: string;
  updated_at: string;
  active_verified_layout_id?: string | null;
}

export interface LayoutAuditEvent {
  id: string;
  layout_revision_id: string;
  event_type: string;
  prior_status?: string | null;
  new_status: string;
  local_operator_label?: string | null;
  note?: string | null;
  created_at: string;
}

export interface ParkingLayoutRevision {
  id: string;
  camera_id: string;
  revision_number: number;
  status: LayoutRevisionStatus;
  canonical_sha256?: string | null;
  reference_image_sha256?: string | null;
  reference_width?: number | null;
  reference_height?: number | null;
  created_at: string;
  submitted_at?: string | null;
  verified_at?: string | null;
  invalidated_at?: string | null;
  invalidation_reason?: string | null;
  parking_spaces: ParkingSpace[];
  approach_zones: ApproachZone[];
  audit_events: LayoutAuditEvent[];
}

export interface LayoutValidationError {
  rule_id: string;
  message: string;
  space_id?: string | null;
  space_label?: string | null;
}

export interface LayoutValidationResult {
  is_valid: boolean;
  errors: LayoutValidationError[];
  spaces_count: number;
  approach_zones_count: number;
}
