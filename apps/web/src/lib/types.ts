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

export type StabilityDecision = 'STABLE' | 'UNSTABLE' | 'INSUFFICIENT_EVIDENCE' | 'ERROR';
export type OperationalGate = 'ALLOWED' | 'BLOCKED';
export type StabilityStatus = 'QUEUED' | 'VALIDATING' | 'ANALYZING' | 'COMPLETE' | 'FAILED' | 'CANCELLED';

export interface SampleMeasurement {
  sample_index: number;
  timestamp_seconds: number;
  frame_index: number;
  matched_features: number;
  inlier_count: number;
  inlier_ratio: number;
  translation_px_x: number;
  translation_px_y: number;
  translation_magnitude_px: number;
  translation_normalized: number;
  scale_factor: number;
  scale_change: number;
  rotation_degrees: number;
  perspective_distortion: number;
  reprojection_error: number;
  decision: StabilityDecision;
  rejection_reasons: string[];
}

export interface StabilityAssessment {
  id: string;
  camera_id: string;
  layout_revision_id?: string | null;
  layout_canonical_sha256?: string | null;
  reference_image_sha256: string;
  video_sha256?: string | null;
  status: StabilityStatus;
  progress_pct: number;
  stage_message?: string | null;
  failure_code?: string | null;
  failure_message?: string | null;
  config_version?: string | null;
  config_sha256?: string | null;
  algorithm_version?: string | null;
  opencv_version?: string | null;
  thresholds_snapshot?: Record<string, any> | null;
  sample_measurements: SampleMeasurement[];
  aggregate_decision?: StabilityDecision | null;
  operational_gate?: OperationalGate | null;
  gate_reasons: string[];
  summary_metrics?: Record<string, any> | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  operator_acknowledged_at?: string | null;
  operator_label?: string | null;
  operator_note?: string | null;
}

export interface CameraStabilityAuditEvent {
  id: string;
  assessment_id: string;
  camera_id: string;
  event_type: string;
  operator_identity: string;
  explicit_reason: string;
  note?: string | null;
  previous_gate_state?: string | null;
  resulting_gate_state: string;
  previous_calibration_status?: string | null;
  resulting_calibration_status: string;
  assessment_sha256?: string | null;
  config_sha256?: string | null;
  reference_image_sha256: string;
  layout_canonical_sha256?: string | null;
  created_at: string;
}

export interface CameraOperationalGate {
  camera_id: string;
  operational_gate: OperationalGate;
  gate_reasons: string[];
  aggregate_decision?: StabilityDecision | null;
  assessment_id?: string | null;
  reference_image_sha256?: string | null;
  layout_canonical_sha256?: string | null;
  created_at?: string | null;
  is_fresh: boolean;
}

// ============================================================================
// Gate-Controlled Parking Occupancy (Phase 2B)
// ============================================================================

export type OccupancyJobStatus =
  | 'QUEUED'
  | 'VALIDATING'
  | 'DETECTING'
  | 'TRACKING'
  | 'CLASSIFYING_OCCUPANCY'
  | 'RENDERING'
  | 'ENCODING'
  | 'COMPLETE'
  | 'FAILED'
  | 'CANCELLED'
  | 'BLOCKED_BY_STABILITY_GATE';

export type BayOccupancyState = 'UNKNOWN' | 'VACANT' | 'OCCUPIED' | 'OCCLUDED';

export interface BaySummaryItem {
  operator_label: string;
  space_type: SpaceType;
  final_state: BayOccupancyState;
  final_confidence: number;
  transitions_count: number;
  last_vehicle_track_id?: number | null;
}

export interface ParkingOccupancyJob {
  id: string;
  camera_id: string;
  site_id: string;
  layout_revision_id?: string | null;
  stability_assessment_id?: string | null;

  status: OccupancyJobStatus;
  progress_pct: number;
  stage_message?: string | null;
  failure_code?: string | null;
  failure_message?: string | null;

  gate_decision?: string | null;
  gate_reasons?: string[] | null;

  input_video_sha256?: string | null;
  output_video_sha256?: string | null;
  reference_image_sha256?: string | null;
  layout_canonical_sha256?: string | null;
  detector_checkpoint_sha256?: string | null;
  occupancy_config_sha256?: string | null;

  total_frames: number;
  processed_frames: number;
  fps: number;
  duration_seconds: number;
  video_width: number;
  video_height: number;

  total_bays: number;
  final_occupied_count: number;
  final_vacant_count: number;
  final_unknown_count: number;
  final_occluded_count: number;
  total_state_transitions: number;

  has_annotated_video: boolean;
  has_timeline: boolean;
  has_manifest: boolean;
  has_summary?: boolean;
  bay_summary?: Record<string, BaySummaryItem> | null;

  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface ParkingOccupancyJobCancelResponse {
  job_id: string;
  status: string;
  cancelled: boolean;
  message: string;
}

export interface ValidationCheckItem {
  check_name: string;
  category: string;
  expected: any;
  observed: any;
  passed: boolean;
  details?: string;
}

export interface ParkingValidationEvidenceReport {
  run_id: string;
  is_synthetic_fixture: boolean;
  disclaimer: string;
  camera_id: string;
  site_id: string;
  job_id: string;
  input_video_sha256: string;
  reference_image_sha256: string;
  layout_canonical_sha256: string;
  stability_assessment_id: string;
  stability_config_sha256: string;
  occupancy_config_sha256: string;
  detector_mode: string;
  detector_checkpoint_sha256: string;
  bytetrack_config_sha256: string;
  bytetrack_frame_rate: number;
  output_video_sha256: string;
  timeline_sha256: string;
  summary_sha256: string;
  manifest_sha256: string;
  operational_gate: string;
  gate_reasons: string[];
  total_frames: number;
  processed_frames: number;
  fps: number;
  duration_seconds: number;
  total_bays: number;
  final_occupied_count: number;
  final_vacant_count: number;
  final_unknown_count: number;
  final_occluded_count: number;
  total_state_transitions: number;
  checks: ValidationCheckItem[];
  passed: boolean;
  failure_reasons: string[];
  software_versions: Record<string, string>;
  git_sha?: string;
  git_commit_sha?: string;
  created_at: string;
  completed_at: string;
}

// ============================================================================
// Safe Usable Capacity & Hazard Association (Phase 3A)
// ============================================================================

export type CapacityState =
  | 'OCCUPIED'
  | 'USABLE_AVAILABLE'
  | 'HAZARD_BLOCKED'
  | 'APPROACH_BLOCKED'
  | 'VACANT_UNASSESSED'
  | 'OCCLUDED'
  | 'UNKNOWN'
  | 'INFERENCE_BLOCKED';

export type HazardTarget = 'BAY' | 'APPROACH_ZONE';
export type HazardReviewState = 'UNREVIEWED' | 'CONFIRMED' | 'REJECTED' | 'NEEDS_REVIEW';
export type HazardLifecycleState = 'ACTIVE' | 'MITIGATED' | 'RESOLVED' | 'EXPIRED' | 'SUPERSEDED';

export interface HazardAssociation {
  id: string;
  camera_id: string;
  parking_space_id: string;
  approach_zone_id?: string | null;
  target_type: HazardTarget;
  road_event_id?: string | null;
  hazard_label: string;
  review_state: HazardReviewState;
  lifecycle_state: HazardLifecycleState;
  severity_label?: string | null;
  notes?: string | null;
  created_by: string;
  reviewed_by?: string | null;
  reviewed_at?: string | null;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface HazardAuditEvent {
  id: string;
  association_id: string;
  event_type: string;
  operator_identity: string;
  prior_review_state?: string | null;
  new_review_state?: string | null;
  prior_lifecycle_state?: string | null;
  new_lifecycle_state?: string | null;
  explicit_reason: string;
  notes?: string | null;
  created_at: string;
}

export interface PavementInspectionRecord {
  id: string;
  camera_id: string;
  session_id?: string | null;
  inspected_at: string;
  inspector_label: string;
  notes?: string | null;
  created_at: string;
}

export interface BayCapacityDecisionItem {
  bay_id: string;
  operator_label: string;
  space_type: SpaceType;
  capacity_state: CapacityState;
  is_usable: boolean;
  reason_codes: string[];
  raw_occupancy_state: string;
  has_active_bay_hazard: boolean;
  has_active_approach_hazard: boolean;
  has_unverified_hazard: boolean;
  pavement_inspection_fresh: boolean;
  inspection_age_seconds?: number | null;
  evidence_ids: Record<string, any>;
}

export interface CapacitySnapshot {
  camera_id: string;
  site_id: string;
  layout_revision_id?: string | null;
  layout_canonical_sha256?: string | null;
  policy_config_sha256: string;
  operational_gate: string;
  gate_reasons: string[];
  evaluated_at: string;
  total_bays: number;
  physical_vacant_count: number;
  usable_available_count: number;
  occupied_count: number;
  hazard_blocked_count: number;
  approach_blocked_count: number;
  vacant_unassessed_count: number;
  unknown_count: number;
  occluded_count: number;
  inference_blocked_count: number;
  snapshot_sha256: string;
  decisions: Record<string, BayCapacityDecisionItem>;
}
