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
  first_seen_seconds: float;
  last_seen_seconds: float;
  first_frame_index: number;
  last_frame_index: number;
  representative_detection_id?: string | null;
  representative_confidence: number;
  representative_bbox: {
    x_min: number;
    y_min: number;
    x_max: number;
    y_max: number;
  };
  support_count: number;
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
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  road_event_id?: string | null;
  created_at: string;
}

export interface DriveSession {
  id: string;
  mode: 'upload' | 'live';
  source_filename: string;
  source_hash: string;
  processing_state: 'queued' | 'validating' | 'sampling' | 'model_loading' | 'processing' | 'rendering' | 'complete' | 'failed';
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
  model_provenance?: {
    run_id: string;
    checkpoint_sha256: string;
    git_sha: string;
    device: string;
    confidence_threshold: number;
    iou_threshold: number;
  } | null;
  route_telemetry?: Array<{ lat: number; lon: number; timestamp: number }> | null;
}

export interface Artifact {
  id: string;
  session_id: string;
  artifact_type: 'raw_video' | 'annotated_video' | 'report_zip' | 'detections_csv' | 'metadata_json';
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
  stage: 'queued' | 'validating' | 'sampling' | 'model_loading' | 'processing' | 'rendering' | 'complete' | 'failed';
  processed_frames: number;
  total_frames: number;
  percentage: number;
  current_fps?: number | null;
  message?: string | null;
  detections_found: number;
}

export type float = number;
