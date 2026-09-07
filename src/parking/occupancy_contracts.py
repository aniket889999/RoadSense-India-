"""Data contracts, enums, and models for gated parking occupancy inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class OccupancyState(str, Enum):
    """Deterministic operational parking space occupancy status."""
    UNKNOWN = "UNKNOWN"
    VACANT = "VACANT"
    OCCUPIED = "OCCUPIED"
    OCCLUDED = "OCCLUDED"


@dataclass(frozen=True)
class VehicleDetection:
    """A single vehicle detection bounding box on a frame."""
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in absolute pixel coords
    track_id: Optional[int] = None
    center_xy: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox_xyxy": [round(c, 2) for c in self.bbox_xyxy],
            "track_id": self.track_id,
            "center_xy": [round(c, 2) for c in self.center_xy],
        }


@dataclass
class BayOccupancyEvidence:
    """Geometric and detection evidence evaluating a single parking space on one frame."""
    bay_id: str
    operator_label: str
    frame_index: int
    timestamp_seconds: float
    intersection_area_px: float
    bay_area_px: float
    vehicle_box_area_px: float
    bay_coverage_ratio: float
    vehicle_overlap_ratio: float
    is_center_inside: bool
    max_confidence: float
    contributing_track_ids: List[int]
    is_instant_evidence_occupied: bool
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bay_id": self.bay_id,
            "operator_label": self.operator_label,
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "intersection_area_px": round(self.intersection_area_px, 1),
            "bay_area_px": round(self.bay_area_px, 1),
            "vehicle_box_area_px": round(self.vehicle_box_area_px, 1),
            "bay_coverage_ratio": round(self.bay_coverage_ratio, 4),
            "vehicle_overlap_ratio": round(self.vehicle_overlap_ratio, 4),
            "is_center_inside": self.is_center_inside,
            "max_confidence": round(self.max_confidence, 4),
            "contributing_track_ids": self.contributing_track_ids,
            "is_instant_evidence_occupied": self.is_instant_evidence_occupied,
            "rejection_reasons": self.rejection_reasons,
        }


@dataclass
class BayStateSummary:
    """Current state and temporal tracking status for an individual parking bay."""
    bay_id: str
    operator_label: str
    space_type: str
    current_state: OccupancyState
    consecutive_frames_in_state: int
    confidence: float
    last_transition_frame: int
    last_transition_timestamp: float
    polygon_normalized: List[Dict[str, float]]
    contributing_track_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bay_id": self.bay_id,
            "operator_label": self.operator_label,
            "space_type": self.space_type,
            "current_state": self.current_state.value,
            "consecutive_frames_in_state": self.consecutive_frames_in_state,
            "confidence": round(self.confidence, 4),
            "last_transition_frame": self.last_transition_frame,
            "last_transition_timestamp": round(self.last_transition_timestamp, 3),
            "polygon_normalized": self.polygon_normalized,
            "contributing_track_ids": self.contributing_track_ids,
        }


@dataclass
class FrameOccupancyResult:
    """Full frame-level evaluation result including all bays and vehicle detections."""
    frame_index: int
    timestamp_seconds: float
    bay_states: Dict[str, BayStateSummary]
    vehicle_detections: List[VehicleDetection]
    total_bays: int
    occupied_count: int
    vacant_count: int
    unknown_count: int
    occluded_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "bay_states": {k: v.to_dict() for k, v in self.bay_states.items()},
            "vehicle_detections": [d.to_dict() for d in self.vehicle_detections],
            "total_bays": self.total_bays,
            "occupied_count": self.occupied_count,
            "vacant_count": self.vacant_count,
            "unknown_count": self.unknown_count,
            "occluded_count": self.occluded_count,
        }


@dataclass
class OccupancyTimelineEntry:
    """Recorded state change event for a parking space across time."""
    frame_index: int
    timestamp_seconds: float
    bay_id: str
    operator_label: str
    previous_state: OccupancyState
    new_state: OccupancyState
    trigger_reason: str
    confidence: float
    contributing_track_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "bay_id": self.bay_id,
            "operator_label": self.operator_label,
            "previous_state": self.previous_state.value,
            "new_state": self.new_state.value,
            "trigger_reason": self.trigger_reason,
            "confidence": round(self.confidence, 4),
            "contributing_track_ids": self.contributing_track_ids,
        }


@dataclass
class ParkingJobManifest:
    """Comprehensive machine-readable provenance manifest for an annotated parking video job."""
    job_id: str
    camera_id: str
    site_id: str
    input_video_sha256: str
    output_video_sha256: str
    reference_image_sha256: str
    layout_canonical_sha256: str
    stability_assessment_id: str
    stability_config_sha256: str
    detector_checkpoint_sha256: str
    occupancy_config_sha256: str
    algorithm_version: str
    opencv_version: str
    ultralytics_version: str
    total_frames: int
    processed_frames: int
    fps: float
    duration_seconds: float
    video_width: int
    video_height: int
    total_bays_evaluated: int
    final_occupied_count: int
    final_vacant_count: int
    final_unknown_count: int
    final_occluded_count: int
    total_state_transitions: int
    job_started_at: str
    job_completed_at: str
    errors_or_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
