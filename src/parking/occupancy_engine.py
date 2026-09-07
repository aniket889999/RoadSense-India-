"""Deterministic parking occupancy engine with geometric polygon overlap and temporal hysteresis."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.parking.occupancy_config import ParkingOccupancyConfig, ScoringConfig, TemporalConfig
from src.parking.occupancy_contracts import (
    BayOccupancyEvidence,
    BayStateSummary,
    FrameOccupancyResult,
    OccupancyState,
    OccupancyTimelineEntry,
    VehicleDetection,
)

logger = logging.getLogger(__name__)


def normalized_polygon_to_pixels(
    polygon_normalized: Sequence[Dict[str, float]],
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """
    Scale normalized polygon vertices [{"x": float, "y": float}, ...] into integer pixel coordinates.
    Clamps within frame boundaries [0, width], [0, height].
    """
    pts = []
    for p in polygon_normalized:
        x = max(0, min(frame_width - 1, int(round(p["x"] * frame_width))))
        y = max(0, min(frame_height - 1, int(round(p["y"] * frame_height))))
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)


def calculate_bay_vehicle_overlap(
    bay_polygon_px: np.ndarray,
    detection: VehicleDetection,
    scoring: ScoringConfig,
) -> BayOccupancyEvidence:
    """
    Calculate exact geometric overlap between a parking polygon and a detected vehicle bounding box.
    Uses bounded ROI rasterization to compute exact pixel-accurate intersection.
    """
    rejection_reasons: List[str] = []
    bay_pts = bay_polygon_px.reshape((-1, 2))

    if len(bay_pts) < 3:
        return BayOccupancyEvidence(
            bay_id="",
            operator_label="",
            frame_index=0,
            timestamp_seconds=0.0,
            intersection_area_px=0.0,
            bay_area_px=0.0,
            vehicle_box_area_px=0.0,
            bay_coverage_ratio=0.0,
            vehicle_overlap_ratio=0.0,
            is_center_inside=False,
            max_confidence=detection.confidence,
            contributing_track_ids=[detection.track_id] if detection.track_id is not None else [],
            is_instant_evidence_occupied=False,
            rejection_reasons=["DEGENERATE_POLYGON: Polygon has fewer than 3 vertices."],
        )

    bx1, by1, bx2, by2 = detection.bbox_xyxy
    bx1_i = int(math.floor(bx1))
    by1_i = int(math.floor(by1))
    bx2_i = int(math.ceil(bx2))
    by2_i = int(math.ceil(by2))

    vehicle_box_area = max(1.0, float((bx2_i - bx1_i) * (by2_i - by1_i)))

    poly_min_x = int(np.min(bay_pts[:, 0]))
    poly_max_x = int(np.max(bay_pts[:, 0]))
    poly_min_y = int(np.min(bay_pts[:, 1]))
    poly_max_y = int(np.max(bay_pts[:, 1]))

    # Compute bounding ROI enclosing both polygon and vehicle box
    roi_min_x = min(poly_min_x, bx1_i)
    roi_max_x = max(poly_max_x, bx2_i)
    roi_min_y = min(poly_min_y, by1_i)
    roi_max_y = max(poly_max_y, by2_i)

    roi_w = max(1, roi_max_x - roi_min_x + 1)
    roi_h = max(1, roi_max_y - roi_min_y + 1)

    # Check for early disjoint bounding box rejection
    if bx2_i < poly_min_x or bx1_i > poly_max_x or by2_i < poly_min_y or by1_i > poly_max_y:
        # Disjoint
        poly_area = max(1.0, float(abs(cv2.contourArea(bay_pts))))
        return BayOccupancyEvidence(
            bay_id="",
            operator_label="",
            frame_index=0,
            timestamp_seconds=0.0,
            intersection_area_px=0.0,
            bay_area_px=poly_area,
            vehicle_box_area_px=vehicle_box_area,
            bay_coverage_ratio=0.0,
            vehicle_overlap_ratio=0.0,
            is_center_inside=False,
            max_confidence=detection.confidence,
            contributing_track_ids=[detection.track_id] if detection.track_id is not None else [],
            is_instant_evidence_occupied=False,
            rejection_reasons=["DISJOINT_BOUNDS: Vehicle bounding box does not intersect bay envelope."],
        )

    # Rasterize bay polygon in ROI
    mask_poly = np.zeros((roi_h, roi_w), dtype=np.uint8)
    rel_bay_pts = bay_pts - np.array([roi_min_x, roi_min_y])
    cv2.fillPoly(mask_poly, [rel_bay_pts], 1)
    poly_area_px = float(np.count_nonzero(mask_poly))
    if poly_area_px <= 0:
        poly_area_px = max(1.0, float(abs(cv2.contourArea(bay_pts))))

    # Rasterize vehicle box in ROI
    mask_box = np.zeros((roi_h, roi_w), dtype=np.uint8)
    rel_bx1 = max(0, bx1_i - roi_min_x)
    rel_by1 = max(0, by1_i - roi_min_y)
    rel_bx2 = min(roi_w, bx2_i - roi_min_x)
    rel_by2 = min(roi_h, by2_i - roi_min_y)
    if rel_bx2 > rel_bx1 and rel_by2 > rel_by1:
        cv2.rectangle(mask_box, (rel_bx1, rel_by1), (rel_bx2, rel_by2), 1, -1)

    intersection_area_px = float(np.count_nonzero(mask_poly & mask_box))
    bay_coverage_ratio = intersection_area_px / max(1.0, poly_area_px)
    vehicle_overlap_ratio = intersection_area_px / max(1.0, vehicle_box_area)

    # Point polygon test for vehicle center
    center_x, center_y = detection.center_xy
    is_center_inside = cv2.pointPolygonTest(bay_pts.astype(np.float32), (float(center_x), float(center_y)), False) >= 0

    # Decision evaluation against scoring thresholds
    if detection.confidence < scoring.min_detection_confidence:
        rejection_reasons.append(
            f"LOW_CONFIDENCE: {detection.confidence:.2f} < {scoring.min_detection_confidence:.2f}"
        )

    if bay_coverage_ratio < scoring.min_bay_coverage_ratio:
        rejection_reasons.append(
            f"INSUFFICIENT_BAY_COVERAGE: {bay_coverage_ratio:.2%} < {scoring.min_bay_coverage_ratio:.2%}"
        )

    if vehicle_overlap_ratio < scoring.min_vehicle_overlap_ratio:
        rejection_reasons.append(
            f"INSUFFICIENT_VEHICLE_OVERLAP: {vehicle_overlap_ratio:.2%} < {scoring.min_vehicle_overlap_ratio:.2%}"
        )

    if scoring.require_center_or_coverage and not is_center_inside and bay_coverage_ratio < scoring.high_coverage_threshold:
        rejection_reasons.append(
            f"CENTER_OUTSIDE_AND_LOW_COVERAGE: Center outside and coverage {bay_coverage_ratio:.2%} < {scoring.high_coverage_threshold:.2%}"
        )

    is_occupied_evidence = (len(rejection_reasons) == 0)

    return BayOccupancyEvidence(
        bay_id="",
        operator_label="",
        frame_index=0,
        timestamp_seconds=0.0,
        intersection_area_px=intersection_area_px,
        bay_area_px=poly_area_px,
        vehicle_box_area_px=vehicle_box_area,
        bay_coverage_ratio=bay_coverage_ratio,
        vehicle_overlap_ratio=vehicle_overlap_ratio,
        is_center_inside=is_center_inside,
        max_confidence=detection.confidence,
        contributing_track_ids=[detection.track_id] if detection.track_id is not None else [],
        is_instant_evidence_occupied=is_occupied_evidence,
        rejection_reasons=rejection_reasons,
    )


class TemporalBayTracker:
    """
    Manages operational occupancy state and temporal hysteresis for a single parking space.
    Prevents high-frequency flickering through enter/exit frame counters and dropout tolerance.
    """

    def __init__(
        self,
        bay_id: str,
        operator_label: str,
        space_type: str,
        polygon_normalized: List[Dict[str, float]],
        temporal_config: TemporalConfig,
    ) -> None:
        self.bay_id = bay_id
        self.operator_label = operator_label
        self.space_type = space_type
        self.polygon_normalized = polygon_normalized
        self.config = temporal_config

        self.current_state: OccupancyState = OccupancyState.UNKNOWN
        self.consecutive_occupied_evidence: int = 0
        self.consecutive_vacant_evidence: int = 0
        self.dropout_frames_count: int = 0
        self.confidence: float = 0.0
        self.last_transition_frame: int = -1
        self.last_transition_timestamp: float = -1.0
        self.contributing_track_ids: List[int] = []
        self.history: List[BayOccupancyEvidence] = []

    def update_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        best_evidence: Optional[BayOccupancyEvidence],
    ) -> Tuple[BayStateSummary, Optional[OccupancyTimelineEntry]]:
        """
        Ingest the best geometric evidence for the current frame and update state machine.
        Returns the updated BayStateSummary and an optional OccupancyTimelineEntry on state transitions.
        """
        timeline_entry: Optional[OccupancyTimelineEntry] = None
        prev_state = self.current_state

        has_occupied_evidence = (best_evidence is not None and best_evidence.is_instant_evidence_occupied)

        if best_evidence is not None:
            best_evidence.bay_id = self.bay_id
            best_evidence.operator_label = self.operator_label
            best_evidence.frame_index = frame_idx
            best_evidence.timestamp_seconds = timestamp_sec
            self.history.append(best_evidence)
            self.confidence = best_evidence.max_confidence
            if best_evidence.contributing_track_ids:
                for tid in best_evidence.contributing_track_ids:
                    if tid not in self.contributing_track_ids:
                        self.contributing_track_ids.append(tid)

        if has_occupied_evidence:
            self.consecutive_occupied_evidence += 1
            self.consecutive_vacant_evidence = 0
            self.dropout_frames_count = 0

            # Transition to OCCUPIED if sustained for required frames or recovering from OCCLUDED
            if self.consecutive_occupied_evidence >= self.config.min_frames_occupied or (prev_state == OccupancyState.OCCLUDED and self.consecutive_occupied_evidence >= 1):
                if self.current_state != OccupancyState.OCCUPIED:
                    self.current_state = OccupancyState.OCCUPIED
                    self.last_transition_frame = frame_idx
                    self.last_transition_timestamp = timestamp_sec
                    timeline_entry = OccupancyTimelineEntry(
                        frame_index=frame_idx,
                        timestamp_seconds=timestamp_sec,
                        bay_id=self.bay_id,
                        operator_label=self.operator_label,
                        previous_state=prev_state,
                        new_state=OccupancyState.OCCUPIED,
                        trigger_reason=f"Sustained vehicle detection for {self.consecutive_occupied_evidence} consecutive frames",
                        confidence=self.confidence,
                        contributing_track_ids=list(self.contributing_track_ids),
                    )
        else:
            # No occupied evidence in this frame
            if self.current_state == OccupancyState.OCCUPIED:
                self.dropout_frames_count += 1
                if self.dropout_frames_count <= self.config.dropout_tolerance_frames:
                    # Enter temporary OCCLUDED state within dropout window
                    self.current_state = OccupancyState.OCCLUDED
                    self.last_transition_frame = frame_idx
                    self.last_transition_timestamp = timestamp_sec
                    timeline_entry = OccupancyTimelineEntry(
                        frame_index=frame_idx,
                        timestamp_seconds=timestamp_sec,
                        bay_id=self.bay_id,
                        operator_label=self.operator_label,
                        previous_state=prev_state,
                        new_state=OccupancyState.OCCLUDED,
                        trigger_reason=f"Temporary detector dropout / occlusion (frame {self.dropout_frames_count}/{self.config.dropout_tolerance_frames})",
                        confidence=self.confidence,
                        contributing_track_ids=list(self.contributing_track_ids),
                    )
                else:
                    self.consecutive_vacant_evidence += 1
                    self.consecutive_occupied_evidence = 0
            elif self.current_state == OccupancyState.OCCLUDED:
                self.dropout_frames_count += 1
                if self.dropout_frames_count > self.config.dropout_tolerance_frames:
                    self.consecutive_vacant_evidence += 1
                    self.consecutive_occupied_evidence = 0
            else:
                self.consecutive_vacant_evidence += 1
                self.consecutive_occupied_evidence = 0

            # Transition to VACANT if clear for required frames
            if self.consecutive_vacant_evidence >= self.config.min_frames_vacant:
                if self.current_state != OccupancyState.VACANT:
                    self.current_state = OccupancyState.VACANT
                    self.last_transition_frame = frame_idx
                    self.last_transition_timestamp = timestamp_sec
                    self.contributing_track_ids = []
                    self.confidence = 0.0
                    timeline_entry = OccupancyTimelineEntry(
                        frame_index=frame_idx,
                        timestamp_seconds=timestamp_sec,
                        bay_id=self.bay_id,
                        operator_label=self.operator_label,
                        previous_state=prev_state,
                        new_state=OccupancyState.VACANT,
                        trigger_reason=f"Space clear of vehicles for {self.consecutive_vacant_evidence} consecutive frames",
                        confidence=0.0,
                        contributing_track_ids=[],
                    )

        consecutive_frames_in_curr_state = (
            self.consecutive_occupied_evidence
            if self.current_state == OccupancyState.OCCUPIED
            else self.consecutive_vacant_evidence
        )

        summary = BayStateSummary(
            bay_id=self.bay_id,
            operator_label=self.operator_label,
            space_type=self.space_type,
            current_state=self.current_state,
            consecutive_frames_in_state=consecutive_frames_in_curr_state,
            confidence=self.confidence,
            last_transition_frame=self.last_transition_frame,
            last_transition_timestamp=self.last_transition_timestamp,
            polygon_normalized=self.polygon_normalized,
            contributing_track_ids=list(self.contributing_track_ids),
        )

        return summary, timeline_entry


class ParkingOccupancyEngine:
    """
    Main evaluation pipeline coordinating vehicle detections, polygon overlap,
    and temporal state tracking across a video stream.
    """

    def __init__(
        self,
        config: ParkingOccupancyConfig,
        parking_spaces: Sequence[Dict[str, Any]],
        video_width: int,
        video_height: int,
    ) -> None:
        self.config = config
        self.video_width = video_width
        self.video_height = video_height

        self.trackers: Dict[str, TemporalBayTracker] = {}
        self.bay_polygons_px: Dict[str, np.ndarray] = {}
        self.timeline: List[OccupancyTimelineEntry] = []

        for space in parking_spaces:
            bay_id = str(space.get("id", ""))
            label = str(space.get("operator_label", f"Bay-{bay_id[:4]}"))
            stype = str(space.get("space_type", "STANDARD"))
            poly_norm = space.get("polygon_normalized", [])

            tracker = TemporalBayTracker(
                bay_id=bay_id,
                operator_label=label,
                space_type=stype,
                polygon_normalized=poly_norm,
                temporal_config=self.config.temporal,
            )
            self.trackers[bay_id] = tracker
            self.bay_polygons_px[bay_id] = normalized_polygon_to_pixels(poly_norm, video_width, video_height)

    def process_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        detections: List[VehicleDetection],
    ) -> FrameOccupancyResult:
        """
        Evaluate all parking bays on a single frame with vehicle detections.
        """
        bay_summaries: Dict[str, BayStateSummary] = {}
        frame_transitions: List[OccupancyTimelineEntry] = []

        for bay_id, tracker in self.trackers.items():
            bay_poly = self.bay_polygons_px[bay_id]

            # Find best overlapping vehicle detection for this bay
            best_evidence: Optional[BayOccupancyEvidence] = None
            max_score = -1.0

            for det in detections:
                ev = calculate_bay_vehicle_overlap(bay_poly, det, self.config.scoring)
                # Score combines coverage, vehicle overlap, and confidence
                score = (ev.bay_coverage_ratio * 0.5) + (ev.vehicle_overlap_ratio * 0.3) + (ev.max_confidence * 0.2)
                if ev.is_center_inside:
                    score += 0.2

                if score > max_score:
                    max_score = score
                    best_evidence = ev

            summary, timeline_entry = tracker.update_frame(frame_idx, timestamp_sec, best_evidence)
            bay_summaries[bay_id] = summary
            if timeline_entry:
                self.timeline.append(timeline_entry)
                frame_transitions.append(timeline_entry)

        occupied = sum(1 for s in bay_summaries.values() if s.current_state == OccupancyState.OCCUPIED)
        vacant = sum(1 for s in bay_summaries.values() if s.current_state == OccupancyState.VACANT)
        unknown = sum(1 for s in bay_summaries.values() if s.current_state == OccupancyState.UNKNOWN)
        occluded = sum(1 for s in bay_summaries.values() if s.current_state == OccupancyState.OCCLUDED)

        return FrameOccupancyResult(
            frame_index=frame_idx,
            timestamp_seconds=timestamp_sec,
            bay_states=bay_summaries,
            vehicle_detections=detections,
            total_bays=len(bay_summaries),
            occupied_count=occupied,
            vacant_count=vacant,
            unknown_count=unknown,
            occluded_count=occluded,
            state_transitions=frame_transitions,
        )
