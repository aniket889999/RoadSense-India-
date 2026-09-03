"""Road Event Fusion: Temporal, Spatial, and ByteTrack Clustering for Repeated Pothole Detections."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel

from src.tracking.bytetrack_adapter import TrackSummary


class DetectionCandidate(BaseModel):
    id: str | None = None
    frame_index: int
    timestamp_seconds: float
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    track_id: int | None = None


class ProposedRoadEvent(BaseModel):
    first_seen_seconds: float
    last_seen_seconds: float
    first_frame_index: int
    last_frame_index: int
    representative_detection_id: str | None = None
    representative_confidence: float
    representative_bbox: dict[str, float]
    support_count: int
    track_id: int | None = None
    detections: List[DetectionCandidate] = []


def _compute_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def _compute_centroid_distance(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    c1_x = (box1[0] + box1[2]) / 2.0
    c1_y = (box1[1] + box1[3]) / 2.0
    c2_x = (box2[0] + box2[2]) / 2.0
    c2_y = (box2[1] + box2[3]) / 2.0
    return math.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)


def fuse_tracks_into_road_events(
    tracks: Sequence[TrackSummary],
    *,
    min_confidence: float = 0.25,
) -> List[ProposedRoadEvent]:
    """Convert stable ByteTrack tracks into candidate ProposedRoadEvent records."""
    events: List[ProposedRoadEvent] = []

    for track in tracks:
        if not track.is_stable or track.max_confidence < min_confidence:
            continue

        rep_box = track.representative_bbox
        candidates = [
            DetectionCandidate(
                frame_index=obs.frame_number,
                timestamp_seconds=obs.timestamp_seconds,
                confidence=obs.confidence,
                x_min=obs.bbox[0],
                y_min=obs.bbox[1],
                x_max=obs.bbox[2],
                y_max=obs.bbox[3],
                track_id=track.track_id,
            )
            for obs in track.observations
        ]

        event = ProposedRoadEvent(
            first_seen_seconds=track.first_seen_seconds,
            last_seen_seconds=track.last_seen_seconds,
            first_frame_index=track.first_frame,
            last_frame_index=track.last_frame,
            representative_confidence=track.max_confidence,
            representative_bbox={
                "x_min": rep_box[0],
                "y_min": rep_box[1],
                "x_max": rep_box[2],
                "y_max": rep_box[3],
            },
            support_count=track.observation_count,
            track_id=track.track_id,
            detections=candidates,
        )
        events.append(event)

    return sorted(events, key=lambda e: (e.first_seen_seconds, e.first_frame_index))


def cluster_detections_into_road_events(
    detections: Sequence[DetectionCandidate],
    *,
    max_time_gap_seconds: float = 1.5,
    min_iou: float = 0.15,
    max_centroid_distance_px: float = 80.0,
) -> List[ProposedRoadEvent]:
    """Cluster consecutive frame detections into deduplicated Road Events.

    If detections have track_id assigned, they group by track_id first.
    Otherwise, temporal and spatial clustering (IoU or centroid distance) is applied.
    """
    if not detections:
        return []

    # 1. Check if track_ids are present
    tracked_groups: Dict[int, List[DetectionCandidate]] = {}
    untracked_dets: List[DetectionCandidate] = []

    for d in detections:
        if d.track_id is not None:
            tracked_groups.setdefault(d.track_id, []).append(d)
        else:
            untracked_dets.append(d)

    road_events: List[ProposedRoadEvent] = []

    # Group tracked detections
    for track_id, group in tracked_groups.items():
        sorted_group = sorted(group, key=lambda d: (d.timestamp_seconds, d.frame_index))
        best_det = max(sorted_group, key=lambda d: d.confidence)
        first_det = sorted_group[0]
        last_det = sorted_group[-1]

        event = ProposedRoadEvent(
            first_seen_seconds=first_det.timestamp_seconds,
            last_seen_seconds=last_det.timestamp_seconds,
            first_frame_index=first_det.frame_index,
            last_frame_index=last_det.frame_index,
            representative_detection_id=best_det.id,
            representative_confidence=best_det.confidence,
            representative_bbox={
                "x_min": best_det.x_min,
                "y_min": best_det.y_min,
                "x_max": best_det.x_max,
                "y_max": best_det.y_max,
            },
            support_count=len(sorted_group),
            track_id=track_id,
            detections=sorted_group,
        )
        road_events.append(event)

    # 2. Cluster untracked detections with temporal/spatial logic
    if untracked_dets:
        sorted_dets = sorted(untracked_dets, key=lambda d: (d.timestamp_seconds, d.frame_index))
        clusters: List[List[DetectionCandidate]] = []

        for det in sorted_dets:
            box = (det.x_min, det.y_min, det.x_max, det.y_max)
            matched_cluster = None

            for cluster in reversed(clusters):
                last_det = cluster[-1]
                time_gap = det.timestamp_seconds - last_det.timestamp_seconds

                if time_gap <= max_time_gap_seconds:
                    last_box = (last_det.x_min, last_det.y_min, last_det.x_max, last_det.y_max)
                    iou = _compute_iou(box, last_box)
                    dist = _compute_centroid_distance(box, last_box)

                    if iou >= min_iou or dist <= max_centroid_distance_px:
                        matched_cluster = cluster
                        break
                else:
                    break

            if matched_cluster is not None:
                matched_cluster.append(det)
            else:
                clusters.append([det])

        for cluster in clusters:
            best_det = max(cluster, key=lambda d: d.confidence)
            first_det = cluster[0]
            last_det = cluster[-1]

            event = ProposedRoadEvent(
                first_seen_seconds=first_det.timestamp_seconds,
                last_seen_seconds=last_det.timestamp_seconds,
                first_frame_index=first_det.frame_index,
                last_frame_index=last_det.frame_index,
                representative_detection_id=best_det.id,
                representative_confidence=best_det.confidence,
                representative_bbox={
                    "x_min": best_det.x_min,
                    "y_min": best_det.y_min,
                    "x_max": best_det.x_max,
                    "y_max": best_det.y_max,
                },
                support_count=len(cluster),
                track_id=None,
                detections=cluster,
            )
            road_events.append(event)

    return sorted(road_events, key=lambda e: (e.first_seen_seconds, e.first_frame_index))
