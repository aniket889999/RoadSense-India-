"""Road Event Fusion: Temporal and Spatial Clustering for Repeated Pothole Detections."""

from __future__ import annotations

import math
from typing import List, Sequence
from pydantic import BaseModel


class DetectionCandidate(BaseModel):
    id: str | None = None
    frame_index: int
    timestamp_seconds: float
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class ProposedRoadEvent(BaseModel):
    first_seen_seconds: float
    last_seen_seconds: float
    first_frame_index: int
    last_frame_index: int
    representative_detection_id: str | None = None
    representative_confidence: float
    representative_bbox: dict[str, float]
    support_count: int
    detections: List[DetectionCandidate]


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


def cluster_detections_into_road_events(
    detections: Sequence[DetectionCandidate],
    *,
    max_time_gap_seconds: float = 1.5,
    min_iou: float = 0.15,
    max_centroid_distance_px: float = 80.0,
) -> List[ProposedRoadEvent]:
    """Cluster consecutive frame detections into deduplicated Road Events.

    Detections are sorted by timestamp/frame. A detection joins the current cluster if:
    1. Time gap from the cluster's latest detection <= max_time_gap_seconds AND
    2. Spatial overlap: IoU >= min_iou OR centroid distance <= max_centroid_distance_px.
    Otherwise, a new Road Event cluster is started.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: (d.timestamp_seconds, d.frame_index))
    clusters: List[List[DetectionCandidate]] = []

    for det in sorted_dets:
        box = (det.x_min, det.y_min, det.x_max, det.y_max)
        matched_cluster = None

        # Check existing active clusters (most recent first)
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
                # Detections are sorted, so further clusters are even older
                break

        if matched_cluster is not None:
            matched_cluster.append(det)
        else:
            clusters.append([det])

    # Convert clusters to ProposedRoadEvent models
    road_events: List[ProposedRoadEvent] = []
    for cluster in clusters:
        # Find representative detection (highest confidence)
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
            detections=cluster,
        )
        road_events.append(event)

    return road_events
