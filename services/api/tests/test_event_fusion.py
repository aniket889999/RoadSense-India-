"""Unit tests for the Road Event Fusion clustering algorithm."""

import pytest
from services.api.app.services.event_fusion import (
    DetectionCandidate,
    cluster_detections_into_road_events,
)


def test_empty_detections():
    events = cluster_detections_into_road_events([])
    assert events == []


def test_single_detection():
    det = DetectionCandidate(
        id="det-1",
        frame_index=10,
        timestamp_seconds=2.0,
        confidence=0.85,
        x_min=100.0,
        y_min=100.0,
        x_max=200.0,
        y_max=200.0,
    )
    events = cluster_detections_into_road_events([det])
    assert len(events) == 1
    ev = events[0]
    assert ev.first_seen_seconds == 2.0
    assert ev.last_seen_seconds == 2.0
    assert ev.representative_confidence == 0.85
    assert ev.support_count == 1
    assert ev.representative_detection_id == "det-1"


def test_clustering_consecutive_overlapping_detections():
    # 3 detections of the same pothole over 3 consecutive frames
    det1 = DetectionCandidate(
        id="det-1", frame_index=10, timestamp_seconds=2.0, confidence=0.70,
        x_min=100.0, y_min=100.0, x_max=200.0, y_max=200.0,
    )
    det2 = DetectionCandidate(
        id="det-2", frame_index=11, timestamp_seconds=2.2, confidence=0.92,
        x_min=105.0, y_min=105.0, x_max=205.0, y_max=205.0,
    )
    det3 = DetectionCandidate(
        id="det-3", frame_index=12, timestamp_seconds=2.4, confidence=0.65,
        x_min=110.0, y_min=110.0, x_max=210.0, y_max=210.0,
    )
    events = cluster_detections_into_road_events([det1, det2, det3])
    assert len(events) == 1
    ev = events[0]
    assert ev.first_seen_seconds == 2.0
    assert ev.last_seen_seconds == 2.4
    assert ev.first_frame_index == 10
    assert ev.last_frame_index == 12
    assert ev.support_count == 3
    # Representative detection must be det-2 with confidence 0.92
    assert ev.representative_confidence == 0.92
    assert ev.representative_detection_id == "det-2"


def test_clustering_separate_potholes_by_time():
    # Two detections far apart in time (> 1.5s gap)
    det1 = DetectionCandidate(
        id="det-1", frame_index=10, timestamp_seconds=2.0, confidence=0.80,
        x_min=100.0, y_min=100.0, x_max=200.0, y_max=200.0,
    )
    det2 = DetectionCandidate(
        id="det-2", frame_index=30, timestamp_seconds=6.0, confidence=0.75,
        x_min=100.0, y_min=100.0, x_max=200.0, y_max=200.0,
    )
    events = cluster_detections_into_road_events([det1, det2])
    assert len(events) == 2
    assert events[0].first_seen_seconds == 2.0
    assert events[1].first_seen_seconds == 6.0


def test_clustering_separate_potholes_by_space():
    # Two detections in the same frame but far apart spatially
    det1 = DetectionCandidate(
        id="det-1", frame_index=10, timestamp_seconds=2.0, confidence=0.80,
        x_min=10.0, y_min=10.0, x_max=50.0, y_max=50.0,
    )
    det2 = DetectionCandidate(
        id="det-2", frame_index=10, timestamp_seconds=2.0, confidence=0.85,
        x_min=500.0, y_min=500.0, x_max=600.0, y_max=600.0,
    )
    events = cluster_detections_into_road_events([det1, det2])
    assert len(events) == 2
