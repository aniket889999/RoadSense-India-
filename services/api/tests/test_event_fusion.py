"""Unit tests for the Road Event Fusion clustering algorithm."""

import pytest
from services.api.app.services.event_fusion import (
    DetectionCandidate,
    cluster_detections_into_road_events,
    fuse_tracks_into_road_events,
)
from src.tracking.bytetrack_adapter import TrackObservation, TrackSummary


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


def test_clustering_with_bytetrack_ids():
    # Detections carrying ByteTrack track IDs
    det1 = DetectionCandidate(
        id="det-1", frame_index=5, timestamp_seconds=1.0, confidence=0.80,
        x_min=100.0, y_min=100.0, x_max=200.0, y_max=200.0, track_id=42,
    )
    det2 = DetectionCandidate(
        id="det-2", frame_index=6, timestamp_seconds=1.2, confidence=0.90,
        x_min=102.0, y_min=101.0, x_max=202.0, y_max=201.0, track_id=42,
    )
    events = cluster_detections_into_road_events([det1, det2])
    assert len(events) == 1
    assert events[0].track_id == 42
    assert events[0].support_count == 2
    assert events[0].representative_confidence == 0.90


def test_fuse_tracks_into_road_events():
    obs1 = TrackObservation(
        session_id="sess-1", track_id=7, frame_number=2, timestamp_seconds=0.4,
        bbox=(50.0, 50.0, 120.0, 120.0), confidence=0.80,
    )
    obs2 = TrackObservation(
        session_id="sess-1", track_id=7, frame_number=3, timestamp_seconds=0.6,
        bbox=(52.0, 51.0, 122.0, 121.0), confidence=0.94,
    )
    track = TrackSummary(
        track_id=7, session_id="sess-1", first_seen_seconds=0.4, last_seen_seconds=0.6,
        first_frame=2, last_frame=3, observation_count=2, max_confidence=0.94,
        avg_confidence=0.87, representative_frame=3, representative_bbox=(52.0, 51.0, 122.0, 121.0),
        observations=[obs1, obs2], is_stable=True,
    )

    events = fuse_tracks_into_road_events([track])
    assert len(events) == 1
    assert events[0].track_id == 7
    assert events[0].representative_confidence == 0.94
    assert events[0].support_count == 2
