"""Unit tests for ByteTrack adapter and temporal tracking."""

import tempfile
from pathlib import Path
import pytest

from src.tracking.bytetrack_adapter import (
    ByteTrackAdapter,
    ByteTrackConfig,
    load_bytetrack_config,
)


def test_load_bytetrack_config():
    config_path = Path("configs/tracking/bytetrack_default.yaml")
    cfg = load_bytetrack_config(config_path)

    assert cfg.track_high_thresh > 0.0
    assert cfg.match_thresh > 0.0
    assert len(cfg.config_sha256) == 64
    assert cfg.min_hits >= 1


def test_bytetrack_tracking_lifecycle():
    config_path = Path("configs/tracking/bytetrack_default.yaml")
    cfg = load_bytetrack_config(config_path)
    tracker = ByteTrackAdapter(cfg)

    # Frame 0: New pothole detection
    obs0 = tracker.update(
        frame_number=0,
        timestamp_seconds=0.0,
        detections=[{"x_min": 100.0, "y_min": 100.0, "x_max": 180.0, "y_max": 160.0, "confidence": 0.85}],
        session_id="sess-test-1",
        model_sha256="test-sha",
    )
    assert len(obs0) == 1
    track_id = obs0[0].track_id

    # Frame 1: Same pothole slightly shifted
    obs1 = tracker.update(
        frame_number=1,
        timestamp_seconds=0.1,
        detections=[{"x_min": 102.0, "y_min": 101.0, "x_max": 182.0, "y_max": 161.0, "confidence": 0.88}],
        session_id="sess-test-1",
        model_sha256="test-sha",
    )
    assert len(obs1) == 1
    assert obs1[0].track_id == track_id

    # Stable tracks should contain track_id
    stable = tracker.get_stable_tracks()
    assert any(t.track_id == track_id for t in stable)


def test_tracker_resets_on_timestamp_discontinuity():
    config_path = Path("configs/tracking/bytetrack_default.yaml")
    cfg = load_bytetrack_config(config_path)
    tracker = ByteTrackAdapter(cfg)

    tracker.update(
        frame_number=0,
        timestamp_seconds=5.0,
        detections=[{"x_min": 50, "y_min": 50, "x_max": 100, "y_max": 100, "confidence": 0.9}],
        session_id="sess-test-2",
    )
    assert len(tracker.get_all_tracks()) == 1

    # Jump backward in time (e.g. video seek) -> resets
    tracker.update(
        frame_number=1,
        timestamp_seconds=1.0,
        detections=[{"x_min": 50, "y_min": 50, "x_max": 100, "y_max": 100, "confidence": 0.9}],
        session_id="sess-test-2",
    )
    # The old track should be cleared
    assert len(tracker.get_all_tracks()) == 1
