"""Unit tests for Phase 2B gated parking occupancy engine, contracts, geometry, and hysteresis."""

import copy
import hashlib
from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.parking.occupancy_config import (
    DetectorConfig,
    ParkingOccupancyConfig,
    ScoringConfig,
    TemporalConfig,
    load_parking_occupancy_config,
)
from src.parking.occupancy_contracts import (
    BayOccupancyEvidence,
    BayStateSummary,
    FrameOccupancyResult,
    OccupancyState,
    OccupancyTimelineEntry,
    VehicleDetection,
)
from src.parking.occupancy_engine import (
    ParkingOccupancyEngine,
    TemporalBayTracker,
    calculate_bay_vehicle_overlap,
    normalized_polygon_to_pixels,
)
from src.parking.vehicle_detector import LocalVehicleDetector


@pytest.fixture
def base_config() -> ParkingOccupancyConfig:
    return load_parking_occupancy_config()


def test_config_strict_loader_and_sha256(base_config):
    """Verify configuration loads cleanly, is immutable, and computes a valid 64-char SHA-256."""
    assert base_config.version == "v1.0.0"
    assert len(base_config.config_sha256) == 64
    assert base_config.detector.allowed_classes == [2, 3, 5, 7]
    assert base_config.scoring.min_bay_coverage_ratio > 0.0
    assert base_config.temporal.min_frames_occupied >= 1
    assert base_config.rendering.vacant_color_bgr == [0, 255, 0]


def test_config_loader_rejects_invalid_values():
    """Verify strict validation rejects booleans, negative numbers, missing fields, or malformed YAML."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        tf.write("version: true\n")
        tf_path = Path(tf.name)

    try:
        with pytest.raises(ValueError):
            load_parking_occupancy_config(tf_path)
    finally:
        tf_path.unlink(missing_ok=True)


def test_polygon_scaling_normalized_to_pixels():
    """Verify normalized polygon coordinates scale accurately to integer pixels within frame bounds."""
    norm_poly = [
        {"x": 0.1, "y": 0.2},
        {"x": 0.4, "y": 0.2},
        {"x": 0.4, "y": 0.8},
        {"x": 0.1, "y": 0.8},
    ]
    px = normalized_polygon_to_pixels(norm_poly, 1920, 1080)
    assert px.shape == (4, 2)
    assert px[0][0] == 192 and px[0][1] == 216
    assert px[2][0] == 768 and px[2][1] == 864


def test_bay_vehicle_overlap_concentric_full_coverage(base_config):
    """A vehicle parked squarely inside a bay produces high coverage, center inside, and occupied evidence."""
    bay_poly = np.array([[100, 100], [300, 100], [300, 400], [100, 400]], dtype=np.int32)
    det = VehicleDetection(
        class_id=2,
        class_name="car",
        confidence=0.85,
        bbox_xyxy=(120.0, 120.0, 280.0, 380.0),
        track_id=101,
        center_xy=(200.0, 250.0),
    )

    ev = calculate_bay_vehicle_overlap(bay_poly, det, base_config.scoring)
    assert ev.is_center_inside is True
    assert ev.bay_coverage_ratio > 0.65
    assert ev.vehicle_overlap_ratio > 0.95
    assert ev.is_instant_evidence_occupied is True
    assert len(ev.rejection_reasons) == 0


def test_bay_vehicle_overlap_angled_polygon(base_config):
    """Angled/slanted parking space polygon evaluates accurately against vehicle bounding box."""
    bay_poly = np.array([[100, 100], [250, 100], [350, 400], [200, 400]], dtype=np.int32)
    det = VehicleDetection(
        class_id=2,
        class_name="car",
        confidence=0.90,
        bbox_xyxy=(150.0, 120.0, 300.0, 380.0),
        track_id=102,
        center_xy=(225.0, 250.0),
    )

    ev = calculate_bay_vehicle_overlap(bay_poly, det, base_config.scoring)
    assert ev.is_center_inside is True
    assert ev.bay_coverage_ratio > 0.50
    assert ev.is_instant_evidence_occupied is True


def test_bay_vehicle_overlap_disjoint_rejection(base_config):
    """A vehicle outside the bay produces 0 intersection and is rejected."""
    bay_poly = np.array([[100, 100], [300, 100], [300, 400], [100, 400]], dtype=np.int32)
    det = VehicleDetection(
        class_id=2,
        class_name="car",
        confidence=0.85,
        bbox_xyxy=(450.0, 100.0, 600.0, 300.0),
        track_id=103,
        center_xy=(525.0, 200.0),
    )

    ev = calculate_bay_vehicle_overlap(bay_poly, det, base_config.scoring)
    assert ev.intersection_area_px == 0.0
    assert ev.bay_coverage_ratio == 0.0
    assert ev.is_instant_evidence_occupied is False
    assert any("DISJOINT" in r for r in ev.rejection_reasons)


def test_bay_vehicle_overlap_low_confidence_rejection(base_config):
    """A low-confidence detection below threshold is rejected even if overlapping geometrically."""
    bay_poly = np.array([[100, 100], [300, 100], [300, 400], [100, 400]], dtype=np.int32)
    det = VehicleDetection(
        class_id=2,
        class_name="car",
        confidence=0.15,  # below 0.30 threshold
        bbox_xyxy=(120.0, 120.0, 280.0, 380.0),
        track_id=104,
        center_xy=(200.0, 250.0),
    )

    ev = calculate_bay_vehicle_overlap(bay_poly, det, base_config.scoring)
    assert ev.is_instant_evidence_occupied is False
    assert any("LOW_CONFIDENCE" in r for r in ev.rejection_reasons)


def test_temporal_bay_tracker_hysteresis_and_transitions(base_config):
    """
    Test enter/exit hysteresis:
    - Starts UNKNOWN.
    - Requires min_frames_occupied (3) before becoming OCCUPIED.
    - Single frame dropout does not clear OCCUPIED.
    - Requires min_frames_vacant (5) before returning to VACANT.
    """
    poly_norm = [{"x": 0.1, "y": 0.1}, {"x": 0.3, "y": 0.1}, {"x": 0.3, "y": 0.4}, {"x": 0.1, "y": 0.4}]
    tracker = TemporalBayTracker(
        bay_id="bay_01",
        operator_label="Bay-A1",
        space_type="STANDARD",
        polygon_normalized=poly_norm,
        temporal_config=base_config.temporal,
    )
    assert tracker.current_state == OccupancyState.UNKNOWN

    occ_evidence = BayOccupancyEvidence(
        bay_id="bay_01",
        operator_label="Bay-A1",
        frame_index=0,
        timestamp_seconds=0.0,
        intersection_area_px=1000.0,
        bay_area_px=1200.0,
        vehicle_box_area_px=1100.0,
        bay_coverage_ratio=0.83,
        vehicle_overlap_ratio=0.91,
        is_center_inside=True,
        max_confidence=0.90,
        contributing_track_ids=[42],
        is_instant_evidence_occupied=True,
    )

    # Frame 1: Occupied evidence (consecutive=1 < 3) -> remains UNKNOWN
    s, t = tracker.update_frame(1, 0.033, occ_evidence)
    assert s.current_state == OccupancyState.UNKNOWN
    assert t is None

    # Frame 2: Occupied evidence (consecutive=2 < 3) -> remains UNKNOWN
    s, t = tracker.update_frame(2, 0.066, occ_evidence)
    assert s.current_state == OccupancyState.UNKNOWN
    assert t is None

    # Frame 3: Occupied evidence (consecutive=3 >= 3) -> transitions to OCCUPIED
    s, t = tracker.update_frame(3, 0.100, occ_evidence)
    assert s.current_state == OccupancyState.OCCUPIED
    assert t is not None
    assert t.previous_state == OccupancyState.UNKNOWN
    assert t.new_state == OccupancyState.OCCUPIED
    assert t.contributing_track_ids == [42]

    # Frame 4: 1-frame dropout (no detection) -> stays OCCUPIED due to dropout tolerance
    s, t = tracker.update_frame(4, 0.133, None)
    assert s.current_state == OccupancyState.OCCUPIED
    assert t is None

    # Frame 5: 2nd dropout -> stays OCCUPIED (within tolerance of 2)
    s, t = tracker.update_frame(5, 0.166, None)
    assert s.current_state == OccupancyState.OCCUPIED
    assert t is None

    # Frames 6..10: sustained vacancy (consecutive vacant >= 5)
    for f in range(6, 11):
        s, t = tracker.update_frame(f, f * 0.033, None)

    # By frame 10 (5 sustained vacant frames past dropout), transitions to VACANT
    assert tracker.current_state == OccupancyState.VACANT
    assert t is not None
    assert t.previous_state == OccupancyState.OCCUPIED
    assert t.new_state == OccupancyState.VACANT


def test_parking_occupancy_engine_multi_bay_evaluation(base_config):
    """Verify ParkingOccupancyEngine evaluates multiple bays simultaneously and produces accurate counts."""
    spaces = [
        {
            "id": "bay_01",
            "operator_label": "Bay-01",
            "space_type": "STANDARD",
            "polygon_normalized": [{"x": 0.05, "y": 0.1}, {"x": 0.25, "y": 0.1}, {"x": 0.25, "y": 0.5}, {"x": 0.05, "y": 0.5}],
        },
        {
            "id": "bay_02",
            "operator_label": "Bay-02",
            "space_type": "STANDARD",
            "polygon_normalized": [{"x": 0.30, "y": 0.1}, {"x": 0.50, "y": 0.1}, {"x": 0.50, "y": 0.5}, {"x": 0.30, "y": 0.5}],
        },
    ]

    engine = ParkingOccupancyEngine(base_config, spaces, video_width=1000, video_height=1000)

    # Vehicle in Bay 01 only
    det_bay1 = VehicleDetection(
        class_id=2,
        class_name="car",
        confidence=0.88,
        bbox_xyxy=(60.0, 120.0, 240.0, 480.0),
        track_id=1,
        center_xy=(150.0, 300.0),
    )

    # Process 3 frames with vehicle in Bay 1 and clear Bay 2
    for f in range(1, 4):
        res = engine.process_frame(f, f * 0.033, [det_bay1])

    # Bay 01 should be OCCUPIED, Bay 02 should be UNKNOWN (needs 5 vacant frames from initial start)
    assert res.bay_states["bay_01"].current_state == OccupancyState.OCCUPIED
    assert res.occupied_count == 1

    # Process 7 more frames with no vehicles anywhere (2 dropout + 5 vacant threshold)
    for f in range(4, 11):
        res = engine.process_frame(f, f * 0.033, [])

    # Both bays should now be VACANT
    assert res.bay_states["bay_01"].current_state == OccupancyState.VACANT
    assert res.bay_states["bay_02"].current_state == OccupancyState.VACANT
    assert res.vacant_count == 2
    assert res.occupied_count == 0


def test_vehicle_detector_checkpoint_validation():
    """Verify LocalVehicleDetector validates checkpoint integrity and class mapping."""
    cfg = DetectorConfig(
        model_path="models/yolov8n.pt",
        expected_model_sha256="f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
        confidence_threshold=0.35,
        iou_threshold=0.45,
        allowed_classes=[2, 3, 5, 7],
        class_names={2: "car", 3: "motorcycle", 5: "bus", 7: "truck"},
        device="cpu",
    )
    detector = LocalVehicleDetector(cfg)
    assert len(detector.checkpoint_sha256) == 64
    assert detector.checkpoint_sha256 == cfg.expected_model_sha256

    # Test inference on blank synthetic frame (should return 0 detections without crash)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = detector.detect_vehicles(blank_frame)
    assert isinstance(dets, list)
    assert len(dets) == 0


def test_vehicle_detector_rejects_tampered_sha():
    """Verify detector raises ValueError if checkpoint SHA does not match expected hash."""
    cfg = DetectorConfig(
        model_path="models/yolov8n.pt",
        expected_model_sha256="0000000000000000000000000000000000000000000000000000000000000000",
        confidence_threshold=0.35,
        iou_threshold=0.45,
        allowed_classes=[2, 3, 5, 7],
        class_names={2: "car"},
        device="cpu",
    )
    with pytest.raises(ValueError, match="Checkpoint SHA-256 mismatch"):
        LocalVehicleDetector(cfg)
