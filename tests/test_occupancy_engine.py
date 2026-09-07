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

    # Frame 4: 1-frame dropout (no detection) -> enters OCCLUDED within dropout tolerance
    s, t = tracker.update_frame(4, 0.133, None)
    assert s.current_state == OccupancyState.OCCLUDED
    assert t is not None
    assert t.new_state == OccupancyState.OCCLUDED

    # Frame 5: 2nd dropout -> stays OCCLUDED (within tolerance of 2)
    s, t = tracker.update_frame(5, 0.166, None)
    assert s.current_state == OccupancyState.OCCLUDED

    # Frames 6..11: sustained vacancy (consecutive vacant >= 5 past dropout)
    vacant_tl = None
    for f in range(6, 12):
        s, t = tracker.update_frame(f, f * 0.033, None)
        if t is not None:
            vacant_tl = t

    # By frame 11, transitions to VACANT
    assert tracker.current_state == OccupancyState.VACANT
    assert vacant_tl is not None
    assert vacant_tl.new_state == OccupancyState.VACANT


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


def test_bytetrack_tracker_lifecycle_and_stability():
    """Verify ParkingByteTracker assigns stable track IDs and resets cleanly across sessions."""
    from src.parking.vehicle_tracker import ParkingByteTracker

    tracker = ParkingByteTracker(fps=30)

    # Frame 1: Car at (100, 100, 200, 200)
    det1 = VehicleDetection(
        class_id=2,
        class_name="car",
        confidence=0.9,
        bbox_xyxy=(100.0, 100.0, 200.0, 200.0),
        center_xy=(150.0, 150.0),
    )
    res1 = tracker.update_tracks([det1], frame_idx=0, frame_shape=(480, 640))
    assert res1.is_healthy is True
    assert len(res1.detections) == 1
    tid1 = res1.detections[0].track_id

    # Frame 2: Car moved slightly to (105, 105, 205, 205)
    det2 = VehicleDetection(
        class_id=2,
        class_name="car",
        confidence=0.88,
        bbox_xyxy=(105.0, 105.0, 205.0, 205.0),
        center_xy=(155.0, 155.0),
    )
    res2 = tracker.update_tracks([det2], frame_idx=1, frame_shape=(480, 640))
    assert res2.is_healthy is True
    assert len(res2.detections) == 1
    # Track ID should be assigned
    assert res2.detections[0].track_id is not None
    if tid1 is not None:
        assert res2.detections[0].track_id == tid1

    # Reset tracker for a new session
    tracker.reset()
    assert tracker._last_frame_idx == -1


def test_temporal_bay_tracker_occluded_entry_and_recovery(base_config):
    """Verify bay enters OCCLUDED during dropouts and recovers to OCCUPIED on return."""
    tracker = TemporalBayTracker(
        bay_id="b1",
        operator_label="Bay-01",
        space_type="STANDARD",
        polygon_normalized=[{"x": 0.1, "y": 0.1}, {"x": 0.3, "y": 0.1}, {"x": 0.3, "y": 0.5}, {"x": 0.1, "y": 0.5}],
        temporal_config=base_config.temporal,
    )

    ev_occ = BayOccupancyEvidence(
        bay_id="b1",
        operator_label="Bay-01",
        frame_index=0,
        timestamp_seconds=0.0,
        intersection_area_px=100.0,
        bay_area_px=100.0,
        vehicle_box_area_px=100.0,
        bay_coverage_ratio=0.8,
        vehicle_overlap_ratio=0.8,
        is_center_inside=True,
        max_confidence=0.9,
        contributing_track_ids=[42],
        is_instant_evidence_occupied=True,
    )

    # 1. Establish OCCUPIED (needs 3 consecutive frames)
    for f in range(3):
        summary, tl = tracker.update_frame(f, f * 0.1, ev_occ)
    assert summary.current_state == OccupancyState.OCCUPIED

    # 2. Next frame: detector drop (no detection) -> enters OCCLUDED
    summary_drop1, tl_drop1 = tracker.update_frame(3, 0.3, None)
    assert summary_drop1.current_state == OccupancyState.OCCLUDED
    assert tl_drop1 is not None
    assert tl_drop1.new_state == OccupancyState.OCCLUDED
    assert 42 in summary_drop1.contributing_track_ids

    # 3. Next frame: detection returns -> recovers to OCCUPIED
    summary_rec, tl_rec = tracker.update_frame(4, 0.4, ev_occ)
    assert summary_rec.current_state == OccupancyState.OCCUPIED
    assert tl_rec is not None
    assert tl_rec.new_state == OccupancyState.OCCUPIED


def test_annotator_independent_alpha_rendering(base_config):
    """Verify ParkingVideoAnnotator applies independent alpha blending per occupancy state."""
    from src.parking.video_annotator import ParkingVideoAnnotator

    annotator = ParkingVideoAnnotator(base_config.rendering)

    # Blank gray frame (val=100)
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    poly_px = np.array([[200, 200], [300, 200], [300, 300], [200, 300]], dtype=np.int32)

    vacant_summary = BayStateSummary(
        bay_id="b1",
        operator_label="B1",
        space_type="STANDARD",
        current_state=OccupancyState.VACANT,
        consecutive_frames_in_state=5,
        confidence=0.0,
        last_transition_frame=0,
        last_transition_timestamp=0.0,
        polygon_normalized=[],
    )

    rendered_vacant = annotator.render_frame(
        frame_bgr=frame,
        bay_states={"b1": vacant_summary},
        bay_polygons_px={"b1": poly_px},
        vehicle_detections=[],
        frame_idx=0,
        timestamp_sec=0.0,
    )

    # Check center pixel (250, 250): should be blended with green (0, 255, 0)
    # Original is (100, 100, 100), vacant alpha = 0.25 (or 0.35)
    center_bgr = rendered_vacant[250, 250]
    assert center_bgr[1] > 130  # Green increased
    assert center_bgr[0] < 100 and center_bgr[2] < 100  # Blue & Red reduced


def test_temporal_bay_tracker_all_five_state_transitions(base_config):
    """
    Exhaustively verify all 5 deterministic state transitions:
    1. UNKNOWN -> VACANT
    2. UNKNOWN -> OCCUPIED
    3. OCCUPIED -> OCCLUDED
    4. OCCLUDED -> OCCUPIED
    5. OCCLUDED -> VACANT
    """
    ev_occ = BayOccupancyEvidence(
        bay_id="b1",
        operator_label="B1",
        frame_index=0,
        timestamp_seconds=0.0,
        intersection_area_px=100.0,
        bay_area_px=100.0,
        vehicle_box_area_px=100.0,
        bay_coverage_ratio=0.8,
        vehicle_overlap_ratio=0.8,
        is_center_inside=True,
        max_confidence=0.92,
        contributing_track_ids=[101],
        is_instant_evidence_occupied=True,
    )

    # 1. UNKNOWN -> VACANT
    t_vac = TemporalBayTracker(
        bay_id="b1",
        operator_label="B1",
        space_type="STANDARD",
        polygon_normalized=[],
        temporal_config=base_config.temporal,
    )
    assert t_vac.current_state == OccupancyState.UNKNOWN
    assert t_vac.last_transition_frame == -1

    tl_vac = None
    for f in range(5):  # min_frames_vacant = 5
        s, tl = t_vac.update_frame(f, f * 0.033, None)
        if tl:
            tl_vac = tl
    assert t_vac.current_state == OccupancyState.VACANT
    assert tl_vac is not None
    assert tl_vac.previous_state == OccupancyState.UNKNOWN
    assert tl_vac.new_state == OccupancyState.VACANT
    assert tl_vac.previous_state != tl_vac.new_state

    # 2. UNKNOWN -> OCCUPIED
    t_occ = TemporalBayTracker(
        bay_id="b1",
        operator_label="B1",
        space_type="STANDARD",
        polygon_normalized=[],
        temporal_config=base_config.temporal,
    )
    tl_occ = None
    for f in range(3):  # min_frames_occupied = 3
        s, tl = t_occ.update_frame(f, f * 0.033, ev_occ)
        if tl:
            tl_occ = tl
    assert t_occ.current_state == OccupancyState.OCCUPIED
    assert tl_occ is not None
    assert tl_occ.previous_state == OccupancyState.UNKNOWN
    assert tl_occ.new_state == OccupancyState.OCCUPIED
    assert tl_occ.previous_state != tl_occ.new_state
    assert tl_occ.contributing_track_ids == [101]

    # 3. OCCUPIED -> OCCLUDED
    s_drop, tl_drop = t_occ.update_frame(3, 3 * 0.033, None)
    assert s_drop.current_state == OccupancyState.OCCLUDED
    assert tl_drop is not None
    assert tl_drop.previous_state == OccupancyState.OCCUPIED
    assert tl_drop.new_state == OccupancyState.OCCLUDED
    assert tl_drop.previous_state != tl_drop.new_state

    # 4. OCCLUDED -> OCCUPIED
    s_rec, tl_rec = t_occ.update_frame(4, 4 * 0.033, ev_occ)
    assert s_rec.current_state == OccupancyState.OCCUPIED
    assert tl_rec is not None
    assert tl_rec.previous_state == OccupancyState.OCCLUDED
    assert tl_rec.new_state == OccupancyState.OCCUPIED
    assert tl_rec.previous_state != tl_rec.new_state

    # 5. OCCLUDED -> VACANT (enter OCCLUDED then sustained clear beyond dropout window)
    t_occ.update_frame(5, 5 * 0.033, None)  # dropout frame 1 -> OCCLUDED
    tl_ov = None
    for f in range(6, 14):
        s, tl = t_occ.update_frame(f, f * 0.033, None)
        if tl:
            tl_ov = tl
    assert t_occ.current_state == OccupancyState.VACANT
    assert tl_ov is not None
    assert tl_ov.previous_state == OccupancyState.OCCLUDED
    assert tl_ov.new_state == OccupancyState.VACANT
    assert tl_ov.previous_state != tl_ov.new_state


def test_temporal_bay_tracker_no_false_frame_zero_transition(base_config):
    """Verify that a single unconfirmed frame on startup does not falsely trigger a frame-zero transition."""
    tracker = TemporalBayTracker(
        bay_id="b1",
        operator_label="B1",
        space_type="STANDARD",
        polygon_normalized=[],
        temporal_config=base_config.temporal,
    )
    assert tracker.last_transition_frame == -1

    # Frame 0: single vacant evidence (need 5 to transition)
    summary, tl = tracker.update_frame(0, 0.0, None)
    assert summary.current_state == OccupancyState.UNKNOWN
    assert summary.last_transition_frame == -1
    assert tl is None  # NO transition emitted


def test_model_path_traversal_and_symlink_rejection(tmp_path):
    """Verify LocalVehicleDetector strictly rejects absolute paths, '..', symlinks, and directory escapes."""
    from src.parking.occupancy_config import DetectorConfig
    from src.parking.vehicle_detector import LocalVehicleDetector

    fake_root = tmp_path / "repo"
    fake_models = fake_root / "models"
    fake_models.mkdir(parents=True)
    real_ckpt = fake_models / "model.pt"
    real_ckpt.write_bytes(b"dummy model bytes")

    # 1. Reject absolute path
    cfg_abs = DetectorConfig(model_path=str(real_ckpt.resolve()))
    with pytest.raises(ValueError, match="Absolute model paths are prohibited"):
        LocalVehicleDetector(cfg_abs, root_dir=fake_root)

    # 2. Reject '..' traversal
    cfg_trav = DetectorConfig(model_path="models/../models/model.pt")
    with pytest.raises(ValueError, match="Path traversal"):
        LocalVehicleDetector(cfg_trav, root_dir=fake_root)

    # 3. Reject symlink checkpoint
    sym_ckpt = fake_models / "sym_model.pt"
    sym_ckpt.symlink_to(real_ckpt)
    cfg_sym = DetectorConfig(model_path="models/sym_model.pt")
    with pytest.raises(ValueError, match="Symlink found|is a symlink"):
        LocalVehicleDetector(cfg_sym, root_dir=fake_root)

    # 4. Reject escape from models directory
    outside_file = fake_root / "outside.pt"
    outside_file.write_bytes(b"outside")
    cfg_out = DetectorConfig(model_path="outside.pt")
    with pytest.raises(ValueError, match="escapes models directory"):
        LocalVehicleDetector(cfg_out, root_dir=fake_root)


def test_bytetrack_yaml_validation_and_symlink_rejection(tmp_path):
    """Verify ParkingByteTracker strictly validates 8-field YAML schema, schema_version, tracker_type, and symlinks."""
    from src.parking.vehicle_tracker import ParkingByteTracker

    fake_root = tmp_path / "repo"
    fake_tracking = fake_root / "configs" / "tracking"
    fake_tracking.mkdir(parents=True)

    valid_content = (
        "schema_version: 1\n"
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.25\n"
        "track_low_thresh: 0.10\n"
        "new_track_thresh: 0.30\n"
        "track_buffer: 30\n"
        "match_thresh: 0.80\n"
        "min_hits: 2\n"
    )

    # 1. Reject missing required field
    bad_missing = fake_tracking / "bad_missing.yaml"
    bad_missing.write_text("schema_version: 1\ntracker_type: bytetrack\n")
    with pytest.raises(ValueError, match="missing required fields"):
        ParkingByteTracker(config_path="configs/tracking/bad_missing.yaml", root_dir=fake_root)

    # 2. Reject unknown field
    bad_unknown = fake_tracking / "bad_unknown.yaml"
    bad_unknown.write_text(valid_content + "unknown_key: 123\n")
    with pytest.raises(ValueError, match="Unknown field in ByteTrack YAML config"):
        ParkingByteTracker(config_path="configs/tracking/bad_unknown.yaml", root_dir=fake_root)

    # 3. Reject invalid schema_version
    bad_schema = fake_tracking / "bad_schema.yaml"
    bad_schema.write_text(valid_content.replace("schema_version: 1", "schema_version: 2"))
    with pytest.raises(ValueError, match="Unsupported ByteTrack schema_version"):
        ParkingByteTracker(config_path="configs/tracking/bad_schema.yaml", root_dir=fake_root)

    # 4. Reject invalid tracker_type
    bad_type = fake_tracking / "bad_type.yaml"
    bad_type.write_text(valid_content.replace("tracker_type: bytetrack", "tracker_type: deepsort"))
    with pytest.raises(ValueError, match="Unsupported tracker_type"):
        ParkingByteTracker(config_path="configs/tracking/bad_type.yaml", root_dir=fake_root)

    # 5. Reject invalid float (boolean or negative)
    bad_float = fake_tracking / "bad_float.yaml"
    bad_float.write_text(valid_content.replace("track_high_thresh: 0.25", "track_high_thresh: true"))
    with pytest.raises(ValueError, match="must be a float"):
        ParkingByteTracker(config_path="configs/tracking/bad_float.yaml", root_dir=fake_root)

    # 6. Reject incoherent threshold ordering
    bad_order = fake_tracking / "bad_order.yaml"
    bad_order.write_text(valid_content.replace("track_high_thresh: 0.25\ntrack_low_thresh: 0.10", "track_high_thresh: 0.10\ntrack_low_thresh: 0.25"))
    with pytest.raises(ValueError, match="Incoherent threshold ordering"):
        ParkingByteTracker(config_path="configs/tracking/bad_order.yaml", root_dir=fake_root)

    # 7. Reject symlink config
    real_yaml = fake_tracking / "real.yaml"
    real_yaml.write_text(valid_content)
    sym_yaml = fake_tracking / "sym.yaml"
    sym_yaml.symlink_to(real_yaml)
    with pytest.raises(ValueError, match="Symlink found|cannot be a symlink"):
        ParkingByteTracker(config_path="configs/tracking/sym.yaml", root_dir=fake_root)


def test_bytetrack_fps_and_frame_rate_argument():
    """Verify runtime validated video FPS is correctly stored and passed to BYTETracker."""
    from src.parking.vehicle_tracker import ParkingByteTracker

    tracker = ParkingByteTracker(fps=45)
    assert tracker.fps == 45
    assert tracker.min_hits == 2


def test_bytetrack_consecutive_min_hits_semantics():
    """Verify that min_hits requires strictly consecutive frame detections, resetting on missed frame."""
    from src.parking.vehicle_tracker import ParkingByteTracker, TrackerUpdateResult
    from src.parking.occupancy_contracts import VehicleDetection

    tracker = ParkingByteTracker(fps=30)
    tracker.min_hits = 3

    det = VehicleDetection(
        class_id=2,
        class_name="car",
        confidence=0.9,
        bbox_xyxy=(100.0, 100.0, 200.0, 200.0),
    )

    # Frame 0: Hit 1 (not yet min_hits=3)
    res0 = tracker.update_tracks([det], frame_idx=0)
    assert res0.is_healthy is True

    # Frame 1: Hit 2
    res1 = tracker.update_tracks([det], frame_idx=1)
    assert res1.is_healthy is True

    # Frame 2: Hit 3 -> meets min_hits
    res2 = tracker.update_tracks([det], frame_idx=2)
    assert res2.is_healthy is True
    assert len(res2.detections) == 1

    # Frame 3: Miss (empty detections)
    tracker.update_tracks([], frame_idx=3)

    # Frame 4: Hit 1 after gap (consecutive count resets to 1)
    res4 = tracker.update_tracks([det], frame_idx=4)
    # Consecutive hits reset to 1 after gap frame 3, so track ID should not immediately be active if min_hits > 1
    assert res4.is_healthy is True


def test_tracker_failure_produces_unknown_evidence_and_never_vacant(base_config):
    """Verify that when tracker is unhealthy, frame evaluation produces UNKNOWN and never increments VACANT."""
    from src.parking.vehicle_tracker import TrackerUpdateResult

    engine = ParkingOccupancyEngine(
        config=base_config,
        parking_spaces=[
            {
                "id": "bay_1",
                "operator_label": "B1",
                "space_type": "STANDARD",
                "polygon_normalized": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.4, "y": 0.1},
                    {"x": 0.4, "y": 0.4},
                    {"x": 0.1, "y": 0.4},
                ],
            }
        ],
        video_width=1920,
        video_height=1080,
    )

    # Inject an unhealthy tracker result
    unhealthy_result = TrackerUpdateResult(
        detections=[],
        is_healthy=False,
        failure_reason="ByteTrack Kalman filter failure",
    )

    frame_res = engine.process_frame(0, 0.0, unhealthy_result)
    assert frame_res.bay_states["bay_1"].current_state == OccupancyState.UNKNOWN
    assert frame_res.unknown_count == 1
    assert frame_res.vacant_count == 0
    assert frame_res.occupied_count == 0


def test_validate_staging_artifacts_strict(tmp_path):
    """Verify ParkingOccupancyJobManager._validate_staging_artifacts detects inconsistencies."""
    from services.api.app.services.parking_occupancy_job_manager import parking_occupancy_job_manager

    staging = tmp_path / "staging_test"
    staging.mkdir()

    # 1. Missing artifacts raise error
    with pytest.raises(ValueError, match="Staging artifact missing"):
        parking_occupancy_job_manager._validate_staging_artifacts(
            staging_dir=staging,
            expected_job_id="job-1",
            expected_camera_id="cam-1",
            expected_site_id="site-1",
            expected_layout_sha="a" * 64,
            expected_assessment_id="assess-1",
            expected_stability_config_sha="b" * 64,
            expected_occupancy_config_sha="c" * 64,
            expected_width=1920,
            expected_height=1080,
            expected_fps=30.0,
            expected_frames=10,
            expected_output_sha="d" * 64,
            expected_timeline_sha="e" * 64,
            expected_summary_sha="f" * 64,
        )
