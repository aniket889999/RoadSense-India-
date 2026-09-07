"""Tests for Phase 2C stationary parking camera validation harness."""

import hashlib
import json
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest
from sqlalchemy import select

from services.api.app.core.config import settings
from services.api.app.db.session import async_session_factory
from services.api.app.models.entities import Camera, CameraStabilityAssessment, ParkingOccupancyJob, Site
from src.parking.contracts import OperationalGate, StabilityDecision
from src.parking.occupancy_config import RenderConfig, load_parking_occupancy_config
from src.parking.occupancy_contracts import BayStateSummary, OccupancyState, VehicleDetection
from src.parking.stability_config import load_stability_config
from src.parking.stability_engine import evaluate_video_camera_stability
from src.parking.synthetic_scene_generator import generate_synthetic_parking_fixture
from src.parking.validation_evidence import ParkingValidationEvidenceReport
from src.parking.validation_runner import run_stable_parking_e2e_validation
from src.parking.vehicle_detector import DeterministicVehicleDetectorDouble
from src.parking.video_annotator import ParkingVideoAnnotator


@pytest.mark.anyio
async def test_synthetic_parking_fixture_generation_and_stability(tmp_path):
    """Verify synthetic stationary parking fixture produces a valid MP4/JPEG and passes stability assessment."""
    fixture_dir = tmp_path / "fixture"
    fixture = generate_synthetic_parking_fixture(
        output_dir=fixture_dir,
        width=1280,
        height=720,
        fps=30.0,
        duration_seconds=2.0,
    )

    assert fixture.video_path.is_file()
    assert fixture.reference_image_path.is_file()
    assert len(fixture.parking_spaces) == 4
    assert len(fixture.approach_zones) == 4
    assert fixture.total_frames == 60

    # Verify stability assessment on stationary synthetic video vs reference image
    ref_bytes = fixture.reference_image_path.read_bytes()
    ref_sha = hashlib.sha256(ref_bytes).hexdigest()
    stability_cfg = load_stability_config()
    
    from src.parking.layout_serialization import compute_canonical_layout_sha256
    canonical_layout_sha = compute_canonical_layout_sha256(
        schema_version="1.0.0",
        camera_id="test_cam_01",
        reference_image_sha256=ref_sha,
        parking_spaces=fixture.parking_spaces,
        approach_zones=fixture.approach_zones,
    )
    result = evaluate_video_camera_stability(
        video_path=fixture.video_path,
        reference_image_bytes=ref_bytes,
        expected_reference_sha256=ref_sha,
        active_layout_canonical_sha256=canonical_layout_sha,
        config=stability_cfg,
    )

    assert result.aggregate_decision == StabilityDecision.STABLE
    assert result.operational_gate == OperationalGate.ALLOWED


@pytest.mark.anyio
async def test_full_stable_parking_e2e_validation_runner(tmp_path, monkeypatch):
    """Verify complete end-to-end validation runner generates valid evidence report and all artifacts."""
    fake_media = tmp_path / "media_root"
    fake_media.mkdir(parents=True)
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(fake_media))

    work_dir = tmp_path / "val_work"
    report = await run_stable_parking_e2e_validation(
        work_dir=work_dir,
        use_synthetic_detector_double=True,
    )

    assert isinstance(report, ParkingValidationEvidenceReport)
    assert report.is_synthetic_fixture is True
    assert "SYNTHETIC TEST EVIDENCE" in report.disclaimer
    assert report.operational_gate == "ALLOWED"
    assert report.passed is True
    assert len(report.failure_reasons) == 0

    # Check that published artifacts exist and have non-empty hashes
    assert report.output_video_sha256 != ""
    assert report.timeline_sha256 != ""
    assert report.summary_sha256 != ""
    assert report.manifest_sha256 != ""

    # Check bay counts
    assert report.total_bays == 4
    assert report.final_occupied_count == 2
    assert report.final_vacant_count == 2


def test_visual_annotator_color_palette_and_overlays():
    """Verify ParkingVideoAnnotator conforms to standard visual palette: Red=Occupied, Green=Vacant, Amber=Occluded, Grey=Unknown."""
    occ_cfg = load_parking_occupancy_config()
    render_cfg = occ_cfg.rendering
    annotator = ParkingVideoAnnotator(render_cfg)

    # Palette assertions in BGR
    assert render_cfg.occupied_color_bgr == [0, 0, 255]    # Red
    assert render_cfg.vacant_color_bgr == [0, 255, 0]      # Green
    assert render_cfg.occluded_color_bgr == [0, 165, 255]  # Amber/Orange
    assert render_cfg.unknown_color_bgr == [128, 128, 128] # Grey

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    bay_polys_px = {
        "b1": np.array([[50, 50], [250, 50], [250, 350], [50, 350]], dtype=np.int32),
        "b2": np.array([[300, 50], [500, 50], [500, 350], [300, 350]], dtype=np.int32),
    }
    bay_states = {
        "b1": BayStateSummary(
            bay_id="b1",
            operator_label="Bay-01",
            space_type="STANDARD",
            current_state=OccupancyState.OCCUPIED,
            consecutive_frames_in_state=10,
            confidence=0.95,
            last_transition_frame=0,
            last_transition_timestamp=0.0,
            polygon_normalized=[{"x": 0.05, "y": 0.05}],
            contributing_track_ids=[1],
        ),
        "b2": BayStateSummary(
            bay_id="b2",
            operator_label="Bay-02",
            space_type="STANDARD",
            current_state=OccupancyState.VACANT,
            consecutive_frames_in_state=10,
            confidence=0.92,
            last_transition_frame=0,
            last_transition_timestamp=0.0,
            polygon_normalized=[{"x": 0.25, "y": 0.05}],
            contributing_track_ids=[],
        ),
    }
    vehicle_dets = [
        VehicleDetection(
            class_id=2,
            class_name="car",
            confidence=0.95,
            bbox_xyxy=(60.0, 60.0, 240.0, 340.0),
            track_id=1,
            center_xy=(150.0, 200.0),
        )
    ]

    rendered = annotator.render_frame(
        frame_bgr=frame,
        bay_states=bay_states,
        bay_polygons_px=bay_polys_px,
        vehicle_detections=vehicle_dets,
        frame_idx=1,
        timestamp_sec=0.033,
    )

    assert rendered.shape == (720, 1280, 3)
    # Check that pixels inside Bay 1 have high Red channel
    b1_center_pixel = rendered[200, 150]
    assert b1_center_pixel[2] > b1_center_pixel[0]  # R > B in BGR
    # Check that pixels inside Bay 2 have high Green channel
    b2_center_pixel = rendered[200, 400]
    assert b2_center_pixel[1] > b2_center_pixel[0]  # G > B in BGR


def test_deterministic_vehicle_detector_double():
    """Verify DeterministicVehicleDetectorDouble returns correct frame-level detections and track_id reset."""
    dets_frame_0 = [
        VehicleDetection(
            class_id=2,
            class_name="car",
            confidence=0.9,
            bbox_xyxy=(10.0, 10.0, 100.0, 100.0),
            track_id=None,
        )
    ]
    detector = DeterministicVehicleDetectorDouble(
        frame_detections={0: dets_frame_0},
        checkpoint_sha256="1" * 64,
    )
    assert detector.checkpoint_sha256 == "1" * 64

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out0 = detector.detect_vehicles(dummy_frame)
    assert len(out0) == 1
    assert out0[0].class_name == "car"
    assert out0[0].track_id is None

    out1 = detector.detect_vehicles(dummy_frame)
    assert len(out1) == 0  # No detections on frame 1
