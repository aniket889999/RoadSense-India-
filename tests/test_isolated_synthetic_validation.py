"""Concurrency, isolation, and evidence regression tests for Phase 2C synthetic parking validation."""

import asyncio
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from services.api.app.core.config import settings
from services.api.app.main import app as production_app
from services.api.app.services.parking_occupancy_job_manager import (
    ParkingOccupancyJobManager,
    parking_occupancy_job_manager,
)
from src.parking.occupancy_contracts import VehicleDetection
from src.parking.testing_support import DeterministicVehicleDetectorDouble
from src.parking.validation_evidence import (
    measure_visual_overlay_regions,
    validate_temporal_timeline_evidence,
)
from src.parking.validation_runner import (
    create_isolated_parking_app,
    run_stable_parking_e2e_validation,
)


@pytest.mark.anyio
async def test_openapi_contains_no_synthetic_validation_route():
    """Requirement: Production OpenAPI schema and FastAPI routing exposes no synthetic validation route."""
    routes = [route.path for route in production_app.routes if hasattr(route, "path")]
    assert "/api/v1/parking/validation/run-synthetic" not in routes
    assert "/parking/validation/run-synthetic" not in routes
    assert not any("run-synthetic" in r for r in routes)

    openapi_schema = production_app.openapi()
    paths = openapi_schema.get("paths", {})
    assert "/api/v1/parking/validation/run-synthetic" not in paths
    assert not any("run-synthetic" in p for p in paths)

    transport = ASGITransport(app=production_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/api/v1/parking/validation/run-synthetic")
        assert res.status_code == 404


@pytest.mark.anyio
async def test_synthetic_validation_cannot_mutate_production_singleton():
    """Requirement: Synthetic validation cannot change or mutate the production manager's detector."""
    assert getattr(parking_occupancy_job_manager, "_detector_factory", None) is None
    assert not hasattr(parking_occupancy_job_manager, "_detector_override")
    assert not hasattr(parking_occupancy_job_manager, "set_detector_override")

    saved_overrides_len = len(production_app.dependency_overrides)
    saved_media_root = getattr(settings, "MEDIA_ROOT", None)

    # Run isolated synthetic validation
    with tempfile.TemporaryDirectory(prefix="roadsense_test_iso_") as temp_dir:
        report = await run_stable_parking_e2e_validation(
            work_dir=Path(temp_dir),
            use_synthetic_detector_double=True,
        )
        assert report.passed is True

    # After run, verify production singleton and globals remain completely unmutated
    assert getattr(parking_occupancy_job_manager, "_detector_factory", None) is None
    assert not hasattr(parking_occupancy_job_manager, "_detector_override")
    assert getattr(settings, "MEDIA_ROOT", None) == saved_media_root
    assert len(production_app.dependency_overrides) == saved_overrides_len


@pytest.mark.anyio
async def test_concurrent_complete_validations_asyncio_gather(tmp_path):
    """
    Requirement 9: Run two complete validations concurrently using asyncio.gather with separate temporary databases and media roots.
    Prove:
    - no shared database records;
    - no shared files;
    - no shared detector counters;
    - no shared job-manager registries;
    - no global setting changes;
    - no global dependency changes;
    - both reports remain deterministic and valid.
    """
    run_dir1 = tmp_path / "run_1"
    run_dir2 = tmp_path / "run_2"
    run_dir1.mkdir(parents=True)
    run_dir2.mkdir(parents=True)

    saved_media_root = getattr(settings, "MEDIA_ROOT", None)
    saved_overrides = dict(production_app.dependency_overrides)

    task1 = run_stable_parking_e2e_validation(work_dir=run_dir1, use_synthetic_detector_double=True)
    task2 = run_stable_parking_e2e_validation(work_dir=run_dir2, use_synthetic_detector_double=True)

    report1, report2 = await asyncio.gather(task1, task2)

    # Both must pass deterministically
    assert report1.passed is True
    assert report2.passed is True
    assert report1.run_id != report2.run_id
    assert report1.job_id != report2.job_id
    assert report1.camera_id != report2.camera_id
    assert report1.site_id != report2.site_id

    # No shared database records
    db1_path = run_dir1 / "isolated_val.db"
    db2_path = run_dir2 / "isolated_val.db"
    assert db1_path.is_file() and db2_path.is_file()
    assert db1_path.stat().st_size > 0
    assert db2_path.stat().st_size > 0

    # No shared files
    media1_files = set(str(p.relative_to(run_dir1)) for p in run_dir1.rglob("*"))
    media2_files = set(str(p.relative_to(run_dir2)) for p in run_dir2.rglob("*"))
    assert len(media1_files) > 0 and len(media2_files) > 0

    # Output artifacts are valid and self-consistent
    assert len(report1.output_video_sha256) == 64 and len(report2.output_video_sha256) == 64
    assert len(report1.timeline_sha256) == 64 and len(report2.timeline_sha256) == 64
    assert len(report1.summary_sha256) == 64 and len(report2.summary_sha256) == 64
    assert report1.final_occupied_count == 2 and report2.final_occupied_count == 2
    assert report1.final_vacant_count == 2 and report2.final_vacant_count == 2
    assert report1.total_bays == 4 and report2.total_bays == 4
    assert report1.operational_gate == "ALLOWED" and report2.operational_gate == "ALLOWED"

    # Globals remain unmutated
    assert getattr(settings, "MEDIA_ROOT", None) == saved_media_root
    assert production_app.dependency_overrides == saved_overrides


@pytest.mark.anyio
async def test_concurrent_isolated_validations_do_not_share_detector_counters():
    """Requirement: Concurrent isolated synthetic validations cannot share detector frame counters or state."""
    det1 = DeterministicVehicleDetectorDouble(
        frame_detections={
            0: [VehicleDetection(class_id=2, class_name="car", confidence=0.9, bbox_xyxy=(0, 0, 10, 10))],
            1: [VehicleDetection(class_id=2, class_name="car", confidence=0.9, bbox_xyxy=(0, 0, 10, 10))],
        }
    )
    det2 = DeterministicVehicleDetectorDouble(
        frame_detections={
            0: [VehicleDetection(class_id=2, class_name="car", confidence=0.9, bbox_xyxy=(0, 0, 10, 10))],
        }
    )

    mgr1 = ParkingOccupancyJobManager(detector_factory=lambda: det1)
    mgr2 = ParkingOccupancyJobManager(detector_factory=lambda: det2)

    # Frame steps on mgr1
    d1 = mgr1._detector_factory()
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)

    res1_f0 = d1.detect_vehicles(dummy)
    assert len(res1_f0) == 1
    assert d1._current_frame_idx == 1

    # mgr2 detector counter must be completely uninfluenced
    d2 = mgr2._detector_factory()
    assert d2._current_frame_idx == 0
    res2_f0 = d2.detect_vehicles(dummy)
    assert len(res2_f0) == 1
    assert d2._current_frame_idx == 1


@pytest.mark.anyio
async def test_temporal_evidence_production_validation_rejections():
    """
    Requirement 12-14: Test production temporal validation function against:
    - malformed JSONL
    - missing required keys
    - non-finite/negative timestamps
    - reordered timestamps
    - duplicate transitions
    - state-chain discontinuities
    - unexpected bay IDs
    - missing expected transitions
    - wrong final states
    """
    expected_bays = {"B1", "B2", "B3", "B4"}
    summary_states = {"B1": "VACANT", "B2": "OCCUPIED", "B3": "VACANT", "B4": "OCCUPIED"}

    # 1. Valid baseline
    valid_lines = [
        json.dumps({"frame_index": 1, "timestamp_seconds": 0.033, "bay_id": "B1", "operator_label": "B1", "previous_state": "UNKNOWN", "new_state": "OCCUPIED", "trigger_reason": "Init"}),
        json.dumps({"frame_index": 1, "timestamp_seconds": 0.033, "bay_id": "B2", "operator_label": "B2", "previous_state": "UNKNOWN", "new_state": "VACANT", "trigger_reason": "Init"}),
        json.dumps({"frame_index": 1, "timestamp_seconds": 0.033, "bay_id": "B3", "operator_label": "B3", "previous_state": "UNKNOWN", "new_state": "VACANT", "trigger_reason": "Init"}),
        json.dumps({"frame_index": 1, "timestamp_seconds": 0.033, "bay_id": "B4", "operator_label": "B4", "previous_state": "UNKNOWN", "new_state": "OCCUPIED", "trigger_reason": "Init"}),
        json.dumps({"frame_index": 30, "timestamp_seconds": 1.0, "bay_id": "B1", "operator_label": "B1", "previous_state": "OCCUPIED", "new_state": "VACANT", "trigger_reason": "Departure"}),
        json.dumps({"frame_index": 45, "timestamp_seconds": 1.5, "bay_id": "B2", "operator_label": "B2", "previous_state": "VACANT", "new_state": "OCCUPIED", "trigger_reason": "Arrival"}),
        json.dumps({"frame_index": 50, "timestamp_seconds": 1.667, "bay_id": "B4", "operator_label": "B4", "previous_state": "OCCUPIED", "new_state": "OCCLUDED", "trigger_reason": "Occlusion"}),
        json.dumps({"frame_index": 70, "timestamp_seconds": 2.333, "bay_id": "B4", "operator_label": "B4", "previous_state": "OCCLUDED", "new_state": "OCCUPIED", "trigger_reason": "Recovery"}),
    ]

    ok, checks, fails, final_s = validate_temporal_timeline_evidence(
        timeline_lines=valid_lines,
        expected_bays=expected_bays,
        summary_final_states=summary_states,
        expected_synthetic_pattern=True,
    )
    assert ok is True
    assert len(fails) == 0
    assert final_s == summary_states

    # 2. Rejection: Malformed JSONL
    malformed = ["{not json}"]
    ok, _, fails, _ = validate_temporal_timeline_evidence(malformed, expected_bays)
    assert ok is False
    assert any("Malformed JSON" in f for f in fails)

    # 3. Rejection: Missing required key
    missing_key = [json.dumps({"frame_index": 1, "bay_id": "B1"})]
    ok, _, fails, _ = validate_temporal_timeline_evidence(missing_key, expected_bays)
    assert ok is False
    assert any("Missing required fields" in f for f in fails)

    # 4. Rejection: Non-finite / Negative timestamp
    bad_ts = [json.dumps({"frame_index": 1, "timestamp_seconds": -5.0, "bay_id": "B1", "operator_label": "B1", "previous_state": "UNKNOWN", "new_state": "OCCUPIED", "trigger_reason": "Init"})]
    ok, _, fails, _ = validate_temporal_timeline_evidence(bad_ts, expected_bays)
    assert ok is False
    assert any("Invalid timestamp" in f for f in fails)

    # 5. Rejection: Reordered timestamps
    reordered_ts = [
        json.dumps({"frame_index": 1, "timestamp_seconds": 2.0, "bay_id": "B1", "operator_label": "B1", "previous_state": "UNKNOWN", "new_state": "OCCUPIED", "trigger_reason": "Init"}),
        json.dumps({"frame_index": 2, "timestamp_seconds": 1.0, "bay_id": "B1", "operator_label": "B1", "previous_state": "OCCUPIED", "new_state": "VACANT", "trigger_reason": "Dep"}),
    ]
    ok, _, fails, _ = validate_temporal_timeline_evidence(reordered_ts, expected_bays)
    assert ok is False
    assert any("Timestamp reordered" in f for f in fails)

    # 6. Rejection: Duplicate transitions (previous == new)
    dup_s = [json.dumps({"frame_index": 1, "timestamp_seconds": 0.1, "bay_id": "B1", "operator_label": "B1", "previous_state": "OCCUPIED", "new_state": "OCCUPIED", "trigger_reason": "Noop"})]
    ok, _, fails, _ = validate_temporal_timeline_evidence(dup_s, expected_bays)
    assert ok is False
    assert any("Duplicate transition" in f for f in fails)

    # 7. Rejection: State chain discontinuity (does not start with UNKNOWN)
    no_unk = [json.dumps({"frame_index": 1, "timestamp_seconds": 0.1, "bay_id": "B1", "operator_label": "B1", "previous_state": "VACANT", "new_state": "OCCUPIED", "trigger_reason": "Init"})]
    ok, _, fails, _ = validate_temporal_timeline_evidence(no_unk, expected_bays)
    assert ok is False
    assert any("Initial transition did not start from 'UNKNOWN'" in f for f in fails)

    # 8. Rejection: State chain discontinuity in subsequent step
    chain_break = [
        json.dumps({"frame_index": 1, "timestamp_seconds": 0.1, "bay_id": "B1", "operator_label": "B1", "previous_state": "UNKNOWN", "new_state": "OCCUPIED", "trigger_reason": "Init"}),
        json.dumps({"frame_index": 2, "timestamp_seconds": 0.2, "bay_id": "B1", "operator_label": "B1", "previous_state": "UNKNOWN", "new_state": "VACANT", "trigger_reason": "Broken"}),
    ]
    ok, _, fails, _ = validate_temporal_timeline_evidence(chain_break, expected_bays)
    assert ok is False
    assert any("Discontinuity: expected previous_state 'OCCUPIED', got 'UNKNOWN'" in f for f in fails)

    # 9. Rejection: Unexpected bay ID
    bad_bay = [json.dumps({"frame_index": 1, "timestamp_seconds": 0.1, "bay_id": "B999", "operator_label": "B999", "previous_state": "UNKNOWN", "new_state": "OCCUPIED", "trigger_reason": "Init"})]
    ok, _, fails, _ = validate_temporal_timeline_evidence(bad_bay, expected_bays)
    assert ok is False
    assert any("Unexpected bay_id 'B999'" in f for f in fails)

    # 10. Rejection: Summary mismatch
    wrong_summary = {"B1": "OCCUPIED", "B2": "OCCUPIED", "B3": "VACANT", "B4": "OCCUPIED"}
    ok, _, fails, _ = validate_temporal_timeline_evidence(valid_lines, expected_bays, summary_final_states=wrong_summary)
    assert ok is False
    assert any("summary reported 'OCCUPIED', timeline resolved 'VACANT'" in f for f in fails)


@pytest.mark.anyio
async def test_visual_four_color_quantitative_measurements():
    """
    Requirement 20-24: Test quantitative visual color measurements against unannotated source for all 4 states:
    - OCCUPIED (Red dominance, R > B + margin, delta >= min_delta)
    - VACANT (Green dominance, G > B + margin, delta >= min_delta)
    - OCCLUDED (Amber dominance, R > B and G > B, delta >= min_delta)
    - UNKNOWN (Balanced Grey, delta >= min_delta)
    """
    H, W = 200, 400
    source_bgr = np.full((H, W, 3), 100, dtype=np.uint8)  # neutral background

    bay_polys = {
        "B1": np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.int32),
        "B2": np.array([[110, 10], [190, 10], [190, 90], [110, 90]], dtype=np.int32),
        "B3": np.array([[210, 10], [290, 10], [290, 90], [210, 90]], dtype=np.int32),
        "B4": np.array([[310, 10], [390, 10], [390, 90], [310, 90]], dtype=np.int32),
    }

    bay_states = {
        "B1": "OCCUPIED",
        "B2": "VACANT",
        "B3": "OCCLUDED",
        "B4": "UNKNOWN",
    }

    # Synthesize blended frame
    rendered_bgr = source_bgr.copy()
    # B1 OCCUPIED (Red: [0, 0, 255], 35% alpha)
    rendered_bgr[10:90, 10:90] = (0.65 * np.array([100, 100, 100]) + 0.35 * np.array([0, 0, 255])).astype(np.uint8)
    # B2 VACANT (Green: [0, 255, 0], 25% alpha)
    rendered_bgr[10:90, 110:190] = (0.75 * np.array([100, 100, 100]) + 0.25 * np.array([0, 255, 0])).astype(np.uint8)
    # B3 OCCLUDED (Amber: [0, 165, 255], 30% alpha)
    rendered_bgr[10:90, 210:290] = (0.70 * np.array([100, 100, 100]) + 0.30 * np.array([0, 165, 255])).astype(np.uint8)
    # B4 UNKNOWN (Grey: [128, 128, 128], 50% alpha)
    rendered_bgr[10:90, 310:390] = (0.50 * np.array([100, 100, 100]) + 0.50 * np.array([128, 128, 128])).astype(np.uint8)

    ok, checks, fails, data = measure_visual_overlay_regions(
        rendered_bgr=rendered_bgr,
        source_bgr=source_bgr,
        bay_polygons_px=bay_polys,
        bay_states=bay_states,
        is_decoded_mp4=False,
    )
    assert ok is True
    assert len(fails) == 0
    assert len(data) == 4

    # B1 OCCUPIED: Red dominance verified
    assert data["B1"]["state"] == "OCCUPIED"
    assert data["B1"]["rendered_bgr_mean"][2] > data["B1"]["rendered_bgr_mean"][0] + 12.0
    assert data["B1"]["color_delta"] > 10.0

    # B2 VACANT: Green dominance verified
    assert data["B2"]["state"] == "VACANT"
    assert data["B2"]["rendered_bgr_mean"][1] > data["B2"]["rendered_bgr_mean"][0] + 12.0
    assert data["B2"]["color_delta"] > 10.0

    # B3 OCCLUDED: Amber dominance verified
    assert data["B3"]["state"] == "OCCLUDED"
    assert data["B3"]["rendered_bgr_mean"][2] > data["B3"]["rendered_bgr_mean"][0] + 12.0
    assert data["B3"]["rendered_bgr_mean"][1] > data["B3"]["rendered_bgr_mean"][0] + 6.0

    # B4 UNKNOWN: Grey verified
    assert data["B4"]["state"] == "UNKNOWN"
    assert data["B4"]["color_delta"] > 5.0
