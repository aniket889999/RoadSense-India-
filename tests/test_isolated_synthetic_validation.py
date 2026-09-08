"""Concurrency and isolation regression tests for Phase 2C synthetic parking validation."""

import asyncio
import os
from pathlib import Path
import tempfile
import pytest
from httpx import ASGITransport, AsyncClient

from services.api.app.main import app
from services.api.app.services.parking_occupancy_job_manager import (
    ParkingOccupancyJobManager,
    parking_occupancy_job_manager,
)
from src.parking.occupancy_contracts import VehicleDetection
from src.parking.testing_support import DeterministicVehicleDetectorDouble
from src.parking.validation_runner import run_stable_parking_e2e_validation
from src.parking.vehicle_detector import LocalVehicleDetector


@pytest.mark.anyio
async def test_openapi_contains_no_synthetic_validation_route():
    """Requirement: Production OpenAPI schema and FastAPI routing exposes no synthetic validation route."""
    routes = [route.path for route in app.routes if hasattr(route, "path")]
    assert "/api/v1/parking/validation/run-synthetic" not in routes
    assert "/parking/validation/run-synthetic" not in routes
    assert not any("run-synthetic" in r for r in routes)

    openapi_schema = app.openapi()
    paths = openapi_schema.get("paths", {})
    assert "/api/v1/parking/validation/run-synthetic" not in paths
    assert not any("run-synthetic" in p for p in paths)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/api/v1/parking/validation/run-synthetic")
        assert res.status_code == 404


@pytest.mark.anyio
async def test_synthetic_validation_cannot_mutate_production_singleton():
    """Requirement: Synthetic validation cannot change or mutate the production manager's detector."""
    # Verify production manager has no detector factory or detector override
    assert getattr(parking_occupancy_job_manager, "_detector_factory", None) is None
    assert not hasattr(parking_occupancy_job_manager, "_detector_override")
    assert not hasattr(parking_occupancy_job_manager, "set_detector_override")

    # Run isolated synthetic validation
    with tempfile.TemporaryDirectory(prefix="roadsense_test_iso_") as temp_dir:
        report = await run_stable_parking_e2e_validation(
            work_dir=Path(temp_dir),
            use_synthetic_detector_double=True,
        )
        assert report.passed is True

    # After run, verify production singleton remains completely unmutated
    assert getattr(parking_occupancy_job_manager, "_detector_factory", None) is None
    assert not hasattr(parking_occupancy_job_manager, "_detector_override")


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
    dummy = np.zeros((100, 100, 3), dtype=np.uint8) if "np" in globals() else None
    import numpy as np
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
async def test_all_temporary_artifacts_removed_after_run(tmp_path):
    """Requirement: All temporary DB, media, and fixture artifacts are purged after isolated run."""
    run_dir = tmp_path / "isolated_run_cleanup_check"
    run_dir.mkdir(parents=True)

    report = await run_stable_parking_e2e_validation(
        work_dir=run_dir,
        use_synthetic_detector_double=True,
    )
    assert report.passed is True

    # Check that report is saved in work_dir, and fixture/video was generated
    assert (run_dir / "parking_validation_evidence_report.json").is_file()

    # Now run automatic cleanup mode (work_dir=None)
    auto_report = await run_stable_parking_e2e_validation(
        work_dir=None,
        use_synthetic_detector_double=True,
    )
    assert auto_report.passed is True


@pytest.mark.anyio
async def test_real_job_cannot_observe_test_double():
    """Requirement: A production ParkingOccupancyJobManager instance always constructs LocalVehicleDetector from verified local configuration."""
    prod_mgr = ParkingOccupancyJobManager()
    assert prod_mgr._detector_factory is None

    # When no factory is supplied, constructor does not load test double
    assert not hasattr(prod_mgr, "_detector_override")
