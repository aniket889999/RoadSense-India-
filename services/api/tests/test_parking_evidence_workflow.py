"""Tests for Phase 2C parking occupancy evidence workflow, retention controls, and negative validation."""

import hashlib
import json
from pathlib import Path
import tempfile
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from services.api.app.core.config import settings
from services.api.app.db.session import async_session_factory
from services.api.app.main import app
from services.api.app.models.entities import Camera, ParkingOccupancyJob, Site
from src.parking.contracts import OperationalGate, StabilityDecision
from src.parking.stability_config import load_stability_config
from src.parking.stability_engine import evaluate_video_camera_stability


@pytest.mark.anyio
async def test_safe_job_deletion_and_retention_controls(tmp_path, monkeypatch):
    """Verify safe job deletion endpoint purges confined artifact directory and database record."""
    fake_media = tmp_path / "media_root"
    fake_media.mkdir(parents=True)
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(fake_media))

    async with async_session_factory() as db:
        site = Site(id="site_del_01", name="Delete Test Site")
        cam = Camera(id="cam_del_01", site_id="site_del_01", name="Cam Delete")
        job = ParkingOccupancyJob(
            id="job_del_01",
            camera_id="cam_del_01",
            site_id="site_del_01",
            status="COMPLETE",
            progress_pct=100.0,
            output_video_sha256="a" * 64,
        )
        db.add_all([site, cam, job])
        await db.commit()

    # Create dummy artifact dir
    job_dir = fake_media / "parking_jobs" / "job_del_01"
    job_dir.mkdir(parents=True)
    (job_dir / "annotated.mp4").write_bytes(b"dummy")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.delete("/api/v1/parking/jobs/job_del_01")
        assert res.status_code == 200
        assert res.json()["deleted"] is True

        # Check directory deleted
        assert not job_dir.exists()

        # Check 404 on subsequent get
        res_get = await client.get("/api/v1/parking/jobs/job_del_01")
        assert res_get.status_code == 404


@pytest.mark.anyio
async def test_run_synthetic_validation_endpoint():
    """Verify POST /api/v1/parking/validation/run-synthetic executes the automated harness."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/api/v1/parking/validation/run-synthetic")
        assert res.status_code == 200
        data = res.json()
        assert data["is_synthetic_fixture"] is True
        assert "SYNTHETIC TEST EVIDENCE" in data["disclaimer"]
        assert data["passed"] is True
        assert data["operational_gate"] == "ALLOWED"
        assert len(data["checks"]) >= 4


@pytest.mark.anyio
async def test_unstable_negative_video_remains_strictly_blocked():
    """
    Negative Verification:
    The moving camera test video: /Users/aniket/Downloads/PARKING LOT TEST.mp4
    SHA-256: 080d4bbed653023718ae972b6734f457bb9c5ffe67a081424ae3818352bbcc0a
    MUST remain strictly UNSTABLE and BLOCKED_BY_STABILITY_GATE.
    No model inference, ByteTrack, or occupancy counts may be generated.
    """
    video_path = Path("/Users/aniket/Downloads/PARKING LOT TEST.mp4")
    if not video_path.is_file():
        pytest.skip("Negative test video file not present in local downloads")

    # Verify sha256
    raw_bytes = video_path.read_bytes()
    vid_sha = hashlib.sha256(raw_bytes).hexdigest()
    assert vid_sha == "080d4bbed653023718ae972b6734f457bb9c5ffe67a081424ae3818352bbcc0a"

    # Stability assessment on unstable video vs reference frame 0
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    ret, frame_0 = cap.read()
    cap.release()
    assert ret is True

    _, ref_jpg_bytes = cv2.imencode(".jpg", frame_0)
    ref_bytes = ref_jpg_bytes.tobytes()
    ref_sha = hashlib.sha256(ref_bytes).hexdigest()

    cfg = load_stability_config()
    result = evaluate_video_camera_stability(
        video_path=video_path,
        reference_image_bytes=ref_bytes,
        expected_reference_sha256=ref_sha,
        active_layout_canonical_sha256="fake_canonical_layout_sha",
        config=cfg,
    )

    # Must be UNSTABLE and BLOCKED
    assert result.aggregate_decision == StabilityDecision.UNSTABLE
    assert result.operational_gate == OperationalGate.BLOCKED
    assert any("UNSTABLE" in r for r in result.gate_reasons)

    # Video bytes must remain unmodified
    assert hashlib.sha256(video_path.read_bytes()).hexdigest() == vid_sha
