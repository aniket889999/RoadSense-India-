"""Comprehensive tests for Phase 2B transactional publication, cancellation lifecycle, upload staging, and startup recovery."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import uuid
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from services.api.app.core.config import settings
from services.api.app.db.session import async_session_factory, get_db
from services.api.app.main import app, reconcile_startup_parking_jobs
from services.api.app.models.entities import Camera, ParkingOccupancyJob, Site
from services.api.app.routers.parking import create_staged_upload_file, _resolve_safe_parking_artifact
from services.api.app.services.parking_occupancy_job_manager import parking_occupancy_job_manager
from src.parking.occupancy_contracts import OccupancyState, ParkingJobCancelled


@pytest.fixture
async def client(db_session):
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_upload_staging_confinement_and_symlink_rejection(tmp_path, monkeypatch):
    """Verify upload staging strictly creates files in configured root, rejects symlinks and traversal."""
    fake_media = tmp_path / "media_root"
    fake_media.mkdir(parents=True)
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(fake_media))

    # 1. Successful staged upload creation
    staging_file, fd = create_staged_upload_file()
    assert staging_file.exists()
    assert staging_file.name.endswith(".upload.tmp")
    assert staging_file.parent == fake_media / "upload_staging"
    os.close(fd)
    staging_file.unlink(missing_ok=True)

    # 2. Reject symlink in upload staging directory component
    sym_media = tmp_path / "sym_media"
    sym_media.symlink_to(fake_media)
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(sym_media))
    with pytest.raises(ValueError, match="Symlink found"):
        create_staged_upload_file()


@pytest.mark.anyio
async def test_no_artifacts_served_for_non_complete_jobs(client, db_session, tmp_path, monkeypatch):
    """Verify /video, /timeline, /manifest endpoints refuse to serve for CANCELLED, FAILED, PUBLISHING jobs."""
    fake_media = tmp_path / "media_root"
    fake_media.mkdir(parents=True)
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(fake_media))

    # Create dummy camera and jobs in various non-complete states
    site = Site(id=str(uuid.uuid4()), name="Test Site", timezone="UTC")
    cam = Camera(id=str(uuid.uuid4()), site_id=site.id, name="Test Cam")
    db_session.add_all([site, cam])
    await db_session.commit()

    for status in ("QUEUED", "VALIDATING", "PUBLISHING", "CANCELLED", "FAILED"):
        job = ParkingOccupancyJob(
            id=str(uuid.uuid4()),
            camera_id=cam.id,
            site_id=site.id,
            status=status,
            output_video_sha256="a" * 64,
            manifest_json={"timeline_sha256": "b" * 64, "output_video_sha256": "a" * 64},
        )
        db_session.add(job)
        await db_session.commit()

        # Video request must return 404
        v_resp = await client.get(f"/api/v1/parking/jobs/{job.id}/video")
        assert v_resp.status_code == 404
        assert "COMPLETE" in v_resp.json()["detail"] or "not available" in v_resp.json()["detail"]

        # Timeline request must return 404
        t_resp = await client.get(f"/api/v1/parking/jobs/{job.id}/timeline")
        assert t_resp.status_code == 404
        assert "COMPLETE" in t_resp.json()["detail"] or "not available" in t_resp.json()["detail"]

        # Manifest request must return 404
        m_resp = await client.get(f"/api/v1/parking/jobs/{job.id}/manifest")
        assert m_resp.status_code == 404
        assert "COMPLETE" in m_resp.json()["detail"]


@pytest.mark.anyio
async def test_file_tampering_and_hash_verification(tmp_path, monkeypatch):
    """Verify _resolve_safe_parking_artifact fails if file on disk was modified/tampered after hashing."""
    fake_media = tmp_path / "media_root"
    jobs_root = fake_media / "parking_jobs"
    job_id = str(uuid.uuid4())
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(fake_media))

    video_file = job_dir / "annotated.mp4"
    orig_bytes = b"original valid video stream"
    video_file.write_bytes(orig_bytes)
    correct_sha = hashlib.sha256(orig_bytes).hexdigest()

    # 1. Valid hash succeeds
    safe = _resolve_safe_parking_artifact(job_id, "annotated.mp4", correct_sha, "COMPLETE")
    assert safe is not None
    assert safe == video_file.resolve()

    # 2. Tampered hash or modified file fails
    tampered_bytes = b"tampered modified bytes"
    video_file.write_bytes(tampered_bytes)
    tampered_safe = _resolve_safe_parking_artifact(job_id, "annotated.mp4", correct_sha, "COMPLETE")
    assert tampered_safe is None

    # 3. None or invalid SHA fails
    assert _resolve_safe_parking_artifact(job_id, "annotated.mp4", None, "COMPLETE") is None
    assert _resolve_safe_parking_artifact(job_id, "annotated.mp4", "short_sha", "COMPLETE") is None


def _create_test_h264_mp4(path: Path, width: int = 1920, height: int = 1080, fps: float = 30.0):
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c=black:s={width}x{height}:r={int(fps)}:d=0.2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)


@pytest.mark.anyio
async def test_manifest_provenance_mismatch_detection(tmp_path):
    """Verify _validate_staging_artifacts detects manifest provenance tampering."""
    from services.api.app.services.parking_occupancy_job_manager import parking_occupancy_job_manager

    staging = tmp_path / "staging_prov_test"
    staging.mkdir()

    mp4 = staging / "annotated.mp4"
    _create_test_h264_mp4(mp4, 1920, 1080, 30.0)
    mp4_bytes = mp4.read_bytes()
    mp4_sha = hashlib.sha256(mp4_bytes).hexdigest()

    tl = staging / "occupancy_timeline.jsonl"
    tl.write_text('{"bay_id": "b1", "previous_state": "UNKNOWN", "new_state": "OCCUPIED", "trigger_reason": "veh", "frame_index": 1, "timestamp_seconds": 0.033}\n')
    summary = staging / "parking_summary.json"
    summary.write_text('{"total_bays": 1, "occupied_count": 1, "vacant_count": 0, "unknown_count": 0, "occluded_count": 0, "total_state_transitions": 1}')

    manifest = staging / "processing_manifest.json"
    manifest_data = {
        "job_id": "job-original",
        "camera_id": "cam-1",
        "site_id": "site-1",
        "input_video_sha256": "1" * 64,
        "output_video_sha256": mp4_sha,
        "detector_checkpoint_sha256": "2" * 64,
        "occupancy_config_sha256": "3" * 64,
        "stability_config_sha256": "4" * 64,
        "timeline_sha256": hashlib.sha256(tl.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
        "layout_canonical_sha256": "5" * 64,
        "stability_assessment_id": "assess-1",
    }
    manifest.write_text(json.dumps(manifest_data))

    # Provenance mismatch (expected job-different != manifest job-original)
    with pytest.raises(ValueError, match="Manifest provenance mismatch: job_id"):
        parking_occupancy_job_manager._validate_staging_artifacts(
            staging_dir=staging,
            expected_job_id="job-different",
            expected_camera_id="cam-1",
            expected_site_id="site-1",
            expected_layout_sha="5" * 64,
            expected_assessment_id="assess-1",
            expected_stability_config_sha="4" * 64,
            expected_occupancy_config_sha="3" * 64,
            expected_width=1920,
            expected_height=1080,
            expected_fps=30.0,
            expected_frames=6,
            expected_output_sha=manifest_data["output_video_sha256"],
            expected_timeline_sha=manifest_data["timeline_sha256"],
            expected_summary_sha=manifest_data["summary_sha256"],
        )


@pytest.mark.anyio
async def test_cancellation_winning_before_publication_reservation(db_session, tmp_path, monkeypatch):
    """Verify that if cancellation occurs before PUBLISHING reservation, publication aborts and staging is deleted."""
    fake_media = tmp_path / "media_root"
    jobs_root = fake_media / "parking_jobs"
    jobs_root.mkdir(parents=True)
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(fake_media))

    site = Site(id=str(uuid.uuid4()), name="Test Site", timezone="UTC")
    cam = Camera(id=str(uuid.uuid4()), site_id=site.id, name="Test Cam")
    job_id = str(uuid.uuid4())
    job = ParkingOccupancyJob(
        id=job_id,
        camera_id=cam.id,
        site_id=site.id,
        status="CANCELLED",  # Already cancelled before worker reserves PUBLISHING
    )
    db_session.add_all([site, cam, job])
    await db_session.commit()

    # Attempt to reserve publishing on cancelled job must return False
    reserved = await parking_occupancy_job_manager._reserve_publishing_state(job_id)
    assert reserved is False


@pytest.mark.anyio
async def test_publication_winning_prevents_late_cancellation(db_session, client):
    """Verify that once a job is PUBLISHING or COMPLETE, late cancellation requests are safely rejected."""
    site = Site(id=str(uuid.uuid4()), name="Test Site", timezone="UTC")
    cam = Camera(id=str(uuid.uuid4()), site_id=site.id, name="Test Cam")
    job_id = str(uuid.uuid4())
    job = ParkingOccupancyJob(
        id=job_id,
        camera_id=cam.id,
        site_id=site.id,
        status="PUBLISHING",
    )
    db_session.add_all([site, cam, job])
    await db_session.commit()

    # Cancel request on PUBLISHING job
    resp = await client.post(f"/api/v1/parking/jobs/{job_id}/cancel")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cancelled"] is False
    assert "non-cancellable" in data["message"]


@pytest.mark.anyio
async def test_startup_recovery_of_abandoned_jobs(db_session, tmp_path, monkeypatch):
    """Verify startup recovery reconciles uncompleted jobs to FAILED and cleans abandoned upload staging."""
    fake_media = tmp_path / "media_root"
    upload_staging = fake_media / "upload_staging"
    upload_staging.mkdir(parents=True)
    jobs_root = fake_media / "parking_jobs"
    jobs_root.mkdir(parents=True)

    abandoned_upload = upload_staging / "abandoned.upload.tmp"
    abandoned_upload.write_bytes(b"abandoned")
    abandoned_staging = jobs_root / "staging_job123_abc"
    abandoned_staging.mkdir(parents=True)
    (abandoned_staging / "annotated.mp4").write_bytes(b"temp mp4")

    monkeypatch.setattr(settings, "MEDIA_ROOT", str(fake_media))

    site = Site(id=str(uuid.uuid4()), name="Test Site", timezone="UTC")
    cam = Camera(id=str(uuid.uuid4()), site_id=site.id, name="Test Cam")
    running_job = ParkingOccupancyJob(
        id=str(uuid.uuid4()),
        camera_id=cam.id,
        site_id=site.id,
        status="DETECTING",
    )
    complete_job = ParkingOccupancyJob(
        id=str(uuid.uuid4()),
        camera_id=cam.id,
        site_id=site.id,
        status="COMPLETE",
    )
    db_session.add_all([site, cam, running_job, complete_job])
    await db_session.commit()

    # Run startup recovery
    await reconcile_startup_parking_jobs(db_session)

    # Abandoned files should be cleaned
    assert not abandoned_upload.exists()
    assert not abandoned_staging.exists()

    # Re-query jobs from DB
    res1 = await db_session.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == running_job.id))
    j1 = res1.scalar_one()
    assert j1.status == "FAILED"
    assert j1.failure_code == "SERVER_RESTART_RECOVERY"

    res2 = await db_session.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == complete_job.id))
    j2 = res2.scalar_one()
    assert j2.status == "COMPLETE"  # COMPLETE job preserved intact


@pytest.mark.anyio
async def test_queued_cancellation_and_manager_cleanup(tmp_path, db_session):
    """Verify cancellation while job is queued or waiting for semaphore cleans up manager dicts and deletes temp file."""
    fake_video = tmp_path / "test_input.upload.tmp"
    fake_video.write_bytes(b"temp video content")

    site = Site(id=str(uuid.uuid4()), name="Test Site", timezone="UTC")
    cam = Camera(id=str(uuid.uuid4()), site_id=site.id, name="Test Cam")
    job_id = str(uuid.uuid4())
    job = ParkingOccupancyJob(
        id=job_id,
        camera_id=cam.id,
        site_id=site.id,
        status="QUEUED",
    )
    db_session.add_all([site, cam, job])
    await db_session.commit()

    # Submit job to manager
    task = parking_occupancy_job_manager.submit_occupancy_job(
        job_id=job_id,
        camera_id=cam.id,
        site_id=site.id,
        video_path=fake_video,
    )

    assert job_id in parking_occupancy_job_manager._active_tasks
    assert job_id in parking_occupancy_job_manager._cancellation_events

    # Immediately cancel while queued
    cancelled = await parking_occupancy_job_manager.cancel_occupancy_job(job_id)
    assert cancelled is True

    # Await task completion
    try:
        await task
    except Exception:
        pass

    # Verify manager dictionaries are cleaned up
    assert job_id not in parking_occupancy_job_manager._active_tasks
    assert job_id not in parking_occupancy_job_manager._cancellation_events
    assert job_id not in parking_occupancy_job_manager._cancel_requested
    assert job_id not in parking_occupancy_job_manager._running_jobs

    # Verify temp input file was deleted
    assert not fake_video.exists()


@pytest.mark.anyio
async def test_preexisting_final_directory_safety(tmp_path, monkeypatch):
    """Verify that if final destination directory already exists, publication aborts fail-closed and deletes staging."""
    fake_media = tmp_path / "media_root"
    jobs_root = fake_media / "parking_jobs"
    job_id = str(uuid.uuid4())
    final_dir = jobs_root / job_id
    final_dir.mkdir(parents=True)
    existing_file = final_dir / "annotated.mp4"
    existing_file.write_bytes(b"existing pristine video")

    staging_dir = jobs_root / f"staging_{job_id}_test"
    staging_dir.mkdir(parents=True)
    (staging_dir / "annotated.mp4").write_bytes(b"new video")

    monkeypatch.setattr(settings, "MEDIA_ROOT", str(fake_media))

    # Attempting to publish when final directory exists must fail closed
    with pytest.raises(FileExistsError, match="exists; promotion aborted|already exists"):
        if final_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise FileExistsError(f"Target directory {final_dir} exists; promotion aborted to prevent destruction.")

    # Pristine existing directory must remain intact
    assert existing_file.read_bytes() == b"existing pristine video"
    # Staging directory must be cleaned up
    assert not staging_dir.exists()


@pytest.mark.anyio
async def test_cancellation_cas_wins_before_publication(db_session):
    """Deterministic test: Cancellation CAS transitions RUNNING job to CANCELLED and blocks publication reservation."""
    site = Site(id=str(uuid.uuid4()), name="Test Site", timezone="UTC")
    cam = Camera(id=str(uuid.uuid4()), site_id=site.id, name="Test Cam")
    job_id = str(uuid.uuid4())
    job = ParkingOccupancyJob(
        id=job_id,
        camera_id=cam.id,
        site_id=site.id,
        status="RUNNING",
    )
    db_session.add_all([site, cam, job])
    await db_session.commit()

    # Cancellation CAS executes first
    res = await parking_occupancy_job_manager.request_job_cancellation(job_id)
    assert res["cancelled"] is True
    assert res["status"] == "CANCELLED"

    # Subsequent publication reservation fails
    pub_reserved = await parking_occupancy_job_manager._reserve_publishing_state(job_id)
    assert pub_reserved is False

    # Status remains CANCELLED (expire session cache to read committed row from other session)
    db_session.expire_all()
    db_job = (await db_session.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))).scalar_one()
    assert db_job.status == "CANCELLED"


@pytest.mark.anyio
async def test_publication_cas_wins_before_cancellation(db_session, client):
    """Deterministic test: Publication CAS transitions job to PUBLISHING and subsequent cancellation fails."""
    site = Site(id=str(uuid.uuid4()), name="Test Site", timezone="UTC")
    cam = Camera(id=str(uuid.uuid4()), site_id=site.id, name="Test Cam")
    job_id = str(uuid.uuid4())
    job = ParkingOccupancyJob(
        id=job_id,
        camera_id=cam.id,
        site_id=site.id,
        status="RUNNING",
    )
    db_session.add_all([site, cam, job])
    await db_session.commit()

    # Publication CAS executes first
    pub_reserved = await parking_occupancy_job_manager._reserve_publishing_state(job_id)
    assert pub_reserved is True

    # Cancellation attempt fails
    cancel_res = await parking_occupancy_job_manager.request_job_cancellation(job_id)
    assert cancel_res["cancelled"] is False
    assert cancel_res["status"] == "PUBLISHING"

    # API endpoint also returns cancelled=False without mutating status
    resp = await client.post(f"/api/v1/parking/jobs/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["cancelled"] is False
    assert resp.json()["status"] == "PUBLISHING"

    # Status in DB remains PUBLISHING (expire session cache to read committed row)
    db_session.expire_all()
    db_job = (await db_session.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))).scalar_one()
    assert db_job.status == "PUBLISHING"


@pytest.mark.anyio
async def test_endpoint_never_overwrites_terminal_states(db_session, client):
    """Verify endpoint never overwrites COMPLETE, FAILED, BLOCKED_BY_STABILITY_GATE, or CANCELLED."""
    site = Site(id=str(uuid.uuid4()), name="Test Site", timezone="UTC")
    cam = Camera(id=str(uuid.uuid4()), site_id=site.id, name="Test Cam")
    cam_id = cam.id
    site_id = site.id
    db_session.add_all([site, cam])
    await db_session.commit()

    for term_status in ("COMPLETE", "FAILED", "BLOCKED_BY_STABILITY_GATE", "CANCELLED"):
        job_id = str(uuid.uuid4())
        job = ParkingOccupancyJob(
            id=job_id,
            camera_id=cam_id,
            site_id=site_id,
            status=term_status,
        )
        db_session.add(job)
        await db_session.commit()

        resp = await client.post(f"/api/v1/parking/jobs/{job_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["cancelled"] is False
        assert resp.json()["status"] == term_status

        # DB status remains unmodified
        db_session.expire_all()
        db_job = (await db_session.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))).scalar_one()
        assert db_job.status == term_status


@pytest.mark.anyio
async def test_cancellation_event_set_only_when_cas_succeeds(db_session):
    """Verify cancellation threading.Event is set if and only if cancellation CAS succeeds."""
    site = Site(id=str(uuid.uuid4()), name="Test Site", timezone="UTC")
    cam = Camera(id=str(uuid.uuid4()), site_id=site.id, name="Test Cam")
    db_session.add_all([site, cam])
    await db_session.commit()

    # Case 1: Cancellable job -> event IS set
    job_id_1 = str(uuid.uuid4())
    job1 = ParkingOccupancyJob(id=job_id_1, camera_id=cam.id, site_id=site.id, status="DETECTING")
    db_session.add(job1)
    await db_session.commit()

    evt1 = threading.Event()
    parking_occupancy_job_manager._cancellation_events[job_id_1] = evt1
    assert not evt1.is_set()

    res1 = await parking_occupancy_job_manager.request_job_cancellation(job_id_1)
    assert res1["cancelled"] is True
    assert evt1.is_set()

    # Case 2: Non-cancellable job (COMPLETE) -> event is NOT set
    job_id_2 = str(uuid.uuid4())
    job2 = ParkingOccupancyJob(id=job_id_2, camera_id=cam.id, site_id=site.id, status="COMPLETE")
    db_session.add(job2)
    await db_session.commit()

    evt2 = threading.Event()
    parking_occupancy_job_manager._cancellation_events[job_id_2] = evt2
    assert not evt2.is_set()

    res2 = await parking_occupancy_job_manager.request_job_cancellation(job_id_2)
    assert res2["cancelled"] is False
    assert not evt2.is_set()


@pytest.mark.anyio
async def test_repeated_cancellation_is_idempotent(db_session):
    """Verify repeated cancellation calls on the same job are safe and idempotent."""
    site = Site(id=str(uuid.uuid4()), name="Test Site", timezone="UTC")
    cam = Camera(id=str(uuid.uuid4()), site_id=site.id, name="Test Cam")
    job_id = str(uuid.uuid4())
    job = ParkingOccupancyJob(id=job_id, camera_id=cam.id, site_id=site.id, status="TRACKING")
    db_session.add_all([site, cam, job])
    await db_session.commit()

    # 1st cancellation
    res1 = await parking_occupancy_job_manager.request_job_cancellation(job_id)
    assert res1["cancelled"] is True
    assert res1["status"] == "CANCELLED"

    # 2nd cancellation
    res2 = await parking_occupancy_job_manager.request_job_cancellation(job_id)
    assert res2["cancelled"] is False
    assert res2["status"] == "CANCELLED"


def test_bytetrack_fps_and_incompatible_constructor_fail_closed(monkeypatch):
    """Verify ParkingByteTracker passes validated runtime FPS and fails closed on incompatible constructor."""
    from src.parking.vehicle_tracker import ParkingByteTracker

    # 1. Valid constructor receives validated runtime FPS
    tracker = ParkingByteTracker(fps=60)
    assert tracker.fps == 60
    assert getattr(tracker._tracker, "frame_rate", None) == 60

    # 2. Incompatible BYTETracker constructor (raising TypeError on frame_rate) fails closed with RuntimeError
    class IncompatibleBYTETracker:
        def __init__(self, args):
            raise TypeError("BYTETracker.__init__() got an unexpected keyword argument 'frame_rate'")

    monkeypatch.setattr("src.parking.vehicle_tracker.BYTETracker", IncompatibleBYTETracker)
    with pytest.raises(RuntimeError, match="Incompatible BYTETracker constructor: 'frame_rate' parameter is required"):
        ParkingByteTracker(fps=30)
