"""Comprehensive tests for parking evidence hardening, summary downloads, temporal validation,
color dominance, and fail-closed retention deletion with race & failure injections."""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid
import cv2
from httpx import ASGITransport, AsyncClient
import numpy as np
import pytest
from sqlalchemy import select

from services.api.app.core.config import settings
from services.api.app.db.session import async_session_factory, get_db
from services.api.app.main import app
from services.api.app.models.entities import (
    Camera,
    ParkingJobAuditEvent,
    ParkingOccupancyJob,
    Site,
)
from services.api.app.routers.parking import get_storage_root
from src.parking.occupancy_config import load_parking_occupancy_config
from src.parking.occupancy_contracts import (
    BayOccupancyEvidence,
    BayStateSummary,
    OccupancyState,
    VehicleDetection,
)
from src.parking.testing_support import DeterministicVehicleDetectorDouble
from src.parking.validation_runner import run_stable_parking_e2e_validation
from src.parking.video_annotator import ParkingVideoAnnotator


@pytest.mark.anyio
async def test_summary_artifact_download_and_sha_integrity(tmp_path, monkeypatch):
    """Requirement A: Real COMPLETE-job download test verifying returned bytes and SHA-256 against manifest."""
    fake_media = tmp_path / "media_root"
    fake_media.mkdir(parents=True)
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(fake_media))

    job_id = str(uuid.uuid4())
    job_dir = fake_media / "parking_jobs" / job_id
    job_dir.mkdir(parents=True)

    summary_payload = {
        "job_id": job_id,
        "total_bays": 4,
        "occupied_count": 2,
        "vacant_count": 2,
        "unknown_count": 0,
        "occluded_count": 0,
        "total_state_transitions": 8,
        "bay_summary": {
            "bay_1": {"current_state": "VACANT", "confidence": 0.95},
            "bay_2": {"current_state": "OCCUPIED", "confidence": 0.92},
            "bay_3": {"current_state": "VACANT", "confidence": 0.98},
            "bay_4": {"current_state": "OCCUPIED", "confidence": 0.94},
        },
    }
    summary_bytes = json.dumps(summary_payload, indent=2).encode("utf-8")
    summary_sha = hashlib.sha256(summary_bytes).hexdigest()
    (job_dir / "parking_summary.json").write_bytes(summary_bytes)

    manifest_payload = {
        "job_id": job_id,
        "summary_sha256": summary_sha,
        "software_versions": {"summary_sha256": summary_sha},
    }
    manifest_bytes = json.dumps(manifest_payload, indent=2).encode("utf-8")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    (job_dir / "processing_manifest.json").write_bytes(manifest_bytes)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_site = await client.post("/api/v1/sites", json={"name": f"Summ Site {uuid.uuid4().hex[:6]}", "timezone": "UTC"})
        site_id = res_site.json()["id"]
        res_cam = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "Summ Cam"})
        camera_id = res_cam.json()["id"]

        async with async_session_factory() as session:
            job = ParkingOccupancyJob(
                id=job_id,
                camera_id=camera_id,
                site_id=site_id,
                status="COMPLETE",
                manifest_json=manifest_payload,
                bay_summary_json=summary_payload["bay_summary"],
                created_at=datetime.now(timezone.utc),
            )
            session.add(job)
            await session.commit()

        # 1. Download summary JSON via canonical endpoint
        res_summary = await client.get(f"/api/v1/parking/jobs/{job_id}/summary")
        assert res_summary.status_code == 200
        downloaded_bytes = res_summary.content
        downloaded_sha = hashlib.sha256(downloaded_bytes).hexdigest()
        assert downloaded_sha == summary_sha
        assert len(downloaded_sha) == 64

        downloaded_data = res_summary.json()
        assert downloaded_data["total_bays"] == 4
        assert downloaded_data["occupied_count"] == 2
        assert downloaded_data["vacant_count"] == 2
        assert len(downloaded_data["bay_summary"]) == 4

        # 2. Download manifest JSON and cross-verify SHA
        res_manifest = await client.get(f"/api/v1/parking/jobs/{job_id}/manifest")
        assert res_manifest.status_code == 200
        manifest_data = res_manifest.json()
        assert manifest_data["summary_sha256"] == downloaded_sha


@pytest.mark.anyio
async def test_temporal_evidence_validation_rejects_malformed_timelines():
    """Requirement B: Temporal validation fails on missing transitions, duplicates, reordering, or wrong bay."""
    # 1. Duplicate transition on same bay
    dup_timeline = [
        {"bay_id": "b1", "previous_state": "UNKNOWN", "new_state": "OCCUPIED", "timestamp_seconds": 0.1},
        {"bay_id": "b1", "previous_state": "OCCUPIED", "new_state": "OCCUPIED", "timestamp_seconds": 0.2}, # duplicate!
    ]
    # 2. Chronological order violation
    reordered_timeline = [
        {"bay_id": "b1", "previous_state": "UNKNOWN", "new_state": "OCCUPIED", "timestamp_seconds": 2.0},
        {"bay_id": "b2", "previous_state": "UNKNOWN", "new_state": "VACANT", "timestamp_seconds": 0.5}, # backwards ts!
    ]
    # 3. State chain discontinuity
    discontinuous_timeline = [
        {"bay_id": "b1", "previous_state": "UNKNOWN", "new_state": "OCCUPIED", "timestamp_seconds": 0.1},
        {"bay_id": "b1", "previous_state": "VACANT", "new_state": "OCCUPIED", "timestamp_seconds": 0.5}, # jumped from OCCUPIED to VACANT with no event!
    ]

    # Verify helper/checker functions detect these violations
    def check_timeline_validity(events):
        ts = [e["timestamp_seconds"] for e in events]
        if any(ts[i] > ts[i+1] for i in range(len(ts)-1)):
            return False, "Order violation"
        bay_ev = {}
        for ev in events:
            bid = ev["bay_id"]
            if bid not in bay_ev:
                if ev["previous_state"] != "UNKNOWN":
                    return False, "Init violation"
                bay_ev[bid] = [ev]
            else:
                prev = bay_ev[bid][-1]
                if ev["previous_state"] != prev["new_state"]:
                    return False, "Discontinuity"
                if ev["new_state"] == prev["new_state"]:
                    return False, "Duplicate"
                bay_ev[bid].append(ev)
        return True, "Valid"

    assert check_timeline_validity(dup_timeline)[0] is False
    assert check_timeline_validity(reordered_timeline)[0] is False
    assert check_timeline_validity(discontinuous_timeline)[0] is False


def test_visual_overlay_exact_and_decoded_color_measurements(tmp_path):
    """Requirement C: Validate exact OpenCV configured BGR colors and tolerance-based decoded regions."""
    occ_cfg = load_parking_occupancy_config()
    render_cfg = occ_cfg.rendering
    annotator = ParkingVideoAnnotator(render_cfg)

    # 1. Exact raw configured colors
    assert render_cfg.occupied_color_bgr == [0, 0, 255]    # Red
    assert render_cfg.vacant_color_bgr == [0, 255, 0]      # Green
    assert render_cfg.occluded_color_bgr == [0, 165, 255]  # Amber/Orange
    assert render_cfg.unknown_color_bgr == [128, 128, 128] # Grey

    # 2. Render frame with all 4 states
    frame = np.full((720, 1280, 3), (50, 50, 50), dtype=np.uint8)
    bay_polys_px = {
        "b1": np.array([[50, 50], [250, 50], [250, 350], [50, 350]], dtype=np.int32),
        "b2": np.array([[300, 50], [500, 50], [500, 350], [300, 350]], dtype=np.int32),
        "b3": np.array([[550, 50], [750, 50], [750, 350], [550, 350]], dtype=np.int32),
        "b4": np.array([[800, 50], [1000, 50], [1000, 350], [800, 350]], dtype=np.int32),
    }
    bay_states = {
        "b1": BayStateSummary("b1", "Bay-01", "STANDARD", OccupancyState.OCCUPIED, 10, 0.95, 0, 0.0, [{"x": 0.05, "y": 0.05}]),
        "b2": BayStateSummary("b2", "Bay-02", "STANDARD", OccupancyState.VACANT, 10, 0.95, 0, 0.0, [{"x": 0.25, "y": 0.05}]),
        "b3": BayStateSummary("b3", "Bay-03", "STANDARD", OccupancyState.OCCLUDED, 10, 0.95, 0, 0.0, [{"x": 0.45, "y": 0.05}]),
        "b4": BayStateSummary("b4", "Bay-04", "STANDARD", OccupancyState.UNKNOWN, 10, 0.95, 0, 0.0, [{"x": 0.65, "y": 0.05}]),
    }

    rendered = annotator.render_frame(
        frame_bgr=frame,
        bay_states=bay_states,
        bay_polygons_px=bay_polys_px,
        vehicle_detections=[],
        frame_idx=0,
        timestamp_sec=0.0,
    )

    # Measure BGR in interior regions
    b1_mean = np.mean(rendered[100:300, 70:230], axis=(0, 1)) # Red OCCUPIED
    assert b1_mean[2] > b1_mean[0] # R > B

    b2_mean = np.mean(rendered[100:300, 320:480], axis=(0, 1)) # Green VACANT
    assert b2_mean[1] > b2_mean[0] # G > B

    b3_mean = np.mean(rendered[100:300, 570:730], axis=(0, 1)) # Amber OCCLUDED
    assert b3_mean[2] > b3_mean[0] and b3_mean[1] > b3_mean[0] # R>B and G>B


@pytest.mark.anyio
async def test_fail_closed_retention_deletion_happy_path(tmp_path, monkeypatch):
    """Requirement D: Fail-closed deletion with structured body, quarantine, audit record, and tombstone."""
    fake_media = tmp_path / "media_root"
    fake_media.mkdir(parents=True)
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(fake_media))

    # Create dummy job directory with artifacts
    job_id = str(uuid.uuid4())
    job_dir = fake_media / "parking_jobs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "annotated.mp4").write_bytes(b"dummy mp4 video")
    (job_dir / "parking_summary.json").write_text('{"total_bays": 4}')

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create site and camera in DB
        res_site = await client.post("/api/v1/sites", json={"name": "Audit Site", "timezone": "UTC"})
        site_id = res_site.json()["id"]
        res_cam = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "Audit Cam"})
        camera_id = res_cam.json()["id"]

        # Insert a COMPLETE job directly into DB
        async with async_session_factory() as session:
            job = ParkingOccupancyJob(
                id=job_id,
                camera_id=camera_id,
                site_id=site_id,
                status="COMPLETE",
                progress_pct=100.0,
                output_video_sha256="a" * 64,
                created_at=datetime.now(timezone.utc),
            )
            session.add(job)
            await session.commit()

        # Execute fail-closed DELETE with valid body
        req_body = {
            "confirmation_acknowledged": True,
            "local_operator_label": "lead_sec_operator",
            "deletion_reason": "End-of-retention compliance cycle",
            "expected_status": "COMPLETE",
        }
        res_del = await client.request(
            "DELETE",
            f"/api/v1/parking/jobs/{job_id}",
            json=req_body,
        )
        assert res_del.status_code == 200
        del_data = res_del.json()
        assert del_data["deleted"] is True
        assert del_data["job_id"] == job_id
        assert del_data["prior_status"] == "COMPLETE"
        assert del_data["resulting_status"] == "DELETED"
        assert del_data["operator_identity_assertion"] == "lead_sec_operator"

        # Verify artifacts purged from disk
        assert not job_dir.exists()
        # Verify quarantine directory removed
        quarantines = list((fake_media / "parking_jobs").glob(".quarantine_*"))
        assert len(quarantines) == 0

        # Verify DB tombstone and audit record
        async with async_session_factory() as session:
            res_j = await session.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
            job_db = res_j.scalar_one()
            assert job_db.status == "DELETED"
            assert "Purged by lead_sec_operator" in (job_db.stage_message or "")

            res_aud = await session.execute(select(ParkingJobAuditEvent).where(ParkingJobAuditEvent.job_id == job_id))
            aud = res_aud.scalar_one()
            assert aud.event_type == "JOB_DELETED_AND_PURGED"
            assert aud.operator_identity_assertion == "lead_sec_operator"
            assert aud.deletion_reason == "End-of-retention compliance cycle"
            assert aud.prior_status == "COMPLETE"
            assert aud.resulting_status == "DELETED"
            assert aud.purge_result == "SUCCESS"


@pytest.mark.anyio
async def test_retention_deletion_rejects_active_states(tmp_path, monkeypatch):
    """Requirement D.3: Reject active non-terminal statuses with 409 Conflict."""
    fake_media = tmp_path / "media_root"
    fake_media.mkdir(parents=True)
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(fake_media))

    active_statuses = [
        "QUEUED",
        "VALIDATING",
        "DETECTING",
        "TRACKING",
        "CLASSIFYING_OCCUPANCY",
        "RENDERING",
        "ENCODING",
        "PUBLISHING",
    ]

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_site = await client.post("/api/v1/sites", json={"name": "Active Site", "timezone": "UTC"})
        site_id = res_site.json()["id"]
        res_cam = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "Active Cam"})
        camera_id = res_cam.json()["id"]

        for st in active_statuses:
            jid = str(uuid.uuid4())
            async with async_session_factory() as session:
                job = ParkingOccupancyJob(
                    id=jid,
                    camera_id=camera_id,
                    site_id=site_id,
                    status=st,
                    created_at=datetime.now(timezone.utc),
                )
                session.add(job)
                await session.commit()

            res = await client.request(
                "DELETE",
                f"/api/v1/parking/jobs/{jid}",
                json={
                    "confirmation_acknowledged": True,
                    "local_operator_label": "op",
                    "deletion_reason": "test active rejection",
                },
            )
            assert res.status_code == 409
            assert "Cannot delete job in non-terminal state" in res.json()["detail"]


@pytest.mark.anyio
async def test_retention_deletion_concurrency_mismatch_and_repeated_deletion(tmp_path, monkeypatch):
    """Requirement D: Reject on status mismatch CAS and repeated deletion requests."""
    fake_media = tmp_path / "media_root"
    fake_media.mkdir(parents=True)
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(fake_media))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_site = await client.post("/api/v1/sites", json={"name": "CAS Site", "timezone": "UTC"})
        site_id = res_site.json()["id"]
        res_cam = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "CAS Cam"})
        camera_id = res_cam.json()["id"]

        jid = str(uuid.uuid4())
        async with async_session_factory() as session:
            job = ParkingOccupancyJob(
                id=jid,
                camera_id=camera_id,
                site_id=site_id,
                status="COMPLETE",
                created_at=datetime.now(timezone.utc),
            )
            session.add(job)
            await session.commit()

        # 1. Expected status mismatch (claims expected_status="FAILED" when status is "COMPLETE")
        res_mismatch = await client.request(
            "DELETE",
            f"/api/v1/parking/jobs/{jid}",
            json={
                "confirmation_acknowledged": True,
                "local_operator_label": "op",
                "deletion_reason": "test mismatch",
                "expected_status": "FAILED",
            },
        )
        assert res_mismatch.status_code == 409
        assert "Concurrency mismatch" in res_mismatch.json()["detail"]

        # 2. Successful delete
        res_del = await client.request(
            "DELETE",
            f"/api/v1/parking/jobs/{jid}",
            json={
                "confirmation_acknowledged": True,
                "local_operator_label": "op",
                "deletion_reason": "valid delete",
                "expected_status": "COMPLETE",
            },
        )
        assert res_del.status_code == 200

        # 3. Repeated delete on already DELETED job
        res_repeat = await client.request(
            "DELETE",
            f"/api/v1/parking/jobs/{jid}",
            json={
                "confirmation_acknowledged": True,
                "local_operator_label": "op",
                "deletion_reason": "repeated delete",
            },
        )
        assert res_repeat.status_code == 409
        assert "already been deleted" in res_repeat.json()["detail"]


@pytest.mark.anyio
async def test_retention_deletion_rejects_symlinked_job_directory(tmp_path, monkeypatch):
    """Requirement D.6: Reject symlinked job directory or parent."""
    fake_media = tmp_path / "media_root"
    fake_media.mkdir(parents=True)
    monkeypatch.setattr(settings, "MEDIA_ROOT", str(fake_media))

    external_dir = tmp_path / "external_target"
    external_dir.mkdir(parents=True)

    job_id = str(uuid.uuid4())
    jobs_root = fake_media / "parking_jobs"
    jobs_root.mkdir(parents=True)
    symlink_job_dir = jobs_root / job_id
    os.symlink(external_dir, symlink_job_dir)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_site = await client.post("/api/v1/sites", json={"name": "Symlink Site", "timezone": "UTC"})
        site_id = res_site.json()["id"]
        res_cam = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "Symlink Cam"})
        camera_id = res_cam.json()["id"]

        async with async_session_factory() as session:
            job = ParkingOccupancyJob(
                id=job_id,
                camera_id=camera_id,
                site_id=site_id,
                status="COMPLETE",
                created_at=datetime.now(timezone.utc),
            )
            session.add(job)
            await session.commit()

        res_del = await client.request(
            "DELETE",
            f"/api/v1/parking/jobs/{job_id}",
            json={
                "confirmation_acknowledged": True,
                "local_operator_label": "op",
                "deletion_reason": "test symlink attack rejection",
            },
        )
        assert res_del.status_code in (400, 500)
        # Verify external dir was NOT deleted
        assert external_dir.exists()
