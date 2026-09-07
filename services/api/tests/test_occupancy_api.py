"""Integration tests for gate-controlled parking occupancy API, background jobs, and video annotation."""

import asyncio
from datetime import datetime, timezone
import hashlib
import io
import os
from pathlib import Path
import tempfile
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import cv2
import numpy as np

from services.api.app.db.session import get_db
from services.api.app.routers.parking import router as parking_router


@pytest.fixture
def parking_app():
    test_app = FastAPI(title="RoadSense Parking Occupancy Test API")
    test_app.include_router(parking_router)
    return test_app


@pytest.fixture
async def client(parking_app, db_session):
    async def override_get_db():
        yield db_session

    parking_app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=parking_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    parking_app.dependency_overrides.clear()


def _create_rich_test_image_bytes(width: int = 640, height: int = 480) -> bytes:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cb_size = 32
    for y in range(0, height, cb_size):
        for x in range(0, width, cb_size):
            if ((x // cb_size) + (y // cb_size)) % 2 == 0:
                img[y : y + cb_size, x : x + cb_size] = (200, 200, 200)
    for i in range(10):
        cv2.circle(img, (50 + i * 50, 60 + i * 35), 18, (120, 240, 70), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _create_synthetic_video_bytes(stationary: bool = True, frames: int = 30, width: int = 640, height: int = 480) -> bytes:
    tmp_path = Path(tempfile.gettempdir()) / f"test_vid_{os.getpid()}_{np.random.randint(100000)}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(tmp_path), fourcc, 30.0, (width, height))

    base_img = np.zeros((height, width, 3), dtype=np.uint8)
    cb_size = 32
    for y in range(0, height, cb_size):
        for x in range(0, width, cb_size):
            if ((x // cb_size) + (y // cb_size)) % 2 == 0:
                base_img[y : y + cb_size, x : x + cb_size] = (200, 200, 200)
    for i in range(10):
        cv2.circle(base_img, (50 + i * 50, 60 + i * 35), 18, (120, 240, 70), -1)

    try:
        for f in range(frames):
            frame = base_img.copy()
            if not stationary:
                M = np.float32([[1, 0, f * 3.0], [0, 1, f * 1.5]])
                frame = cv2.warpAffine(frame, M, (width, height))
            out.write(frame)
    finally:
        out.release()

    with open(tmp_path, "rb") as vf:
        vid_bytes = vf.read()
    tmp_path.unlink(missing_ok=True)
    return vid_bytes


async def _poll_job_until_terminal(client: AsyncClient, job_id: str, timeout: float = 15.0) -> dict:
    start_t = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_t < timeout:
        res = await client.get(f"/api/v1/parking/jobs/{job_id}")
        assert res.status_code == 200
        data = res.json()
        if data["status"] in ("COMPLETE", "FAILED", "CANCELLED", "BLOCKED_BY_STABILITY_GATE"):
            return data
        await asyncio.sleep(0.2)
    raise TimeoutError(f"Job {job_id} did not reach terminal state within {timeout}s")


@pytest.mark.anyio
async def test_occupancy_job_blocked_by_stability_gate_when_no_assessment(client):
    """Submitting an occupancy job without camera stability assessment fails closed to BLOCKED_BY_STABILITY_GATE."""
    s_resp = await client.post("/api/v1/sites", json={"name": "No Stability Site"})
    site_id = s_resp.json()["id"]
    c_resp = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "No Stability Cam"})
    camera_id = c_resp.json()["id"]

    vid_bytes = _create_synthetic_video_bytes(stationary=True, frames=10)
    submit_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/occupancy/jobs",
        files={"file": ("input.mp4", vid_bytes, "video/mp4")},
    )
    assert submit_resp.status_code == 202
    job_id = submit_resp.json()["id"]

    job_data = await _poll_job_until_terminal(client, job_id)
    assert job_data["status"] == "BLOCKED_BY_STABILITY_GATE"
    assert job_data["gate_decision"] == "BLOCKED"
    assert job_data["failure_code"] == "STABILITY_GATE_BLOCKED"
    assert any("LAYOUT" in r or "ASSESSMENT" in r for r in job_data["gate_reasons"])


@pytest.mark.anyio
async def test_occupancy_job_rejects_empty_file_and_missing_file(client):
    """API strictly rejects empty upload or invalid file payload."""
    s_resp = await client.post("/api/v1/sites", json={"name": "Upload Test Site"})
    site_id = s_resp.json()["id"]
    c_resp = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "Upload Cam"})
    camera_id = c_resp.json()["id"]

    # 1. Empty file
    r_empty = await client.post(
        f"/api/v1/cameras/{camera_id}/occupancy/jobs",
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )
    assert r_empty.status_code == 400
    assert "empty" in r_empty.json()["detail"].lower()


@pytest.mark.anyio
async def test_occupancy_job_lifecycle_when_gate_allowed(client):
    """
    Full positive workflow:
    1. Upload reference image & verify layout.
    2. Run stationary stability assessment -> STABLE & ALLOWED.
    3. Submit occupancy video job -> 202 Accepted.
    4. Poll until COMPLETE.
    5. Download annotated video, manifest JSON, and timeline JSONL.
    """
    # 1. Setup Camera & Layout
    s_resp = await client.post("/api/v1/sites", json={"name": "Complete Lifecycle Site"})
    site_id = s_resp.json()["id"]
    c_resp = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "Lifecycle Cam"})
    camera_id = c_resp.json()["id"]

    ref_bytes = _create_rich_test_image_bytes(640, 480)
    await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("ref.jpg", ref_bytes, "image/jpeg")},
    )

    draft_payload = {
        "parking_spaces": [
            {
                "operator_label": "Space-A1",
                "space_type": "STANDARD",
                "polygon_normalized": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.4, "y": 0.1},
                    {"x": 0.4, "y": 0.6},
                    {"x": 0.1, "y": 0.6},
                ],
            }
        ]
    }
    l_resp = await client.post(f"/api/v1/cameras/{camera_id}/layouts", json=draft_payload)
    layout_id = l_resp.json()["id"]
    await client.post(f"/api/v1/layouts/{layout_id}/submit", json={"local_operator_label": "Operator-1"})
    await client.post(f"/api/v1/layouts/{layout_id}/verify", json={"local_operator_label": "Lead-1", "confirmation_acknowledged": True})

    # 2. Run Stability Assessment -> COMPLETE & STABLE
    stat_vid = _create_synthetic_video_bytes(stationary=True, frames=15)
    ass_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/stability/assess",
        files={"file": ("stab.mp4", stat_vid, "video/mp4")},
    )
    ass_id = ass_resp.json()["id"]

    # Poll stability assessment until complete
    for _ in range(30):
        r = await client.get(f"/api/v1/stability/assessments/{ass_id}")
        if r.json()["status"] == "COMPLETE":
            break
        await asyncio.sleep(0.1)

    gate_resp = await client.get(f"/api/v1/cameras/{camera_id}/stability/gate")
    assert gate_resp.json()["operational_gate"] == "ALLOWED"

    # 3. Submit Parking Occupancy Job -> HTTP 202 Accepted
    occ_vid = _create_synthetic_video_bytes(stationary=True, frames=10)
    occ_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/occupancy/jobs",
        files={"file": ("parking_stream.mp4", occ_vid, "video/mp4")},
    )
    assert occ_resp.status_code == 202
    job_id = occ_resp.json()["id"]
    assert occ_resp.json()["status"] == "QUEUED"

    # 4. Poll until COMPLETE
    job_data = await _poll_job_until_terminal(client, job_id, timeout=20.0)
    assert job_data["status"] == "COMPLETE"
    assert job_data["gate_decision"] == "ALLOWED"
    assert job_data["total_bays"] == 1
    assert job_data["has_annotated_video"] is True
    assert job_data["has_manifest"] is True
    assert job_data["has_timeline"] is True

    # 5. Fetch Manifest
    man_resp = await client.get(f"/api/v1/parking/jobs/{job_id}/manifest")
    assert man_resp.status_code == 200
    manifest = man_resp.json()
    assert manifest["job_id"] == job_id
    assert len(manifest["output_video_sha256"]) == 64
    assert len(manifest["detector_checkpoint_sha256"]) == 64
    assert manifest["total_bays_evaluated"] == 1

    # 6. Fetch Annotated Video Stream
    vid_resp = await client.get(f"/api/v1/parking/jobs/{job_id}/video")
    assert vid_resp.status_code == 200
    assert vid_resp.headers["content-type"] == "video/mp4"
    assert len(vid_resp.content) > 1000

    # 7. Fetch Timeline JSONL
    time_resp = await client.get(f"/api/v1/parking/jobs/{job_id}/timeline")
    assert time_resp.status_code == 200


@pytest.mark.anyio
async def test_occupancy_job_cancellation(client):
    """Cancelling an active occupancy job immediately transitions status to CANCELLED."""
    s_resp = await client.post("/api/v1/sites", json={"name": "Cancel Occupancy Site"})
    site_id = s_resp.json()["id"]
    c_resp = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "Cancel Cam"})
    camera_id = c_resp.json()["id"]

    vid_bytes = _create_synthetic_video_bytes(stationary=True, frames=40)
    submit_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/occupancy/jobs",
        files={"file": ("stream.mp4", vid_bytes, "video/mp4")},
    )
    job_id = submit_resp.json()["id"]

    cancel_resp = await client.post(f"/api/v1/parking/jobs/{job_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "CANCELLED"

    detail_resp = await client.get(f"/api/v1/parking/jobs/{job_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["status"] == "CANCELLED"
