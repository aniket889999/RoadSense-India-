"""Integration tests for camera stability API endpoints, asynchronous jobs, and fail-closed operational gate."""

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
    test_app = FastAPI(title="RoadSense Stability Test API")
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

    cv2.circle(img, (width // 4, height // 4), 40, (50, 100, 200), -1)
    cv2.rectangle(img, (width // 2, height // 3), (width // 2 + 100, height // 3 + 80), (80, 180, 50), -1)
    cv2.putText(img, "CALIBRATION TARGET", (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _create_synthetic_video_bytes(stationary: bool = True, frames: int = 30) -> bytes:
    img_bytes = _create_rich_test_image_bytes(640, 480)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
        tmp_path = Path(tmp_f.name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(tmp_path), fourcc, 30.0, (640, 480))

    try:
        for i in range(frames):
            if stationary:
                out.write(img)
            else:
                M = np.float32([[1, 0, float(i * 1.5)], [0, 1, 0]])
                shifted = cv2.warpAffine(img, M, (640, 480))
                out.write(shifted)
        out.release()

        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
        return video_bytes
    finally:
        tmp_path.unlink(missing_ok=True)


async def _poll_assessment_until_complete(client: AsyncClient, assessment_id: str, timeout: float = 10.0) -> dict:
    start_t = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_t < timeout:
        res = await client.get(f"/api/v1/stability/assessments/{assessment_id}")
        assert res.status_code == 200
        data = res.json()
        if data["status"] in ("COMPLETE", "FAILED", "CANCELLED"):
            return data
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Assessment {assessment_id} did not complete within {timeout}s")


@pytest.mark.anyio
async def test_gate_fail_closed_before_assessment(client):
    """A camera with verified layout must still have gate BLOCKED until stable assessment runs."""
    s_resp = await client.post("/api/v1/sites", json={"name": "Gate Test Site"})
    site_id = s_resp.json()["id"]
    c_resp = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "Gate Cam"})
    camera_id = c_resp.json()["id"]

    # Check gate before reference image -> BLOCKED
    gate_init = await client.get(f"/api/v1/cameras/{camera_id}/stability/gate")
    assert gate_init.status_code == 200
    assert gate_init.json()["operational_gate"] == "BLOCKED"
    assert any("NO_ASSESSMENT" in r for r in gate_init.json()["gate_reasons"])

    # Upload reference image
    img_bytes = _create_rich_test_image_bytes(640, 480)
    await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("ref.jpg", img_bytes, "image/jpeg")},
    )

    # Gate still BLOCKED
    gate_with_ref = await client.get(f"/api/v1/cameras/{camera_id}/stability/gate")
    assert gate_with_ref.json()["operational_gate"] == "BLOCKED"


@pytest.mark.anyio
async def test_api_rejects_empty_file_and_missing_file(client):
    """Public API strictly requires non-empty multipart upload and rejects local_video_name."""
    s_resp = await client.post("/api/v1/sites", json={"name": "Validation Site"})
    site_id = s_resp.json()["id"]
    c_resp = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "Validation Cam"})
    camera_id = c_resp.json()["id"]

    img_bytes = _create_rich_test_image_bytes(640, 480)
    await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("ref.jpg", img_bytes, "image/jpeg")},
    )

    # 1. Empty file (0 bytes) -> HTTP 400
    empty_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/stability/assess",
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )
    assert empty_resp.status_code == 400
    assert "empty" in empty_resp.json()["detail"].lower()

    # 2. Passing local_video_name instead of file -> HTTP 422 Unprocessable Entity
    legacy_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/stability/assess",
        data={"local_video_name": "PARKING LOT TEST.mp4"},
    )
    assert legacy_resp.status_code == 422


@pytest.mark.anyio
async def test_async_stability_assessment_and_operational_gate_lifecycle(client):
    """Full async lifecycle: submit video (202) -> poll COMPLETE -> STABLE -> ALLOWED -> replace image -> BLOCKED."""
    # 1. Setup camera & layout
    s_resp = await client.post("/api/v1/sites", json={"name": "Full Lifecycle Site"})
    site_id = s_resp.json()["id"]
    c_resp = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "Full Cam"})
    camera_id = c_resp.json()["id"]

    img_bytes = _create_rich_test_image_bytes(640, 480)
    await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("ref.jpg", img_bytes, "image/jpeg")},
    )

    draft_payload = {
        "parking_spaces": [
            {
                "operator_label": "B1",
                "space_type": "STANDARD",
                "polygon_normalized": [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.1}, {"x": 0.2, "y": 0.2}, {"x": 0.1, "y": 0.2}],
            }
        ]
    }
    l_resp = await client.post(f"/api/v1/cameras/{camera_id}/layouts", json=draft_payload)
    layout_id = l_resp.json()["id"]
    await client.post(f"/api/v1/layouts/{layout_id}/submit", json={"local_operator_label": "Alice"})
    await client.post(f"/api/v1/layouts/{layout_id}/verify", json={"local_operator_label": "Bob", "confirmation_acknowledged": True})

    # 2. Submit stationary video assessment -> HTTP 202 Accepted
    stat_vid = _create_synthetic_video_bytes(stationary=True, frames=30)
    assess_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/stability/assess",
        files={"file": ("stream.mp4", stat_vid, "video/mp4")},
    )
    assert assess_resp.status_code == 202
    ass_data = assess_resp.json()
    assert ass_data["status"] in ("QUEUED", "VALIDATING", "ANALYZING", "COMPLETE")
    assessment_id = ass_data["id"]

    # 3. Poll until completed
    complete_data = await _poll_assessment_until_complete(client, assessment_id)
    assert complete_data["status"] == "COMPLETE"
    assert complete_data["aggregate_decision"] == "STABLE"
    assert complete_data["operational_gate"] == "ALLOWED"
    assert complete_data["config_version"] is not None
    assert complete_data["config_sha256"] is not None
    assert len(complete_data["sample_measurements"]) > 0

    # 4. Operational gate endpoint returns ALLOWED
    gate_resp = await client.get(f"/api/v1/cameras/{camera_id}/stability/gate")
    assert gate_resp.status_code == 200
    gate_data = gate_resp.json()
    assert gate_data["operational_gate"] == "ALLOWED"
    assert gate_data["is_fresh"] is True

    # 5. List assessments for camera
    list_resp = await client.get(f"/api/v1/cameras/{camera_id}/stability/assessments")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1

    # 6. STABLE assessment cannot trigger invalidation -> HTTP 400
    invalid_req = await client.post(
        f"/api/v1/stability/assessments/{assessment_id}/acknowledge",
        json={
            "local_operator_label": "Alice",
            "trigger_calibration_invalidation": True,
            "confirm_invalidation": True,
            "invalidation_reason": "Trying to invalidate a stable camera",
        },
    )
    assert invalid_req.status_code == 400
    assert "strictly allowed only for unstable" in invalid_req.json()["detail"].lower()

    # 7. Replacing reference image flips gate to BLOCKED
    new_img = _create_rich_test_image_bytes(640, 480)
    await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("ref2.jpg", new_img, "image/jpeg")},
        data={"confirm_replacement": "true", "operator_label": "Alice", "reason": "Camera realigned"},
    )

    gate_after_replace = await client.get(f"/api/v1/cameras/{camera_id}/stability/gate")
    assert gate_after_replace.json()["operational_gate"] == "BLOCKED"
    assert any("STALE_REFERENCE_SHA" in r or "NO_VERIFIED_LAYOUT" in r for r in gate_after_replace.json()["gate_reasons"])


@pytest.mark.anyio
async def test_unstable_assessment_and_immutable_audit_logging(client):
    """Unstable video yields BLOCKED; operator can acknowledge and invalidate calibration with audit log."""
    s_resp = await client.post("/api/v1/sites", json={"name": "Unstable Site"})
    site_id = s_resp.json()["id"]
    c_resp = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "Unstable Cam"})
    camera_id = c_resp.json()["id"]

    img_bytes = _create_rich_test_image_bytes(640, 480)
    await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("ref.jpg", img_bytes, "image/jpeg")},
    )

    draft_payload = {
        "parking_spaces": [
            {
                "operator_label": "B1",
                "space_type": "STANDARD",
                "polygon_normalized": [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.1}, {"x": 0.2, "y": 0.2}, {"x": 0.1, "y": 0.2}],
            }
        ]
    }
    l_resp = await client.post(f"/api/v1/cameras/{camera_id}/layouts", json=draft_payload)
    layout_id = l_resp.json()["id"]
    await client.post(f"/api/v1/layouts/{layout_id}/submit", json={"local_operator_label": "Alice"})
    await client.post(f"/api/v1/layouts/{layout_id}/verify", json={"local_operator_label": "Bob", "confirmation_acknowledged": True})

    # Submit moving/unstable video
    moving_vid = _create_synthetic_video_bytes(stationary=False, frames=30)
    assess_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/stability/assess",
        files={"file": ("moving.mp4", moving_vid, "video/mp4")},
    )
    assert assess_resp.status_code == 202
    assessment_id = assess_resp.json()["id"]

    complete_data = await _poll_assessment_until_complete(client, assessment_id)
    assert complete_data["status"] == "COMPLETE"
    assert complete_data["aggregate_decision"] == "UNSTABLE"
    assert complete_data["operational_gate"] == "BLOCKED"

    # Invalidation without explicit confirmation fails -> 400
    fail_ack = await client.post(
        f"/api/v1/stability/assessments/{assessment_id}/acknowledge",
        json={
            "local_operator_label": "Charlie-Lead",
            "trigger_calibration_invalidation": True,
            "confirm_invalidation": False,
            "invalidation_reason": "Drift detected",
        },
    )
    assert fail_ack.status_code == 400

    # Invalidation without nonblank reason fails -> 400
    fail_ack2 = await client.post(
        f"/api/v1/stability/assessments/{assessment_id}/acknowledge",
        json={
            "local_operator_label": "Charlie-Lead",
            "trigger_calibration_invalidation": True,
            "confirm_invalidation": True,
            "invalidation_reason": "   ",
        },
    )
    assert fail_ack2.status_code == 400

    # Valid operator acknowledgement + invalidation
    ack_resp = await client.post(
        f"/api/v1/stability/assessments/{assessment_id}/acknowledge",
        json={
            "local_operator_label": "Charlie-Lead",
            "note": "Camera confirmed drifted by operator",
            "trigger_calibration_invalidation": True,
            "confirm_invalidation": True,
            "invalidation_reason": "Excessive camera pan detected in automated stability guard",
        },
    )
    assert ack_resp.status_code == 201
    audit_data = ack_resp.json()
    assert audit_data["event_type"] == "CALIBRATION_INVALIDATED"
    assert audit_data["operator_identity"] == "Charlie-Lead"
    assert audit_data["resulting_calibration_status"] == "INVALIDATED"

    # Query audit events list endpoint
    audits_list = await client.get(f"/api/v1/cameras/{camera_id}/stability/audit-events")
    assert audits_list.status_code == 200
    assert len(audits_list.json()) == 1
    assert audits_list.json()[0]["event_type"] == "CALIBRATION_INVALIDATED"

    # Verify camera calibration was updated to INVALIDATED
    cam_check = await client.get(f"/api/v1/cameras/{camera_id}")
    assert cam_check.json()["calibration_status"] == "INVALIDATED"

    # Verify layout was updated to INVALIDATED
    layout_check = await client.get(f"/api/v1/layouts/{layout_id}")
    assert layout_check.json()["status"] == "INVALIDATED"


@pytest.mark.anyio
async def test_cancel_in_progress_assessment(client):
    """Cancelling an active assessment transitions status to CANCELLED and blocks gate."""
    s_resp = await client.post("/api/v1/sites", json={"name": "Cancel Site"})
    site_id = s_resp.json()["id"]
    c_resp = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "Cancel Cam"})
    camera_id = c_resp.json()["id"]

    img_bytes = _create_rich_test_image_bytes(640, 480)
    await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("ref.jpg", img_bytes, "image/jpeg")},
    )

    stat_vid = _create_synthetic_video_bytes(stationary=True, frames=60)
    assess_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/stability/assess",
        files={"file": ("stream.mp4", stat_vid, "video/mp4")},
    )
    assert assess_resp.status_code == 202
    assessment_id = assess_resp.json()["id"]

    # Cancel assessment
    cancel_resp = await client.post(f"/api/v1/stability/assessments/{assessment_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "CANCELLED"

    await asyncio.sleep(0.2)

    detail_resp = await client.get(f"/api/v1/stability/assessments/{assessment_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["status"] == "CANCELLED"
    assert detail_resp.json()["operational_gate"] == "BLOCKED"
