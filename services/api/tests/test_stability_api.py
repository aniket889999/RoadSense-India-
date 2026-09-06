"""Integration tests for camera stability API endpoints and fail-closed operational gate."""

import io
import os
from pathlib import Path
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import cv2
import numpy as np
from PIL import Image

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

    tmp_path = Path(tempfile.mktemp(suffix=".mp4"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(tmp_path), fourcc, 30.0, (640, 480))

    try:
        for i in range(frames):
            if stationary:
                out.write(img)
            else:
                M = np.float32([[1, 0, float(i * 1.0)], [0, 1, 0]])
                shifted = cv2.warpAffine(img, M, (640, 480))
                out.write(shifted)
        out.release()

        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
        return video_bytes
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


import tempfile


@pytest.mark.anyio
async def test_gate_fail_closed_before_assessment(client):
    """A camera with verified layout must still have gate BLOCKED until stable assessment runs."""
    # 1. Create site and camera
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
async def test_stability_assessment_upload_and_operational_gate_lifecycle(client):
    """Full lifecycle: upload stationary video -> STABLE -> ALLOWED -> replace image -> BLOCKED."""
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

    # Submit and verify layout
    await client.post(f"/api/v1/layouts/{layout_id}/submit", json={"local_operator_label": "Alice"})
    await client.post(f"/api/v1/layouts/{layout_id}/verify", json={"local_operator_label": "Bob", "confirmation_acknowledged": True})

    # 2. Upload stationary video assessment
    stat_vid = _create_synthetic_video_bytes(stationary=True, frames=30)
    assess_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/stability/assess",
        files={"file": ("stream.mp4", stat_vid, "video/mp4")},
    )
    assert assess_resp.status_code == 201
    ass_data = assess_resp.json()
    assert ass_data["aggregate_decision"] == "STABLE"
    assert ass_data["operational_gate"] == "ALLOWED"
    assessment_id = ass_data["id"]

    # 3. Query operational gate endpoint
    gate_resp = await client.get(f"/api/v1/cameras/{camera_id}/stability/gate")
    assert gate_resp.status_code == 200
    gate_data = gate_resp.json()
    assert gate_data["operational_gate"] == "ALLOWED"
    assert gate_data["is_fresh"] is True

    # 4. Query assessment details
    detail_resp = await client.get(f"/api/v1/stability/assessments/{assessment_id}")
    assert detail_resp.status_code == 200
    assert len(detail_resp.json()["sample_measurements"]) > 0

    # 5. List assessments for camera
    list_resp = await client.get(f"/api/v1/cameras/{camera_id}/stability/assessments")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1

    # 6. Replacing reference image must invalidate earlier assessment and flip gate to BLOCKED
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
async def test_unstable_assessment_and_operator_acknowledgement(client):
    """Unstable video yields BLOCKED; operator can acknowledge and invalidate calibration."""
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

    # Upload moving/unstable video
    moving_vid = _create_synthetic_video_bytes(stationary=False, frames=30)
    assess_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/stability/assess",
        files={"file": ("moving.mp4", moving_vid, "video/mp4")},
    )
    assert assess_resp.status_code == 201
    ass_data = assess_resp.json()
    assert ass_data["aggregate_decision"] == "UNSTABLE"
    assert ass_data["operational_gate"] == "BLOCKED"
    assessment_id = ass_data["id"]

    # Gate is BLOCKED
    gate_resp = await client.get(f"/api/v1/cameras/{camera_id}/stability/gate")
    assert gate_resp.json()["operational_gate"] == "BLOCKED"

    # Operator acknowledges and triggers audited calibration invalidation
    ack_resp = await client.post(
        f"/api/v1/stability/assessments/{assessment_id}/acknowledge",
        json={
            "local_operator_label": "Charlie-Lead",
            "note": "Camera confirmed drifted by operator",
            "trigger_calibration_invalidation": True,
            "invalidation_reason": "Excessive camera pan detected in automated stability guard",
        },
    )
    assert ack_resp.status_code == 200
    assert ack_resp.json()["operator_acknowledged_at"] is not None
    assert ack_resp.json()["operator_label"] == "Charlie-Lead"

    # Verify camera calibration was updated to INVALIDATED
    cam_check = await client.get(f"/api/v1/cameras/{camera_id}")
    assert cam_check.json()["calibration_status"] == "INVALIDATED"

    # Verify layout was updated to INVALIDATED
    layout_check = await client.get(f"/api/v1/layouts/{layout_id}")
    assert layout_check.json()["status"] == "INVALIDATED"
