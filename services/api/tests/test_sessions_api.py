"""Integration tests for Drive Sessions and Review APIs."""

import io
import pytest
from services.api.app.models.entities import DriveSession, RoadEvent


@pytest.mark.asyncio
async def test_upload_rejects_invalid_file(client):
    # Upload text data pretending to be video
    fake_file = io.BytesIO(b"This is not a video file.")
    response = await client.post(
        "/api/v1/sessions/upload",
        files={"file": ("test.mp4", fake_file, "video/mp4")},
    )
    assert response.status_code == 400
    assert "signature" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_valid_container_header(client):
    # Valid MP4 ftyp container header
    mp4_bytes = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 100
    file_obj = io.BytesIO(mp4_bytes)

    response = await client.post(
        "/api/v1/sessions/upload",
        files={"file": ("dashcam_sample.mp4", file_obj, "video/mp4")},
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["source_filename"] == "dashcam_sample.mp4"
    assert data["processing_state"] == "queued"


@pytest.mark.asyncio
async def test_list_and_get_session(client, db_session):
    # Insert a dummy session
    session = DriveSession(
        mode="upload",
        source_filename="test_drive.mp4",
        source_hash="a" * 64,
        processing_state="complete",
    )
    db_session.add(session)
    await db_session.commit()
    await db_session.refresh(session)

    list_res = await client.get("/api/v1/sessions")
    assert list_res.status_code == 200
    sessions = list_res.json()
    assert len(sessions) >= 1

    detail_res = await client.get(f"/api/v1/sessions/{session.id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["source_filename"] == "test_drive.mp4"


@pytest.mark.asyncio
async def test_review_road_event_state_machine(client, db_session):
    session = DriveSession(
        mode="upload",
        source_filename="test_drive.mp4",
        source_hash="b" * 64,
        processing_state="complete",
    )
    db_session.add(session)
    await db_session.flush()

    event = RoadEvent(
        session_id=session.id,
        first_seen_seconds=1.0,
        last_seen_seconds=2.0,
        first_frame_index=5,
        last_frame_index=10,
        representative_confidence=0.88,
        representative_bbox={"x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200},
        support_count=3,
        review_status="PENDING_REVIEW",
    )
    db_session.add(event)
    await db_session.commit()
    await db_session.refresh(event)

    # 1. Confirm event
    confirm_res = await client.patch(
        f"/api/v1/road-events/{event.id}/review",
        json={"action": "CONFIRM", "reviewer_note": "Verified by Inspector Alice"},
    )
    assert confirm_res.status_code == 200
    data = confirm_res.json()
    assert data["review_status"] == "CONFIRMED"
    assert data["reviewer_note"] == "Verified by Inspector Alice"
    assert len(data["review_actions"]) == 1
    assert data["review_actions"][0]["action"] == "CONFIRM"
    assert data["review_actions"][0]["previous_status"] == "PENDING_REVIEW"
    assert data["review_actions"][0]["new_status"] == "CONFIRMED"

    # 2. Reject event
    reject_res = await client.patch(
        f"/api/v1/road-events/{event.id}/review",
        json={"action": "REJECT", "reviewer_note": "False positive - manhole cover"},
    )
    assert reject_res.status_code == 200
    assert reject_res.json()["review_status"] == "REJECTED"


@pytest.mark.asyncio
async def test_live_camera_boundary(client):
    response = await client.post("/api/v1/live/offer", json={"sdp": "dummy_sdp"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "not_connected"
    assert data["supported"] is False
