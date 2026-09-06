import io
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from PIL import Image

from services.api.app.db.session import get_db
from services.api.app.routers.parking import router as parking_router


@pytest.fixture
def parking_app():
    test_app = FastAPI(title="RoadSense Parking Test API")
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


def _create_synthetic_image_bytes(width: int = 640, height: int = 480, format: str = "JPEG") -> bytes:
    img = Image.new("RGB", (width, height), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


@pytest.mark.anyio
async def test_site_and_camera_crud(client):

    # 1. Create site
    site_payload = {
        "name": "Central Hospital Parking Deck",
        "description": "Multi-level visitor and ambulance parking",
        "timezone": "Asia/Kolkata",
    }
    resp = await client.post("/api/v1/sites", json=site_payload)
    assert resp.status_code == 201
    site_data = resp.json()
    assert site_data["name"] == site_payload["name"]
    site_id = site_data["id"]

    # 2. List sites
    list_resp = await client.get("/api/v1/sites")
    assert list_resp.status_code == 200
    assert any(s["id"] == site_id for s in list_resp.json())

    # 3. Create camera under site
    cam_payload = {
        "name": "CCTV-Deck-North-01",
        "description": "Pole mounted 1080p camera covering bays 1-10",
        "camera_position_description": "North mast at 5.5m elevation",
    }
    cam_resp = await client.post(f"/api/v1/sites/{site_id}/cameras", json=cam_payload)
    assert cam_resp.status_code == 201
    cam_data = cam_resp.json()
    assert cam_data["name"] == cam_payload["name"]
    assert cam_data["calibration_status"] == "NOT_CONFIGURED"
    camera_id = cam_data["id"]

    # 4. Get camera detail
    cam_get = await client.get(f"/api/v1/cameras/{camera_id}")
    assert cam_get.status_code == 200
    assert cam_get.json()["id"] == camera_id


@pytest.mark.anyio
async def test_reference_image_upload_and_security(client):
    # Create site & camera
    s_resp = await client.post("/api/v1/sites", json={"name": "Test Site"})
    site_id = s_resp.json()["id"]
    c_resp = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "Test Cam"})
    camera_id = c_resp.json()["id"]

    # 1. Upload valid reference JPEG
    img_bytes = _create_synthetic_image_bytes(800, 600, "JPEG")
    upload_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("reference.jpg", img_bytes, "image/jpeg")},
    )
    assert upload_resp.status_code == 200
    cam_data = upload_resp.json()
    assert cam_data["reference_width"] == 800
    assert cam_data["reference_height"] == 600
    assert len(cam_data["reference_image_sha256"]) == 64
    assert not cam_data["reference_image_path"].startswith("/")  # No absolute path exposed

    # 2. Get reference image
    get_img = await client.get(f"/api/v1/cameras/{camera_id}/reference-image")
    assert get_img.status_code == 200
    assert get_img.headers["content-type"] in ("image/jpeg", "image/png")
    assert len(get_img.content) > 0

    # 3. Reject malformed image bytes
    bad_upload = await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("corrupt.jpg", b"not-a-real-image-payload", "image/jpeg")},
    )
    assert bad_upload.status_code == 400


@pytest.mark.anyio
async def test_parking_layout_lifecycle_and_validation(client):

    # Setup camera with reference image
    s_resp = await client.post("/api/v1/sites", json={"name": "Lifecycle Test Site"})
    site_id = s_resp.json()["id"]
    c_resp = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "Lifecycle Cam"})
    camera_id = c_resp.json()["id"]
    img_bytes = _create_synthetic_image_bytes(1280, 720, "JPEG")
    await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("ref.jpg", img_bytes, "image/jpeg")},
    )

    # 1. Create draft layout with 2 valid bays and 1 approach zone
    draft_payload = {
        "parking_spaces": [
            {
                "operator_label": "Bay-01",
                "space_type": "STANDARD",
                "polygon_normalized": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.25, "y": 0.1},
                    {"x": 0.25, "y": 0.4},
                    {"x": 0.1, "y": 0.4},
                ],
            },
            {
                "operator_label": "Bay-02",
                "space_type": "ACCESSIBLE",
                "polygon_normalized": [
                    {"x": 0.3, "y": 0.1},
                    {"x": 0.45, "y": 0.1},
                    {"x": 0.45, "y": 0.4},
                    {"x": 0.3, "y": 0.4},
                ],
            },
        ],
        "approach_zones": [
            {
                "parking_space_id": "client_0",
                "polygon_normalized": [
                    {"x": 0.1, "y": 0.4},
                    {"x": 0.25, "y": 0.4},
                    {"x": 0.25, "y": 0.55},
                    {"x": 0.1, "y": 0.55},
                ],
            }
        ],
    }

    create_resp = await client.post(f"/api/v1/cameras/{camera_id}/layouts", json=draft_payload)
    assert create_resp.status_code == 201
    layout = create_resp.json()
    layout_id = layout["id"]
    assert layout["status"] == "DRAFT"
    assert len(layout["parking_spaces"]) == 2
    assert len(layout["approach_zones"]) == 1
    assert len(layout["audit_events"]) == 1

    # 2. Validate layout
    val_resp = await client.post(f"/api/v1/layouts/{layout_id}/validate")
    assert val_resp.status_code == 200
    val_data = val_resp.json()
    assert val_data["is_valid"] is True
    assert len(val_data["errors"]) == 0

    # 3. Submit layout for review
    sub_resp = await client.post(f"/api/v1/layouts/{layout_id}/submit", json={"local_operator_label": "Alice"})
    assert sub_resp.status_code == 200
    assert sub_resp.json()["status"] == "PENDING_REVIEW"

    # 4. Verification requires explicit confirmation acknowledgement
    bad_verify = await client.post(
        f"/api/v1/layouts/{layout_id}/verify",
        json={"local_operator_label": "Bob", "confirmation_acknowledged": False},
    )
    assert bad_verify.status_code == 400

    # 5. Verify layout successfully
    verify_resp = await client.post(
        f"/api/v1/layouts/{layout_id}/verify",
        json={"local_operator_label": "Bob", "confirmation_acknowledged": True, "note": "Verified on ground truth"},
    )
    assert verify_resp.status_code == 200
    verified_layout = verify_resp.json()
    assert verified_layout["status"] == "VERIFIED"
    assert len(verified_layout["canonical_sha256"]) == 64

    # Verify camera calibration updated to VERIFIED
    cam_after = await client.get(f"/api/v1/cameras/{camera_id}")
    assert cam_after.json()["calibration_status"] == "VERIFIED"

    # 6. Mutating a VERIFIED layout branches to a new DRAFT revision
    update_payload = {
        "parking_spaces": [
            {
                "operator_label": "Bay-01-Edited",
                "space_type": "STANDARD",
                "polygon_normalized": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.28, "y": 0.1},
                    {"x": 0.28, "y": 0.4},
                    {"x": 0.1, "y": 0.4},
                ],
            }
        ]
    }
    branch_resp = await client.put(f"/api/v1/layouts/{layout_id}", json=update_payload)
    assert branch_resp.status_code == 200
    branched_layout = branch_resp.json()
    assert branched_layout["id"] != layout_id
    assert branched_layout["revision_number"] == 2
    assert branched_layout["status"] == "DRAFT"

    # 7. Invalidate camera calibration
    inval_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/invalidate-calibration",
        json={"local_operator_label": "Alice", "invalidation_reason": "Camera was knocked by high wind"},
    )
    assert inval_resp.status_code == 200
    assert inval_resp.json()["calibration_status"] == "INVALIDATED"

    # Check that original verified layout is now INVALIDATED
    old_layout_check = await client.get(f"/api/v1/layouts/{layout_id}")
    assert old_layout_check.json()["status"] == "INVALIDATED"


@pytest.mark.anyio
async def test_initial_reference_image_upload_sets_pending_review(client):
    """STEP 3.1: A first successful reference-image upload must set camera calibration to PENDING_REVIEW, never VERIFIED."""
    s_resp = await client.post("/api/v1/sites", json={"name": "Initial State Site"})
    site_id = s_resp.json()["id"]
    c_resp = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "Pending Review Cam"})
    camera_id = c_resp.json()["id"]
    assert c_resp.json()["calibration_status"] == "NOT_CONFIGURED"

    img_bytes = _create_synthetic_image_bytes(800, 600, "JPEG")
    up_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("ref.jpg", img_bytes, "image/jpeg")},
    )
    assert up_resp.status_code == 200
    cam_data = up_resp.json()
    assert cam_data["calibration_status"] == "PENDING_REVIEW"
    assert cam_data["calibration_status"] != "VERIFIED"


@pytest.mark.anyio
async def test_layout_snapshot_and_stale_rejection(client):
    """STEP 3.2 & 3.4: Layout revisions snapshot image metadata; validation/submission fail on snapshot mismatch."""
    s_resp = await client.post("/api/v1/sites", json={"name": "Snapshot Site"})
    site_id = s_resp.json()["id"]
    c_resp = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "Snapshot Cam"})
    camera_id = c_resp.json()["id"]

    # Try creating layout before camera has reference image -> must fail
    fail_create = await client.post(
        f"/api/v1/cameras/{camera_id}/layouts",
        json={"parking_spaces": [{"operator_label": "B1", "space_type": "STANDARD", "polygon_normalized": [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.1}, {"x": 0.2, "y": 0.2}, {"x": 0.1, "y": 0.2}]}]},
    )
    assert fail_create.status_code == 400

    # Upload reference image 1
    img1 = _create_synthetic_image_bytes(640, 480, "JPEG")
    up1 = await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("ref1.jpg", img1, "image/jpeg")},
    )
    assert up1.status_code == 200
    sha1 = up1.json()["reference_image_sha256"]

    # Create layout revision 1
    create_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/layouts",
        json={"parking_spaces": [{"operator_label": "B1", "space_type": "STANDARD", "polygon_normalized": [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.1}, {"x": 0.2, "y": 0.2}, {"x": 0.1, "y": 0.2}]}]},
    )
    assert create_resp.status_code == 201
    layout1 = create_resp.json()
    assert layout1["reference_image_sha256"] == sha1
    assert layout1["reference_width"] == 640
    assert layout1["reference_height"] == 480
    layout1_id = layout1["id"]

    # Replace reference image with image 2
    img2 = _create_synthetic_image_bytes(800, 600, "PNG")
    # Unconfirmed replacement must be rejected (400)
    unconf_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("ref2.png", img2, "image/png")},
    )
    assert unconf_resp.status_code == 400
    assert "CONFIRMATION_REQUIRED" in unconf_resp.json()["detail"]

    # Replacement without operator or reason must be rejected
    no_op_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("ref2.png", img2, "image/png")},
        data={"confirm_replacement": "true", "operator_label": "   ", "reason": "Camera repositioned"},
    )
    assert no_op_resp.status_code == 400

    # Valid confirmed replacement
    conf_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("ref2.png", img2, "image/png")},
        data={"confirm_replacement": "true", "operator_label": "alice-sec", "reason": "Camera lens swapped and repositioned"},
    )
    assert conf_resp.status_code == 200
    sha2 = conf_resp.json()["reference_image_sha256"]
    assert sha2 != sha1

    # Check that layout 1 has become INVALIDATED
    l1_check = await client.get(f"/api/v1/layouts/{layout1_id}")
    assert l1_check.json()["status"] == "INVALIDATED"

    # Validation of layout 1 must fail due to snapshot mismatch or invalid status
    val_fail = await client.post(f"/api/v1/layouts/{layout1_id}/validate")
    assert val_fail.status_code == 400

    # Submission of layout 1 must fail
    sub_fail = await client.post(f"/api/v1/layouts/{layout1_id}/submit", json={"local_operator_label": "alice-sec"})
    assert sub_fail.status_code == 400

    # Verification of layout 1 must fail
    ver_fail = await client.post(f"/api/v1/layouts/{layout1_id}/verify", json={"local_operator_label": "bob-lead", "confirmation_acknowledged": True})
    assert ver_fail.status_code == 400


@pytest.mark.anyio
async def test_verification_requires_pending_review_and_nonblank_operator(client):
    """STEP 3.8 & 3.9: Verification must accept PENDING_REVIEW only. Nonblank operator required."""
    s_resp = await client.post("/api/v1/sites", json={"name": "Strict Verification Site"})
    site_id = s_resp.json()["id"]
    c_resp = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "Strict Cam"})
    camera_id = c_resp.json()["id"]

    img_bytes = _create_synthetic_image_bytes(640, 480, "JPEG")
    await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("ref.jpg", img_bytes, "image/jpeg")},
    )

    create_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/layouts",
        json={"parking_spaces": [{"operator_label": "Bay-01", "space_type": "STANDARD", "polygon_normalized": [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.1}, {"x": 0.2, "y": 0.2}, {"x": 0.1, "y": 0.2}]}]},
    )
    layout_id = create_resp.json()["id"]

    # 1. Attempt DRAFT -> VERIFIED directly must fail
    bad_direct_verify = await client.post(
        f"/api/v1/layouts/{layout_id}/verify",
        json={"local_operator_label": "alice", "confirmation_acknowledged": True},
    )
    assert bad_direct_verify.status_code == 400
    assert "PENDING_REVIEW" in bad_direct_verify.json()["detail"]

    # 2. Blank operator label in submit must be rejected
    blank_submit = await client.post(
        f"/api/v1/layouts/{layout_id}/submit",
        json={"local_operator_label": "   "},
    )
    assert blank_submit.status_code == 422

    # 3. Valid submit
    valid_submit = await client.post(
        f"/api/v1/layouts/{layout_id}/submit",
        json={"local_operator_label": "alice-op"},
    )
    assert valid_submit.status_code == 200
    assert valid_submit.json()["status"] == "PENDING_REVIEW"

    # 4. Blank operator label in verify must be rejected
    blank_verify = await client.post(
        f"/api/v1/layouts/{layout_id}/verify",
        json={"local_operator_label": "   ", "confirmation_acknowledged": True},
    )
    assert blank_verify.status_code == 422

    # 5. Successful verify from PENDING_REVIEW
    ok_verify = await client.post(
        f"/api/v1/layouts/{layout_id}/verify",
        json={"local_operator_label": "bob-lead", "confirmation_acknowledged": True},
    )
    assert ok_verify.status_code == 200
    assert ok_verify.json()["status"] == "VERIFIED"


@pytest.mark.anyio
async def test_input_validation_and_whitespace_trimming(client):
    """STEP 4.5 & 4.6: Constrained enum validation and whitespace-only rejection."""
    # Blank site name
    resp = await client.post("/api/v1/sites", json={"name": "   "})
    assert resp.status_code == 422

    # Valid site
    s_resp = await client.post("/api/v1/sites", json={"name": "  Sanitized Site  "})
    assert s_resp.status_code == 201
    assert s_resp.json()["name"] == "Sanitized Site"
    site_id = s_resp.json()["id"]

    # Blank camera name
    c_fail = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "   "})
    assert c_fail.status_code == 422

    # Valid camera
    c_ok = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "  North Cam  "})
    assert c_ok.status_code == 201
    assert c_ok.json()["name"] == "North Cam"
    camera_id = c_ok.json()["id"]

    img_bytes = _create_synthetic_image_bytes(640, 480, "JPEG")
    await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("ref.jpg", img_bytes, "image/jpeg")},
    )

    # Invalid space type enum
    bad_space = await client.post(
        f"/api/v1/cameras/{camera_id}/layouts",
        json={"parking_spaces": [{"operator_label": "B1", "space_type": "NOT_A_VALID_TYPE", "polygon_normalized": [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.1}, {"x": 0.2, "y": 0.2}, {"x": 0.1, "y": 0.2}]}]},
    )
    assert bad_space.status_code == 422

    # Whitespace-only operator label for space
    blank_label = await client.post(
        f"/api/v1/cameras/{camera_id}/layouts",
        json={"parking_spaces": [{"operator_label": "   ", "space_type": "STANDARD", "polygon_normalized": [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.1}, {"x": 0.2, "y": 0.2}, {"x": 0.1, "y": 0.2}]}]},
    )
    assert blank_label.status_code == 422
