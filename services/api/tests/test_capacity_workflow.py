import io
import pytest
from PIL import Image


def _create_synthetic_image_bytes(width: int = 640, height: int = 480) -> bytes:
    img = Image.new("RGB", (width, height), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


async def _setup_camera_with_verified_layout(client):
    """Helper to setup site, camera, reference image, and verified layout."""
    # 1. Site & Camera
    s_resp = await client.post("/api/v1/sites", json={"name": "Phase 3A Site", "timezone": "Asia/Kolkata"})
    assert s_resp.status_code == 201
    site_id = s_resp.json()["id"]

    c_resp = await client.post(f"/api/v1/sites/{site_id}/cameras", json={"name": "CAM-Phase3A"})
    assert c_resp.status_code == 201
    camera_id = c_resp.json()["id"]

    # 2. Reference Image
    img_bytes = _create_synthetic_image_bytes(800, 600)
    ref_resp = await client.post(
        f"/api/v1/cameras/{camera_id}/reference-image",
        files={"file": ("ref.jpg", img_bytes, "image/jpeg")},
    )
    assert ref_resp.status_code == 200

    # 3. Layout with 2 bays and 1 approach zone
    layout_payload = {
        "parking_spaces": [
            {
                "operator_label": "Bay 1",
                "space_type": "STANDARD",
                "polygon_normalized": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.3, "y": 0.1},
                    {"x": 0.3, "y": 0.4},
                    {"x": 0.1, "y": 0.4},
                ],
            },
            {
                "operator_label": "Bay 2",
                "space_type": "STANDARD",
                "polygon_normalized": [
                    {"x": 0.4, "y": 0.1},
                    {"x": 0.6, "y": 0.1},
                    {"x": 0.6, "y": 0.4},
                    {"x": 0.4, "y": 0.4},
                ],
            },
        ],
        "approach_zones": [
            {
                "parking_space_id": "client_0",
                "polygon_normalized": [
                    {"x": 0.1, "y": 0.4},
                    {"x": 0.3, "y": 0.4},
                    {"x": 0.3, "y": 0.55},
                    {"x": 0.1, "y": 0.55},
                ],
            }
        ],
    }
    layout_resp = await client.post(f"/api/v1/cameras/{camera_id}/layouts", json=layout_payload)
    assert layout_resp.status_code == 201
    layout = layout_resp.json()
    layout_id = layout["id"]

    # 4. Submit & Verify
    await client.post(f"/api/v1/layouts/{layout_id}/submit", json={"local_operator_label": "Operator Alice"})
    verify_resp = await client.post(
        f"/api/v1/layouts/{layout_id}/verify",
        json={"local_operator_label": "Reviewer Bob", "confirmation_acknowledged": True},
    )
    assert verify_resp.status_code == 200

    # Retrieve fresh layout with assigned parking spaces
    get_layout = await client.get(f"/api/v1/layouts/{layout_id}")
    assert get_layout.status_code == 200
    spaces = get_layout.json()["parking_spaces"]

    return camera_id, spaces


@pytest.mark.anyio
async def test_hazard_association_lifecycle_and_audit(client):
    """
    Test Phase 3A operations workflow:
    - Create hazard association
    - Guarded review transition with optimistic concurrency (version checking)
    - Audit event trail emission
    - Lifecycle transitions (ACTIVE -> MITIGATED -> RESOLVED -> ACTIVE)
    """
    camera_id, spaces = await _setup_camera_with_verified_layout(client)
    space_1 = spaces[0]

    # 1. Create a manual hazard association
    create_payload = {
        "parking_space_id": space_1["id"],
        "target_type": "BAY",
        "hazard_label": "Severe Pothole at bay entrance",
        "severity_label": "HIGH",
        "initial_review_state": "UNREVIEWED",
        "initial_lifecycle_state": "ACTIVE",
        "notes": "Observed in synthetic test fixture",
        "created_by": "site_operator_1",
    }
    resp = await client.post(f"/api/v1/cameras/{camera_id}/hazards/associations", json=create_payload)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assoc_id = data["id"]
    assert data["hazard_label"] == create_payload["hazard_label"]
    assert data["review_state"] == "UNREVIEWED"
    assert data["lifecycle_state"] == "ACTIVE"
    assert data["version"] == 1
    assert data["created_by"] == "site_operator_1"

    # 2. List hazard associations
    list_resp = await client.get(f"/api/v1/cameras/{camera_id}/hazards/associations")
    assert list_resp.status_code == 200
    associations = list_resp.json()
    assert len(associations) == 1
    assert associations[0]["id"] == assoc_id

    # 3. Human-Review: Confirm the hazard (review transition)
    review_payload = {
        "review_state": "CONFIRMED",
        "reviewer_identity": "qa_supervisor",
        "explicit_reason": "Ground evidence verified via inspection photo",
        "expected_version": 1,
    }
    rev_resp = await client.post(f"/api/v1/hazards/associations/{assoc_id}/review", json=review_payload)
    assert rev_resp.status_code == 200, rev_resp.text
    rev_data = rev_resp.json()
    assert rev_data["review_state"] == "CONFIRMED"
    assert rev_data["reviewed_by"] == "qa_supervisor"
    assert rev_data["version"] == 2

    # 4. Test Optimistic Concurrency Failure:
    # Stale version check (expected_version=1 while version is now 2) must return 409 Conflict
    stale_payload = {
        "review_state": "REJECTED",
        "reviewer_identity": "competing_operator",
        "explicit_reason": "Conflicting decision",
        "expected_version": 1,
    }
    conflict_resp = await client.post(f"/api/v1/hazards/associations/{assoc_id}/review", json=stale_payload)
    assert conflict_resp.status_code == 409
    assert "Version mismatch" in conflict_resp.json()["detail"]

    # 5. Lifecycle transition: Mark Mitigated
    lifecycle_payload = {
        "lifecycle_state": "MITIGATED",
        "operator_identity": "field_maintenance",
        "explicit_reason": "Cold asphalt patch applied temporarily",
        "expected_version": 2,
    }
    lc_resp = await client.post(f"/api/v1/hazards/associations/{assoc_id}/lifecycle", json=lifecycle_payload)
    assert lc_resp.status_code == 200, lc_resp.text
    lc_data = lc_resp.json()
    assert lc_data["lifecycle_state"] == "MITIGATED"
    assert lc_data["version"] == 3

    # 6. Audit trail check: check that immutable events were logged
    audit_resp = await client.get(f"/api/v1/cameras/{camera_id}/hazards/audit-events")
    assert audit_resp.status_code == 200
    events = audit_resp.json()
    assert len(events) >= 3  # CREATED, REVIEWED, LIFECYCLE_TRANSITION
    event_types = [e["event_type"] for e in events]
    assert "CREATED" in event_types
    assert "REVIEWED" in event_types
    assert "LIFECYCLE_TRANSITION" in event_types


@pytest.mark.anyio
async def test_pavement_inspection_and_capacity_snapshot(client):
    """
    Test Phase 3A pavement inspection registration and safe usable capacity snapshot:
    - Record fresh inspection
    - Calculate safe capacity snapshot
    - Verify rule enforcement and deterministic SHA256 fingerprints
    """
    camera_id, spaces = await _setup_camera_with_verified_layout(client)

    # 1. Record pavement inspection
    insp_payload = {
        "session_id": "SURVEY-2026-09-13-001",
        "inspector_label": "lead_inspector_aniket",
        "notes": "Visual pavement scan confirmed asphalt surface intact",
    }
    insp_resp = await client.post(f"/api/v1/cameras/{camera_id}/inspection/record", json=insp_payload)
    assert insp_resp.status_code == 201, insp_resp.text
    insp_data = insp_resp.json()
    assert insp_data["session_id"] == insp_payload["session_id"]
    assert insp_data["inspector_label"] == insp_payload["inspector_label"]

    # 2. List pavement inspections
    list_insp_resp = await client.get(f"/api/v1/cameras/{camera_id}/inspection/records")
    assert list_insp_resp.status_code == 200
    inspections = list_insp_resp.json()
    assert len(inspections) == 1
    assert inspections[0]["id"] == insp_data["id"]

    # 3. Request Capacity Snapshot
    snap_resp = await client.get(f"/api/v1/cameras/{camera_id}/capacity/snapshot")
    assert snap_resp.status_code == 200, snap_resp.text
    snap = snap_resp.json()

    assert snap["camera_id"] == camera_id
    assert len(snap["policy_config_sha256"]) == 64
    assert len(snap["snapshot_sha256"]) == 64
    assert "decisions" in snap
    for space in spaces:
        assert space["id"] in snap["decisions"]
        decision = snap["decisions"][space["id"]]
        assert decision["bay_id"] == space["id"]
        assert decision["operator_label"] == space["operator_label"]
        # Gate defaults to BLOCKED until stability assessment passes -> INFERENCE_BLOCKED
        assert decision["capacity_state"] == "INFERENCE_BLOCKED"
        assert "GATE_BLOCKED" in decision["reason_codes"]


@pytest.mark.anyio
async def test_unverified_ai_suggestions_never_affect_capacity(client):
    """
    Test Rule 8: Unverified AI suggestions appear in queue but never independently
    alter official capacity (fails-closed or ignored until confirmed).
    """
    camera_id, spaces = await _setup_camera_with_verified_layout(client)
    space_1 = spaces[0]

    # Create unverified AI suggestion
    suggestion_payload = {
        "parking_space_id": space_1["id"],
        "target_type": "BAY",
        "hazard_label": "Pothole detected by mobile inference",
        "severity_label": "MEDIUM",
        "initial_review_state": "UNREVIEWED",
        "initial_lifecycle_state": "ACTIVE",
        "notes": "Synthetic AI candidate awaiting human supervisor verification",
        "created_by": "ai_detector_service",
    }
    resp = await client.post(f"/api/v1/cameras/{camera_id}/hazards/associations", json=suggestion_payload)
    assert resp.status_code == 201
    assoc = resp.json()
    assert assoc["review_state"] == "UNREVIEWED"

    # Evaluate snapshot
    snap_resp = await client.get(f"/api/v1/cameras/{camera_id}/capacity/snapshot")
    assert snap_resp.status_code == 200
    snap = snap_resp.json()
    decision = snap["decisions"][space_1["id"]]

    # Has unverified hazard is True, but has_active_bay_hazard is False!
    assert decision["has_unverified_hazard"] is True
    assert decision["has_active_bay_hazard"] is False
    assert "UNVERIFIED_HAZARD_IGNORED" in decision["reason_codes"]
