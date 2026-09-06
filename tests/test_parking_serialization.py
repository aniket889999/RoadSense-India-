"""Unit tests for layout serialization and canonical fingerprinting."""

import json
from src.parking.contracts import CalibrationStatus, LayoutRevisionStatus, SpaceType
from src.parking.layout_serialization import (
    CanonicalLayout,
    CanonicalParkingSpace,
    compute_canonical_layout_sha256,
    deserialize_canonical_layout,
    serialize_canonical_layout,
)


def test_fingerprint_determinism():
    spaces_1 = [
        {"id": "sp_1", "operator_label": "Bay-01", "space_type": "STANDARD", "polygon_normalized": [(0.1, 0.1), (0.3, 0.1), (0.3, 0.4), (0.1, 0.4)]},
        {"id": "sp_2", "operator_label": "Bay-02", "space_type": "ACCESSIBLE", "polygon_normalized": [(0.4, 0.1), (0.6, 0.1), (0.6, 0.4), (0.4, 0.4)]},
    ]
    # Reverse order in list, polygon points rotated
    spaces_2 = [
        {"id": "sp_2", "operator_label": "Bay-02", "space_type": "ACCESSIBLE", "polygon_normalized": [(0.6, 0.4), (0.4, 0.4), (0.4, 0.1), (0.6, 0.1)]},
        {"id": "sp_1", "operator_label": "Bay-01", "space_type": "STANDARD", "polygon_normalized": [(0.3, 0.4), (0.1, 0.4), (0.1, 0.1), (0.3, 0.1)]},
    ]

    hash_1 = compute_canonical_layout_sha256("1.0.0", "cam_01", "ref_sha_abc", spaces_1)
    hash_2 = compute_canonical_layout_sha256("1.0.0", "cam_01", "ref_sha_abc", spaces_2)

    assert hash_1 == hash_2
    assert len(hash_1) == 64


def test_fingerprint_changes_on_geometry_change():
    spaces_1 = [
        {"id": "sp_1", "operator_label": "Bay-01", "space_type": "STANDARD", "polygon_normalized": [(0.1, 0.1), (0.3, 0.1), (0.3, 0.4), (0.1, 0.4)]},
    ]
    spaces_2 = [
        {"id": "sp_1", "operator_label": "Bay-01", "space_type": "STANDARD", "polygon_normalized": [(0.1, 0.1), (0.35, 0.1), (0.35, 0.4), (0.1, 0.4)]},
    ]

    hash_1 = compute_canonical_layout_sha256("1.0.0", "cam_01", "ref_sha_abc", spaces_1)
    hash_2 = compute_canonical_layout_sha256("1.0.0", "cam_01", "ref_sha_abc", spaces_2)

    assert hash_1 != hash_2


def test_serialize_and_deserialize_roundtrip():
    layout = CanonicalLayout(
        schema_version="1.0.0",
        site_id="site_123",
        camera_id="cam_456",
        layout_revision_id="rev_789",
        revision_number=1,
        reference_image_width=1920,
        reference_image_height=1080,
        reference_image_sha256="ref_sha_123",
        calibration_status=CalibrationStatus.VERIFIED.value,
        layout_status=LayoutRevisionStatus.DRAFT.value,
        created_at="2026-09-06T12:00:00Z",
        parking_spaces=[
            CanonicalParkingSpace(
                id="sp_1",
                operator_label="Bay-01",
                space_type=SpaceType.STANDARD.value,
                polygon_normalized=[{"x": 0.1, "y": 0.1}, {"x": 0.3, "y": 0.1}, {"x": 0.3, "y": 0.4}, {"x": 0.1, "y": 0.4}],
            )
        ],
    )

    serialized = serialize_canonical_layout(layout)
    deserialized = deserialize_canonical_layout(serialized)

    assert deserialized.site_id == "site_123"
    assert deserialized.camera_id == "cam_456"
    assert len(deserialized.parking_spaces) == 1
    assert deserialized.parking_spaces[0].operator_label == "Bay-01"
    assert len(deserialized.canonical_layout_sha256) == 64
