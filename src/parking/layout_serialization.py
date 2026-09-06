"""Canonical layout schema, serialization, and deterministic SHA-256 fingerprinting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from src.parking.contracts import CalibrationStatus, LayoutRevisionStatus, SpaceType
from src.parking.geometry import NormalizedPolygon, canonicalize_polygon


@dataclass(frozen=True)
class CanonicalParkingSpace:
    id: str
    operator_label: str
    space_type: str
    polygon_normalized: list[dict[str, float]]
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "operator_label": self.operator_label,
            "space_type": self.space_type,
            "polygon_normalized": self.polygon_normalized,
            "active": self.active,
        }


@dataclass(frozen=True)
class CanonicalApproachZone:
    id: str
    parking_space_id: str
    polygon_normalized: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parking_space_id": self.parking_space_id,
            "polygon_normalized": self.polygon_normalized,
        }


@dataclass
class CanonicalLayout:
    schema_version: str = "1.0.0"
    site_id: str = ""
    camera_id: str = ""
    layout_revision_id: str = ""
    revision_number: int = 1
    reference_image_width: int = 0
    reference_image_height: int = 0
    reference_image_sha256: str = ""
    calibration_status: str = CalibrationStatus.NOT_CONFIGURED.value
    layout_status: str = LayoutRevisionStatus.DRAFT.value
    created_at: str = ""
    updated_at: Optional[str] = None
    submitted_at: Optional[str] = None
    verified_at: Optional[str] = None
    invalidated_at: Optional[str] = None
    invalidation_reason: Optional[str] = None
    parking_spaces: list[CanonicalParkingSpace] = field(default_factory=list)
    approach_zones: list[CanonicalApproachZone] = field(default_factory=list)
    canonical_layout_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "site_id": self.site_id,
            "camera_id": self.camera_id,
            "layout_revision_id": self.layout_revision_id,
            "revision_number": self.revision_number,
            "reference_image_width": self.reference_image_width,
            "reference_image_height": self.reference_image_height,
            "reference_image_sha256": self.reference_image_sha256,
            "calibration_status": self.calibration_status,
            "layout_status": self.layout_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "submitted_at": self.submitted_at,
            "verified_at": self.verified_at,
            "invalidated_at": self.invalidated_at,
            "invalidation_reason": self.invalidation_reason,
            "parking_spaces": [sp.to_dict() for sp in self.parking_spaces],
            "approach_zones": [az.to_dict() for az in self.approach_zones],
            "canonical_layout_sha256": self.canonical_layout_sha256,
        }


def compute_canonical_layout_sha256(
    schema_version: str,
    camera_id: str,
    reference_image_sha256: str,
    parking_spaces: list[dict[str, Any] | CanonicalParkingSpace],
    approach_zones: Optional[list[dict[str, Any] | CanonicalApproachZone]] = None,
) -> str:
    """
    Compute a deterministic SHA-256 fingerprint for a parking layout geometry.

    The hash input contains:
    - schema_version
    - camera_id
    - reference_image_sha256
    - canonicalized, sorted parking spaces (id, operator_label, space_type, canonical polygon)
    - canonicalized, sorted approach zones (id, parking_space_id, canonical polygon)

    Timestamps and volatile IDs are intentionally excluded from the hash input.
    """
    canonical_spaces: list[dict[str, Any]] = []
    for sp in parking_spaces:
        if isinstance(sp, CanonicalParkingSpace):
            sp_dict = sp.to_dict()
        else:
            sp_dict = dict(sp)

        # Canonicalize polygon coordinates to 6 decimal places
        pts = sp_dict.get("polygon_normalized") or []
        try:
            poly = NormalizedPolygon.from_list(pts)
            c_poly = canonicalize_polygon(poly)
            c_pts = c_poly.to_list()
        except Exception:
            c_pts = [{"x": round(float(p.get("x", 0)), 6), "y": round(float(p.get("y", 0)), 6)} for p in pts]

        canonical_spaces.append({
            "id": str(sp_dict.get("id", "")),
            "operator_label": str(sp_dict.get("operator_label", "")).strip(),
            "space_type": str(sp_dict.get("space_type", SpaceType.STANDARD.value)),
            "polygon_normalized": c_pts,
            "active": bool(sp_dict.get("active", True)),
        })

    # Sort parking spaces by operator_label, then id
    canonical_spaces.sort(key=lambda s: (s["operator_label"], s["id"]))

    canonical_approaches: list[dict[str, Any]] = []
    if approach_zones:
        for az in approach_zones:
            if isinstance(az, CanonicalApproachZone):
                az_dict = az.to_dict()
            else:
                az_dict = dict(az)

            pts = az_dict.get("polygon_normalized") or []
            try:
                poly = NormalizedPolygon.from_list(pts)
                c_poly = canonicalize_polygon(poly)
                c_pts = c_poly.to_list()
            except Exception:
                c_pts = [{"x": round(float(p.get("x", 0)), 6), "y": round(float(p.get("y", 0)), 6)} for p in pts]

            canonical_approaches.append({
                "id": str(az_dict.get("id", "")),
                "parking_space_id": str(az_dict.get("parking_space_id", "")),
                "polygon_normalized": c_pts,
            })

        # Sort approach zones by parking_space_id, then id
        canonical_approaches.sort(key=lambda a: (a["parking_space_id"], a["id"]))

    fingerprint_payload = {
        "schema_version": schema_version,
        "camera_id": camera_id,
        "reference_image_sha256": reference_image_sha256,
        "parking_spaces": canonical_spaces,
        "approach_zones": canonical_approaches,
    }

    # Strict deterministic JSON encoding (sorted keys, no extra whitespace)
    canonical_json = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def serialize_canonical_layout(layout: CanonicalLayout) -> str:
    """Serialize CanonicalLayout to formatted JSON with embedded fingerprint."""
    if not layout.canonical_layout_sha256:
        layout.canonical_layout_sha256 = compute_canonical_layout_sha256(
            schema_version=layout.schema_version,
            camera_id=layout.camera_id,
            reference_image_sha256=layout.reference_image_sha256,
            parking_spaces=layout.parking_spaces,
            approach_zones=layout.approach_zones,
        )
    return json.dumps(layout.to_dict(), indent=2, sort_keys=True)


def deserialize_canonical_layout(json_str: str) -> CanonicalLayout:
    """Deserialize JSON string into CanonicalLayout dataclass."""
    data = json.loads(json_str)

    spaces = [
        CanonicalParkingSpace(
            id=sp["id"],
            operator_label=sp["operator_label"],
            space_type=sp.get("space_type", SpaceType.STANDARD.value),
            polygon_normalized=sp["polygon_normalized"],
            active=sp.get("active", True),
        )
        for sp in data.get("parking_spaces", [])
    ]

    approaches = [
        CanonicalApproachZone(
            id=az["id"],
            parking_space_id=az["parking_space_id"],
            polygon_normalized=az["polygon_normalized"],
        )
        for az in data.get("approach_zones", [])
    ]

    return CanonicalLayout(
        schema_version=data.get("schema_version", "1.0.0"),
        site_id=data.get("site_id", ""),
        camera_id=data.get("camera_id", ""),
        layout_revision_id=data.get("layout_revision_id", ""),
        revision_number=data.get("revision_number", 1),
        reference_image_width=data.get("reference_image_width", 0),
        reference_image_height=data.get("reference_image_height", 0),
        reference_image_sha256=data.get("reference_image_sha256", ""),
        calibration_status=data.get("calibration_status", CalibrationStatus.NOT_CONFIGURED.value),
        layout_status=data.get("layout_status", LayoutRevisionStatus.DRAFT.value),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at"),
        submitted_at=data.get("submitted_at"),
        verified_at=data.get("verified_at"),
        invalidated_at=data.get("invalidated_at"),
        invalidation_reason=data.get("invalidation_reason"),
        parking_spaces=spaces,
        approach_zones=approaches,
        canonical_layout_sha256=data.get("canonical_layout_sha256", ""),
    )
