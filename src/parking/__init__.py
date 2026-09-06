"""RoadSense SiteOps — Parking Domain Package."""

from src.parking.contracts import (
    CalibrationStatus,
    DerivedAvailability,
    LayoutRevisionStatus,
    OccupancyState,
    ObstructionState,
    ReviewState,
    SpaceType,
    SurfaceState,
    derive_operator_availability,
)
from src.parking.geometry import (
    NormalizedPoint,
    NormalizedPolygon,
    calculate_iou,
    calculate_overlap_ratio,
    calculate_polygon_area,
    calculate_polygon_intersection_area,
)
from src.parking.layout_serialization import (
    CanonicalLayout,
    CanonicalParkingSpace,
    compute_canonical_layout_sha256,
    deserialize_canonical_layout,
    serialize_canonical_layout,
)
from src.parking.layout_validation import (
    LayoutValidationError,
    validate_parking_layout,
)

__all__ = [
    "CalibrationStatus",
    "LayoutRevisionStatus",
    "OccupancyState",
    "SurfaceState",
    "ObstructionState",
    "ReviewState",
    "DerivedAvailability",
    "SpaceType",
    "derive_operator_availability",
    "NormalizedPoint",
    "NormalizedPolygon",
    "calculate_polygon_area",
    "calculate_polygon_intersection_area",
    "calculate_overlap_ratio",
    "calculate_iou",
    "LayoutValidationError",
    "validate_parking_layout",
    "CanonicalParkingSpace",
    "CanonicalLayout",
    "serialize_canonical_layout",
    "deserialize_canonical_layout",
    "compute_canonical_layout_sha256",
]
