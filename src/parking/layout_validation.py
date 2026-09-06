"""Structured layout and polygon validation rules for RoadSense SiteOps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from src.parking.geometry import (
    NormalizedPoint,
    NormalizedPolygon,
    calculate_iou,
    calculate_overlap_ratio,
    calculate_polygon_area,
    has_self_intersections,
    is_polygon_convex,
)


@dataclass(frozen=True)
class LayoutValidationError:
    """Structured validation error for a parking layout or polygon element."""
    rule_id: str
    message: str
    space_id: Optional[str] = None
    space_label: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "message": self.message,
            "space_id": self.space_id,
            "space_label": self.space_label,
        }


def validate_single_polygon(
    polygon: NormalizedPolygon,
    element_name: str = "Polygon",
    space_id: Optional[str] = None,
    space_label: Optional[str] = None,
    min_area: float = 1e-6,
) -> list[LayoutValidationError]:
    """Validate a single polygon for coordinate bounds, vertex count, convexity, and self-intersection."""
    errors: list[LayoutValidationError] = []
    pts = polygon.points

    if len(pts) < 3:
        errors.append(
            LayoutValidationError(
                rule_id="INSUFFICIENT_VERTICES",
                message=f"{element_name} must contain at least 3 vertices, got {len(pts)}.",
                space_id=space_id,
                space_label=space_label,
            )
        )
        return errors

    # Check finite and bounds [0.0, 1.0]
    for idx, p in enumerate(pts):
        if not (math.isfinite(p.x) and math.isfinite(p.y)):
            errors.append(
                LayoutValidationError(
                    rule_id="NON_FINITE_COORDINATES",
                    message=f"{element_name} vertex {idx} contains non-finite coordinates: ({p.x}, {p.y}).",
                    space_id=space_id,
                    space_label=space_label,
                )
            )
        elif not (0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0):
            errors.append(
                LayoutValidationError(
                    rule_id="OUT_OF_BOUNDS_COORDINATES",
                    message=f"{element_name} vertex {idx} ({p.x:.4f}, {p.y:.4f}) is out of normalized bounds [0.0, 1.0].",
                    space_id=space_id,
                    space_label=space_label,
                )
            )

    # Check adjacent duplicates
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        if abs(pts[i].x - pts[j].x) < 1e-7 and abs(pts[i].y - pts[j].y) < 1e-7:
            errors.append(
                LayoutValidationError(
                    rule_id="DUPLICATE_ADJACENT_VERTICES",
                    message=f"{element_name} contains duplicate adjacent vertices at indices {i} and {j}.",
                    space_id=space_id,
                    space_label=space_label,
                )
            )

    # Check area
    area = calculate_polygon_area(polygon)
    if area < min_area:
        errors.append(
            LayoutValidationError(
                rule_id="DEGENERATE_ZERO_AREA",
                message=f"{element_name} has degenerate or zero area ({area:.8f} < {min_area}).",
                space_id=space_id,
                space_label=space_label,
            )
        )

    # Check self-intersection
    if has_self_intersections(polygon):
        errors.append(
            LayoutValidationError(
                rule_id="SELF_INTERSECTING_POLYGON",
                message=f"{element_name} edges intersect with each other.",
                space_id=space_id,
                space_label=space_label,
            )
        )

    # Check convexity
    if not is_polygon_convex(polygon):
        errors.append(
            LayoutValidationError(
                rule_id="NON_CONVEX_POLYGON",
                message=f"{element_name} is non-convex. Phase 1 requires simple convex polygons.",
                space_id=space_id,
                space_label=space_label,
            )
        )

    return errors


def validate_parking_layout(
    parking_spaces: Sequence[dict[str, Any]],
    approach_zones: Optional[Sequence[dict[str, Any]]] = None,
    reference_width: Optional[int] = None,
    reference_height: Optional[int] = None,
    max_inter_space_overlap_ratio: float = 0.05,
) -> list[LayoutValidationError]:
    """
    Perform comprehensive validation of a complete parking layout configuration.

    Checks:
    1. Reference frame dimensions (> 0).
    2. Unique space IDs and unique operator labels.
    3. Valid simple convex normalized polygons for each space.
    4. Valid simple convex normalized polygons for each approach zone.
    5. Approach zones link to exactly one existing space ID.
    6. Excessive overlap between distinct parking spaces is rejected.
    """
    errors: list[LayoutValidationError] = []

    # Check reference dimensions if provided
    if reference_width is not None and reference_width <= 0:
        errors.append(
            LayoutValidationError(
                rule_id="INVALID_REFERENCE_DIMENSION",
                message=f"Reference image width must be positive, got {reference_width}.",
            )
        )
    if reference_height is not None and reference_height <= 0:
        errors.append(
            LayoutValidationError(
                rule_id="INVALID_REFERENCE_DIMENSION",
                message=f"Reference image height must be positive, got {reference_height}.",
            )
        )

    if not parking_spaces:
        errors.append(
            LayoutValidationError(
                rule_id="EMPTY_LAYOUT",
                message="Parking layout must contain at least one configured parking space.",
            )
        )
        return errors

    seen_ids: set[str] = set()
    seen_labels: set[str] = set()
    parsed_polygons: list[tuple[str, str, NormalizedPolygon]] = []

    for idx, sp in enumerate(parking_spaces):
        sp_id = str(sp.get("id") or f"space_{idx}")
        label = str(sp.get("operator_label") or "").strip()

        # Check unique ID
        if sp_id in seen_ids:
            errors.append(
                LayoutValidationError(
                    rule_id="DUPLICATE_SPACE_ID",
                    message=f"Duplicate parking space ID '{sp_id}' detected.",
                    space_id=sp_id,
                    space_label=label,
                )
            )
        seen_ids.add(sp_id)

        # Check unique label
        if not label:
            errors.append(
                LayoutValidationError(
                    rule_id="EMPTY_SPACE_LABEL",
                    message=f"Parking space at index {idx} has an empty operator label.",
                    space_id=sp_id,
                )
            )
        elif label.lower() in seen_labels:
            errors.append(
                LayoutValidationError(
                    rule_id="DUPLICATE_SPACE_LABEL",
                    message=f"Duplicate parking space label '{label}' detected.",
                    space_id=sp_id,
                    space_label=label,
                )
            )
        else:
            seen_labels.add(label.lower())

        # Validate polygon geometry
        raw_points = sp.get("polygon_normalized") or []
        try:
            poly = NormalizedPolygon.from_list(raw_points)
            poly_errors = validate_single_polygon(
                poly,
                element_name=f"Parking space '{label or sp_id}'",
                space_id=sp_id,
                space_label=label,
            )
            errors.extend(poly_errors)
            if not poly_errors:
                parsed_polygons.append((sp_id, label, poly))
        except Exception as e:
            errors.append(
                LayoutValidationError(
                    rule_id="MALFORMED_POLYGON_DATA",
                    message=f"Failed to parse polygon for space '{label or sp_id}': {str(e)}",
                    space_id=sp_id,
                    space_label=label,
                )
            )

    # Check pairwise parking space overlaps
    num_polys = len(parsed_polygons)
    for i in range(num_polys):
        id_i, label_i, poly_i = parsed_polygons[i]
        for j in range(i + 1, num_polys):
            id_j, label_j, poly_j = parsed_polygons[j]
            overlap_ij = calculate_overlap_ratio(poly_i, poly_j)
            overlap_ji = calculate_overlap_ratio(poly_j, poly_i)
            iou = calculate_iou(poly_i, poly_j)

            if overlap_ij > max_inter_space_overlap_ratio or overlap_ji > max_inter_space_overlap_ratio or iou > max_inter_space_overlap_ratio:
                errors.append(
                    LayoutValidationError(
                        rule_id="EXCESSIVE_SPACE_OVERLAP",
                        message=(
                            f"Excessive overlap between parking space '{label_i}' and '{label_j}' "
                            f"(Overlap: {max(overlap_ij, overlap_ji)*100:.1f}%, IoU: {iou*100:.1f}% > {max_inter_space_overlap_ratio*100:.1f}%)."
                        ),
                        space_id=id_i,
                        space_label=label_i,
                    )
                )

    # Validate approach zones
    if approach_zones:
        seen_approach_spaces: set[str] = set()
        for idx, az in enumerate(approach_zones):
            sp_id = str(az.get("parking_space_id") or "")
            az_id = str(az.get("id") or f"approach_{idx}")

            if not sp_id or sp_id not in seen_ids:
                errors.append(
                    LayoutValidationError(
                        rule_id="ORPHAN_APPROACH_ZONE",
                        message=f"Approach zone '{az_id}' links to non-existent space ID '{sp_id}'.",
                        space_id=sp_id,
                    )
                )
            elif sp_id in seen_approach_spaces:
                errors.append(
                    LayoutValidationError(
                        rule_id="DUPLICATE_APPROACH_ZONE",
                        message=f"Multiple approach zones configured for parking space ID '{sp_id}'. Exactly one approach zone allowed per space.",
                        space_id=sp_id,
                    )
                )
            else:
                seen_approach_spaces.add(sp_id)

            raw_points = az.get("polygon_normalized") or []
            try:
                poly = NormalizedPolygon.from_list(raw_points)
                az_errors = validate_single_polygon(
                    poly,
                    element_name=f"Approach zone for space '{sp_id}'",
                    space_id=sp_id,
                )
                errors.extend(az_errors)
            except Exception as e:
                errors.append(
                    LayoutValidationError(
                        rule_id="MALFORMED_APPROACH_POLYGON",
                        message=f"Failed to parse polygon for approach zone '{az_id}': {str(e)}",
                        space_id=sp_id,
                    )
                )

    return errors
