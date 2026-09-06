"""Deterministic geometry primitives and calculations for RoadSense SiteOps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class NormalizedPoint:
    """A 2D point normalized to reference image coordinates [0.0, 1.0]."""
    x: float
    y: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.x) and math.isfinite(self.y)):
            raise ValueError(f"Coordinates must be finite numbers, got ({self.x}, {self.y})")
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            raise ValueError(f"Normalized coordinates must be within [0.0, 1.0], got ({self.x}, {self.y})")

    def to_dict(self) -> dict[str, float]:
        return {"x": round(self.x, 6), "y": round(self.y, 6)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedPoint:
        return cls(x=float(data["x"]), y=float(data["y"]))


@dataclass(frozen=True)
class NormalizedPolygon:
    """A simple convex polygon in normalized [0.0, 1.0] coordinates."""
    points: Tuple[NormalizedPoint, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError(f"A polygon requires at least 3 points, got {len(self.points)}")

    def to_list(self) -> list[dict[str, float]]:
        return [p.to_dict() for p in self.points]

    @classmethod
    def from_list(cls, points_data: Sequence[dict[str, Any] | Tuple[float, float] | NormalizedPoint]) -> NormalizedPolygon:
        pts: list[NormalizedPoint] = []
        for item in points_data:
            if isinstance(item, NormalizedPoint):
                pts.append(item)
            elif isinstance(item, dict):
                pts.append(NormalizedPoint.from_dict(item))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                pts.append(NormalizedPoint(x=float(item[0]), y=float(item[1])))
            else:
                raise ValueError(f"Invalid point representation: {item}")
        return cls(points=tuple(pts))


def calculate_polygon_area(polygon: NormalizedPolygon) -> float:
    """Compute the non-negative area of a polygon in normalized space using Shoelace formula."""
    pts = polygon.points
    n = len(pts)
    if n < 3:
        return 0.0
    area2 = 0.0
    for i in range(n):
        j = (i + 1) % n
        area2 += pts[i].x * pts[j].y - pts[j].x * pts[i].y
    return abs(area2) / 2.0


def is_polygon_convex(polygon: NormalizedPolygon, allow_colinear: bool = False, eps: float = 1e-9) -> bool:
    """
    Check if a polygon is strictly convex (or non-degenerate convex).
    Returns True if all cross-products of consecutive edges share the same sign.
    """
    pts = polygon.points
    n = len(pts)
    if n < 3:
        return False

    sign = 0
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]

        dx1 = p1.x - p0.x
        dy1 = p1.y - p0.y
        dx2 = p2.x - p1.x
        dy2 = p2.y - p1.y

        cross = dx1 * dy2 - dy1 * dx2

        if abs(cross) > eps:
            current_sign = 1 if cross > 0 else -1
            if sign == 0:
                sign = current_sign
            elif sign != current_sign:
                return False
        elif not allow_colinear:
            # Colinear adjacent edges not permitted in strictly convex polygon
            return False

    return sign != 0


def has_self_intersections(polygon: NormalizedPolygon, eps: float = 1e-9) -> bool:
    """Check if any non-adjacent edges of a polygon intersect."""
    pts = polygon.points
    n = len(pts)
    if n < 4:
        return False

    def ccw(A: NormalizedPoint, B: NormalizedPoint, C: NormalizedPoint) -> float:
        return (C.y - A.y) * (B.x - A.x) - (B.y - A.y) * (C.x - A.x)

    def intersect(A: NormalizedPoint, B: NormalizedPoint, C: NormalizedPoint, D: NormalizedPoint) -> bool:
        # Check if segment AB strictly intersects segment CD
        ccw1 = ccw(A, B, C)
        ccw2 = ccw(A, B, D)
        ccw3 = ccw(C, D, A)
        ccw4 = ccw(C, D, B)

        if ((ccw1 > eps and ccw2 < -eps) or (ccw1 < -eps and ccw2 > eps)) and \
           ((ccw3 > eps and ccw4 < -eps) or (ccw3 < -eps and ccw4 > eps)):
            return True
        return False

    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if intersect(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]):
                return True

    return False


def canonicalize_polygon(polygon: NormalizedPolygon) -> NormalizedPolygon:
    """
    Produce a canonical representation of a convex polygon:
    1. Removes duplicate adjacent points.
    2. Ensures clockwise winding order.
    3. Starts from the lexicographically smallest vertex (min y, then min x).
    """
    pts = list(polygon.points)
    if not pts:
        return polygon

    # Remove duplicate adjacent points
    unique_pts: list[NormalizedPoint] = []
    for p in pts:
        if not unique_pts or (abs(p.x - unique_pts[-1].x) > 1e-7 or abs(p.y - unique_pts[-1].y) > 1e-7):
            unique_pts.append(p)

    if len(unique_pts) > 1 and abs(unique_pts[0].x - unique_pts[-1].x) <= 1e-7 and abs(unique_pts[0].y - unique_pts[-1].y) <= 1e-7:
        unique_pts.pop()

    if len(unique_pts) < 3:
        return NormalizedPolygon(tuple(unique_pts))

    # Calculate signed area to determine winding order
    signed_area2 = 0.0
    n = len(unique_pts)
    for i in range(n):
        j = (i + 1) % n
        signed_area2 += unique_pts[i].x * unique_pts[j].y - unique_pts[j].x * unique_pts[i].y

    # For standard screen coordinates (y downwards):
    # signed_area2 < 0 is clockwise, signed_area2 > 0 is counter-clockwise
    # Normalize to clockwise (signed_area2 < 0)
    if signed_area2 > 0:
        unique_pts.reverse()

    # Find lexicographically smallest point (min y, then min x)
    min_idx = 0
    for i in range(1, len(unique_pts)):
        if (unique_pts[i].y, unique_pts[i].x) < (unique_pts[min_idx].y, unique_pts[min_idx].x):
            min_idx = i

    # Rotate list so that lexicographically smallest point is at index 0
    canonical_pts = unique_pts[min_idx:] + unique_pts[:min_idx]
    return NormalizedPolygon(tuple(canonical_pts))


def _clip_polygon_with_halfplane(
    subject_points: Sequence[NormalizedPoint],
    cp1: NormalizedPoint,
    cp2: NormalizedPoint,
) -> list[NormalizedPoint]:
    """Sutherland-Hodgman clipping of subject polygon with directed edge cp1 -> cp2."""
    def is_inside(p: NormalizedPoint) -> bool:
        # Returns True if p is on the left side of line cp1 -> cp2
        return (cp2.x - cp1.x) * (p.y - cp1.y) - (cp2.y - cp1.y) * (p.x - cp1.x) >= -1e-9

    def compute_intersection(s: NormalizedPoint, e: NormalizedPoint) -> NormalizedPoint:
        dc_x = cp1.x - cp2.x
        dc_y = cp1.y - cp2.y
        dp_x = s.x - e.x
        dp_y = s.y - e.y

        n1 = cp1.x * cp2.y - cp1.y * cp2.x
        n2 = s.x * e.y - s.y * e.x
        denom = dc_x * dp_y - dc_y * dp_x

        if abs(denom) < 1e-11:
            return s
        ix = (n1 * dp_x - n2 * dc_x) / denom
        iy = (n1 * dp_y - n2 * dc_y) / denom
        return NormalizedPoint(x=max(0.0, min(1.0, ix)), y=max(0.0, min(1.0, iy)))

    output_list: list[NormalizedPoint] = []
    if not subject_points:
        return output_list

    s = subject_points[-1]
    for e in subject_points:
        if is_inside(e):
            if not is_inside(s):
                output_list.append(compute_intersection(s, e))
            output_list.append(e)
        elif is_inside(s):
            output_list.append(compute_intersection(s, e))
        s = e

    return output_list


def calculate_polygon_intersection_area(poly1: NormalizedPolygon, poly2: NormalizedPolygon) -> float:
    """
    Calculate the exact intersection area of two convex polygons using Sutherland-Hodgman clipping.
    """
    if calculate_polygon_area(poly1) <= 1e-9 or calculate_polygon_area(poly2) <= 1e-9:
        return 0.0

    # Ensure counter-clockwise order for clipping halfplane convention
    def ensure_ccw(poly: NormalizedPolygon) -> list[NormalizedPoint]:
        pts = list(poly.points)
        area2 = sum(pts[i].x * pts[(i+1)%len(pts)].y - pts[(i+1)%len(pts)].x * pts[i].y for i in range(len(pts)))
        if area2 < 0:
            pts.reverse()
        return pts

    subject = ensure_ccw(poly1)
    clipper = ensure_ccw(poly2)

    clipped: Sequence[NormalizedPoint] = subject
    for i in range(len(clipper)):
        cp1 = clipper[i]
        cp2 = clipper[(i + 1) % len(clipper)]
        clipped = _clip_polygon_with_halfplane(clipped, cp1, cp2)
        if len(clipped) < 3:
            return 0.0

    # Calculate area of clipped polygon
    if len(clipped) < 3:
        return 0.0
    area2 = sum(clipped[i].x * clipped[(i+1)%len(clipped)].y - clipped[(i+1)%len(clipped)].x * clipped[i].y for i in range(len(clipped)))
    return abs(area2) / 2.0


def calculate_overlap_ratio(source_poly: NormalizedPolygon, target_poly: NormalizedPolygon) -> float:
    """
    Calculate the overlap ratio: Intersection Area / Area(source_poly).
    Represents the fraction of source_poly that is overlapped by target_poly.
    """
    source_area = calculate_polygon_area(source_poly)
    if source_area <= 1e-9:
        return 0.0
    intersection_area = calculate_polygon_intersection_area(source_poly, target_poly)
    return min(1.0, intersection_area / source_area)


def calculate_iou(poly1: NormalizedPolygon, poly2: NormalizedPolygon) -> float:
    """
    Calculate the Intersection over Union (IoU) between two convex polygons:
    Intersection Area / (Area(poly1) + Area(poly2) - Intersection Area).
    """
    area1 = calculate_polygon_area(poly1)
    area2 = calculate_polygon_area(poly2)
    intersection = calculate_polygon_intersection_area(poly1, poly2)
    union = area1 + area2 - intersection
    if union <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, intersection / union))
