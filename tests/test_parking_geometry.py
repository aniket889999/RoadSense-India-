"""Unit tests for deterministic geometry primitives and calculations."""

import math
import pytest
from src.parking.geometry import (
    NormalizedPoint,
    NormalizedPolygon,
    calculate_iou,
    calculate_overlap_ratio,
    calculate_polygon_area,
    calculate_polygon_intersection_area,
    canonicalize_polygon,
    has_self_intersections,
    is_polygon_convex,
)


def test_normalized_point_valid():
    p = NormalizedPoint(0.25, 0.75)
    assert p.x == 0.25
    assert p.y == 0.75
    assert p.to_dict() == {"x": 0.25, "y": 0.75}


def test_normalized_point_bounds():
    with pytest.raises(ValueError, match="within"):
        NormalizedPoint(-0.1, 0.5)

    with pytest.raises(ValueError, match="within"):
        NormalizedPoint(0.5, 1.1)


def test_normalized_point_nan_inf():
    with pytest.raises(ValueError, match="finite"):
        NormalizedPoint(float("nan"), 0.5)

    with pytest.raises(ValueError, match="finite"):
        NormalizedPoint(0.5, float("inf"))


def test_polygon_area_rectangle():
    # Unit rectangle [0.1, 0.1] to [0.5, 0.4] -> width = 0.4, height = 0.3 -> area = 0.12
    poly = NormalizedPolygon.from_list([
        (0.1, 0.1),
        (0.5, 0.1),
        (0.5, 0.4),
        (0.1, 0.4),
    ])
    area = calculate_polygon_area(poly)
    assert pytest.approx(area, rel=1e-5) == 0.12


def test_polygon_area_triangle():
    poly = NormalizedPolygon.from_list([
        (0.0, 0.0),
        (0.4, 0.0),
        (0.0, 0.6),
    ])
    area = calculate_polygon_area(poly)
    assert pytest.approx(area, rel=1e-5) == 0.12


def test_convexity():
    # Convex rectangle
    convex_poly = NormalizedPolygon.from_list([
        (0.1, 0.1),
        (0.5, 0.1),
        (0.5, 0.4),
        (0.1, 0.4),
    ])
    assert is_polygon_convex(convex_poly) is True

    # Concave polygon (arrow / chevron shape)
    concave_poly = NormalizedPolygon.from_list([
        (0.1, 0.1),
        (0.5, 0.1),
        (0.3, 0.25),
        (0.5, 0.4),
        (0.1, 0.4),
    ])
    assert is_polygon_convex(concave_poly) is False


def test_self_intersection():
    # Figure 8 polygon (self-intersecting)
    hourglass = NormalizedPolygon.from_list([
        (0.1, 0.1),
        (0.5, 0.5),
        (0.5, 0.1),
        (0.1, 0.5),
    ])
    assert has_self_intersections(hourglass) is True

    # Simple rectangle
    simple_box = NormalizedPolygon.from_list([
        (0.1, 0.1),
        (0.5, 0.1),
        (0.5, 0.5),
        (0.1, 0.5),
    ])
    assert has_self_intersections(simple_box) is False


def test_canonicalize_polygon_determinism():
    # Two polygons with same vertices in different order / winding
    poly1 = NormalizedPolygon.from_list([
        (0.1, 0.4),
        (0.5, 0.4),
        (0.5, 0.1),
        (0.1, 0.1),
    ])
    poly2 = NormalizedPolygon.from_list([
        (0.1, 0.1),
        (0.5, 0.1),
        (0.5, 0.4),
        (0.1, 0.4),
    ])
    c1 = canonicalize_polygon(poly1)
    c2 = canonicalize_polygon(poly2)
    assert c1.to_list() == c2.to_list()


def test_polygon_intersection_and_iou():
    # Box 1: [0.1, 0.1] to [0.3, 0.3], area = 0.04
    poly1 = NormalizedPolygon.from_list([
        (0.1, 0.1),
        (0.3, 0.1),
        (0.3, 0.3),
        (0.1, 0.3),
    ])
    # Box 2: [0.2, 0.1] to [0.4, 0.3], area = 0.04
    # Overlap box: [0.2, 0.1] to [0.3, 0.3], area = 0.02
    poly2 = NormalizedPolygon.from_list([
        (0.2, 0.1),
        (0.4, 0.1),
        (0.4, 0.3),
        (0.2, 0.3),
    ])

    inter_area = calculate_polygon_intersection_area(poly1, poly2)
    assert pytest.approx(inter_area, rel=1e-5) == 0.02

    overlap_1 = calculate_overlap_ratio(poly1, poly2)
    assert pytest.approx(overlap_1, rel=1e-5) == 0.5

    iou = calculate_iou(poly1, poly2)
    # Union area = 0.04 + 0.04 - 0.02 = 0.06 -> IoU = 0.02 / 0.06 = 1/3 ~ 0.333333
    assert pytest.approx(iou, rel=1e-5) == (1.0 / 3.0)


def test_disjoint_polygons_intersection():
    poly1 = NormalizedPolygon.from_list([(0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)])
    poly2 = NormalizedPolygon.from_list([(0.5, 0.5), (0.6, 0.5), (0.6, 0.6), (0.5, 0.6)])

    assert calculate_polygon_intersection_area(poly1, poly2) == 0.0
    assert calculate_overlap_ratio(poly1, poly2) == 0.0
    assert calculate_iou(poly1, poly2) == 0.0
