"""Unit tests for parking layout validation rules."""

import pytest
from src.parking.layout_validation import validate_parking_layout, validate_single_polygon
from src.parking.geometry import NormalizedPolygon


def test_validate_single_polygon_valid():
    poly = NormalizedPolygon.from_list([(0.1, 0.1), (0.3, 0.1), (0.3, 0.4), (0.1, 0.4)])
    errors = validate_single_polygon(poly)
    assert len(errors) == 0


def test_validate_single_polygon_insufficient_vertices():
    with pytest.raises(ValueError, match="at least 3"):
        NormalizedPolygon.from_list([(0.1, 0.1), (0.2, 0.2)])


def test_validate_single_polygon_non_convex():
    poly = NormalizedPolygon.from_list([
        (0.1, 0.1),
        (0.5, 0.1),
        (0.3, 0.25),
        (0.5, 0.4),
        (0.1, 0.4),
    ])
    errors = validate_single_polygon(poly)
    assert any(e.rule_id == "NON_CONVEX_POLYGON" for e in errors)


def test_validate_single_polygon_self_intersecting():
    poly = NormalizedPolygon.from_list([
        (0.1, 0.1),
        (0.5, 0.5),
        (0.5, 0.1),
        (0.1, 0.5),
    ])
    errors = validate_single_polygon(poly)
    assert any(e.rule_id == "SELF_INTERSECTING_POLYGON" for e in errors)


def test_validate_parking_layout_empty():
    errors = validate_parking_layout([])
    assert any(e.rule_id == "EMPTY_LAYOUT" for e in errors)


def test_validate_parking_layout_duplicate_labels():
    spaces = [
        {"id": "sp_1", "operator_label": "Bay-01", "polygon_normalized": [(0.1, 0.1), (0.2, 0.1), (0.2, 0.3), (0.1, 0.3)]},
        {"id": "sp_2", "operator_label": "Bay-01", "polygon_normalized": [(0.3, 0.1), (0.4, 0.1), (0.4, 0.3), (0.3, 0.3)]},
    ]
    errors = validate_parking_layout(spaces)
    assert any(e.rule_id == "DUPLICATE_SPACE_LABEL" for e in errors)


def test_validate_parking_layout_excessive_overlap():
    # Two heavily overlapping spaces
    spaces = [
        {"id": "sp_1", "operator_label": "Bay-01", "polygon_normalized": [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]},
        {"id": "sp_2", "operator_label": "Bay-02", "polygon_normalized": [(0.15, 0.1), (0.35, 0.1), (0.35, 0.3), (0.15, 0.3)]},
    ]
    errors = validate_parking_layout(spaces)
    assert any(e.rule_id == "EXCESSIVE_SPACE_OVERLAP" for e in errors)


def test_validate_parking_layout_valid_boundary_contact():
    # Two adjacent spaces sharing a vertical boundary line (overlap area = 0)
    spaces = [
        {"id": "sp_1", "operator_label": "Bay-01", "polygon_normalized": [(0.1, 0.1), (0.2, 0.1), (0.2, 0.3), (0.1, 0.3)]},
        {"id": "sp_2", "operator_label": "Bay-02", "polygon_normalized": [(0.2, 0.1), (0.3, 0.1), (0.3, 0.3), (0.2, 0.3)]},
    ]
    errors = validate_parking_layout(spaces)
    assert len(errors) == 0


def test_validate_approach_zone_orphan():
    spaces = [
        {"id": "sp_1", "operator_label": "Bay-01", "polygon_normalized": [(0.1, 0.1), (0.2, 0.1), (0.2, 0.3), (0.1, 0.3)]},
    ]
    approaches = [
        {"id": "az_1", "parking_space_id": "sp_non_existent", "polygon_normalized": [(0.1, 0.3), (0.2, 0.3), (0.2, 0.4), (0.1, 0.4)]},
    ]
    errors = validate_parking_layout(spaces, approach_zones=approaches)
    assert any(e.rule_id == "ORPHAN_APPROACH_ZONE" for e in errors)
