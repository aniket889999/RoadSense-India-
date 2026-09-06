"""Unit tests for RoadSense SiteOps parking contracts and availability derivation."""

import pytest
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


def test_enums_completeness():
    assert CalibrationStatus.NOT_CONFIGURED.value == "NOT_CONFIGURED"
    assert CalibrationStatus.PENDING_REVIEW.value == "PENDING_REVIEW"
    assert CalibrationStatus.VERIFIED.value == "VERIFIED"
    assert CalibrationStatus.INVALIDATED.value == "INVALIDATED"

    assert LayoutRevisionStatus.DRAFT.value == "DRAFT"
    assert LayoutRevisionStatus.PENDING_REVIEW.value == "PENDING_REVIEW"
    assert LayoutRevisionStatus.VERIFIED.value == "VERIFIED"
    assert LayoutRevisionStatus.SUPERSEDED.value == "SUPERSEDED"
    assert LayoutRevisionStatus.INVALIDATED.value == "INVALIDATED"

    assert OccupancyState.FREE.value == "FREE"
    assert OccupancyState.OCCUPIED.value == "OCCUPIED"
    assert OccupancyState.UNKNOWN.value == "UNKNOWN"

    assert SurfaceState.SAFE.value == "SAFE"
    assert SurfaceState.UNSAFE.value == "UNSAFE"
    assert SurfaceState.UNKNOWN.value == "UNKNOWN"

    assert ObstructionState.CLEAR.value == "CLEAR"
    assert ObstructionState.BLOCKED.value == "BLOCKED"
    assert ObstructionState.UNKNOWN.value == "UNKNOWN"

    assert ReviewState.UNREVIEWED.value == "UNREVIEWED"
    assert ReviewState.CONFIRMED.value == "CONFIRMED"
    assert ReviewState.REJECTED.value == "REJECTED"
    assert ReviewState.NEEDS_REVIEW.value == "NEEDS_REVIEW"

    assert DerivedAvailability.FREE_SAFE.value == "FREE_SAFE"
    assert DerivedAvailability.FREE_UNSAFE.value == "FREE_UNSAFE"
    assert DerivedAvailability.OCCUPIED.value == "OCCUPIED"
    assert DerivedAvailability.BLOCKED.value == "BLOCKED"
    assert DerivedAvailability.UNKNOWN.value == "UNKNOWN"


def test_derive_availability_precedence_rule_1_unknown_occupancy():
    # If occupancy is UNKNOWN, derived status must be UNKNOWN
    assert derive_operator_availability(
        OccupancyState.UNKNOWN, SurfaceState.SAFE, ObstructionState.CLEAR, ReviewState.CONFIRMED
    ) == DerivedAvailability.UNKNOWN

    assert derive_operator_availability(
        OccupancyState.UNKNOWN, SurfaceState.UNSAFE, ObstructionState.BLOCKED, ReviewState.CONFIRMED
    ) == DerivedAvailability.UNKNOWN


def test_derive_availability_precedence_rule_2_blocked():
    # If obstruction is BLOCKED and occupancy is not UNKNOWN, derived status is BLOCKED
    assert derive_operator_availability(
        OccupancyState.FREE, SurfaceState.SAFE, ObstructionState.BLOCKED, ReviewState.CONFIRMED
    ) == DerivedAvailability.BLOCKED

    assert derive_operator_availability(
        OccupancyState.FREE, SurfaceState.UNSAFE, ObstructionState.BLOCKED, ReviewState.CONFIRMED
    ) == DerivedAvailability.BLOCKED


def test_derive_availability_precedence_rule_3_occupied():
    # If occupied, status is OCCUPIED
    assert derive_operator_availability(
        OccupancyState.OCCUPIED, SurfaceState.SAFE, ObstructionState.CLEAR, ReviewState.CONFIRMED
    ) == DerivedAvailability.OCCUPIED

    assert derive_operator_availability(
        OccupancyState.OCCUPIED, SurfaceState.UNSAFE, ObstructionState.CLEAR, ReviewState.CONFIRMED
    ) == DerivedAvailability.OCCUPIED


def test_derive_availability_precedence_rule_4_confirmed_hazard():
    # If FREE and confirmed hazard present, derived status is FREE_UNSAFE
    assert derive_operator_availability(
        OccupancyState.FREE, SurfaceState.UNSAFE, ObstructionState.CLEAR, ReviewState.CONFIRMED
    ) == DerivedAvailability.FREE_UNSAFE

    assert derive_operator_availability(
        OccupancyState.FREE, SurfaceState.SAFE, ObstructionState.CLEAR, ReviewState.CONFIRMED, has_confirmed_hazard=True
    ) == DerivedAvailability.FREE_UNSAFE


def test_derive_availability_precedence_rule_5_free_safe():
    # If FREE, CLEAR, SAFE and CONFIRMED -> FREE_SAFE
    assert derive_operator_availability(
        OccupancyState.FREE, SurfaceState.SAFE, ObstructionState.CLEAR, ReviewState.CONFIRMED
    ) == DerivedAvailability.FREE_SAFE


def test_derive_availability_precedence_rule_6_unreviewed_not_safe():
    # UNREVIEWED surface cannot establish SAFE
    assert derive_operator_availability(
        OccupancyState.FREE, SurfaceState.SAFE, ObstructionState.CLEAR, ReviewState.UNREVIEWED
    ) == DerivedAvailability.UNKNOWN


def test_derive_availability_precedence_rejected_hazard():
    # When candidate hazard is REJECTED, space is SAFE
    assert derive_operator_availability(
        OccupancyState.FREE, SurfaceState.UNSAFE, ObstructionState.CLEAR, ReviewState.REJECTED
    ) == DerivedAvailability.FREE_SAFE


def test_derive_availability_precedence_needs_review():
    # NEEDS_REVIEW yields UNKNOWN
    assert derive_operator_availability(
        OccupancyState.FREE, SurfaceState.SAFE, ObstructionState.CLEAR, ReviewState.NEEDS_REVIEW
    ) == DerivedAvailability.UNKNOWN
