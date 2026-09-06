"""Domain contracts, enums, and pure state derivation functions for RoadSense SiteOps."""

from enum import Enum
from typing import Optional


class CalibrationStatus(str, Enum):
    """Camera geometric calibration lifecycle status."""
    NOT_CONFIGURED = "NOT_CONFIGURED"
    PENDING_REVIEW = "PENDING_REVIEW"
    VERIFIED = "VERIFIED"
    INVALIDATED = "INVALIDATED"


class LayoutRevisionStatus(str, Enum):
    """Parking layout revision lifecycle status."""
    DRAFT = "DRAFT"
    PENDING_REVIEW = "PENDING_REVIEW"
    VERIFIED = "VERIFIED"
    SUPERSEDED = "SUPERSEDED"
    INVALIDATED = "INVALIDATED"


class OccupancyState(str, Enum):
    """Factored physical vehicle occupancy dimension."""
    FREE = "FREE"
    OCCUPIED = "OCCUPIED"
    UNKNOWN = "UNKNOWN"


class SurfaceState(str, Enum):
    """Factored pavement surface condition dimension."""
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    UNKNOWN = "UNKNOWN"


class ObstructionState(str, Enum):
    """Factored physical obstacle and access dimension."""
    CLEAR = "CLEAR"
    BLOCKED = "BLOCKED"
    UNKNOWN = "UNKNOWN"


class ReviewState(str, Enum):
    """Human-in-the-loop inspection and review workflow dimension."""
    UNREVIEWED = "UNREVIEWED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class DerivedAvailability(str, Enum):
    """Derived operator-facing availability badge."""
    FREE_SAFE = "FREE_SAFE"
    FREE_UNSAFE = "FREE_UNSAFE"
    OCCUPIED = "OCCUPIED"
    BLOCKED = "BLOCKED"
    UNKNOWN = "UNKNOWN"


class SpaceType(str, Enum):
    """Designated parking stall functional classification."""
    STANDARD = "STANDARD"
    ACCESSIBLE = "ACCESSIBLE"
    EV_CHARGING = "EV_CHARGING"
    LOADING = "LOADING"
    EMERGENCY = "EMERGENCY"
    OTHER = "OTHER"


def derive_operator_availability(
    occupancy: OccupancyState,
    surface: SurfaceState,
    obstruction: ObstructionState,
    review: ReviewState,
    has_confirmed_hazard: bool = False,
) -> DerivedAvailability:
    """
    Derive operator-facing availability status from factored state dimensions.

    Strict Precedence Rules:
    1. If evidence quality is insufficient or occupancy is UNKNOWN:
       derived status = UNKNOWN
    2. If obstruction is BLOCKED:
       derived status = BLOCKED
    3. If occupancy is OCCUPIED:
       derived status = OCCUPIED (surface hazard is preserved as a separate flag)
    4. If occupancy is FREE and a CONFIRMED surface hazard affects the bay or approach zone:
       derived status = FREE_UNSAFE
    5. If occupancy is FREE, obstruction is CLEAR, and the reviewed surface state is SAFE (confirmed):
       derived status = FREE_SAFE
    6. An UNREVIEWED surface observation must not establish SAFE.
    7. NEEDS_REVIEW changes workflow presentation / yields UNKNOWN if unverified, but must not erase known physical facts.
    """
    # Rule 1: Unknown occupancy or insufficient evidence
    if occupancy == OccupancyState.UNKNOWN:
        return DerivedAvailability.UNKNOWN

    # Rule 2: Obstruction blocks access regardless of surface state
    if obstruction == ObstructionState.BLOCKED:
        return DerivedAvailability.BLOCKED

    # Rule 3: Vehicle physically present
    if occupancy == OccupancyState.OCCUPIED:
        return DerivedAvailability.OCCUPIED

    # Rule 4 & 5: When occupancy is FREE
    if occupancy == OccupancyState.FREE:
        # Rule 7 / triage check: if workflow requires review and surface/obstruction is unverified
        if review == ReviewState.NEEDS_REVIEW:
            return DerivedAvailability.UNKNOWN

        # Rule 4: Confirmed hazard presence
        if has_confirmed_hazard or (surface == SurfaceState.UNSAFE and review == ReviewState.CONFIRMED):
            return DerivedAvailability.FREE_UNSAFE

        # If a candidate hazard was formally rejected by reviewer, surface is treated as clear
        if surface == SurfaceState.UNSAFE and review == ReviewState.REJECTED:
            if obstruction == ObstructionState.CLEAR:
                return DerivedAvailability.FREE_SAFE
            return DerivedAvailability.UNKNOWN

        # Rule 5: Confirmed safe and clear
        if obstruction == ObstructionState.CLEAR and surface == SurfaceState.SAFE and review == ReviewState.CONFIRMED:
            return DerivedAvailability.FREE_SAFE

        # Rule 6: Unreviewed surface cannot establish SAFE
        if surface == SurfaceState.SAFE and review == ReviewState.UNREVIEWED:
            return DerivedAvailability.UNKNOWN

        # Unknown surface or obstruction
        if obstruction == ObstructionState.UNKNOWN or surface == SurfaceState.UNKNOWN:
            return DerivedAvailability.UNKNOWN

    return DerivedAvailability.UNKNOWN
