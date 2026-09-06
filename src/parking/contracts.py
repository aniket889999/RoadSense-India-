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


class StabilityDecision(str, Enum):
    """Automated camera stability assessment outcome."""
    STABLE = "STABLE"
    UNSTABLE = "UNSTABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    ERROR = "ERROR"


class OperationalGate(str, Enum):
    """Operational inference gate status controlling downstream analytics."""
    ALLOWED = "ALLOWED"
    BLOCKED = "BLOCKED"


from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class StabilityThresholds:
    """
    Configurable thresholds for geometric camera stability validation.

    Note: These are operational research heuristics designed for fail-closed safety,
    not scientifically universal invariants across all lenses and fields of view.
    """
    max_translation_px: float = 8.0
    max_translation_norm: float = 0.01
    max_rotation_deg: float = 1.0
    max_scale_change: float = 0.03
    max_perspective_distortion: float = 0.0005
    min_matches: int = 25
    min_inliers: int = 15
    min_inlier_ratio: float = 0.25
    max_reprojection_error: float = 2.5
    max_assessment_age_seconds: int = 86400  # 24 hours validity window


def evaluate_operational_gate(
    decision: Optional[StabilityDecision],
    assessment_reference_sha: Optional[str],
    current_reference_sha: Optional[str],
    assessment_layout_sha: Optional[str],
    current_layout_sha: Optional[str],
    assessment_timestamp: Optional[datetime] = None,
    current_timestamp: Optional[datetime] = None,
    max_age_seconds: int = 86400,
) -> tuple[OperationalGate, list[str]]:
    """
    Derive fail-closed operational inference gate status.

    Rules:
    - ALLOWED only if:
      1. Aggregate decision is STABLE.
      2. Assessment reference image SHA exactly matches the camera's current reference SHA.
      3. Assessment layout canonical SHA exactly matches the current verified layout SHA.
      4. Assessment is fresh (within max_age_seconds if timestamps provided).
    - BLOCKED for all other conditions (UNSTABLE, INSUFFICIENT_EVIDENCE, ERROR,
      stale assessment, missing assessment, or SHA mismatch).
    """
    reasons: list[str] = []

    if decision is None:
        reasons.append("NO_ASSESSMENT: Camera has no recorded stability assessment.")
        return OperationalGate.BLOCKED, reasons

    if decision != StabilityDecision.STABLE:
        reasons.append(f"UNSTABLE_DECISION: Stability assessment decision is {decision.value}.")

    if not assessment_reference_sha or not current_reference_sha:
        reasons.append("MISSING_REFERENCE_SHA: Reference image SHA is missing from camera or assessment.")
    elif assessment_reference_sha != current_reference_sha:
        reasons.append(
            f"STALE_REFERENCE_SHA: Assessment reference SHA ({assessment_reference_sha[:10]}...) "
            f"differs from current camera reference SHA ({current_reference_sha[:10]}...)."
        )

    if not current_layout_sha:
        reasons.append("NO_VERIFIED_LAYOUT: Camera has no active verified layout revision.")
    elif not assessment_layout_sha or assessment_layout_sha != current_layout_sha:
        reasons.append(
            f"STALE_LAYOUT_SHA: Assessment layout SHA ({str(assessment_layout_sha)[:10]}...) "
            f"differs from active verified layout SHA ({str(current_layout_sha)[:10]}...)."
        )

    if assessment_timestamp and current_timestamp:
        t_curr = current_timestamp.astimezone(timezone.utc) if current_timestamp.tzinfo else current_timestamp.replace(tzinfo=timezone.utc)
        t_ass = assessment_timestamp.astimezone(timezone.utc) if assessment_timestamp.tzinfo else assessment_timestamp.replace(tzinfo=timezone.utc)
        age = (t_curr - t_ass).total_seconds()
        if age > max_age_seconds:
            reasons.append(f"EXPIRED_ASSESSMENT: Assessment age ({int(age)}s) exceeds max allowed age ({max_age_seconds}s).")

    if reasons:
        return OperationalGate.BLOCKED, reasons

    return OperationalGate.ALLOWED, ["Camera geometry verified stable and aligned with active verified layout."]
