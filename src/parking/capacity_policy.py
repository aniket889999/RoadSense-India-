"""Pure, deterministic, and versioned domain policy for Safe Usable Parking Capacity fusion.

RoadSense SiteOps Phase 3A:
Combines fixed-camera parking occupancy with current, human-reviewed pavement-hazard
evidence to calculate honest safe usable parking capacity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class CapacityState(str, Enum):
    """Deterministic safe usable capacity state for a single parking bay."""
    OCCUPIED = "OCCUPIED"
    USABLE_AVAILABLE = "USABLE_AVAILABLE"
    HAZARD_BLOCKED = "HAZARD_BLOCKED"
    APPROACH_BLOCKED = "APPROACH_BLOCKED"
    VACANT_UNASSESSED = "VACANT_UNASSESSED"
    OCCLUDED = "OCCLUDED"
    UNKNOWN = "UNKNOWN"
    INFERENCE_BLOCKED = "INFERENCE_BLOCKED"


class OperationalGate(str, Enum):
    """Operational camera gate allowance status."""
    ALLOWED = "ALLOWED"
    BLOCKED = "BLOCKED"


class HazardTarget(str, Enum):
    """Geometric target for hazard association."""
    BAY = "BAY"
    APPROACH_ZONE = "APPROACH_ZONE"


class HazardReviewState(str, Enum):
    """Human-in-the-loop review state for hazard candidates."""
    UNREVIEWED = "UNREVIEWED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class HazardLifecycleState(str, Enum):
    """Lifecycle disposition of a confirmed hazard."""
    ACTIVE = "ACTIVE"
    MITIGATED = "MITIGATED"
    RESOLVED = "RESOLVED"
    EXPIRED = "EXPIRED"
    SUPERSEDED = "SUPERSEDED"


# Reason Codes (Stable, Machine-Readable)
REASON_GATE_BLOCKED = "GATE_BLOCKED"
REASON_LAYOUT_NOT_VERIFIED = "LAYOUT_NOT_VERIFIED"
REASON_OCCUPANCY_UNKNOWN = "OCCUPANCY_UNKNOWN"
REASON_OCCUPANCY_OCCLUDED = "OCCUPANCY_OCCLUDED"
REASON_OCCUPANCY_OCCUPIED = "OCCUPANCY_OCCUPIED"
REASON_OCCUPANCY_EVIDENCE_STALE = "OCCUPANCY_EVIDENCE_STALE"
REASON_OCCUPANCY_EVIDENCE_MISSING = "OCCUPANCY_EVIDENCE_MISSING"
REASON_INSPECTION_MISSING = "INSPECTION_MISSING"
REASON_INSPECTION_STALE = "INSPECTION_STALE"
REASON_BAY_HAZARD_ACTIVE = "BAY_HAZARD_ACTIVE"
REASON_APPROACH_HAZARD_ACTIVE = "APPROACH_HAZARD_ACTIVE"
REASON_UNVERIFIED_HAZARD_IGNORED = "UNVERIFIED_HAZARD_IGNORED"
REASON_INACTIVE_HAZARD_LOGGED = "INACTIVE_HAZARD_LOGGED"
REASON_USABLE_AVAILABLE = "USABLE_AVAILABLE"
REASON_INVALID_INPUT_FAIL_CLOSED = "INVALID_INPUT_FAIL_CLOSED"


@dataclass(frozen=True)
class CapacityPolicyConfig:
    """Versioned configuration governing safe usable capacity evaluation."""
    policy_version: str = "3A.1"
    max_occupancy_age_seconds: float = 300.0  # 5 minutes
    max_inspection_age_seconds: float = 86400.0 * 30.0  # 30 days
    require_verified_layout: bool = True
    fail_closed_on_missing_evidence: bool = True

    def compute_sha256(self) -> str:
        """Compute deterministic SHA-256 fingerprint of the configuration."""
        canonical_dict = {
            "policy_version": self.policy_version,
            "max_occupancy_age_seconds": round(float(self.max_occupancy_age_seconds), 3),
            "max_inspection_age_seconds": round(float(self.max_inspection_age_seconds), 3),
            "require_verified_layout": bool(self.require_verified_layout),
            "fail_closed_on_missing_evidence": bool(self.fail_closed_on_missing_evidence),
        }
        encoded = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class HazardAssociationInput:
    """Associated hazard evidence for a bay or its approach zone."""
    association_id: str
    hazard_id: str
    target_type: Union[HazardTarget, str]
    target_id: str
    review_state: Union[HazardReviewState, str]
    lifecycle_state: Union[HazardLifecycleState, str]
    severity_label: Optional[str] = None  # Qualitative tag (e.g. LOW/MED/HIGH); NO fabricated depth
    evidence_id: Optional[str] = None
    created_at: Optional[Union[datetime, str]] = None
    reviewed_at: Optional[Union[datetime, str]] = None
    reviewed_by: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "association_id": self.association_id,
            "hazard_id": self.hazard_id,
            "target_type": str(self.target_type.value if isinstance(self.target_type, HazardTarget) else self.target_type),
            "target_id": self.target_id,
            "review_state": str(self.review_state.value if isinstance(self.review_state, HazardReviewState) else self.review_state),
            "lifecycle_state": str(self.lifecycle_state.value if isinstance(self.lifecycle_state, HazardLifecycleState) else self.lifecycle_state),
            "severity_label": self.severity_label,
            "evidence_id": self.evidence_id,
            "created_at": str(self.created_at) if self.created_at else None,
            "reviewed_at": str(self.reviewed_at) if self.reviewed_at else None,
            "reviewed_by": self.reviewed_by,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class PavementInspectionEvidence:
    """Pavement surface inspection evidence for the parking area."""
    session_id: Optional[str] = None
    inspection_timestamp: Optional[Union[datetime, str, float]] = None
    freshness_age_seconds: Optional[float] = None
    is_fresh: Optional[bool] = None
    inspection_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "inspection_timestamp": str(self.inspection_timestamp) if self.inspection_timestamp else None,
            "freshness_age_seconds": round(self.freshness_age_seconds, 2) if self.freshness_age_seconds is not None else None,
            "is_fresh": self.is_fresh,
            "inspection_notes": self.inspection_notes,
        }


@dataclass(frozen=True)
class BayOccupancyEvidenceInput:
    """Occupancy observation and evidence for an individual parking bay."""
    bay_id: str
    operator_label: str
    space_type: str = "STANDARD"
    occupancy_state: str = "UNKNOWN"  # VACANT, OCCUPIED, OCCLUDED, UNKNOWN
    occupancy_timestamp: Optional[Union[datetime, str, float]] = None
    evidence_age_seconds: Optional[float] = None
    confidence: float = 1.0
    provenance_sha256: Optional[str] = None
    job_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bay_id": self.bay_id,
            "operator_label": self.operator_label,
            "space_type": self.space_type,
            "occupancy_state": self.occupancy_state,
            "occupancy_timestamp": str(self.occupancy_timestamp) if self.occupancy_timestamp else None,
            "evidence_age_seconds": round(self.evidence_age_seconds, 2) if self.evidence_age_seconds is not None else None,
            "confidence": round(self.confidence, 4),
            "provenance_sha256": self.provenance_sha256,
            "job_id": self.job_id,
        }


@dataclass(frozen=True)
class BayCapacityDecision:
    """Deterministic evaluation outcome for an individual bay."""
    bay_id: str
    operator_label: str
    space_type: str
    capacity_state: CapacityState
    is_usable: bool
    reason_codes: List[str]
    evidence_ids: Dict[str, Any]
    policy_config_sha256: str
    evaluated_at: str
    raw_occupancy_state: str
    has_active_bay_hazard: bool
    has_active_approach_hazard: bool
    has_unverified_hazard: bool
    pavement_inspection_fresh: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bay_id": self.bay_id,
            "operator_label": self.operator_label,
            "space_type": self.space_type,
            "capacity_state": self.capacity_state.value,
            "is_usable": self.is_usable,
            "reason_codes": list(self.reason_codes),
            "evidence_ids": self.evidence_ids,
            "policy_config_sha256": self.policy_config_sha256,
            "evaluated_at": self.evaluated_at,
            "raw_occupancy_state": self.raw_occupancy_state,
            "has_active_bay_hazard": self.has_active_bay_hazard,
            "has_active_approach_hazard": self.has_active_approach_hazard,
            "has_unverified_hazard": self.has_unverified_hazard,
            "pavement_inspection_fresh": self.pavement_inspection_fresh,
        }


@dataclass(frozen=True)
class CapacitySnapshot:
    """Canonical Safe Usable Capacity snapshot reconciling per-bay and aggregate totals."""
    camera_id: str
    site_id: str
    layout_revision_id: Optional[str]
    layout_canonical_sha256: Optional[str]
    policy_config_sha256: str
    operational_gate: OperationalGate
    gate_reasons: List[str]
    evaluated_at: str
    decisions: Dict[str, BayCapacityDecision]

    # Reconciled Aggregate Totals
    total_bays: int
    physical_vacant_count: int
    usable_available_count: int
    occupied_count: int
    hazard_blocked_count: int
    approach_blocked_count: int
    vacant_unassessed_count: int
    unknown_count: int
    occluded_count: int
    inference_blocked_count: int

    snapshot_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "site_id": self.site_id,
            "layout_revision_id": self.layout_revision_id,
            "layout_canonical_sha256": self.layout_canonical_sha256,
            "policy_config_sha256": self.policy_config_sha256,
            "operational_gate": self.operational_gate.value,
            "gate_reasons": list(self.gate_reasons),
            "evaluated_at": self.evaluated_at,
            "decisions": {k: v.to_dict() for k, v in self.decisions.items()},
            "total_bays": self.total_bays,
            "physical_vacant_count": self.physical_vacant_count,
            "usable_available_count": self.usable_available_count,
            "occupied_count": self.occupied_count,
            "hazard_blocked_count": self.hazard_blocked_count,
            "approach_blocked_count": self.approach_blocked_count,
            "vacant_unassessed_count": self.vacant_unassessed_count,
            "unknown_count": self.unknown_count,
            "occluded_count": self.occluded_count,
            "inference_blocked_count": self.inference_blocked_count,
            "snapshot_sha256": self.snapshot_sha256,
        }


def _resolve_age_seconds(
    explicit_age: Optional[float],
    timestamp: Optional[Union[datetime, str, float]],
    reference_time: datetime,
) -> Optional[float]:
    """Deterministically resolve age in seconds from explicit age or timestamp."""
    if explicit_age is not None:
        if not math.isfinite(explicit_age) or explicit_age < 0:
            return -1.0  # Signals invalid
        return float(explicit_age)

    if timestamp is None:
        return None

    if isinstance(timestamp, (int, float)):
        if not math.isfinite(timestamp) or timestamp < 0:
            return -1.0
        ref_epoch = reference_time.timestamp()
        age = ref_epoch - float(timestamp)
        return age if age >= 0 else -1.0

    if isinstance(timestamp, str):
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            age = (reference_time - parsed).total_seconds()
            return age if age >= 0 else -1.0
        except Exception:
            return -1.0

    if isinstance(timestamp, datetime):
        dt = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)
        age = (reference_time - dt).total_seconds()
        return age if age >= 0 else -1.0

    return -1.0


def compute_snapshot_sha256(
    camera_id: str,
    site_id: str,
    layout_canonical_sha256: Optional[str],
    policy_config_sha256: str,
    operational_gate: str,
    decisions: Dict[str, BayCapacityDecision],
) -> str:
    """Compute deterministic canonical SHA-256 fingerprint for a capacity snapshot."""
    sorted_bay_items = []
    for bay_id in sorted(decisions.keys()):
        d = decisions[bay_id]
        sorted_bay_items.append({
            "bay_id": d.bay_id,
            "capacity_state": d.capacity_state.value,
            "is_usable": d.is_usable,
            "reason_codes": sorted(d.reason_codes),
            "raw_occupancy_state": d.raw_occupancy_state,
        })

    payload = {
        "camera_id": camera_id,
        "site_id": site_id,
        "layout_canonical_sha256": layout_canonical_sha256 or "",
        "policy_config_sha256": policy_config_sha256,
        "operational_gate": operational_gate,
        "bays": sorted_bay_items,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_bay_capacity(
    bay: BayOccupancyEvidenceInput,
    gate: OperationalGate,
    gate_reasons: Sequence[str],
    layout_verified: bool,
    layout_canonical_sha256: Optional[str],
    hazards: Sequence[HazardAssociationInput],
    inspection: Optional[PavementInspectionEvidence],
    config: CapacityPolicyConfig,
    reference_time: datetime,
    policy_config_sha256: str,
) -> BayCapacityDecision:
    """
    Pure deterministic evaluation of an individual parking space capacity decision.
    Enforces rules 1 through 12.
    """
    eval_iso = reference_time.isoformat()
    reasons: List[str] = []
    evidence_ids: Dict[str, Any] = {
        "job_id": bay.job_id,
        "occupancy_provenance": bay.provenance_sha256,
        "hazard_associations": [],
        "inspection_session_id": inspection.session_id if inspection else None,
        "layout_canonical_sha256": layout_canonical_sha256,
    }

    # Defensive input validation: bay_id and valid enums
    if not bay.bay_id or not bay.operator_label:
        return BayCapacityDecision(
            bay_id=bay.bay_id or "UNKNOWN",
            operator_label=bay.operator_label or "UNKNOWN",
            space_type=bay.space_type or "STANDARD",
            capacity_state=CapacityState.UNKNOWN,
            is_usable=False,
            reason_codes=[REASON_INVALID_INPUT_FAIL_CLOSED],
            evidence_ids=evidence_ids,
            policy_config_sha256=policy_config_sha256,
            evaluated_at=eval_iso,
            raw_occupancy_state=str(bay.occupancy_state),
            has_active_bay_hazard=False,
            has_active_approach_hazard=False,
            has_unverified_hazard=False,
            pavement_inspection_fresh=False,
        )

    # Audit & categorize hazards associated with this bay
    bay_hazards = [
        h for h in hazards
        if h.target_id == bay.bay_id
    ]

    has_active_bay_hazard = False
    has_active_approach_hazard = False
    has_unverified_hazard = False

    for h in bay_hazards:
        target = h.target_type.value if isinstance(h.target_type, HazardTarget) else str(h.target_type).upper()
        review = h.review_state.value if isinstance(h.review_state, HazardReviewState) else str(h.review_state).upper()
        lifecycle = h.lifecycle_state.value if isinstance(h.lifecycle_state, HazardLifecycleState) else str(h.lifecycle_state).upper()

        evidence_ids["hazard_associations"].append(h.to_dict())

        if review == "CONFIRMED":
            if lifecycle == "ACTIVE":
                if target == "BAY":
                    has_active_bay_hazard = True
                elif target == "APPROACH_ZONE":
                    has_active_approach_hazard = True
            else:
                # Rule 9: Resolved, expired and superseded hazards remain auditable but inactive
                if REASON_INACTIVE_HAZARD_LOGGED not in reasons:
                    reasons.append(REASON_INACTIVE_HAZARD_LOGGED)
        elif review in ("UNREVIEWED", "NEEDS_REVIEW"):
            # Rule 8: Unverified AI suggestions never independently change official capacity
            has_unverified_hazard = True
            if REASON_UNVERIFIED_HAZARD_IGNORED not in reasons:
                reasons.append(REASON_UNVERIFIED_HAZARD_IGNORED)

    # Rule 1: A blocked operational gate produces INFERENCE_BLOCKED and no capacity claim
    is_gate_blocked = (
        gate == OperationalGate.BLOCKED or
        (isinstance(gate, str) and str(gate).upper() == "BLOCKED")
    )
    if is_gate_blocked:
        reasons.append(REASON_GATE_BLOCKED)
        for gr in gate_reasons:
            if gr not in reasons:
                reasons.append(gr)
        return BayCapacityDecision(
            bay_id=bay.bay_id,
            operator_label=bay.operator_label,
            space_type=bay.space_type,
            capacity_state=CapacityState.INFERENCE_BLOCKED,
            is_usable=False,
            reason_codes=reasons,
            evidence_ids=evidence_ids,
            policy_config_sha256=policy_config_sha256,
            evaluated_at=eval_iso,
            raw_occupancy_state=str(bay.occupancy_state),
            has_active_bay_hazard=has_active_bay_hazard,
            has_active_approach_hazard=has_active_approach_hazard,
            has_unverified_hazard=has_unverified_hazard,
            pavement_inspection_fresh=False,
        )

    # Layout verification check
    if config.require_verified_layout and (not layout_verified or not layout_canonical_sha256):
        reasons.append(REASON_LAYOUT_NOT_VERIFIED)
        return BayCapacityDecision(
            bay_id=bay.bay_id,
            operator_label=bay.operator_label,
            space_type=bay.space_type,
            capacity_state=CapacityState.UNKNOWN,
            is_usable=False,
            reason_codes=reasons,
            evidence_ids=evidence_ids,
            policy_config_sha256=policy_config_sha256,
            evaluated_at=eval_iso,
            raw_occupancy_state=str(bay.occupancy_state),
            has_active_bay_hazard=has_active_bay_hazard,
            has_active_approach_hazard=has_active_approach_hazard,
            has_unverified_hazard=has_unverified_hazard,
            pavement_inspection_fresh=False,
        )

    # Evaluate Occupancy Evidence Freshness
    raw_occ = str(bay.occupancy_state).upper()
    occ_age = _resolve_age_seconds(bay.evidence_age_seconds, bay.occupancy_timestamp, reference_time)

    if occ_age is None:
        reasons.append(REASON_OCCUPANCY_EVIDENCE_MISSING)
        is_occ_fresh = False
    elif occ_age < 0:
        reasons.append(REASON_INVALID_INPUT_FAIL_CLOSED)
        is_occ_fresh = False
    elif occ_age > config.max_occupancy_age_seconds:
        reasons.append(REASON_OCCUPANCY_EVIDENCE_STALE)
        is_occ_fresh = False
    else:
        is_occ_fresh = True

    # Rule 2: UNKNOWN and OCCLUDED occupancy are never counted as available
    if raw_occ == "OCCLUDED":
        reasons.append(REASON_OCCUPANCY_OCCLUDED)
        return BayCapacityDecision(
            bay_id=bay.bay_id,
            operator_label=bay.operator_label,
            space_type=bay.space_type,
            capacity_state=CapacityState.OCCLUDED,
            is_usable=False,
            reason_codes=reasons,
            evidence_ids=evidence_ids,
            policy_config_sha256=policy_config_sha256,
            evaluated_at=eval_iso,
            raw_occupancy_state=raw_occ,
            has_active_bay_hazard=has_active_bay_hazard,
            has_active_approach_hazard=has_active_approach_hazard,
            has_unverified_hazard=has_unverified_hazard,
            pavement_inspection_fresh=False,
        )

    if raw_occ == "UNKNOWN" or not is_occ_fresh:
        if raw_occ == "UNKNOWN" and REASON_OCCUPANCY_UNKNOWN not in reasons:
            reasons.append(REASON_OCCUPANCY_UNKNOWN)
        return BayCapacityDecision(
            bay_id=bay.bay_id,
            operator_label=bay.operator_label,
            space_type=bay.space_type,
            capacity_state=CapacityState.UNKNOWN,
            is_usable=False,
            reason_codes=reasons,
            evidence_ids=evidence_ids,
            policy_config_sha256=policy_config_sha256,
            evaluated_at=eval_iso,
            raw_occupancy_state=raw_occ,
            has_active_bay_hazard=has_active_bay_hazard,
            has_active_approach_hazard=has_active_approach_hazard,
            has_unverified_hazard=has_unverified_hazard,
            pavement_inspection_fresh=False,
        )

    # Rule 3: OCCUPIED bays are never counted as available
    if raw_occ == "OCCUPIED":
        reasons.append(REASON_OCCUPANCY_OCCUPIED)
        if has_active_bay_hazard:
            reasons.append(REASON_BAY_HAZARD_ACTIVE)
        if has_active_approach_hazard:
            reasons.append(REASON_APPROACH_HAZARD_ACTIVE)

        return BayCapacityDecision(
            bay_id=bay.bay_id,
            operator_label=bay.operator_label,
            space_type=bay.space_type,
            capacity_state=CapacityState.OCCUPIED,
            is_usable=False,
            reason_codes=reasons,
            evidence_ids=evidence_ids,
            policy_config_sha256=policy_config_sha256,
            evaluated_at=eval_iso,
            raw_occupancy_state=raw_occ,
            has_active_bay_hazard=has_active_bay_hazard,
            has_active_approach_hazard=has_active_approach_hazard,
            has_unverified_hazard=has_unverified_hazard,
            pavement_inspection_fresh=False,
        )

    # If not VACANT (or recognized free state), fail closed to UNKNOWN
    if raw_occ not in ("VACANT", "FREE"):
        reasons.append(REASON_INVALID_INPUT_FAIL_CLOSED)
        return BayCapacityDecision(
            bay_id=bay.bay_id,
            operator_label=bay.operator_label,
            space_type=bay.space_type,
            capacity_state=CapacityState.UNKNOWN,
            is_usable=False,
            reason_codes=reasons,
            evidence_ids=evidence_ids,
            policy_config_sha256=policy_config_sha256,
            evaluated_at=eval_iso,
            raw_occupancy_state=raw_occ,
            has_active_bay_hazard=has_active_bay_hazard,
            has_active_approach_hazard=has_active_approach_hazard,
            has_unverified_hazard=has_unverified_hazard,
            pavement_inspection_fresh=False,
        )

    # The bay is physically VACANT.
    # Check pavement inspection evidence freshness
    pavement_fresh = False
    if inspection is None:
        reasons.append(REASON_INSPECTION_MISSING)
    elif inspection.is_fresh is False:
        reasons.append(REASON_INSPECTION_STALE)
    elif inspection.is_fresh is True:
        pavement_fresh = True
    else:
        # Resolve from age or timestamp
        insp_age = _resolve_age_seconds(
            inspection.freshness_age_seconds,
            inspection.inspection_timestamp,
            reference_time,
        )
        if insp_age is None:
            reasons.append(REASON_INSPECTION_MISSING)
        elif insp_age < 0 or insp_age > config.max_inspection_age_seconds:
            reasons.append(REASON_INSPECTION_STALE)
        else:
            pavement_fresh = True

    # Rule 6: Active, human-verified bay hazards produce HAZARD_BLOCKED
    if has_active_bay_hazard:
        reasons.append(REASON_BAY_HAZARD_ACTIVE)
        if has_active_approach_hazard:
            reasons.append(REASON_APPROACH_HAZARD_ACTIVE)

        return BayCapacityDecision(
            bay_id=bay.bay_id,
            operator_label=bay.operator_label,
            space_type=bay.space_type,
            capacity_state=CapacityState.HAZARD_BLOCKED,
            is_usable=False,
            reason_codes=reasons,
            evidence_ids=evidence_ids,
            policy_config_sha256=policy_config_sha256,
            evaluated_at=eval_iso,
            raw_occupancy_state=raw_occ,
            has_active_bay_hazard=True,
            has_active_approach_hazard=has_active_approach_hazard,
            has_unverified_hazard=has_unverified_hazard,
            pavement_inspection_fresh=pavement_fresh,
        )

    # Rule 7: Active, human-verified approach hazards produce APPROACH_BLOCKED
    if has_active_approach_hazard:
        reasons.append(REASON_APPROACH_HAZARD_ACTIVE)
        return BayCapacityDecision(
            bay_id=bay.bay_id,
            operator_label=bay.operator_label,
            space_type=bay.space_type,
            capacity_state=CapacityState.APPROACH_BLOCKED,
            is_usable=False,
            reason_codes=reasons,
            evidence_ids=evidence_ids,
            policy_config_sha256=policy_config_sha256,
            evaluated_at=eval_iso,
            raw_occupancy_state=raw_occ,
            has_active_bay_hazard=False,
            has_active_approach_hazard=True,
            has_unverified_hazard=has_unverified_hazard,
            pavement_inspection_fresh=pavement_fresh,
        )

    # Rule 5: Missing or stale inspection evidence produces VACANT_UNASSESSED
    if not pavement_fresh:
        return BayCapacityDecision(
            bay_id=bay.bay_id,
            operator_label=bay.operator_label,
            space_type=bay.space_type,
            capacity_state=CapacityState.VACANT_UNASSESSED,
            is_usable=False,
            reason_codes=reasons,
            evidence_ids=evidence_ids,
            policy_config_sha256=policy_config_sha256,
            evaluated_at=eval_iso,
            raw_occupancy_state=raw_occ,
            has_active_bay_hazard=False,
            has_active_approach_hazard=False,
            has_unverified_hazard=has_unverified_hazard,
            pavement_inspection_fresh=False,
        )

    # Rule 4: A vacant bay becomes USABLE_AVAILABLE only when all conditions pass
    reasons.append(REASON_USABLE_AVAILABLE)
    return BayCapacityDecision(
        bay_id=bay.bay_id,
        operator_label=bay.operator_label,
        space_type=bay.space_type,
        capacity_state=CapacityState.USABLE_AVAILABLE,
        is_usable=True,
        reason_codes=reasons,
        evidence_ids=evidence_ids,
        policy_config_sha256=policy_config_sha256,
        evaluated_at=eval_iso,
        raw_occupancy_state=raw_occ,
        has_active_bay_hazard=False,
        has_active_approach_hazard=False,
        has_unverified_hazard=has_unverified_hazard,
        pavement_inspection_fresh=True,
    )


def evaluate_safe_usable_capacity(
    camera_id: str,
    site_id: str,
    bays: Sequence[BayOccupancyEvidenceInput],
    gate: OperationalGate,
    gate_reasons: Sequence[str] = (),
    layout_verified: bool = True,
    layout_revision_id: Optional[str] = None,
    layout_canonical_sha256: Optional[str] = None,
    hazards: Sequence[HazardAssociationInput] = (),
    inspection: Optional[PavementInspectionEvidence] = None,
    config: Optional[CapacityPolicyConfig] = None,
    reference_time: Optional[datetime] = None,
) -> CapacitySnapshot:
    """
    Evaluate safe usable capacity for all configured bays of a camera.
    Returns a deterministic, fully reconciled CapacitySnapshot.
    """
    cfg = config or CapacityPolicyConfig()
    cfg_sha256 = cfg.compute_sha256()
    ref_time = reference_time or datetime.now(timezone.utc)
    eval_iso = ref_time.isoformat()

    decisions: Dict[str, BayCapacityDecision] = {}

    for bay in bays:
        dec = evaluate_bay_capacity(
            bay=bay,
            gate=gate,
            gate_reasons=gate_reasons,
            layout_verified=layout_verified,
            layout_canonical_sha256=layout_canonical_sha256,
            hazards=hazards,
            inspection=inspection,
            config=cfg,
            reference_time=ref_time,
            policy_config_sha256=cfg_sha256,
        )
        decisions[bay.bay_id] = dec

    # Reconcile Aggregate Counts
    total_bays = len(decisions)
    usable_available_count = 0
    occupied_count = 0
    hazard_blocked_count = 0
    approach_blocked_count = 0
    vacant_unassessed_count = 0
    unknown_count = 0
    occluded_count = 0
    inference_blocked_count = 0

    physical_vacant_count = 0

    for d in decisions.values():
        st = d.capacity_state
        if st == CapacityState.USABLE_AVAILABLE:
            usable_available_count += 1
            physical_vacant_count += 1
        elif st == CapacityState.OCCUPIED:
            occupied_count += 1
        elif st == CapacityState.HAZARD_BLOCKED:
            hazard_blocked_count += 1
            physical_vacant_count += 1
        elif st == CapacityState.APPROACH_BLOCKED:
            approach_blocked_count += 1
            physical_vacant_count += 1
        elif st == CapacityState.VACANT_UNASSESSED:
            vacant_unassessed_count += 1
            physical_vacant_count += 1
        elif st == CapacityState.UNKNOWN:
            unknown_count += 1
            if d.raw_occupancy_state in ("VACANT", "FREE"):
                physical_vacant_count += 1
        elif st == CapacityState.OCCLUDED:
            occluded_count += 1
        elif st == CapacityState.INFERENCE_BLOCKED:
            inference_blocked_count += 1
            if d.raw_occupancy_state in ("VACANT", "FREE"):
                physical_vacant_count += 1

    # Rule 10: Aggregated totals must reconcile exactly with per-bay decisions
    reconciled_sum = (
        usable_available_count +
        occupied_count +
        hazard_blocked_count +
        approach_blocked_count +
        vacant_unassessed_count +
        unknown_count +
        occluded_count +
        inference_blocked_count
    )
    if reconciled_sum != total_bays:
        raise ValueError(
            f"Aggregation reconciliation mismatch: sum({reconciled_sum}) != total_bays({total_bays})"
        )

    gate_val = gate if isinstance(gate, OperationalGate) else OperationalGate(str(gate).upper())

    snapshot_sha = compute_snapshot_sha256(
        camera_id=camera_id,
        site_id=site_id,
        layout_canonical_sha256=layout_canonical_sha256,
        policy_config_sha256=cfg_sha256,
        operational_gate=gate_val.value,
        decisions=decisions,
    )

    return CapacitySnapshot(
        camera_id=camera_id,
        site_id=site_id,
        layout_revision_id=layout_revision_id,
        layout_canonical_sha256=layout_canonical_sha256,
        policy_config_sha256=cfg_sha256,
        operational_gate=gate_val,
        gate_reasons=list(gate_reasons),
        evaluated_at=eval_iso,
        decisions=decisions,
        total_bays=total_bays,
        physical_vacant_count=physical_vacant_count,
        usable_available_count=usable_available_count,
        occupied_count=occupied_count,
        hazard_blocked_count=hazard_blocked_count,
        approach_blocked_count=approach_blocked_count,
        vacant_unassessed_count=vacant_unassessed_count,
        unknown_count=unknown_count,
        occluded_count=occluded_count,
        inference_blocked_count=inference_blocked_count,
        snapshot_sha256=snapshot_sha,
    )
