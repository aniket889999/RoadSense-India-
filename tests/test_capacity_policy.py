"""Unit tests for Phase 3A Safe Usable Parking Capacity policy, contracts, and rules."""

from datetime import datetime, timedelta, timezone
import pytest

from src.parking.capacity_policy import (
    BayCapacityDecision,
    BayOccupancyEvidenceInput,
    CapacityPolicyConfig,
    CapacitySnapshot,
    CapacityState,
    HazardAssociationInput,
    HazardLifecycleState,
    HazardReviewState,
    HazardTarget,
    OperationalGate,
    PavementInspectionEvidence,
    REASON_APPROACH_HAZARD_ACTIVE,
    REASON_BAY_HAZARD_ACTIVE,
    REASON_GATE_BLOCKED,
    REASON_INACTIVE_HAZARD_LOGGED,
    REASON_INSPECTION_MISSING,
    REASON_INSPECTION_STALE,
    REASON_INVALID_INPUT_FAIL_CLOSED,
    REASON_LAYOUT_NOT_VERIFIED,
    REASON_OCCUPANCY_OCCLUDED,
    REASON_OCCUPANCY_OCCUPIED,
    REASON_OCCUPANCY_EVIDENCE_STALE,
    REASON_OCCUPANCY_UNKNOWN,
    REASON_UNVERIFIED_HAZARD_IGNORED,
    REASON_USABLE_AVAILABLE,
    compute_snapshot_sha256,
    evaluate_bay_capacity,
    evaluate_safe_usable_capacity,
)


@pytest.fixture
def base_time() -> datetime:
    return datetime(2026, 9, 13, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def standard_config() -> CapacityPolicyConfig:
    return CapacityPolicyConfig(
        policy_version="3A.1",
        max_occupancy_age_seconds=300.0,
        max_inspection_age_seconds=86400.0 * 30.0,
        require_verified_layout=True,
    )


@pytest.fixture
def fresh_inspection(base_time: datetime) -> PavementInspectionEvidence:
    return PavementInspectionEvidence(
        session_id="insp_sess_001",
        inspection_timestamp=base_time - timedelta(days=2),
        freshness_age_seconds=86400.0 * 2.0,
        is_fresh=True,
        inspection_notes="Regular mobile inspection passed cleanly",
    )


def test_rule_1_blocked_operational_gate_produces_inference_blocked(
    base_time: datetime,
    standard_config: CapacityPolicyConfig,
    fresh_inspection: PavementInspectionEvidence,
):
    """Rule 1: A blocked operational gate produces INFERENCE_BLOCKED and no capacity claim."""
    bay = BayOccupancyEvidenceInput(
        bay_id="B1",
        operator_label="Bay 1",
        occupancy_state="VACANT",
        evidence_age_seconds=10.0,
    )

    dec = evaluate_bay_capacity(
        bay=bay,
        gate=OperationalGate.BLOCKED,
        gate_reasons=["DRIFT_EXCEEDS_OPERATIONAL_THRESHOLD"],
        layout_verified=True,
        layout_canonical_sha256="canonical_hash_123",
        hazards=[],
        inspection=fresh_inspection,
        config=standard_config,
        reference_time=base_time,
        policy_config_sha256=standard_config.compute_sha256(),
    )

    assert dec.capacity_state == CapacityState.INFERENCE_BLOCKED
    assert dec.is_usable is False
    assert REASON_GATE_BLOCKED in dec.reason_codes
    assert "DRIFT_EXCEEDS_OPERATIONAL_THRESHOLD" in dec.reason_codes


def test_rule_2_unknown_and_occluded_occupancy_never_available(
    base_time: datetime,
    standard_config: CapacityPolicyConfig,
    fresh_inspection: PavementInspectionEvidence,
):
    """Rule 2: UNKNOWN and OCCLUDED occupancy are never counted as available."""
    # Test OCCLUDED
    bay_occluded = BayOccupancyEvidenceInput(
        bay_id="B1",
        operator_label="Bay 1",
        occupancy_state="OCCLUDED",
        evidence_age_seconds=10.0,
    )
    dec_occ = evaluate_bay_capacity(
        bay=bay_occluded,
        gate=OperationalGate.ALLOWED,
        gate_reasons=[],
        layout_verified=True,
        layout_canonical_sha256="hash",
        hazards=[],
        inspection=fresh_inspection,
        config=standard_config,
        reference_time=base_time,
        policy_config_sha256=standard_config.compute_sha256(),
    )
    assert dec_occ.capacity_state == CapacityState.OCCLUDED
    assert dec_occ.is_usable is False
    assert REASON_OCCUPANCY_OCCLUDED in dec_occ.reason_codes

    # Test UNKNOWN
    bay_unknown = BayOccupancyEvidenceInput(
        bay_id="B2",
        operator_label="Bay 2",
        occupancy_state="UNKNOWN",
        evidence_age_seconds=10.0,
    )
    dec_unk = evaluate_bay_capacity(
        bay=bay_unknown,
        gate=OperationalGate.ALLOWED,
        gate_reasons=[],
        layout_verified=True,
        layout_canonical_sha256="hash",
        hazards=[],
        inspection=fresh_inspection,
        config=standard_config,
        reference_time=base_time,
        policy_config_sha256=standard_config.compute_sha256(),
    )
    assert dec_unk.capacity_state == CapacityState.UNKNOWN
    assert dec_unk.is_usable is False
    assert REASON_OCCUPANCY_UNKNOWN in dec_unk.reason_codes


def test_rule_3_occupied_bays_never_available(
    base_time: datetime,
    standard_config: CapacityPolicyConfig,
    fresh_inspection: PavementInspectionEvidence,
):
    """Rule 3: OCCUPIED bays are never counted as available."""
    bay_occupied = BayOccupancyEvidenceInput(
        bay_id="B1",
        operator_label="Bay 1",
        occupancy_state="OCCUPIED",
        evidence_age_seconds=15.0,
    )
    # Even if an active hazard exists, it is recorded in reasons while capacity is OCCUPIED
    hazard = HazardAssociationInput(
        association_id="assoc_1",
        hazard_id="haz_pothole_1",
        target_type=HazardTarget.BAY,
        target_id="B1",
        review_state=HazardReviewState.CONFIRMED,
        lifecycle_state=HazardLifecycleState.ACTIVE,
    )

    dec = evaluate_bay_capacity(
        bay=bay_occupied,
        gate=OperationalGate.ALLOWED,
        gate_reasons=[],
        layout_verified=True,
        layout_canonical_sha256="hash",
        hazards=[hazard],
        inspection=fresh_inspection,
        config=standard_config,
        reference_time=base_time,
        policy_config_sha256=standard_config.compute_sha256(),
    )
    assert dec.capacity_state == CapacityState.OCCUPIED
    assert dec.is_usable is False
    assert REASON_OCCUPANCY_OCCUPIED in dec.reason_codes
    assert REASON_BAY_HAZARD_ACTIVE in dec.reason_codes
    assert dec.has_active_bay_hazard is True


def test_rule_4_usable_available_happy_path(
    base_time: datetime,
    standard_config: CapacityPolicyConfig,
    fresh_inspection: PavementInspectionEvidence,
):
    """Rule 4: A vacant bay becomes USABLE_AVAILABLE when all criteria are met."""
    bay = BayOccupancyEvidenceInput(
        bay_id="B1",
        operator_label="Bay 1",
        occupancy_state="VACANT",
        evidence_age_seconds=20.0,
        provenance_sha256="evidence_sha_123",
        job_id="job_001",
    )

    dec = evaluate_bay_capacity(
        bay=bay,
        gate=OperationalGate.ALLOWED,
        gate_reasons=[],
        layout_verified=True,
        layout_canonical_sha256="canonical_123",
        hazards=[],
        inspection=fresh_inspection,
        config=standard_config,
        reference_time=base_time,
        policy_config_sha256=standard_config.compute_sha256(),
    )

    assert dec.capacity_state == CapacityState.USABLE_AVAILABLE
    assert dec.is_usable is True
    assert REASON_USABLE_AVAILABLE in dec.reason_codes
    assert dec.pavement_inspection_fresh is True
    assert dec.has_active_bay_hazard is False
    assert dec.has_active_approach_hazard is False


def test_rule_5_missing_or_stale_inspection_produces_vacant_unassessed(
    base_time: datetime,
    standard_config: CapacityPolicyConfig,
):
    """Rule 5: Missing or stale inspection evidence produces VACANT_UNASSESSED."""
    bay = BayOccupancyEvidenceInput(
        bay_id="B1",
        operator_label="Bay 1",
        occupancy_state="VACANT",
        evidence_age_seconds=10.0,
    )

    # 1. Missing inspection evidence
    dec_missing = evaluate_bay_capacity(
        bay=bay,
        gate=OperationalGate.ALLOWED,
        gate_reasons=[],
        layout_verified=True,
        layout_canonical_sha256="hash",
        hazards=[],
        inspection=None,
        config=standard_config,
        reference_time=base_time,
        policy_config_sha256=standard_config.compute_sha256(),
    )
    assert dec_missing.capacity_state == CapacityState.VACANT_UNASSESSED
    assert dec_missing.is_usable is False
    assert REASON_INSPECTION_MISSING in dec_missing.reason_codes

    # 2. Stale inspection evidence (older than 30 days)
    stale_inspection = PavementInspectionEvidence(
        session_id="old_sess",
        freshness_age_seconds=86400.0 * 35.0,  # 35 days
    )
    dec_stale = evaluate_bay_capacity(
        bay=bay,
        gate=OperationalGate.ALLOWED,
        gate_reasons=[],
        layout_verified=True,
        layout_canonical_sha256="hash",
        hazards=[],
        inspection=stale_inspection,
        config=standard_config,
        reference_time=base_time,
        policy_config_sha256=standard_config.compute_sha256(),
    )
    assert dec_stale.capacity_state == CapacityState.VACANT_UNASSESSED
    assert dec_stale.is_usable is False
    assert REASON_INSPECTION_STALE in dec_stale.reason_codes


def test_rule_6_active_human_verified_bay_hazard_produces_hazard_blocked(
    base_time: datetime,
    standard_config: CapacityPolicyConfig,
    fresh_inspection: PavementInspectionEvidence,
):
    """Rule 6: Active, human-verified bay hazards produce HAZARD_BLOCKED."""
    bay = BayOccupancyEvidenceInput(
        bay_id="B1",
        operator_label="Bay 1",
        occupancy_state="VACANT",
        evidence_age_seconds=10.0,
    )
    hazard = HazardAssociationInput(
        association_id="assoc_1",
        hazard_id="haz_1",
        target_type=HazardTarget.BAY,
        target_id="B1",
        review_state=HazardReviewState.CONFIRMED,
        lifecycle_state=HazardLifecycleState.ACTIVE,
        severity_label="HIGH",
    )

    dec = evaluate_bay_capacity(
        bay=bay,
        gate=OperationalGate.ALLOWED,
        gate_reasons=[],
        layout_verified=True,
        layout_canonical_sha256="hash",
        hazards=[hazard],
        inspection=fresh_inspection,
        config=standard_config,
        reference_time=base_time,
        policy_config_sha256=standard_config.compute_sha256(),
    )

    assert dec.capacity_state == CapacityState.HAZARD_BLOCKED
    assert dec.is_usable is False
    assert REASON_BAY_HAZARD_ACTIVE in dec.reason_codes
    assert dec.has_active_bay_hazard is True


def test_rule_7_active_human_verified_approach_hazard_produces_approach_blocked(
    base_time: datetime,
    standard_config: CapacityPolicyConfig,
    fresh_inspection: PavementInspectionEvidence,
):
    """Rule 7: Active, human-verified approach hazards produce APPROACH_BLOCKED."""
    bay = BayOccupancyEvidenceInput(
        bay_id="B1",
        operator_label="Bay 1",
        occupancy_state="VACANT",
        evidence_age_seconds=10.0,
    )
    hazard = HazardAssociationInput(
        association_id="assoc_2",
        hazard_id="haz_2",
        target_type=HazardTarget.APPROACH_ZONE,
        target_id="B1",
        review_state=HazardReviewState.CONFIRMED,
        lifecycle_state=HazardLifecycleState.ACTIVE,
    )

    dec = evaluate_bay_capacity(
        bay=bay,
        gate=OperationalGate.ALLOWED,
        gate_reasons=[],
        layout_verified=True,
        layout_canonical_sha256="hash",
        hazards=[hazard],
        inspection=fresh_inspection,
        config=standard_config,
        reference_time=base_time,
        policy_config_sha256=standard_config.compute_sha256(),
    )

    assert dec.capacity_state == CapacityState.APPROACH_BLOCKED
    assert dec.is_usable is False
    assert REASON_APPROACH_HAZARD_ACTIVE in dec.reason_codes
    assert dec.has_active_approach_hazard is True
    assert dec.has_active_bay_hazard is False


def test_rule_8_unverified_ai_suggestions_never_block_official_capacity(
    base_time: datetime,
    standard_config: CapacityPolicyConfig,
    fresh_inspection: PavementInspectionEvidence,
):
    """Rule 8: Unverified AI suggestions never independently change official capacity."""
    bay = BayOccupancyEvidenceInput(
        bay_id="B1",
        operator_label="Bay 1",
        occupancy_state="VACANT",
        evidence_age_seconds=10.0,
    )
    # Candidate defect that is UNREVIEWED
    ai_suggestion = HazardAssociationInput(
        association_id="assoc_unreviewed",
        hazard_id="haz_raw_yolo_1",
        target_type=HazardTarget.BAY,
        target_id="B1",
        review_state=HazardReviewState.UNREVIEWED,
        lifecycle_state=HazardLifecycleState.ACTIVE,
    )
    # Candidate defect in NEEDS_REVIEW triage
    ai_needs_review = HazardAssociationInput(
        association_id="assoc_needs_review",
        hazard_id="haz_raw_yolo_2",
        target_type=HazardTarget.APPROACH_ZONE,
        target_id="B1",
        review_state=HazardReviewState.NEEDS_REVIEW,
        lifecycle_state=HazardLifecycleState.ACTIVE,
    )

    dec = evaluate_bay_capacity(
        bay=bay,
        gate=OperationalGate.ALLOWED,
        gate_reasons=[],
        layout_verified=True,
        layout_canonical_sha256="hash",
        hazards=[ai_suggestion, ai_needs_review],
        inspection=fresh_inspection,
        config=standard_config,
        reference_time=base_time,
        policy_config_sha256=standard_config.compute_sha256(),
    )

    # Officially, still USABLE_AVAILABLE because human has not confirmed the hazard
    assert dec.capacity_state == CapacityState.USABLE_AVAILABLE
    assert dec.is_usable is True
    assert REASON_UNVERIFIED_HAZARD_IGNORED in dec.reason_codes
    assert dec.has_unverified_hazard is True
    assert dec.has_active_bay_hazard is False
    assert dec.has_active_approach_hazard is False


def test_rule_9_resolved_expired_superseded_hazards_remain_auditable_but_inactive(
    base_time: datetime,
    standard_config: CapacityPolicyConfig,
    fresh_inspection: PavementInspectionEvidence,
):
    """Rule 9: Resolved, expired and superseded hazards remain auditable but inactive."""
    bay = BayOccupancyEvidenceInput(
        bay_id="B1",
        operator_label="Bay 1",
        occupancy_state="VACANT",
        evidence_age_seconds=10.0,
    )

    for state in (
        HazardLifecycleState.RESOLVED,
        HazardLifecycleState.EXPIRED,
        HazardLifecycleState.SUPERSEDED,
        HazardLifecycleState.MITIGATED,
    ):
        inactive_hazard = HazardAssociationInput(
            association_id=f"assoc_{state.value}",
            hazard_id=f"haz_{state.value}",
            target_type=HazardTarget.BAY,
            target_id="B1",
            review_state=HazardReviewState.CONFIRMED,
            lifecycle_state=state,
        )

        dec = evaluate_bay_capacity(
            bay=bay,
            gate=OperationalGate.ALLOWED,
            gate_reasons=[],
            layout_verified=True,
            layout_canonical_sha256="hash",
            hazards=[inactive_hazard],
            inspection=fresh_inspection,
            config=standard_config,
            reference_time=base_time,
            policy_config_sha256=standard_config.compute_sha256(),
        )

        assert dec.capacity_state == CapacityState.USABLE_AVAILABLE
        assert dec.is_usable is True
        assert REASON_INACTIVE_HAZARD_LOGGED in dec.reason_codes
        assert dec.has_active_bay_hazard is False


def test_rule_10_and_snapshot_deterministic_aggregation(
    base_time: datetime,
    standard_config: CapacityPolicyConfig,
    fresh_inspection: PavementInspectionEvidence,
):
    """Rule 10: Aggregated totals reconcile exactly with per-bay decisions."""
    # Set up 8 bays covering every possible state
    bays = [
        # 1. USABLE_AVAILABLE
        BayOccupancyEvidenceInput(bay_id="B1", operator_label="B1", occupancy_state="VACANT", evidence_age_seconds=10.0),
        # 2. OCCUPIED
        BayOccupancyEvidenceInput(bay_id="B2", operator_label="B2", occupancy_state="OCCUPIED", evidence_age_seconds=10.0),
        # 3. HAZARD_BLOCKED
        BayOccupancyEvidenceInput(bay_id="B3", operator_label="B3", occupancy_state="VACANT", evidence_age_seconds=10.0),
        # 4. APPROACH_BLOCKED
        BayOccupancyEvidenceInput(bay_id="B4", operator_label="B4", occupancy_state="VACANT", evidence_age_seconds=10.0),
        # 5. VACANT_UNASSESSED (will be tested in sub-call without inspection, or separate bay)
        BayOccupancyEvidenceInput(bay_id="B5", operator_label="B5", occupancy_state="VACANT", evidence_age_seconds=10.0),
        # 6. OCCLUDED
        BayOccupancyEvidenceInput(bay_id="B6", operator_label="B6", occupancy_state="OCCLUDED", evidence_age_seconds=10.0),
        # 7. UNKNOWN
        BayOccupancyEvidenceInput(bay_id="B7", operator_label="B7", occupancy_state="UNKNOWN", evidence_age_seconds=10.0),
    ]

    hazards = [
        # B3 has active bay hazard
        HazardAssociationInput(
            association_id="a3",
            hazard_id="h3",
            target_type=HazardTarget.BAY,
            target_id="B3",
            review_state=HazardReviewState.CONFIRMED,
            lifecycle_state=HazardLifecycleState.ACTIVE,
        ),
        # B4 has active approach hazard
        HazardAssociationInput(
            association_id="a4",
            hazard_id="h4",
            target_type=HazardTarget.APPROACH_ZONE,
            target_id="B4",
            review_state=HazardReviewState.CONFIRMED,
            lifecycle_state=HazardLifecycleState.ACTIVE,
        ),
    ]

    snapshot = evaluate_safe_usable_capacity(
        camera_id="cam_main",
        site_id="site_1",
        bays=bays,
        gate=OperationalGate.ALLOWED,
        gate_reasons=[],
        layout_verified=True,
        layout_canonical_sha256="canonical_sha_abc",
        hazards=hazards,
        inspection=fresh_inspection,
        config=standard_config,
        reference_time=base_time,
    )

    assert snapshot.total_bays == 7
    assert snapshot.usable_available_count == 2  # B1, B5
    assert snapshot.occupied_count == 1          # B2
    assert snapshot.hazard_blocked_count == 1    # B3
    assert snapshot.approach_blocked_count == 1  # B4
    assert snapshot.vacant_unassessed_count == 0
    assert snapshot.occluded_count == 1          # B6
    assert snapshot.unknown_count == 1           # B7
    assert snapshot.inference_blocked_count == 0

    # Total reconciliation
    sum_counts = (
        snapshot.usable_available_count +
        snapshot.occupied_count +
        snapshot.hazard_blocked_count +
        snapshot.approach_blocked_count +
        snapshot.vacant_unassessed_count +
        snapshot.unknown_count +
        snapshot.occluded_count +
        snapshot.inference_blocked_count
    )
    assert sum_counts == snapshot.total_bays

    # Determinism: repeating with identical inputs generates identical snapshot_sha256
    snapshot2 = evaluate_safe_usable_capacity(
        camera_id="cam_main",
        site_id="site_1",
        bays=bays,
        gate=OperationalGate.ALLOWED,
        gate_reasons=[],
        layout_verified=True,
        layout_canonical_sha256="canonical_sha_abc",
        hazards=hazards,
        inspection=fresh_inspection,
        config=standard_config,
        reference_time=base_time,
    )
    assert snapshot.snapshot_sha256 == snapshot2.snapshot_sha256


def test_stale_occupancy_evidence_fails_closed(
    base_time: datetime,
    standard_config: CapacityPolicyConfig,
    fresh_inspection: PavementInspectionEvidence,
):
    """Occupancy evidence older than max_occupancy_age_seconds fails closed to UNKNOWN."""
    bay_stale = BayOccupancyEvidenceInput(
        bay_id="B1",
        operator_label="B1",
        occupancy_state="VACANT",
        evidence_age_seconds=301.0,  # 1s over 300s limit
    )

    dec = evaluate_bay_capacity(
        bay=bay_stale,
        gate=OperationalGate.ALLOWED,
        gate_reasons=[],
        layout_verified=True,
        layout_canonical_sha256="hash",
        hazards=[],
        inspection=fresh_inspection,
        config=standard_config,
        reference_time=base_time,
        policy_config_sha256=standard_config.compute_sha256(),
    )

    assert dec.capacity_state == CapacityState.UNKNOWN
    assert dec.is_usable is False
    assert REASON_OCCUPANCY_EVIDENCE_STALE in dec.reason_codes


def test_conflicting_bay_and_approach_hazards_precedence(
    base_time: datetime,
    standard_config: CapacityPolicyConfig,
    fresh_inspection: PavementInspectionEvidence,
):
    """When both bay hazard and approach hazard exist, bay hazard takes direct precedence and both reasons logged."""
    bay = BayOccupancyEvidenceInput(
        bay_id="B1",
        operator_label="B1",
        occupancy_state="VACANT",
        evidence_age_seconds=10.0,
    )
    h_bay = HazardAssociationInput(
        association_id="a_bay",
        hazard_id="h_bay",
        target_type=HazardTarget.BAY,
        target_id="B1",
        review_state=HazardReviewState.CONFIRMED,
        lifecycle_state=HazardLifecycleState.ACTIVE,
    )
    h_app = HazardAssociationInput(
        association_id="a_app",
        hazard_id="h_app",
        target_type=HazardTarget.APPROACH_ZONE,
        target_id="B1",
        review_state=HazardReviewState.CONFIRMED,
        lifecycle_state=HazardLifecycleState.ACTIVE,
    )

    dec = evaluate_bay_capacity(
        bay=bay,
        gate=OperationalGate.ALLOWED,
        gate_reasons=[],
        layout_verified=True,
        layout_canonical_sha256="hash",
        hazards=[h_bay, h_app],
        inspection=fresh_inspection,
        config=standard_config,
        reference_time=base_time,
        policy_config_sha256=standard_config.compute_sha256(),
    )

    assert dec.capacity_state == CapacityState.HAZARD_BLOCKED
    assert dec.is_usable is False
    assert REASON_BAY_HAZARD_ACTIVE in dec.reason_codes
    assert REASON_APPROACH_HAZARD_ACTIVE in dec.reason_codes
    assert dec.has_active_bay_hazard is True
    assert dec.has_active_approach_hazard is True


def test_unverified_layout_fails_closed(
    base_time: datetime,
    standard_config: CapacityPolicyConfig,
    fresh_inspection: PavementInspectionEvidence,
):
    """If layout is not verified or missing canonical SHA, fails closed to UNKNOWN."""
    bay = BayOccupancyEvidenceInput(
        bay_id="B1",
        operator_label="B1",
        occupancy_state="VACANT",
        evidence_age_seconds=10.0,
    )

    dec = evaluate_bay_capacity(
        bay=bay,
        gate=OperationalGate.ALLOWED,
        gate_reasons=[],
        layout_verified=False,
        layout_canonical_sha256=None,
        hazards=[],
        inspection=fresh_inspection,
        config=standard_config,
        reference_time=base_time,
        policy_config_sha256=standard_config.compute_sha256(),
    )

    assert dec.capacity_state == CapacityState.UNKNOWN
    assert dec.is_usable is False
    assert REASON_LAYOUT_NOT_VERIFIED in dec.reason_codes


def test_invalid_input_fails_closed(
    base_time: datetime,
    standard_config: CapacityPolicyConfig,
    fresh_inspection: PavementInspectionEvidence,
):
    """Invalid input (e.g. empty bay_id or negative age) fails closed safely."""
    # 1. Missing bay_id
    bay_invalid = BayOccupancyEvidenceInput(
        bay_id="",
        operator_label="",
        occupancy_state="VACANT",
    )
    dec = evaluate_bay_capacity(
        bay=bay_invalid,
        gate=OperationalGate.ALLOWED,
        gate_reasons=[],
        layout_verified=True,
        layout_canonical_sha256="hash",
        hazards=[],
        inspection=fresh_inspection,
        config=standard_config,
        reference_time=base_time,
        policy_config_sha256=standard_config.compute_sha256(),
    )
    assert dec.capacity_state == CapacityState.UNKNOWN
    assert dec.is_usable is False
    assert REASON_INVALID_INPUT_FAIL_CLOSED in dec.reason_codes

    # 2. Negative age
    bay_neg_age = BayOccupancyEvidenceInput(
        bay_id="B1",
        operator_label="B1",
        occupancy_state="VACANT",
        evidence_age_seconds=-10.0,
    )
    dec_neg = evaluate_bay_capacity(
        bay=bay_neg_age,
        gate=OperationalGate.ALLOWED,
        gate_reasons=[],
        layout_verified=True,
        layout_canonical_sha256="hash",
        hazards=[],
        inspection=fresh_inspection,
        config=standard_config,
        reference_time=base_time,
        policy_config_sha256=standard_config.compute_sha256(),
    )
    assert dec_neg.capacity_state == CapacityState.UNKNOWN
    assert dec_neg.is_usable is False
    assert REASON_INVALID_INPUT_FAIL_CLOSED in dec_neg.reason_codes
