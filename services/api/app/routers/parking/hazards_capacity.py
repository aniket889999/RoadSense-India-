"""Parking API: hazard associations, pavement inspections and capacity snapshots.

Moved verbatim from the original ``routers/parking.py`` (no behaviour change).
"""

from __future__ import annotations

from services.api.app.routers.parking._common import *  # noqa: F401,F403  (shared imports, models, schemas, router)
from services.api.app.routers.parking._common import _HAZARD_LIFECYCLE_TRANSITIONS, _HAZARD_REVIEW_TRANSITIONS, _read_bay_occupancy_summary  # noqa: F401


@router.post(
    "/cameras/{camera_id}/hazards/associations",
    response_model=HazardAssociationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_hazard_association(
    camera_id: str,
    payload: HazardAssociationCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a manual or linked hazard association for a parking bay or approach zone."""
    cam_res = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = cam_res.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

    # Validate that parking_space_id exists and belongs to this camera's layout
    space_res = await db.execute(
        select(ParkingSpace)
        .join(ParkingLayoutRevision, ParkingSpace.layout_revision_id == ParkingLayoutRevision.id)
        .where(ParkingSpace.id == payload.parking_space_id)
        .where(ParkingLayoutRevision.camera_id == camera_id)
        .where(ParkingLayoutRevision.status == LayoutRevisionStatus.VERIFIED.value)
    )
    space = space_res.scalar_one_or_none()
    if not space:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Parking space '{payload.parking_space_id}' is not part of the active "
                f"verified layout for camera '{camera_id}'."
            ),
        )

    approach_zone_id: Optional[str] = None
    if payload.target_type == "APPROACH_ZONE":
        zone_query = select(ApproachZone).where(ApproachZone.parking_space_id == space.id)
        if payload.approach_zone_id is not None:
            zone_query = zone_query.where(ApproachZone.id == payload.approach_zone_id)
        zone = (await db.execute(zone_query)).scalar_one_or_none()
        if zone is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "APPROACH_ZONE hazards require an approach zone belonging to the "
                    "selected bay in the active verified layout."
                ),
            )
        approach_zone_id = zone.id

    # Optional road event verification
    if payload.road_event_id:
        re_res = await db.execute(select(RoadEvent).where(RoadEvent.id == payload.road_event_id))
        if not re_res.scalar_one_or_none():
            raise HTTPException(
                status_code=400,
                detail=f"Referenced road event '{payload.road_event_id}' not found.",
            )

    assoc = ParkingHazardAssociation(
        camera_id=camera_id,
        parking_space_id=payload.parking_space_id,
        approach_zone_id=approach_zone_id,
        target_type=payload.target_type,
        road_event_id=payload.road_event_id,
        hazard_label=payload.hazard_label,
        review_state="UNREVIEWED",
        lifecycle_state="ACTIVE",
        severity_label=payload.severity_label,
        notes=payload.notes,
        created_by=payload.created_by,
        version=1,
    )
    db.add(assoc)
    await db.flush()

    # Append-only audit trail
    audit = ParkingHazardAuditEvent(
        association_id=assoc.id,
        event_type="CREATED",
        operator_identity=payload.created_by,
        prior_review_state=None,
        new_review_state="UNREVIEWED",
        prior_lifecycle_state=None,
        new_lifecycle_state="ACTIVE",
        explicit_reason="Initial hazard association created",
        notes=payload.notes,
    )
    db.add(audit)
    await db.commit()
    await db.refresh(assoc)
    return assoc


@router.get(
    "/cameras/{camera_id}/hazards/associations",
    response_model=List[HazardAssociationResponse],
)
async def list_hazard_associations(
    camera_id: str,
    review_state: Optional[str] = None,
    lifecycle_state: Optional[str] = None,
    target_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """List all hazard associations for a given camera with optional status filters."""
    query = select(ParkingHazardAssociation).where(ParkingHazardAssociation.camera_id == camera_id)
    if review_state:
        query = query.where(ParkingHazardAssociation.review_state == review_state.upper())
    if lifecycle_state:
        query = query.where(ParkingHazardAssociation.lifecycle_state == lifecycle_state.upper())
    if target_type:
        query = query.where(ParkingHazardAssociation.target_type == target_type.upper())
    query = query.order_by(desc(ParkingHazardAssociation.created_at))

    result = await db.execute(query)
    return result.scalars().all()


@router.get(
    "/hazards/associations/{association_id}",
    response_model=HazardAssociationResponse,
)
async def get_hazard_association(
    association_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve details of a single hazard association."""
    res = await db.execute(select(ParkingHazardAssociation).where(ParkingHazardAssociation.id == association_id))
    assoc = res.scalar_one_or_none()
    if not assoc:
        raise HTTPException(status_code=404, detail="Hazard association not found.")
    return assoc


@router.post(
    "/hazards/associations/{association_id}/review",
    response_model=HazardAssociationResponse,
)
async def review_hazard_association(
    association_id: str,
    payload: HazardAssociationReviewRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Guarded human review action on a candidate hazard association.
    Enforces optimistic concurrency control and records immutable audit log.
    """
    res = await db.execute(select(ParkingHazardAssociation).where(ParkingHazardAssociation.id == association_id))
    assoc = res.scalar_one_or_none()
    if not assoc:
        raise HTTPException(status_code=404, detail="Hazard association not found.")

    if assoc.version != payload.expected_version:
        raise HTTPException(
            status_code=409,
            detail=f"Version mismatch: expected {payload.expected_version}, but current version is {assoc.version}.",
        )

    prior_review_state = assoc.review_state
    if payload.review_state not in _HAZARD_REVIEW_TRANSITIONS.get(prior_review_state, frozenset()):
        raise HTTPException(
            status_code=409,
            detail=f"Review transition {prior_review_state} -> {payload.review_state} is not permitted.",
        )

    reviewed_at = utc_now()
    update_values = {
        "review_state": payload.review_state,
        "reviewed_by": payload.reviewer_identity,
        "reviewed_at": reviewed_at,
        "version": payload.expected_version + 1,
        "updated_at": reviewed_at,
    }
    if payload.notes is not None:
        update_values["notes"] = payload.notes

    cas_result = await db.execute(
        update(ParkingHazardAssociation)
        .where(ParkingHazardAssociation.id == association_id)
        .where(ParkingHazardAssociation.version == payload.expected_version)
        .where(ParkingHazardAssociation.review_state == prior_review_state)
        .values(**update_values)
        .execution_options(synchronize_session=False)
    )
    if cas_result.rowcount != 1:
        await db.rollback()
        current = (
            await db.execute(
                select(ParkingHazardAssociation).where(ParkingHazardAssociation.id == association_id)
            )
        ).scalar_one_or_none()
        if current is None:
            raise HTTPException(status_code=404, detail="Hazard association not found.")
        raise HTTPException(
            status_code=409,
            detail=f"Version mismatch: expected {payload.expected_version}, but current version is {current.version}.",
        )

    audit = ParkingHazardAuditEvent(
        association_id=assoc.id,
        event_type="REVIEWED",
        operator_identity=payload.reviewer_identity,
        prior_review_state=prior_review_state,
        new_review_state=payload.review_state,
        prior_lifecycle_state=assoc.lifecycle_state,
        new_lifecycle_state=assoc.lifecycle_state,
        explicit_reason=payload.explicit_reason,
        notes=payload.notes,
    )
    db.add(audit)
    await db.commit()
    refreshed = await db.execute(
        select(ParkingHazardAssociation)
        .where(ParkingHazardAssociation.id == association_id)
        .execution_options(populate_existing=True)
    )
    return refreshed.scalar_one()


@router.post(
    "/hazards/associations/{association_id}/lifecycle",
    response_model=HazardAssociationResponse,
)
async def transition_hazard_lifecycle(
    association_id: str,
    payload: HazardAssociationLifecycleRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Guarded lifecycle transition (ACTIVE, MITIGATED, RESOLVED, EXPIRED, SUPERSEDED).
    Enforces optimistic concurrency control and records immutable audit log.
    """
    res = await db.execute(select(ParkingHazardAssociation).where(ParkingHazardAssociation.id == association_id))
    assoc = res.scalar_one_or_none()
    if not assoc:
        raise HTTPException(status_code=404, detail="Hazard association not found.")

    if assoc.version != payload.expected_version:
        raise HTTPException(
            status_code=409,
            detail=f"Version mismatch: expected {payload.expected_version}, but current version is {assoc.version}.",
        )

    prior_lifecycle_state = assoc.lifecycle_state
    if payload.lifecycle_state not in _HAZARD_LIFECYCLE_TRANSITIONS.get(
        prior_lifecycle_state, frozenset()
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Lifecycle transition {prior_lifecycle_state} -> "
                f"{payload.lifecycle_state} is not permitted."
            ),
        )

    transitioned_at = utc_now()
    update_values = {
        "lifecycle_state": payload.lifecycle_state,
        "version": payload.expected_version + 1,
        "updated_at": transitioned_at,
    }
    if payload.notes is not None:
        update_values["notes"] = payload.notes

    cas_result = await db.execute(
        update(ParkingHazardAssociation)
        .where(ParkingHazardAssociation.id == association_id)
        .where(ParkingHazardAssociation.version == payload.expected_version)
        .where(ParkingHazardAssociation.lifecycle_state == prior_lifecycle_state)
        .values(**update_values)
        .execution_options(synchronize_session=False)
    )
    if cas_result.rowcount != 1:
        await db.rollback()
        current = (
            await db.execute(
                select(ParkingHazardAssociation).where(ParkingHazardAssociation.id == association_id)
            )
        ).scalar_one_or_none()
        if current is None:
            raise HTTPException(status_code=404, detail="Hazard association not found.")
        raise HTTPException(
            status_code=409,
            detail=f"Version mismatch: expected {payload.expected_version}, but current version is {current.version}.",
        )

    audit = ParkingHazardAuditEvent(
        association_id=assoc.id,
        event_type="LIFECYCLE_TRANSITION",
        operator_identity=payload.operator_identity,
        prior_review_state=assoc.review_state,
        new_review_state=assoc.review_state,
        prior_lifecycle_state=prior_lifecycle_state,
        new_lifecycle_state=payload.lifecycle_state,
        explicit_reason=payload.explicit_reason,
        notes=payload.notes,
    )
    db.add(audit)
    await db.commit()
    refreshed = await db.execute(
        select(ParkingHazardAssociation)
        .where(ParkingHazardAssociation.id == association_id)
        .execution_options(populate_existing=True)
    )
    return refreshed.scalar_one()


@router.get(
    "/cameras/{camera_id}/hazards/audit-events",
    response_model=List[HazardAuditEventResponse],
)
async def list_hazard_audit_events(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
):
    """List append-only audit events for all hazard associations on a camera."""
    query = (
        select(ParkingHazardAuditEvent)
        .join(ParkingHazardAssociation, ParkingHazardAuditEvent.association_id == ParkingHazardAssociation.id)
        .where(ParkingHazardAssociation.camera_id == camera_id)
        .order_by(desc(ParkingHazardAuditEvent.created_at))
    )
    res = await db.execute(query)
    return res.scalars().all()


@router.post(
    "/cameras/{camera_id}/inspection/record",
    response_model=PavementInspectionRecordResponse,
    status_code=status.HTTP_201_CREATED,
)
async def record_pavement_inspection(
    camera_id: str,
    payload: PavementInspectionRecordCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Record a mobile pavement surface inspection event establishing freshness."""
    cam_res = await db.execute(select(Camera).where(Camera.id == camera_id))
    if not cam_res.scalar_one_or_none():
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

    if payload.session_id is not None:
        session_res = await db.execute(
            select(DriveSession).where(DriveSession.id == payload.session_id)
        )
        if session_res.scalar_one_or_none() is None:
            raise HTTPException(
                status_code=400,
                detail=f"Referenced drive session '{payload.session_id}' not found.",
            )

    inspected_at = payload.inspected_at or utc_now()
    if inspected_at > utc_now():
        raise HTTPException(
            status_code=400,
            detail="inspected_at cannot be in the future.",
        )

    rec = PavementInspectionRecord(
        camera_id=camera_id,
        session_id=payload.session_id,
        inspected_at=inspected_at,
        inspector_label=payload.inspector_label,
        notes=payload.notes,
    )
    db.add(rec)
    await db.commit()
    await db.refresh(rec)
    return rec


@router.get(
    "/cameras/{camera_id}/inspection/records",
    response_model=List[PavementInspectionRecordResponse],
)
async def list_pavement_inspections(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
):
    """List pavement inspection records for a given camera, newest first."""
    res = await db.execute(
        select(PavementInspectionRecord)
        .where(PavementInspectionRecord.camera_id == camera_id)
        .order_by(desc(PavementInspectionRecord.inspected_at))
    )
    return res.scalars().all()


@router.get(
    "/cameras/{camera_id}/capacity/snapshot",
    response_model=CapacitySnapshotResponse,
)
async def get_camera_capacity_snapshot(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
    stab_manager: StabilityAssessmentJobManager = Depends(get_stability_job_manager),
):
    """
    Compute and return current Safe Usable Capacity snapshot for a camera.
    Fuses fixed occupancy, camera stability gate, human-reviewed hazards, and pavement inspection freshness.
    """
    cam_res = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = cam_res.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

    # 1. Active verified layout
    layout_res = await db.execute(
        select(ParkingLayoutRevision)
        .options(selectinload(ParkingLayoutRevision.parking_spaces))
        .where(ParkingLayoutRevision.camera_id == camera_id)
        .where(ParkingLayoutRevision.status == LayoutRevisionStatus.VERIFIED.value)
    )
    verified_layout = layout_res.scalar_one_or_none()
    current_layout_sha = verified_layout.canonical_sha256 if verified_layout else None

    # Spaces to evaluate: verified layout spaces, or fallback to latest layout spaces
    spaces: List[ParkingSpace] = []
    if verified_layout:
        spaces = verified_layout.parking_spaces
    else:
        latest_layout_res = await db.execute(
            select(ParkingLayoutRevision)
            .options(selectinload(ParkingLayoutRevision.parking_spaces))
            .where(ParkingLayoutRevision.camera_id == camera_id)
            .order_by(desc(ParkingLayoutRevision.revision_number))
        )
        latest_layout = latest_layout_res.scalars().first()
        if latest_layout:
            spaces = latest_layout.parking_spaces

    # 2. Evaluate operational stability gate
    assess_res = await db.execute(
        select(CameraStabilityAssessment)
        .where(CameraStabilityAssessment.camera_id == camera_id)
        .where(CameraStabilityAssessment.status == "COMPLETE")
        .order_by(desc(CameraStabilityAssessment.created_at))
    )
    latest_assessment = assess_res.scalars().first()

    now = utc_now()
    if not latest_assessment:
        gate_enum, gate_reasons = evaluate_operational_gate(
            decision=None,
            assessment_reference_sha=None,
            current_reference_sha=cam.reference_image_sha256,
            assessment_layout_sha=None,
            current_layout_sha=current_layout_sha,
        )
    else:
        decision_enum = None
        try:
            decision_enum = StabilityDecision(latest_assessment.aggregate_decision)
        except Exception:
            pass
        max_age = (latest_assessment.thresholds_snapshot or {}).get(
            "max_assessment_age_seconds", stab_manager.config.thresholds.max_assessment_age_seconds
        )
        assess_at = latest_assessment.created_at
        if assess_at and assess_at.tzinfo is None:
            assess_at = assess_at.replace(tzinfo=timezone.utc)
        gate_enum, gate_reasons = evaluate_operational_gate(
            decision=decision_enum,
            assessment_reference_sha=latest_assessment.reference_image_sha256,
            current_reference_sha=cam.reference_image_sha256,
            assessment_layout_sha=latest_assessment.layout_canonical_sha256,
            current_layout_sha=current_layout_sha,
            assessment_timestamp=assess_at,
            current_timestamp=now,
            max_age_seconds=max_age,
        )

    # 3. Latest completed occupancy job
    job_res = await db.execute(
        select(ParkingOccupancyJob)
        .where(ParkingOccupancyJob.camera_id == camera_id)
        .where(ParkingOccupancyJob.status == "COMPLETE")
        .order_by(desc(ParkingOccupancyJob.completed_at))
    )
    latest_job = job_res.scalars().first()

    job_evidence_age: Optional[float] = None
    bay_summary: Dict[str, Any] = {}
    if latest_job and latest_job.completed_at:
        job_comp = latest_job.completed_at
        if job_comp.tzinfo is None:
            job_comp = job_comp.replace(tzinfo=timezone.utc)
        job_evidence_age = (now - job_comp).total_seconds()
        bay_summary = latest_job.bay_summary_json or {}

    # 4. Build BayOccupancyEvidenceInput per space
    bay_inputs: List[BayOccupancyEvidenceInput] = []
    for sp in spaces:
        occ_state, conf = _read_bay_occupancy_summary(bay_summary, sp.id)

        bay_inputs.append(
            BayOccupancyEvidenceInput(
                bay_id=sp.id,
                operator_label=sp.operator_label,
                space_type=sp.space_type,
                occupancy_state=occ_state,
                evidence_age_seconds=job_evidence_age,
                confidence=conf,
                provenance_sha256=latest_job.input_video_sha256 if latest_job else None,
                job_id=latest_job.id if latest_job else None,
            )
        )

    # 5. Pavement inspection evidence
    insp_res = await db.execute(
        select(PavementInspectionRecord)
        .where(PavementInspectionRecord.camera_id == camera_id)
        .order_by(desc(PavementInspectionRecord.inspected_at))
    )
    latest_insp = insp_res.scalars().first()
    insp_evidence: Optional[PavementInspectionEvidence] = None
    if latest_insp:
        insp_at = latest_insp.inspected_at
        if insp_at and insp_at.tzinfo is None:
            insp_at = insp_at.replace(tzinfo=timezone.utc)
        insp_age = (now - insp_at).total_seconds()
        insp_evidence = PavementInspectionEvidence(
            session_id=latest_insp.session_id,
            inspection_timestamp=insp_at,
            freshness_age_seconds=insp_age,
            is_fresh=None,
            inspection_notes=latest_insp.notes,
        )

    # 6. Active/reviewed hazard associations
    haz_res = await db.execute(
        select(ParkingHazardAssociation)
        .options(selectinload(ParkingHazardAssociation.approach_zone))
        .where(ParkingHazardAssociation.camera_id == camera_id)
    )
    db_hazards = haz_res.scalars().all()
    hazard_inputs: List[HazardAssociationInput] = []
    for h in db_hazards:
        try:
            target_enum = HazardTarget(h.target_type)
        except Exception:
            target_enum = HazardTarget.BAY
        try:
            rev_enum = HazardReviewState(h.review_state)
        except Exception:
            rev_enum = HazardReviewState.UNREVIEWED
        try:
            life_enum = HazardLifecycleState(h.lifecycle_state)
        except Exception:
            life_enum = HazardLifecycleState.ACTIVE

        hazard_inputs.append(
            HazardAssociationInput(
                association_id=h.id,
                hazard_id=h.road_event_id or h.id,
                target_type=target_enum,
                target_id=(
                    h.approach_zone_id
                    if target_enum == HazardTarget.APPROACH_ZONE
                    else h.parking_space_id
                ),
                affected_bay_id=h.parking_space_id,
                review_state=rev_enum,
                lifecycle_state=life_enum,
                severity_label=h.severity_label,
                notes=h.notes,
                created_at=h.created_at,
                reviewed_at=h.reviewed_at,
                reviewed_by=h.reviewed_by,
            )
        )

    # 7. Evaluate Safe Usable Capacity snapshot
    config = CapacityPolicyConfig()
    snapshot = evaluate_safe_usable_capacity(
        camera_id=camera_id,
        site_id=cam.site_id,
        bays=bay_inputs,
        gate=PolicyOperationalGate(gate_enum.value),
        gate_reasons=gate_reasons,
        layout_verified=bool(verified_layout),
        layout_revision_id=verified_layout.id if verified_layout else None,
        layout_canonical_sha256=current_layout_sha,
        hazards=hazard_inputs,
        inspection=insp_evidence,
        config=config,
        reference_time=now,
    )

    # Convert decisions for response schema
    current_insp_age = round(insp_age, 1) if (latest_insp and insp_age is not None) else None
    decision_responses: Dict[str, BayCapacityDecisionResponse] = {}
    for bay_id, d in snapshot.decisions.items():
        decision_responses[bay_id] = BayCapacityDecisionResponse(
            bay_id=d.bay_id,
            operator_label=d.operator_label,
            space_type=d.space_type,
            capacity_state=d.capacity_state.value,
            is_usable=d.is_usable,
            reason_codes=d.reason_codes,
            raw_occupancy_state=d.raw_occupancy_state,
            has_active_bay_hazard=d.has_active_bay_hazard,
            has_active_approach_hazard=d.has_active_approach_hazard,
            has_unverified_hazard=d.has_unverified_hazard,
            pavement_inspection_fresh=d.pavement_inspection_fresh,
            inspection_age_seconds=current_insp_age,
            evidence_ids=d.evidence_ids,
        )

    return CapacitySnapshotResponse(
        camera_id=snapshot.camera_id,
        site_id=snapshot.site_id,
        layout_revision_id=snapshot.layout_revision_id,
        layout_canonical_sha256=snapshot.layout_canonical_sha256,
        policy_config_sha256=snapshot.policy_config_sha256,
        operational_gate=snapshot.operational_gate.value,
        gate_reasons=snapshot.gate_reasons,
        evaluated_at=snapshot.evaluated_at,
        total_bays=snapshot.total_bays,
        physical_vacant_count=snapshot.physical_vacant_count,
        usable_available_count=snapshot.usable_available_count,
        occupied_count=snapshot.occupied_count,
        hazard_blocked_count=snapshot.hazard_blocked_count,
        approach_blocked_count=snapshot.approach_blocked_count,
        vacant_unassessed_count=snapshot.vacant_unassessed_count,
        unknown_count=snapshot.unknown_count,
        occluded_count=snapshot.occluded_count,
        inference_blocked_count=snapshot.inference_blocked_count,
        snapshot_sha256=snapshot.snapshot_sha256,
        decisions=decision_responses,
    )
