"""Parking API: layout revisions, validation, review and verification.

Moved verbatim from the original ``routers/parking.py`` (no behaviour change).
"""

from __future__ import annotations

from services.api.app.routers.parking._common import *  # noqa: F401,F403  (shared imports, models, schemas, router)


def _dt_sort_key(dt: Optional[datetime]) -> float:
    if dt is None:
        return 0.0
    if dt.tzinfo is not None:
        return dt.timestamp()
    return dt.replace(tzinfo=timezone.utc).timestamp()


def _build_layout_response(rev: ParkingLayoutRevision) -> LayoutRevisionResponse:
    spaces_out: List[ParkingSpaceSchema] = []
    approaches_out: List[ApproachZoneSchema] = []

    for sp in rev.parking_spaces:
        pts = [PointSchema(x=float(p["x"]), y=float(p["y"])) for p in sp.polygon_normalized]
        spaces_out.append(
            ParkingSpaceSchema(
                id=sp.id,
                operator_label=sp.operator_label,
                space_type=sp.space_type,
                polygon_normalized=pts,
                active=sp.active,
            )
        )
        if sp.approach_zone:
            az_pts = [PointSchema(x=float(p["x"]), y=float(p["y"])) for p in sp.approach_zone.polygon_normalized]
            approaches_out.append(
                ApproachZoneSchema(
                    id=sp.approach_zone.id,
                    parking_space_id=sp.id,
                    polygon_normalized=az_pts,
                )
            )

    audits_out = [
        AuditEventResponse(
            id=a.id,
            layout_revision_id=a.layout_revision_id,
            event_type=a.event_type,
            prior_status=a.prior_status,
            new_status=a.new_status,
            local_operator_label=a.local_operator_label,
            note=a.note,
            created_at=a.created_at,
        )
        for a in sorted(rev.audit_events, key=lambda x: _dt_sort_key(x.created_at))
    ]

    return LayoutRevisionResponse(
        id=rev.id,
        camera_id=rev.camera_id,
        revision_number=rev.revision_number,
        status=rev.status,
        canonical_sha256=rev.canonical_sha256,
        reference_image_sha256=rev.reference_image_sha256,
        reference_width=rev.reference_width,
        reference_height=rev.reference_height,
        created_at=rev.created_at,
        submitted_at=rev.submitted_at,
        verified_at=rev.verified_at,
        invalidated_at=rev.invalidated_at,
        invalidation_reason=rev.invalidation_reason,
        parking_spaces=spaces_out,
        approach_zones=approaches_out,
        audit_events=audits_out,
    )


@router.post("/cameras/{camera_id}/layouts", response_model=LayoutRevisionResponse, status_code=status.HTTP_201_CREATED)
async def create_draft_layout(
    camera_id: str,
    payload: LayoutCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a new draft parking layout revision for a camera."""
    cam_res = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = cam_res.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not cam.reference_image_sha256 or not cam.reference_width or not cam.reference_height:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera has no valid reference image. Upload a reference image before creating a layout.",
        )

    # Get max revision number for this camera
    max_rev_res = await db.execute(
        select(func.coalesce(func.max(ParkingLayoutRevision.revision_number), 0)).where(
            ParkingLayoutRevision.camera_id == camera_id
        )
    )
    next_rev = (max_rev_res.scalar() or 0) + 1

    revision_id = str(uuid.uuid4())
    layout_rev = ParkingLayoutRevision(
        id=revision_id,
        camera_id=camera_id,
        revision_number=next_rev,
        status=LayoutRevisionStatus.DRAFT.value,
        reference_image_sha256=cam.reference_image_sha256,
        reference_width=cam.reference_width,
        reference_height=cam.reference_height,
        created_at=utc_now(),
    )
    db.add(layout_rev)

    # Add parking spaces and approach zones
    space_id_map: dict[str, str] = {}
    for idx, sp_data in enumerate(payload.parking_spaces):
        sp_id = sp_data.id or str(uuid.uuid4())
        client_key = sp_data.id or f"client_{idx}"
        space_id_map[client_key] = sp_id
        space_id_map[sp_id] = sp_id

        pts_json = [p.model_dump() for p in sp_data.polygon_normalized]
        sp = ParkingSpace(
            id=sp_id,
            layout_revision_id=revision_id,
            operator_label=sp_data.operator_label.strip(),
            space_type=sp_data.space_type,
            polygon_normalized=pts_json,
            active=sp_data.active,
            created_at=utc_now(),
        )
        db.add(sp)

    if payload.approach_zones:
        for az_data in payload.approach_zones:
            target_sp_id = space_id_map.get(az_data.parking_space_id, az_data.parking_space_id)
            az_pts = [p.model_dump() for p in az_data.polygon_normalized]
            az = ApproachZone(
                id=az_data.id or str(uuid.uuid4()),
                parking_space_id=target_sp_id,
                polygon_normalized=az_pts,
                created_at=utc_now(),
            )
            db.add(az)

    # Initial audit log
    audit = LayoutAuditEvent(
        id=str(uuid.uuid4()),
        layout_revision_id=revision_id,
        event_type="CREATED",
        prior_status=None,
        new_status=LayoutRevisionStatus.DRAFT.value,
        local_operator_label="system",
        note="Initial draft layout created",
        created_at=utc_now(),
    )
    db.add(audit)

    try:
        await db.commit()
    except IntegrityError as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Conflict: A layout revision with this revision number was created concurrently.",
        ) from e

    # Re-fetch with relationships loaded
    full_rev = await db.execute(
        select(ParkingLayoutRevision)
        .options(
            selectinload(ParkingLayoutRevision.parking_spaces).selectinload(ParkingSpace.approach_zone),
            selectinload(ParkingLayoutRevision.audit_events),
        )
        .where(ParkingLayoutRevision.id == revision_id)
    )
    rev_obj = full_rev.scalar_one()
    return _build_layout_response(rev_obj)


@router.get("/cameras/{camera_id}/layouts", response_model=List[LayoutRevisionResponse])
async def list_camera_layouts(camera_id: str, db: AsyncSession = Depends(get_db)):
    """List all layout revisions for a camera."""
    result = await db.execute(
        select(ParkingLayoutRevision)
        .options(
            selectinload(ParkingLayoutRevision.parking_spaces).selectinload(ParkingSpace.approach_zone),
            selectinload(ParkingLayoutRevision.audit_events),
        )
        .where(ParkingLayoutRevision.camera_id == camera_id)
        .order_by(desc(ParkingLayoutRevision.revision_number))
    )
    revisions = result.scalars().all()
    return [_build_layout_response(r) for r in revisions]


@router.get("/layouts/{layout_id}", response_model=LayoutRevisionResponse)
async def get_layout(layout_id: str, db: AsyncSession = Depends(get_db)):
    """Get full details of a parking layout revision."""
    result = await db.execute(
        select(ParkingLayoutRevision)
        .options(
            selectinload(ParkingLayoutRevision.parking_spaces).selectinload(ParkingSpace.approach_zone),
            selectinload(ParkingLayoutRevision.audit_events),
        )
        .where(ParkingLayoutRevision.id == layout_id)
    )
    rev = result.scalar_one_or_none()
    if not rev:
        raise HTTPException(status_code=404, detail="Layout revision not found")
    return _build_layout_response(rev)


@router.put("/layouts/{layout_id}", response_model=LayoutRevisionResponse)
async def update_layout(
    layout_id: str,
    payload: LayoutUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Update a parking layout revision.
    If the layout is DRAFT, updates in place.
    If VERIFIED, PENDING_REVIEW, SUPERSEDED, or INVALIDATED, creates a new DRAFT revision to preserve immutability.
    """
    result = await db.execute(
        select(ParkingLayoutRevision)
        .options(
            selectinload(ParkingLayoutRevision.parking_spaces).selectinload(ParkingSpace.approach_zone),
            selectinload(ParkingLayoutRevision.audit_events),
            selectinload(ParkingLayoutRevision.camera),
        )
        .where(ParkingLayoutRevision.id == layout_id)
    )
    rev = result.scalar_one_or_none()
    if not rev:
        raise HTTPException(status_code=404, detail="Layout revision not found")

    cam = rev.camera
    if not cam or not cam.reference_image_sha256:
        raise HTTPException(status_code=400, detail="Camera has no valid reference image.")

    if rev.status == LayoutRevisionStatus.DRAFT.value:
        # Verify snapshot matches current camera image
        if (
            rev.reference_image_sha256 != cam.reference_image_sha256
            or rev.reference_width != cam.reference_width
            or rev.reference_height != cam.reference_height
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Layout reference snapshot does not match current camera reference image. Geometry is stale.",
            )

        # Update DRAFT in place
        for old_sp in list(rev.parking_spaces):
            await db.delete(old_sp)
        await db.flush()

        space_id_map: dict[str, str] = {}
        for idx, sp_data in enumerate(payload.parking_spaces):
            sp_id = sp_data.id or str(uuid.uuid4())
            client_key = sp_data.id or f"client_{idx}"
            space_id_map[client_key] = sp_id
            space_id_map[sp_id] = sp_id

            pts_json = [p.model_dump() for p in sp_data.polygon_normalized]
            sp = ParkingSpace(
                id=sp_id,
                layout_revision_id=rev.id,
                operator_label=sp_data.operator_label.strip(),
                space_type=sp_data.space_type,
                polygon_normalized=pts_json,
                active=sp_data.active,
                created_at=utc_now(),
            )
            db.add(sp)

        if payload.approach_zones:
            for az_data in payload.approach_zones:
                target_sp_id = space_id_map.get(az_data.parking_space_id, az_data.parking_space_id)
                az_pts = [p.model_dump() for p in az_data.polygon_normalized]
                az = ApproachZone(
                    id=az_data.id or str(uuid.uuid4()),
                    parking_space_id=target_sp_id,
                    polygon_normalized=az_pts,
                    created_at=utc_now(),
                )
                db.add(az)

        audit = LayoutAuditEvent(
            id=str(uuid.uuid4()),
            layout_revision_id=rev.id,
            event_type="UPDATED",
            prior_status=LayoutRevisionStatus.DRAFT.value,
            new_status=LayoutRevisionStatus.DRAFT.value,
            local_operator_label="operator",
            note="Draft geometry updated",
            created_at=utc_now(),
        )
        db.add(audit)
        await db.commit()

        reloaded = await db.execute(
            select(ParkingLayoutRevision)
            .options(
                selectinload(ParkingLayoutRevision.parking_spaces).selectinload(ParkingSpace.approach_zone),
                selectinload(ParkingLayoutRevision.audit_events),
            )
            .where(ParkingLayoutRevision.id == rev.id)
        )
        return _build_layout_response(reloaded.scalar_one())

    else:
        # Non-draft: create a new DRAFT revision without mutating history
        max_rev_res = await db.execute(
            select(func.coalesce(func.max(ParkingLayoutRevision.revision_number), 0)).where(
                ParkingLayoutRevision.camera_id == rev.camera_id
            )
        )
        next_rev = (max_rev_res.scalar() or 0) + 1

        new_rev_id = str(uuid.uuid4())
        new_rev = ParkingLayoutRevision(
            id=new_rev_id,
            camera_id=rev.camera_id,
            revision_number=next_rev,
            status=LayoutRevisionStatus.DRAFT.value,
            reference_image_sha256=cam.reference_image_sha256,
            reference_width=cam.reference_width,
            reference_height=cam.reference_height,
            created_at=utc_now(),
        )
        db.add(new_rev)

        space_id_map = {}
        for idx, sp_data in enumerate(payload.parking_spaces):
            sp_id = str(uuid.uuid4())
            client_key = sp_data.id or f"client_{idx}"
            space_id_map[client_key] = sp_id
            space_id_map[sp_id] = sp_id

            pts_json = [p.model_dump() for p in sp_data.polygon_normalized]
            sp = ParkingSpace(
                id=sp_id,
                layout_revision_id=new_rev_id,
                operator_label=sp_data.operator_label.strip(),
                space_type=sp_data.space_type,
                polygon_normalized=pts_json,
                active=sp_data.active,
                created_at=utc_now(),
            )
            db.add(sp)

        if payload.approach_zones:
            for az_data in payload.approach_zones:
                target_sp_id = space_id_map.get(az_data.parking_space_id, az_data.parking_space_id)
                az_pts = [p.model_dump() for p in az_data.polygon_normalized]
                az = ApproachZone(
                    id=str(uuid.uuid4()),
                    parking_space_id=target_sp_id,
                    polygon_normalized=az_pts,
                    created_at=utc_now(),
                )
                db.add(az)

        audit = LayoutAuditEvent(
            id=str(uuid.uuid4()),
            layout_revision_id=new_rev_id,
            event_type="BRANCHED_FROM_REVISION",
            prior_status=rev.status,
            new_status=LayoutRevisionStatus.DRAFT.value,
            local_operator_label="operator",
            note=f"Created new draft revision {next_rev} branched from revision {rev.revision_number} ({rev.status})",
            created_at=utc_now(),
        )
        db.add(audit)

        try:
            await db.commit()
        except IntegrityError as e:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Conflict: Concurrent layout creation assigned the same revision number.",
            ) from e

        reloaded = await db.execute(
            select(ParkingLayoutRevision)
            .options(
                selectinload(ParkingLayoutRevision.parking_spaces).selectinload(ParkingSpace.approach_zone),
                selectinload(ParkingLayoutRevision.audit_events),
            )
            .where(ParkingLayoutRevision.id == new_rev_id)
        )
        return _build_layout_response(reloaded.scalar_one())


@router.post("/layouts/{layout_id}/validate", response_model=LayoutValidationResponse)
async def validate_layout_endpoint(layout_id: str, db: AsyncSession = Depends(get_db)):
    """Run geometry and topological validation on a layout revision."""
    result = await db.execute(
        select(ParkingLayoutRevision)
        .options(
            selectinload(ParkingLayoutRevision.parking_spaces).selectinload(ParkingSpace.approach_zone),
            selectinload(ParkingLayoutRevision.camera),
        )
        .where(ParkingLayoutRevision.id == layout_id)
    )
    rev = result.scalar_one_or_none()
    if not rev:
        raise HTTPException(status_code=404, detail="Layout revision not found")

    cam = rev.camera
    if not cam or not cam.reference_image_sha256 or not cam.reference_width or not cam.reference_height:
        raise HTTPException(status_code=400, detail="Camera has no valid reference image.")
    if not rev.reference_image_sha256 or not rev.reference_width or not rev.reference_height:
        raise HTTPException(status_code=400, detail="Layout revision lacks reference image snapshot.")
    if (
        rev.reference_image_sha256 != cam.reference_image_sha256
        or rev.reference_width != cam.reference_width
        or rev.reference_height != cam.reference_height
    ):
        raise HTTPException(
            status_code=400,
            detail="Layout reference snapshot does not match current camera reference image. Layout geometry is stale.",
        )

    spaces_dicts = [
        {
            "id": sp.id,
            "operator_label": sp.operator_label,
            "space_type": sp.space_type,
            "polygon_normalized": sp.polygon_normalized,
            "active": sp.active,
        }
        for sp in rev.parking_spaces
    ]

    approaches_dicts = [
        {
            "id": sp.approach_zone.id,
            "parking_space_id": sp.id,
            "polygon_normalized": sp.approach_zone.polygon_normalized,
        }
        for sp in rev.parking_spaces
        if sp.approach_zone
    ]

    raw_errors = validate_parking_layout(
        parking_spaces=spaces_dicts,
        approach_zones=approaches_dicts,
        reference_width=cam.reference_width,
        reference_height=cam.reference_height,
    )

    error_items = [
        LayoutValidationErrorDict(
            rule_id=e.rule_id,
            message=e.message,
            space_id=e.space_id,
            space_label=e.space_label,
        )
        for e in raw_errors
    ]

    return LayoutValidationResponse(
        is_valid=len(error_items) == 0,
        errors=error_items,
        spaces_count=len(spaces_dicts),
        approach_zones_count=len(approaches_dicts),
    )


@router.post("/layouts/{layout_id}/submit", response_model=LayoutRevisionResponse)
async def submit_layout_for_review(
    layout_id: str,
    payload: LayoutSubmitRequest,
    db: AsyncSession = Depends(get_db),
):
    """Submit a DRAFT layout revision for human verification (DRAFT -> PENDING_REVIEW)."""
    result = await db.execute(
        select(ParkingLayoutRevision)
        .options(
            selectinload(ParkingLayoutRevision.parking_spaces).selectinload(ParkingSpace.approach_zone),
            selectinload(ParkingLayoutRevision.audit_events),
            selectinload(ParkingLayoutRevision.camera),
        )
        .where(ParkingLayoutRevision.id == layout_id)
    )
    rev = result.scalar_one_or_none()
    if not rev:
        raise HTTPException(status_code=404, detail="Layout revision not found")

    if rev.status != LayoutRevisionStatus.DRAFT.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot submit layout in '{rev.status}' status. Only DRAFT layouts can be submitted.",
        )

    clean_op = payload.local_operator_label.strip()
    if not clean_op:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Non-blank operator identity is required.")

    cam = rev.camera
    if not cam or not cam.reference_image_sha256 or not cam.reference_width or not cam.reference_height:
        raise HTTPException(status_code=400, detail="Camera has no valid reference image.")
    if not rev.reference_image_sha256 or not rev.reference_width or not rev.reference_height:
        raise HTTPException(status_code=400, detail="Layout revision lacks reference image snapshot.")
    if (
        rev.reference_image_sha256 != cam.reference_image_sha256
        or rev.reference_width != cam.reference_width
        or rev.reference_height != cam.reference_height
    ):
        raise HTTPException(
            status_code=400,
            detail="Layout reference snapshot does not match current camera reference image. Layout is stale.",
        )

    # Perform geometry validation before allowing submission
    spaces_dicts = [
        {
            "id": sp.id,
            "operator_label": sp.operator_label,
            "space_type": sp.space_type,
            "polygon_normalized": sp.polygon_normalized,
            "active": sp.active,
        }
        for sp in rev.parking_spaces
    ]
    approaches_dicts = [
        {
            "id": sp.approach_zone.id,
            "parking_space_id": sp.id,
            "polygon_normalized": sp.approach_zone.polygon_normalized,
        }
        for sp in rev.parking_spaces
        if sp.approach_zone
    ]
    validation_errors = validate_parking_layout(
        parking_spaces=spaces_dicts,
        approach_zones=approaches_dicts,
        reference_width=cam.reference_width,
        reference_height=cam.reference_height,
    )
    if validation_errors:
        err_msgs = "; ".join([e.message for e in validation_errors[:3]])
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Cannot submit layout with geometry validation errors: {err_msgs}",
        )

    rev.status = LayoutRevisionStatus.PENDING_REVIEW.value
    rev.submitted_at = utc_now()

    audit = LayoutAuditEvent(
        id=str(uuid.uuid4()),
        layout_revision_id=rev.id,
        event_type="SUBMITTED",
        prior_status=LayoutRevisionStatus.DRAFT.value,
        new_status=LayoutRevisionStatus.PENDING_REVIEW.value,
        local_operator_label=clean_op,
        note=payload.note or "Submitted for human review",
        created_at=utc_now(),
    )
    db.add(audit)
    await db.commit()
    await db.refresh(rev)

    return _build_layout_response(rev)


@router.post("/layouts/{layout_id}/verify", response_model=LayoutRevisionResponse)
async def verify_layout(
    layout_id: str,
    payload: LayoutVerifyRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Explicitly verify and activate a parking layout revision (PENDING_REVIEW -> VERIFIED).
    Requires local operator identity assertion and explicit confirmation checkbox.
    Computes deterministic canonical fingerprint and marks prior VERIFIED revisions as SUPERSEDED.
    """
    if not payload.confirmation_acknowledged:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification requires explicit confirmation acknowledgement.",
        )

    clean_op = payload.local_operator_label.strip()
    if not clean_op:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Non-blank operator identity is required.")

    result = await db.execute(
        select(ParkingLayoutRevision)
        .options(
            selectinload(ParkingLayoutRevision.parking_spaces).selectinload(ParkingSpace.approach_zone),
            selectinload(ParkingLayoutRevision.audit_events),
            selectinload(ParkingLayoutRevision.camera),
        )
        .where(ParkingLayoutRevision.id == layout_id)
    )
    rev = result.scalar_one_or_none()
    if not rev:
        raise HTTPException(status_code=404, detail="Layout revision not found")

    # STEP 3.8: Verification must accept PENDING_REVIEW only. DRAFT -> VERIFIED directly must fail.
    if rev.status != LayoutRevisionStatus.PENDING_REVIEW.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot verify layout in '{rev.status}' status. Only layouts in PENDING_REVIEW status can be verified.",
        )

    cam = rev.camera
    if not cam or not cam.reference_image_sha256 or not cam.reference_width or not cam.reference_height:
        raise HTTPException(status_code=400, detail="Camera has no valid reference image.")
    if not rev.reference_image_sha256 or not rev.reference_width or not rev.reference_height:
        raise HTTPException(status_code=400, detail="Layout revision lacks reference image snapshot.")
    if (
        rev.reference_image_sha256 != cam.reference_image_sha256
        or rev.reference_width != cam.reference_width
        or rev.reference_height != cam.reference_height
    ):
        raise HTTPException(
            status_code=400,
            detail="Layout reference snapshot does not match current camera reference image. Layout is stale.",
        )

    # Validate layout geometry
    spaces_dicts = [
        {
            "id": sp.id,
            "operator_label": sp.operator_label,
            "space_type": sp.space_type,
            "polygon_normalized": sp.polygon_normalized,
            "active": sp.active,
        }
        for sp in rev.parking_spaces
    ]
    approaches_dicts = [
        {
            "id": sp.approach_zone.id,
            "parking_space_id": sp.id,
            "polygon_normalized": sp.approach_zone.polygon_normalized,
        }
        for sp in rev.parking_spaces
        if sp.approach_zone
    ]
    validation_errors = validate_parking_layout(
        parking_spaces=spaces_dicts,
        approach_zones=approaches_dicts,
        reference_width=cam.reference_width,
        reference_height=cam.reference_height,
    )
    if validation_errors:
        err_msgs = "; ".join([e.message for e in validation_errors[:3]])
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Cannot verify layout with geometry validation errors: {err_msgs}",
        )

    # Compute deterministic canonical fingerprint
    canonical_sha = compute_canonical_layout_sha256(
        schema_version="1.0.0",
        camera_id=rev.camera_id,
        reference_image_sha256=cam.reference_image_sha256,
        parking_spaces=spaces_dicts,
        approach_zones=approaches_dicts,
    )

    prior_status = rev.status

    # Mark any prior VERIFIED revision for this camera as SUPERSEDED in the same transaction
    prior_verified_res = await db.execute(
        select(ParkingLayoutRevision)
        .where(ParkingLayoutRevision.camera_id == rev.camera_id)
        .where(ParkingLayoutRevision.status == LayoutRevisionStatus.VERIFIED.value)
        .where(ParkingLayoutRevision.id != rev.id)
    )
    for prior_rev in prior_verified_res.scalars().all():
        prior_rev.status = LayoutRevisionStatus.SUPERSEDED.value
        supersede_audit = LayoutAuditEvent(
            id=str(uuid.uuid4()),
            layout_revision_id=prior_rev.id,
            event_type="SUPERSEDED",
            prior_status=LayoutRevisionStatus.VERIFIED.value,
            new_status=LayoutRevisionStatus.SUPERSEDED.value,
            local_operator_label=clean_op,
            note=f"Superseded by verified revision {rev.revision_number}",
            created_at=utc_now(),
        )
        db.add(supersede_audit)

    rev.status = LayoutRevisionStatus.VERIFIED.value
    rev.verified_at = utc_now()
    rev.canonical_sha256 = canonical_sha

    # Update camera calibration status to VERIFIED
    cam.calibration_status = CalibrationStatus.VERIFIED.value
    cam.updated_at = utc_now()

    # Record audit event
    audit = LayoutAuditEvent(
        id=str(uuid.uuid4()),
        layout_revision_id=rev.id,
        event_type="VERIFIED",
        prior_status=prior_status,
        new_status=LayoutRevisionStatus.VERIFIED.value,
        local_operator_label=clean_op,
        note=payload.note or f"Manually verified by local operator: {clean_op}",
        created_at=utc_now(),
    )
    db.add(audit)

    try:
        await db.commit()
    except IntegrityError as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Conflict: Another layout revision was verified concurrently for this camera.",
        ) from e

    await db.refresh(rev)
    return _build_layout_response(rev)


@router.post("/layouts/{layout_id}/invalidate", response_model=LayoutRevisionResponse)
async def invalidate_layout(
    layout_id: str,
    payload: LayoutInvalidateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Explicitly invalidate a VERIFIED layout revision (VERIFIED -> INVALIDATED).
    Records reason and audit log.
    """
    result = await db.execute(
        select(ParkingLayoutRevision)
        .options(
            selectinload(ParkingLayoutRevision.parking_spaces).selectinload(ParkingSpace.approach_zone),
            selectinload(ParkingLayoutRevision.audit_events),
            selectinload(ParkingLayoutRevision.camera),
        )
        .where(ParkingLayoutRevision.id == layout_id)
    )
    rev = result.scalar_one_or_none()
    if not rev:
        raise HTTPException(status_code=404, detail="Layout revision not found")

    if rev.status != LayoutRevisionStatus.VERIFIED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot invalidate layout in '{rev.status}' status. Only VERIFIED layouts can be invalidated.",
        )

    prior_status = rev.status
    rev.status = LayoutRevisionStatus.INVALIDATED.value
    rev.invalidated_at = utc_now()
    rev.invalidation_reason = payload.invalidation_reason

    audit = LayoutAuditEvent(
        id=str(uuid.uuid4()),
        layout_revision_id=rev.id,
        event_type="INVALIDATED",
        prior_status=prior_status,
        new_status=LayoutRevisionStatus.INVALIDATED.value,
        local_operator_label=payload.local_operator_label,
        note=payload.note or payload.invalidation_reason,
        created_at=utc_now(),
    )
    db.add(audit)
    await db.commit()
    await db.refresh(rev)

    return _build_layout_response(rev)


