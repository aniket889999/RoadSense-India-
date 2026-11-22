"""Parking API: sites, cameras, reference images and calibration invalidation.

Moved verbatim from the original ``routers/parking.py`` (no behaviour change).
"""

from __future__ import annotations

from services.api.app.routers.parking._common import *  # noqa: F401,F403  (shared imports, models, schemas, router)


@router.post("/sites", response_model=SiteResponse, status_code=status.HTTP_201_CREATED)
async def create_site(payload: SiteCreateRequest, db: AsyncSession = Depends(get_db)):
    """Create a new facility site."""
    site = Site(
        id=str(uuid.uuid4()),
        name=payload.name.strip(),
        description=payload.description.strip() if payload.description else None,
        timezone=payload.timezone.strip(),
        active=True,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    db.add(site)
    await db.commit()
    await db.refresh(site)

    return SiteResponse(
        id=site.id,
        name=site.name,
        description=site.description,
        timezone=site.timezone,
        active=site.active,
        created_at=site.created_at,
        updated_at=site.updated_at,
        cameras_count=0,
    )


@router.get("/sites", response_model=List[SiteResponse])
async def list_sites(db: AsyncSession = Depends(get_db)):
    """List all configured facility sites."""
    result = await db.execute(
        select(Site, func.count(Camera.id).label("cameras_count"))
        .outerjoin(Camera, Camera.site_id == Site.id)
        .group_by(Site.id)
        .order_by(desc(Site.created_at))
    )
    rows = result.all()

    sites_out: List[SiteResponse] = []
    for site, cam_count in rows:
        sites_out.append(
            SiteResponse(
                id=site.id,
                name=site.name,
                description=site.description,
                timezone=site.timezone,
                active=site.active,
                created_at=site.created_at,
                updated_at=site.updated_at,
                cameras_count=cam_count,
            )
        )
    return sites_out


@router.get("/sites/{site_id}", response_model=SiteResponse)
async def get_site(site_id: str, db: AsyncSession = Depends(get_db)):
    """Get site details by ID."""
    result = await db.execute(select(Site).where(Site.id == site_id))
    site = result.scalar_one_or_none()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")

    cam_count_res = await db.execute(select(func.count(Camera.id)).where(Camera.site_id == site_id))
    cam_count = cam_count_res.scalar() or 0

    return SiteResponse(
        id=site.id,
        name=site.name,
        description=site.description,
        timezone=site.timezone,
        active=site.active,
        created_at=site.created_at,
        updated_at=site.updated_at,
        cameras_count=cam_count,
    )


# ============================================================================
# Camera Endpoints
# ============================================================================

@router.post("/sites/{site_id}/cameras", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(site_id: str, payload: CameraCreateRequest, db: AsyncSession = Depends(get_db)):
    """Create a new camera under a facility site."""
    site_res = await db.execute(select(Site).where(Site.id == site_id))
    if not site_res.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Site not found")

    camera = Camera(
        id=str(uuid.uuid4()),
        site_id=site_id,
        name=payload.name.strip(),
        description=payload.description.strip() if payload.description else None,
        calibration_status=CalibrationStatus.NOT_CONFIGURED.value,
        camera_position_description=payload.camera_position_description.strip() if payload.camera_position_description else None,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    db.add(camera)
    await db.commit()
    await db.refresh(camera)

    return CameraResponse(
        id=camera.id,
        site_id=camera.site_id,
        name=camera.name,
        description=camera.description,
        reference_image_path=camera.reference_image_path,
        reference_image_sha256=camera.reference_image_sha256,
        reference_width=camera.reference_width,
        reference_height=camera.reference_height,
        calibration_status=camera.calibration_status,
        camera_position_description=camera.camera_position_description,
        created_at=camera.created_at,
        updated_at=camera.updated_at,
        active_verified_layout_id=None,
    )


@router.get("/sites/{site_id}/cameras", response_model=List[CameraResponse])
async def list_site_cameras(site_id: str, db: AsyncSession = Depends(get_db)):
    """List all cameras associated with a site."""
    result = await db.execute(
        select(Camera).where(Camera.site_id == site_id).order_by(desc(Camera.created_at))
    )
    cameras = result.scalars().all()

    out: List[CameraResponse] = []
    for cam in cameras:
        # Check for active verified layout
        rev_res = await db.execute(
            select(ParkingLayoutRevision.id)
            .where(ParkingLayoutRevision.camera_id == cam.id)
            .where(ParkingLayoutRevision.status == LayoutRevisionStatus.VERIFIED.value)
            .order_by(desc(ParkingLayoutRevision.revision_number))
            .limit(1)
        )
        active_layout_id = rev_res.scalar_one_or_none()

        out.append(
            CameraResponse(
                id=cam.id,
                site_id=cam.site_id,
                name=cam.name,
                description=cam.description,
                reference_image_path=cam.reference_image_path,
                reference_image_sha256=cam.reference_image_sha256,
                reference_width=cam.reference_width,
                reference_height=cam.reference_height,
                calibration_status=cam.calibration_status,
                camera_position_description=cam.camera_position_description,
                created_at=cam.created_at,
                updated_at=cam.updated_at,
                active_verified_layout_id=active_layout_id,
            )
        )
    return out


@router.get("/cameras/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: str, db: AsyncSession = Depends(get_db)):
    """Get camera details by ID."""
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = result.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    rev_res = await db.execute(
        select(ParkingLayoutRevision.id)
        .where(ParkingLayoutRevision.camera_id == cam.id)
        .where(ParkingLayoutRevision.status == LayoutRevisionStatus.VERIFIED.value)
        .order_by(desc(ParkingLayoutRevision.revision_number))
        .limit(1)
    )
    active_layout_id = rev_res.scalar_one_or_none()

    return CameraResponse(
        id=cam.id,
        site_id=cam.site_id,
        name=cam.name,
        description=cam.description,
        reference_image_path=cam.reference_image_path,
        reference_image_sha256=cam.reference_image_sha256,
        reference_width=cam.reference_width,
        reference_height=cam.reference_height,
        calibration_status=cam.calibration_status,
        camera_position_description=cam.camera_position_description,
        created_at=cam.created_at,
        updated_at=cam.updated_at,
        active_verified_layout_id=active_layout_id,
    )


@router.post("/cameras/{camera_id}/reference-image", response_model=CameraResponse)
async def upload_reference_image(
    camera_id: str,
    file: UploadFile = File(...),
    confirm_replacement: bool = Form(False),
    operator_label: Optional[str] = Form(None),
    reason: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    storage_root: Path = Depends(get_storage_root),
):
    """Upload and normalize a reference frame image for a camera."""
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = result.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Check if camera has existing layouts
    layout_res = await db.execute(
        select(ParkingLayoutRevision).where(ParkingLayoutRevision.camera_id == camera_id)
    )
    existing_layouts = layout_res.scalars().all()
    has_active_layouts = any(
        l.status in (LayoutRevisionStatus.DRAFT.value, LayoutRevisionStatus.PENDING_REVIEW.value, LayoutRevisionStatus.VERIFIED.value)
        for l in existing_layouts
    )

    clean_op: Optional[str] = None
    clean_reason: Optional[str] = None

    if has_active_layouts:
        if not confirm_replacement:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CONFIRMATION_REQUIRED: Replacing reference image on a camera with active layouts requires explicit confirmation (confirm_replacement=true), a non-blank operator label, and a meaningful reason.",
            )
        if not operator_label or not operator_label.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Non-blank operator label is required when replacing reference image.",
            )
        if not reason or not reason.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Meaningful reason is required when replacing reference image.",
            )
        clean_op = operator_label.strip()
        clean_reason = reason.strip()

    try:
        processed = await process_and_store_upload_file(
            upload_file=file,
            camera_id=cam.id,
            storage_root=storage_root,
        )
    except ReferenceImageSecurityError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Security error: {str(e)}")
    except ReferenceImageProcessingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error processing reference image upload")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal image processing error")

    old_file_path = cam.reference_image_path
    old_sha = cam.reference_image_sha256

    try:
        # Transactionally invalidate existing layouts tied to prior image
        if has_active_layouts and clean_op and clean_reason:
            for rev in existing_layouts:
                if rev.status in (LayoutRevisionStatus.DRAFT.value, LayoutRevisionStatus.PENDING_REVIEW.value, LayoutRevisionStatus.VERIFIED.value):
                    prior_status = rev.status
                    rev.status = LayoutRevisionStatus.INVALIDATED.value
                    rev.invalidated_at = utc_now()
                    rev.invalidation_reason = f"Reference image replaced by {clean_op}: {clean_reason}"
                    db.add(LayoutAuditEvent(
                        id=str(uuid.uuid4()),
                        layout_revision_id=rev.id,
                        event_type="REFERENCE_IMAGE_REPLACED",
                        prior_status=prior_status,
                        new_status=LayoutRevisionStatus.INVALIDATED.value,
                        local_operator_label=clean_op,
                        note=f"Image SHA changed from {old_sha} to {processed.sha256}. Reason: {clean_reason}",
                        created_at=utc_now(),
                    ))

        # Update camera reference image metadata and set calibration status to PENDING_REVIEW (never VERIFIED)
        cam.reference_image_path = processed.relative_path
        cam.reference_image_sha256 = processed.sha256
        cam.reference_width = processed.width
        cam.reference_height = processed.height
        cam.calibration_status = CalibrationStatus.PENDING_REVIEW.value
        cam.updated_at = utc_now()

        await db.commit()
        await db.refresh(cam)

    except Exception:
        await db.rollback()
        # Clean up the newly created image on DB failure to preserve atomic intake
        try:
            Path(processed.absolute_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise

    # On DB commit success, clean up superseded image if no other camera references it
    if old_file_path and old_file_path != processed.relative_path:
        other_cams = await db.execute(select(Camera.id).where(Camera.reference_image_path == old_file_path))
        if not other_cams.scalars().all():
            try:
                (storage_root / old_file_path).unlink(missing_ok=True)
            except OSError:
                pass

    return CameraResponse(
        id=cam.id,
        site_id=cam.site_id,
        name=cam.name,
        description=cam.description,
        reference_image_path=cam.reference_image_path,
        reference_image_sha256=cam.reference_image_sha256,
        reference_width=cam.reference_width,
        reference_height=cam.reference_height,
        calibration_status=cam.calibration_status,
        camera_position_description=cam.camera_position_description,
        created_at=cam.created_at,
        updated_at=cam.updated_at,
    )


@router.get("/cameras/{camera_id}/reference-image")
async def get_reference_image(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
    storage_root: Path = Depends(get_storage_root),
):
    """Serve the stored reference image for display on the ROI canvas."""
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = result.scalar_one_or_none()
    if not cam or not cam.reference_image_path:
        raise HTTPException(status_code=404, detail="Reference image not found for this camera")
    try:
        image_file = validate_served_file_path(cam.reference_image_path, storage_root)
    except ReferenceImageSecurityError as e:
        logger.warning(f"Security error accessing reference image: {e}")
        raise HTTPException(status_code=404, detail="Image file not found on local storage")

    media_type = "image/png" if image_file.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(
        path=str(image_file),
        media_type=media_type,
        filename=image_file.name,
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/cameras/{camera_id}/invalidate-calibration", response_model=CameraResponse)
async def invalidate_camera_calibration(
    camera_id: str,
    payload: CameraInvalidateCalibrationRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Explicitly invalidate a camera's geometric calibration.
    Invalidates any active verified layout revision for occupancy use and records an audit event.
    """
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = result.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    cam.calibration_status = CalibrationStatus.INVALIDATED.value
    cam.updated_at = utc_now()

    # Find any active VERIFIED layout revision for this camera and mark as INVALIDATED
    rev_res = await db.execute(
        select(ParkingLayoutRevision)
        .where(ParkingLayoutRevision.camera_id == camera_id)
        .where(ParkingLayoutRevision.status == LayoutRevisionStatus.VERIFIED.value)
    )
    verified_revisions = rev_res.scalars().all()
    for rev in verified_revisions:
        rev.status = LayoutRevisionStatus.INVALIDATED.value
        rev.invalidated_at = utc_now()
        rev.invalidation_reason = f"Camera calibration invalidated: {payload.invalidation_reason}"

        audit = LayoutAuditEvent(
            id=str(uuid.uuid4()),
            layout_revision_id=rev.id,
            event_type="CALIBRATION_INVALIDATED",
            prior_status=LayoutRevisionStatus.VERIFIED.value,
            new_status=LayoutRevisionStatus.INVALIDATED.value,
            local_operator_label=payload.local_operator_label,
            note=payload.note or payload.invalidation_reason,
            created_at=utc_now(),
        )
        db.add(audit)

    await db.commit()
    await db.refresh(cam)

    return CameraResponse(
        id=cam.id,
        site_id=cam.site_id,
        name=cam.name,
        description=cam.description,
        reference_image_path=cam.reference_image_path,
        reference_image_sha256=cam.reference_image_sha256,
        reference_width=cam.reference_width,
        reference_height=cam.reference_height,
        calibration_status=cam.calibration_status,
        camera_position_description=cam.camera_position_description,
        created_at=cam.created_at,
        updated_at=cam.updated_at,
        active_verified_layout_id=None,
    )


# ============================================================================
# Parking Layout Revision Endpoints
# ============================================================================

