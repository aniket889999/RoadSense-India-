"""FastAPI router for RoadSense SiteOps parking layout, calibration, and ROI editor endpoints."""

from __future__ import annotations

import io
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import desc, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from services.api.app.core.config import settings, REPO_ROOT
from services.api.app.db.session import get_db
from services.api.app.models.entities import (
    ApproachZone,
    Camera,
    CameraStabilityAssessment,
    LayoutAuditEvent,
    ParkingLayoutRevision,
    ParkingSpace,
    Site,
)
from services.api.app.schemas.parking import (
    AcknowledgeStabilityRequest,
    AuditEventResponse,
    CameraCreateRequest,
    CameraInvalidateCalibrationRequest,
    CameraOperationalGateResponse,
    CameraResponse,
    LayoutCreateRequest,
    LayoutInvalidateRequest,
    LayoutRevisionResponse,
    LayoutSubmitRequest,
    LayoutUpdateRequest,
    LayoutValidationErrorDict,
    LayoutValidationResponse,
    LayoutVerifyRequest,
    ParkingSpaceSchema,
    ApproachZoneSchema,
    PointSchema,
    SampleMeasurementSchema,
    SiteCreateRequest,
    SiteResponse,
    StabilityAssessmentResponse,
)
from services.api.app.services.reference_image_service import (
    ReferenceImageProcessingError,
    ReferenceImageSecurityError,
    process_and_store_upload_file,
    validate_served_file_path,
)

from src.parking.contracts import (
    CalibrationStatus,
    LayoutRevisionStatus,
    OperationalGate,
    SpaceType,
    StabilityDecision,
    StabilityThresholds,
    evaluate_operational_gate,
)
from src.parking.layout_serialization import compute_canonical_layout_sha256
from src.parking.layout_validation import validate_parking_layout
from src.parking.stability_engine import evaluate_video_camera_stability

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["parking"])


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_storage_root() -> Path:
    root = (REPO_ROOT / "outputs").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root



# ============================================================================
# Site Endpoints
# ============================================================================

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
):
    """Upload and normalize a reference frame image for a camera."""
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = result.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    storage_root = get_storage_root()

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
async def get_reference_image(camera_id: str, db: AsyncSession = Depends(get_db)):
    """Serve the stored reference image for display on the ROI canvas."""
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = result.scalar_one_or_none()
    if not cam or not cam.reference_image_path:
        raise HTTPException(status_code=404, detail="Reference image not found for this camera")

    storage_root = get_storage_root()
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


def _build_stability_response(a: CameraStabilityAssessment) -> StabilityAssessmentResponse:
    return StabilityAssessmentResponse(
        id=a.id,
        camera_id=a.camera_id,
        layout_revision_id=a.layout_revision_id,
        layout_canonical_sha256=a.layout_canonical_sha256,
        reference_image_sha256=a.reference_image_sha256,
        video_sha256=a.video_sha256,
        algorithm_version=a.algorithm_version,
        opencv_version=a.opencv_version,
        thresholds_snapshot=a.thresholds_snapshot,
        sample_measurements=[
            SampleMeasurementSchema(
                sample_index=s["sample_index"],
                timestamp_seconds=float(s["timestamp_seconds"]),
                frame_index=int(s["frame_index"]),
                matched_features=int(s["matched_features"]),
                inlier_count=int(s["inlier_count"]),
                inlier_ratio=float(s["inlier_ratio"]),
                translation_px_x=float(s["translation_px_x"]),
                translation_px_y=float(s["translation_px_y"]),
                translation_magnitude_px=float(s["translation_magnitude_px"]),
                translation_normalized=float(s["translation_normalized"]),
                scale_factor=float(s["scale_factor"]),
                scale_change=float(s["scale_change"]),
                rotation_degrees=float(s["rotation_degrees"]),
                perspective_distortion=float(s["perspective_distortion"]),
                reprojection_error=float(s["reprojection_error"]),
                decision=s["decision"],
                rejection_reasons=s.get("rejection_reasons", []),
            )
            for s in a.sample_measurements
        ],
        aggregate_decision=a.aggregate_decision,
        operational_gate=a.operational_gate,
        gate_reasons=a.gate_reasons,
        summary_metrics=a.summary_metrics,
        created_at=a.created_at,
        operator_acknowledged_at=a.operator_acknowledged_at,
        operator_label=a.operator_label,
        operator_note=a.operator_note,
    )


# ============================================================================
# Camera Stability Endpoints
# ============================================================================

@router.post("/cameras/{camera_id}/stability/assess", response_model=StabilityAssessmentResponse, status_code=status.HTTP_201_CREATED)
async def create_camera_stability_assessment(
    camera_id: str,
    file: Optional[UploadFile] = File(None),
    local_video_name: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Execute a bounded local video camera stability assessment against the camera's reference image.
    Computes motion metrics and fail-closed operational gate status.
    """
    cam_res = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = cam_res.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not cam.reference_image_path or not cam.reference_image_sha256:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera has no reference image uploaded. Stability assessment requires a verified reference frame.",
        )

    storage_root = get_storage_root()
    ref_file_path = validate_served_file_path(cam.reference_image_path, storage_root)
    if not ref_file_path.exists():
        raise HTTPException(status_code=404, detail="Reference image file missing from storage.")

    with open(ref_file_path, "rb") as rf:
        ref_bytes = rf.read()

    # Find active verified layout if present
    layout_res = await db.execute(
        select(ParkingLayoutRevision)
        .where(ParkingLayoutRevision.camera_id == camera_id)
        .where(ParkingLayoutRevision.status == LayoutRevisionStatus.VERIFIED.value)
    )
    verified_layout = layout_res.scalar_one_or_none()
    active_layout_id = verified_layout.id if verified_layout else None
    active_layout_sha = verified_layout.canonical_sha256 if verified_layout else None

    # Handle incoming video
    temp_video_path: Optional[Path] = None
    target_video_path: Optional[Path] = None

    try:
        if file is not None:
            # Stream upload bounded to 100MB
            camera_dir = storage_root / "staging" / camera_id
            camera_dir.mkdir(parents=True, exist_ok=True)
            temp_video_path = camera_dir / f"assess_stream_{uuid.uuid4().hex}.mp4"

            MAX_VIDEO_BYTES = 100 * 1024 * 1024
            total_bytes = 0
            with open(temp_video_path, "wb") as vf:
                while chunk := await file.read(65536):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_VIDEO_BYTES:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"Uploaded video exceeds maximum allowed size ({MAX_VIDEO_BYTES // (1024*1024)} MiB).",
                        )
                    vf.write(chunk)
            target_video_path = temp_video_path
        elif local_video_name:
            # Safe local video name lookup
            clean_name = os.path.basename(local_video_name.strip())
            candidate_paths = [
                storage_root / "staging" / camera_id / clean_name,
                Path("/Users/aniket/Downloads") / clean_name,
            ]
            for cp in candidate_paths:
                if cp.exists() and cp.is_file():
                    target_video_path = cp
                    break
            if not target_video_path:
                raise HTTPException(status_code=404, detail=f"Local video '{clean_name}' not found.")
        else:
            raise HTTPException(status_code=400, detail="Either 'file' or 'local_video_name' must be provided.")

        # Run stability assessment engine
        result = evaluate_video_camera_stability(
            video_path=target_video_path,
            reference_image_bytes=ref_bytes,
            expected_reference_sha256=cam.reference_image_sha256,
            active_layout_canonical_sha256=active_layout_sha,
            sample_count=5,
            max_duration_sec=60.0,
            max_bytes=100 * 1024 * 1024,
        )

        assessment = CameraStabilityAssessment(
            id=str(uuid.uuid4()),
            camera_id=camera_id,
            layout_revision_id=active_layout_id,
            layout_canonical_sha256=active_layout_sha,
            reference_image_sha256=result.reference_image_sha256,
            video_sha256=result.video_sha256,
            algorithm_version=result.algorithm_version,
            opencv_version=result.opencv_version,
            thresholds_snapshot=result.thresholds_snapshot,
            sample_measurements=[s.to_dict() for s in result.samples],
            aggregate_decision=result.aggregate_decision.value,
            operational_gate=result.operational_gate.value,
            gate_reasons=result.gate_reasons,
            summary_metrics=result.summary_metrics,
            created_at=utc_now(),
        )
        db.add(assessment)
        await db.commit()
        await db.refresh(assessment)

        return _build_stability_response(assessment)

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    finally:
        if temp_video_path and temp_video_path.exists():
            try:
                temp_video_path.unlink()
            except Exception:
                pass


@router.get("/cameras/{camera_id}/stability/gate", response_model=CameraOperationalGateResponse)
async def get_camera_operational_gate(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve the current fail-closed operational inference gate for a camera.
    Inference is ALLOWED only if a fresh STABLE assessment matches both reference and layout SHAs.
    """
    cam_res = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = cam_res.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    layout_res = await db.execute(
        select(ParkingLayoutRevision)
        .where(ParkingLayoutRevision.camera_id == camera_id)
        .where(ParkingLayoutRevision.status == LayoutRevisionStatus.VERIFIED.value)
    )
    verified_layout = layout_res.scalar_one_or_none()
    current_layout_sha = verified_layout.canonical_sha256 if verified_layout else None

    # Get latest stability assessment
    ass_res = await db.execute(
        select(CameraStabilityAssessment)
        .where(CameraStabilityAssessment.camera_id == camera_id)
        .order_by(desc(CameraStabilityAssessment.created_at))
        .limit(1)
    )
    latest_assessment = ass_res.scalar_one_or_none()

    if not latest_assessment:
        gate, reasons = evaluate_operational_gate(
            decision=None,
            assessment_reference_sha=None,
            current_reference_sha=cam.reference_image_sha256,
            assessment_layout_sha=None,
            current_layout_sha=current_layout_sha,
        )
        return CameraOperationalGateResponse(
            camera_id=camera_id,
            operational_gate=gate.value,
            gate_reasons=reasons,
            aggregate_decision=None,
            assessment_id=None,
            reference_image_sha256=cam.reference_image_sha256,
            layout_canonical_sha256=current_layout_sha,
            created_at=None,
            is_fresh=False,
        )

    decision_enum = None
    try:
        decision_enum = StabilityDecision(latest_assessment.aggregate_decision)
    except Exception:
        pass

    now = utc_now()
    gate, reasons = evaluate_operational_gate(
        decision=decision_enum,
        assessment_reference_sha=latest_assessment.reference_image_sha256,
        current_reference_sha=cam.reference_image_sha256,
        assessment_layout_sha=latest_assessment.layout_canonical_sha256,
        current_layout_sha=current_layout_sha,
        assessment_timestamp=latest_assessment.created_at,
        current_timestamp=now,
    )

    t_ass = latest_assessment.created_at.astimezone(timezone.utc) if latest_assessment.created_at.tzinfo else latest_assessment.created_at.replace(tzinfo=timezone.utc)
    age_seconds = (now - t_ass).total_seconds()
    is_fresh = (age_seconds <= 86400) and (gate == OperationalGate.ALLOWED)

    return CameraOperationalGateResponse(
        camera_id=camera_id,
        operational_gate=gate.value,
        gate_reasons=reasons,
        aggregate_decision=latest_assessment.aggregate_decision,
        assessment_id=latest_assessment.id,
        reference_image_sha256=cam.reference_image_sha256,
        layout_canonical_sha256=current_layout_sha,
        created_at=latest_assessment.created_at,
        is_fresh=is_fresh,
    )


@router.get("/cameras/{camera_id}/stability/assessments", response_model=List[StabilityAssessmentResponse])
async def list_camera_stability_assessments(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
):
    """List historical camera stability assessments ordered by creation time."""
    cam_res = await db.execute(select(Camera).where(Camera.id == camera_id))
    if not cam_res.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Camera not found")

    res = await db.execute(
        select(CameraStabilityAssessment)
        .where(CameraStabilityAssessment.camera_id == camera_id)
        .order_by(desc(CameraStabilityAssessment.created_at))
    )
    assessments = res.scalars().all()
    return [_build_stability_response(a) for a in assessments]


@router.get("/stability/assessments/{assessment_id}", response_model=StabilityAssessmentResponse)
async def get_stability_assessment_detail(
    assessment_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve full stability assessment report including per-sample measurements."""
    res = await db.execute(select(CameraStabilityAssessment).where(CameraStabilityAssessment.id == assessment_id))
    a = res.scalar_one_or_none()
    if not a:
        raise HTTPException(status_code=404, detail="Stability assessment not found")
    return _build_stability_response(a)


@router.post("/stability/assessments/{assessment_id}/acknowledge", response_model=StabilityAssessmentResponse)
async def acknowledge_stability_assessment(
    assessment_id: str,
    payload: AcknowledgeStabilityRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Operator acknowledgement of a stability assessment outcome.
    Optionally triggers audited camera calibration invalidation if the assessment is UNSTABLE.
    """
    res = await db.execute(select(CameraStabilityAssessment).where(CameraStabilityAssessment.id == assessment_id))
    a = res.scalar_one_or_none()
    if not a:
        raise HTTPException(status_code=404, detail="Stability assessment not found")

    a.operator_acknowledged_at = utc_now()
    a.operator_label = payload.local_operator_label
    a.operator_note = payload.note

    if payload.trigger_calibration_invalidation:
        cam_res = await db.execute(select(Camera).where(Camera.id == a.camera_id))
        cam = cam_res.scalar_one_or_none()
        if cam:
            cam.calibration_status = CalibrationStatus.INVALIDATED.value
            cam.updated_at = utc_now()

            # Invalidate any active verified layouts
            rev_res = await db.execute(
                select(ParkingLayoutRevision)
                .where(ParkingLayoutRevision.camera_id == a.camera_id)
                .where(ParkingLayoutRevision.status == LayoutRevisionStatus.VERIFIED.value)
            )
            for rev in rev_res.scalars().all():
                rev.status = LayoutRevisionStatus.INVALIDATED.value
                rev.invalidated_at = utc_now()
                rev.invalidation_reason = f"Camera stability assessment UNSTABLE: {payload.invalidation_reason or payload.note or 'Geometry drift detected'}"

                audit = LayoutAuditEvent(
                    id=str(uuid.uuid4()),
                    layout_revision_id=rev.id,
                    event_type="CALIBRATION_INVALIDATED",
                    prior_status=LayoutRevisionStatus.VERIFIED.value,
                    new_status=LayoutRevisionStatus.INVALIDATED.value,
                    local_operator_label=payload.local_operator_label,
                    note=payload.note or f"Stability assessment {assessment_id} flagged UNSTABLE.",
                    created_at=utc_now(),
                )
                db.add(audit)

    await db.commit()
    await db.refresh(a)
    return _build_stability_response(a)
