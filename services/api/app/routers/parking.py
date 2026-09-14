"""FastAPI router for RoadSense SiteOps parking layout, calibration, and ROI editor endpoints."""

from __future__ import annotations

import hashlib
import io
import logging
import math
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import desc, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from services.api.app.core.config import settings, REPO_ROOT
from services.api.app.db.session import get_db
from services.api.app.models.entities import (
    ApproachZone,
    Camera,
    CameraStabilityAssessment,
    CameraStabilityAuditEvent,
    DriveSession,
    LayoutAuditEvent,
    ParkingHazardAssociation,
    ParkingHazardAuditEvent,
    ParkingJobAuditEvent,
    ParkingLayoutRevision,
    ParkingOccupancyJob,
    ParkingSpace,
    PavementInspectionRecord,
    RoadEvent,
    Site,
)
from services.api.app.schemas.parking import (
    AcknowledgeStabilityRequest,
    AuditEventResponse,
    BayCapacityDecisionResponse,
    CameraCreateRequest,
    CameraInvalidateCalibrationRequest,
    CameraOperationalGateResponse,
    CameraResponse,
    CameraStabilityAuditResponse,
    CapacitySnapshotResponse,
    HazardAssociationCreateRequest,
    HazardAssociationLifecycleRequest,
    HazardAssociationResponse,
    HazardAssociationReviewRequest,
    HazardAuditEventResponse,
    LayoutCreateRequest,
    LayoutInvalidateRequest,
    LayoutRevisionResponse,
    LayoutSubmitRequest,
    LayoutUpdateRequest,
    LayoutValidationErrorDict,
    LayoutValidationResponse,
    LayoutVerifyRequest,
    ParkingOccupancyJobCancelResponse,
    ParkingOccupancyJobDeleteRequest,
    ParkingOccupancyJobDeleteResponse,
    ParkingOccupancyJobResponse,
    ParkingSpaceSchema,
    PavementInspectionRecordCreateRequest,
    PavementInspectionRecordResponse,
    ApproachZoneSchema,
    PointSchema,
    SampleMeasurementSchema,
    SiteCreateRequest,
    SiteResponse,
    StabilityAssessmentCancelResponse,
    StabilityAssessmentResponse,
)
from src.parking.capacity_policy import (
    BayOccupancyEvidenceInput,
    CapacityPolicyConfig,
    HazardAssociationInput,
    HazardLifecycleState,
    HazardReviewState,
    HazardTarget,
    OperationalGate as PolicyOperationalGate,
    PavementInspectionEvidence,
    evaluate_safe_usable_capacity,
)
from services.api.app.services.parking_occupancy_job_manager import (
    ParkingOccupancyJobManager,
    parking_occupancy_job_manager,
)
from services.api.app.services.reference_image_service import (
    ReferenceImageProcessingError,
    ReferenceImageSecurityError,
    process_and_store_upload_file,
    validate_served_file_path,
)
from services.api.app.services.stability_job_manager import (
    StabilityAssessmentJobManager,
    stability_job_manager,
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


_VALID_OCCUPANCY_STATES = frozenset({"VACANT", "OCCUPIED", "OCCLUDED", "UNKNOWN"})
_HAZARD_REVIEW_TRANSITIONS = {
    "UNREVIEWED": frozenset({"CONFIRMED", "REJECTED", "NEEDS_REVIEW"}),
    "NEEDS_REVIEW": frozenset({"CONFIRMED", "REJECTED"}),
    "CONFIRMED": frozenset({"NEEDS_REVIEW"}),
    "REJECTED": frozenset({"NEEDS_REVIEW"}),
}
_HAZARD_LIFECYCLE_TRANSITIONS = {
    "ACTIVE": frozenset({"MITIGATED", "RESOLVED", "EXPIRED", "SUPERSEDED"}),
    "MITIGATED": frozenset({"ACTIVE", "RESOLVED", "EXPIRED", "SUPERSEDED"}),
    "RESOLVED": frozenset({"ACTIVE"}),
    "EXPIRED": frozenset({"ACTIVE", "SUPERSEDED"}),
    "SUPERSEDED": frozenset(),
}


def _read_bay_occupancy_summary(
    bay_summary: object,
    bay_id: str,
) -> Tuple[str, float]:
    """Translate persisted occupancy output into fail-closed capacity evidence."""
    if not isinstance(bay_summary, dict):
        return "UNKNOWN", 0.0

    persisted = bay_summary.get(bay_id)
    if not isinstance(persisted, dict):
        return "UNKNOWN", 0.0

    state_value = persisted.get("current_state")
    if not isinstance(state_value, str):
        return "UNKNOWN", 0.0
    state = state_value.strip().upper()
    if state not in _VALID_OCCUPANCY_STATES:
        return "UNKNOWN", 0.0

    confidence_value = persisted.get("confidence")
    if isinstance(confidence_value, bool) or not isinstance(confidence_value, (int, float)):
        return "UNKNOWN", 0.0
    confidence = float(confidence_value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return "UNKNOWN", 0.0

    return state, confidence

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["parking"])


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_storage_root() -> Path:
    if hasattr(settings, "MEDIA_ROOT") and settings.MEDIA_ROOT:
        root = Path(settings.MEDIA_ROOT)
    else:
        root = REPO_ROOT / "outputs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_parking_job_manager() -> ParkingOccupancyJobManager:
    return parking_occupancy_job_manager


def get_stability_job_manager() -> StabilityAssessmentJobManager:
    return stability_job_manager



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
    samples_out: List[SampleMeasurementSchema] = []
    if a.sample_measurements:
        for s in a.sample_measurements:
            samples_out.append(
                SampleMeasurementSchema(
                    sample_index=int(s["sample_index"]),
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
            )

    return StabilityAssessmentResponse(
        id=a.id,
        camera_id=a.camera_id,
        layout_revision_id=a.layout_revision_id,
        layout_canonical_sha256=a.layout_canonical_sha256,
        reference_image_sha256=a.reference_image_sha256,
        video_sha256=a.video_sha256,
        status=a.status or "COMPLETE",
        progress_pct=a.progress_pct if a.progress_pct is not None else 100.0,
        stage_message=a.stage_message,
        failure_code=a.failure_code,
        failure_message=a.failure_message,
        config_version=a.config_version,
        config_sha256=a.config_sha256,
        algorithm_version=a.algorithm_version,
        opencv_version=a.opencv_version,
        thresholds_snapshot=a.thresholds_snapshot,
        sample_measurements=samples_out,
        aggregate_decision=a.aggregate_decision,
        operational_gate=a.operational_gate,
        gate_reasons=a.gate_reasons or [],
        summary_metrics=a.summary_metrics,
        created_at=a.created_at,
        started_at=a.started_at,
        completed_at=a.completed_at,
        operator_acknowledged_at=a.operator_acknowledged_at,
        operator_label=a.operator_label,
        operator_note=a.operator_note,
    )


# ============================================================================
# Camera Stability Endpoints
# ============================================================================

@router.post("/cameras/{camera_id}/stability/assess", response_model=StabilityAssessmentResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_camera_stability_assessment(
    camera_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    stab_manager: StabilityAssessmentJobManager = Depends(get_stability_job_manager),
    storage_root: Path = Depends(get_storage_root),
):
    """
    Execute an asynchronous bounded camera stability assessment against the camera's reference image.
    Accepts multipart video upload only, returns HTTP 202 with job progress tracking.
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

    # Stream video to secure staging path
    camera_dir = storage_root / "staging" / camera_id
    camera_dir.mkdir(parents=True, exist_ok=True)
    assessment_id = str(uuid.uuid4())
    temp_video_path = camera_dir / f"assess_stream_{assessment_id}.mp4"

    cfg = stab_manager.config
    MAX_VIDEO_BYTES = cfg.intake.max_upload_bytes
    total_bytes = 0

    try:
        with open(temp_video_path, "wb") as vf:
            while chunk := await file.read(65536):
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_BYTES:
                    temp_video_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Uploaded video exceeds configured limit of {MAX_VIDEO_BYTES // (1024*1024)} MB.",
                    )
                vf.write(chunk)

        if total_bytes == 0:
            temp_video_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded video file is empty (0 bytes).",
            )

        assessment = CameraStabilityAssessment(
            id=assessment_id,
            camera_id=camera_id,
            layout_revision_id=active_layout_id,
            layout_canonical_sha256=active_layout_sha,
            reference_image_sha256=cam.reference_image_sha256,
            video_sha256=None,
            status="QUEUED",
            progress_pct=0.0,
            stage_message="Queued for stability evaluation",
            config_version=cfg.version,
            config_sha256=cfg.config_sha256,
            algorithm_version=cfg.algorithm_version,
            opencv_version=None,
            thresholds_snapshot=cfg.thresholds_snapshot,
            sample_measurements=[],
            aggregate_decision=None,
            operational_gate=OperationalGate.BLOCKED.value,
            gate_reasons=["PROCESSING: Assessment job is currently running."],
            summary_metrics=None,
            created_at=utc_now(),
        )
        db.add(assessment)
        await db.commit()
        await db.refresh(assessment)

        # Launch background evaluation worker
        stab_manager.submit_assessment_job(
            assessment_id=assessment_id,
            video_path=temp_video_path,
            camera_id=camera_id,
            reference_image_bytes=ref_bytes,
            expected_reference_sha256=cam.reference_image_sha256,
            active_layout_canonical_sha256=active_layout_sha,
        )

        return _build_stability_response(assessment)

    except HTTPException:
        temp_video_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        temp_video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/stability/assessments/{assessment_id}/cancel", response_model=StabilityAssessmentCancelResponse)
async def cancel_stability_assessment(
    assessment_id: str,
    db: AsyncSession = Depends(get_db),
    stab_manager: StabilityAssessmentJobManager = Depends(get_stability_job_manager),
):
    """Safely cancel an in-progress or queued stability assessment."""
    res = await db.execute(select(CameraStabilityAssessment).where(CameraStabilityAssessment.id == assessment_id))
    a = res.scalar_one_or_none()
    if not a:
        raise HTTPException(status_code=404, detail="Stability assessment not found")

    if a.status in ("COMPLETE", "FAILED", "CANCELLED"):
        return StabilityAssessmentCancelResponse(
            assessment_id=assessment_id,
            status=a.status,
            cancelled=False,
            message=f"Assessment is already in terminal state '{a.status}'.",
        )

    await stab_manager.cancel_assessment_job(assessment_id)
    a.status = "CANCELLED"
    a.progress_pct = 100.0
    a.stage_message = "Assessment cancelled by operator"
    a.failure_code = "CANCELLED"
    a.failure_message = "Assessment cancelled by operator request."
    a.aggregate_decision = StabilityDecision.ERROR.value
    a.operational_gate = OperationalGate.BLOCKED.value
    a.gate_reasons = ["CANCELLED: Assessment cancelled by operator."]
    a.completed_at = utc_now()
    await db.commit()

    return StabilityAssessmentCancelResponse(
        assessment_id=assessment_id,
        status="CANCELLED",
        cancelled=True,
        message="Stability assessment cancelled successfully.",
    )


@router.get("/cameras/{camera_id}/stability/gate", response_model=CameraOperationalGateResponse)
async def get_camera_operational_gate(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
    stab_manager: StabilityAssessmentJobManager = Depends(get_stability_job_manager),
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

    # Find latest completed assessment for camera
    assess_res = await db.execute(
        select(CameraStabilityAssessment)
        .where(CameraStabilityAssessment.camera_id == camera_id)
        .where(CameraStabilityAssessment.status == "COMPLETE")
        .order_by(desc(CameraStabilityAssessment.created_at))
    )
    latest_assessment = assess_res.scalars().first()

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
    cfg = stab_manager.config
    max_age = (latest_assessment.thresholds_snapshot or {}).get("max_assessment_age_seconds", cfg.thresholds.max_assessment_age_seconds)

    gate, reasons = evaluate_operational_gate(
        decision=decision_enum,
        assessment_reference_sha=latest_assessment.reference_image_sha256,
        current_reference_sha=cam.reference_image_sha256,
        assessment_layout_sha=latest_assessment.layout_canonical_sha256,
        current_layout_sha=current_layout_sha,
        assessment_timestamp=latest_assessment.created_at,
        current_timestamp=now,
        max_age_seconds=max_age,
    )

    t_ass = latest_assessment.created_at if latest_assessment.created_at.tzinfo else latest_assessment.created_at.replace(tzinfo=timezone.utc)
    age_seconds = (now - t_ass).total_seconds()
    is_fresh = (0.0 <= age_seconds <= max_age) and (gate == OperationalGate.ALLOWED)

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


@router.post("/stability/assessments/{assessment_id}/acknowledge", response_model=CameraStabilityAuditResponse, status_code=status.HTTP_201_CREATED)
async def acknowledge_stability_assessment(
    assessment_id: str,
    payload: AcknowledgeStabilityRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Operator acknowledgement of a stability assessment outcome.
    Appends an immutable audit event and optionally triggers audited calibration invalidation for UNSTABLE assessments.
    """
    res = await db.execute(
        select(CameraStabilityAssessment)
        .where(CameraStabilityAssessment.id == assessment_id)
    )
    a = res.scalar_one_or_none()
    if not a:
        raise HTTPException(status_code=404, detail="Stability assessment not found")

    if a.status != "COMPLETE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot acknowledge assessment in non-terminal status '{a.status}'.",
        )

    cam_res = await db.execute(select(Camera).where(Camera.id == a.camera_id))
    cam = cam_res.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Associated camera not found")

    # Get active verified layout if present
    layout_res = await db.execute(
        select(ParkingLayoutRevision)
        .where(ParkingLayoutRevision.camera_id == a.camera_id)
        .where(ParkingLayoutRevision.status == LayoutRevisionStatus.VERIFIED.value)
    )
    verified_layouts = layout_res.scalars().all()
    current_layout = verified_layouts[0] if verified_layouts else None
    current_layout_sha = current_layout.canonical_sha256 if current_layout else None

    prev_gate = a.operational_gate or OperationalGate.BLOCKED.value
    prev_calib = cam.calibration_status
    resulting_gate = prev_gate
    resulting_calib = prev_calib
    event_type = "OPERATOR_ACKNOWLEDGED"

    if payload.trigger_calibration_invalidation:
        if not payload.confirm_invalidation:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Explicit confirmation (confirm_invalidation=True) is required to invalidate camera calibration.",
            )
        if not payload.invalidation_reason or not payload.invalidation_reason.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A non-blank invalidation reason is required to invalidate camera calibration.",
            )
        if a.aggregate_decision != StabilityDecision.UNSTABLE.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Calibration invalidation via this endpoint is strictly allowed only for UNSTABLE assessments; current assessment is {a.aggregate_decision}.",
            )
        if a.reference_image_sha256 != cam.reference_image_sha256:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Assessment reference image SHA does not match the camera's current reference image SHA.",
            )
        if current_layout_sha and a.layout_canonical_sha256 != current_layout_sha:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Assessment layout canonical SHA does not match the active verified layout SHA.",
            )

        # Staleness check
        max_age = (a.thresholds_snapshot or {}).get("max_assessment_age_seconds", 86400)
        t_ass = a.created_at if a.created_at.tzinfo else a.created_at.replace(tzinfo=timezone.utc)
        age = (utc_now() - t_ass).total_seconds()
        if age > max_age:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot invalidate calibration from an expired assessment ({int(age)}s old > {max_age}s max age).",
            )

        # Invalidate camera calibration
        cam.calibration_status = CalibrationStatus.INVALIDATED.value
        cam.updated_at = utc_now()
        resulting_calib = CalibrationStatus.INVALIDATED.value
        resulting_gate = OperationalGate.BLOCKED.value
        event_type = "CALIBRATION_INVALIDATED"

        for rev in verified_layouts:
            rev.status = LayoutRevisionStatus.INVALIDATED.value
            rev.invalidated_at = utc_now()
            rev.invalidation_reason = f"Camera stability assessment UNSTABLE: {payload.invalidation_reason.strip()}"
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

    # Compute assessment summary SHA
    summary_str = f"{a.id}:{a.video_sha256}:{a.reference_image_sha256}:{a.aggregate_decision}:{a.operational_gate}"
    ass_sha = hashlib.sha256(summary_str.encode("utf-8")).hexdigest()

    audit_event = CameraStabilityAuditEvent(
        id=str(uuid.uuid4()),
        assessment_id=a.id,
        camera_id=cam.id,
        event_type=event_type,
        operator_identity=payload.local_operator_label,
        explicit_reason=payload.invalidation_reason.strip() if payload.invalidation_reason else (payload.note or "Operator acknowledged stability assessment outcome."),
        note=payload.note,
        previous_gate_state=prev_gate,
        resulting_gate_state=resulting_gate,
        previous_calibration_status=prev_calib,
        resulting_calibration_status=resulting_calib,
        assessment_sha256=ass_sha,
        config_sha256=a.config_sha256,
        reference_image_sha256=cam.reference_image_sha256,
        layout_canonical_sha256=current_layout_sha,
        created_at=utc_now(),
    )
    db.add(audit_event)

    await db.commit()
    await db.refresh(audit_event)

    return CameraStabilityAuditResponse(
        id=audit_event.id,
        assessment_id=audit_event.assessment_id,
        camera_id=audit_event.camera_id,
        event_type=audit_event.event_type,
        operator_identity=audit_event.operator_identity,
        explicit_reason=audit_event.explicit_reason,
        note=audit_event.note,
        previous_gate_state=audit_event.previous_gate_state,
        resulting_gate_state=audit_event.resulting_gate_state,
        previous_calibration_status=audit_event.previous_calibration_status,
        resulting_calibration_status=audit_event.resulting_calibration_status,
        assessment_sha256=audit_event.assessment_sha256,
        config_sha256=audit_event.config_sha256,
        reference_image_sha256=audit_event.reference_image_sha256,
        layout_canonical_sha256=audit_event.layout_canonical_sha256,
        created_at=audit_event.created_at,
    )


@router.get("/cameras/{camera_id}/stability/audit-events", response_model=List[CameraStabilityAuditResponse])
async def list_camera_stability_audit_events(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
):
    """List append-only camera stability audit and acknowledgement events."""
    cam_res = await db.execute(select(Camera).where(Camera.id == camera_id))
    if not cam_res.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Camera not found")

    res = await db.execute(
        select(CameraStabilityAuditEvent)
        .where(CameraStabilityAuditEvent.camera_id == camera_id)
        .order_by(desc(CameraStabilityAuditEvent.created_at))
    )
    events = res.scalars().all()
    return [
        CameraStabilityAuditResponse(
            id=e.id,
            assessment_id=e.assessment_id,
            camera_id=e.camera_id,
            event_type=e.event_type,
            operator_identity=e.operator_identity,
            explicit_reason=e.explicit_reason,
            note=e.note,
            previous_gate_state=e.previous_gate_state,
            resulting_gate_state=e.resulting_gate_state,
            previous_calibration_status=e.previous_calibration_status,
            resulting_calibration_status=e.resulting_calibration_status,
            assessment_sha256=e.assessment_sha256,
            config_sha256=e.config_sha256,
            reference_image_sha256=e.reference_image_sha256,
            layout_canonical_sha256=e.layout_canonical_sha256,
            created_at=e.created_at,
        )
        for e in events
    ]


# ============================================================================
# Phase 2B: Gate-Controlled Parking Occupancy & Video Inference Endpoints
# ============================================================================

def _resolve_safe_parking_artifact(
    job_id: str,
    artifact_filename: str,
    expected_sha256: Optional[str] = None,
    job_status: Optional[str] = "COMPLETE",
    storage_root: Optional[Path] = None,
) -> Optional[Path]:
    """
    Resolves and verifies an artifact path with fail-closed security guarantees:
    - Enforces allowed filename whitelist
    - Requires job status to be COMPLETE
    - Validates lexical confinement under media_root/parking_jobs/{job_id}
    - Rejects any symlinks in path components
    - Verifies byte-level SHA-256 hash match against database provenance
    """
    import re
    if not job_id or not re.match(r"^[a-f0-9\-]{36}$", job_id):
        return None

    ALLOWED_ARTIFACTS = {
        "annotated.mp4",
        "occupancy_timeline.jsonl",
        "processing_manifest.json",
        "parking_summary.json",
    }
    if artifact_filename not in ALLOWED_ARTIFACTS:
        return None

    # Strict authorization: require job status to be exactly COMPLETE
    if job_status != "COMPLETE":
        return None

    if not expected_sha256 or len(expected_sha256) != 64:
        return None

    raw_root = storage_root if storage_root is not None else get_storage_root()
    if raw_root.is_symlink() or os.path.islink(str(raw_root)):
        return None

    media_root = raw_root.resolve()
    jobs_root = (media_root / "parking_jobs").resolve()
    if jobs_root.is_symlink() or os.path.islink(str(jobs_root)):
        return None

    candidate = jobs_root / job_id / artifact_filename
    if ".." in candidate.parts:
        return None

    curr = candidate
    while curr != jobs_root and curr != curr.parent:
        if curr.is_symlink() or os.path.islink(str(curr)):
            return None
        curr = curr.parent

    resolved_jobs_root = jobs_root
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_jobs_root)
    except ValueError:
        return None

    if not resolved.exists() or not resolved.is_file() or resolved.is_symlink():
        return None

    # Recompute file SHA on disk and verify match
    h = hashlib.sha256()
    with open(resolved, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    if h.hexdigest().lower() != expected_sha256.lower():
        logger.warning(f"Artifact SHA-256 verification failed for {resolved}: expected {expected_sha256}, got {h.hexdigest()}")
        return None

    return resolved


def create_staged_upload_file(storage_root: Optional[Path] = None) -> Tuple[Path, int]:
    """
    Creates an exclusively locked staged upload file under configured upload_staging root.
    Rejects symlinks in raw path chain, enforces confinement, and uses a neutral staging suffix.
    """
    raw_root = storage_root if storage_root is not None else get_storage_root()
    if raw_root.is_symlink() or os.path.islink(str(raw_root)):
        raise ValueError(f"Symlink found in media root path component: {raw_root}")

    media_root = raw_root.resolve()
    upload_staging_dir = media_root / "upload_staging"
    if upload_staging_dir.is_symlink() or os.path.islink(str(upload_staging_dir)):
        raise ValueError(f"Symlink found in upload staging path component: {upload_staging_dir}")

    upload_staging_dir.mkdir(parents=True, exist_ok=True)
    resolved_staging = upload_staging_dir.resolve()

    staging_file_name = f"upload_{uuid.uuid4().hex}.upload.tmp"
    staging_file_path = upload_staging_dir / staging_file_name

    if ".." in staging_file_path.parts:
        raise ValueError("Path traversal in staging file path")

    resolved_path = staging_file_path.resolve()
    try:
        resolved_path.relative_to(resolved_staging)
    except ValueError:
        raise ValueError("Staging file escapes upload staging root")

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    fd = os.open(str(staging_file_path), flags, 0o600)
    return staging_file_path, fd


def _build_occupancy_job_response(job: ParkingOccupancyJob, storage_root: Optional[Path] = None) -> ParkingOccupancyJobResponse:
    timeline_sha = (job.manifest_json or {}).get("timeline_sha256") or ((job.manifest_json or {}).get("software_versions") or {}).get("timeline_sha256")
    safe_video = _resolve_safe_parking_artifact(job.id, "annotated.mp4", job.output_video_sha256, job.status, storage_root=storage_root)
    safe_timeline = _resolve_safe_parking_artifact(job.id, "occupancy_timeline.jsonl", timeline_sha, job.status, storage_root=storage_root) if timeline_sha else None
    has_video = safe_video is not None
    has_timeline = safe_timeline is not None
    has_manifest = bool(job.manifest_json is not None and job.status == "COMPLETE")
    return ParkingOccupancyJobResponse(
        id=job.id,
        camera_id=job.camera_id,
        site_id=job.site_id,
        layout_revision_id=job.layout_revision_id,
        stability_assessment_id=job.stability_assessment_id,
        status=job.status,
        progress_pct=job.progress_pct,
        stage_message=job.stage_message,
        failure_code=job.failure_code,
        failure_message=job.failure_message,
        gate_decision=job.gate_decision,
        gate_reasons=job.gate_reasons,
        input_video_sha256=job.input_video_sha256,
        output_video_sha256=job.output_video_sha256,
        reference_image_sha256=job.reference_image_sha256,
        layout_canonical_sha256=job.layout_canonical_sha256,
        detector_checkpoint_sha256=job.detector_checkpoint_sha256,
        occupancy_config_sha256=job.occupancy_config_sha256,
        total_frames=job.total_frames,
        processed_frames=job.processed_frames,
        fps=job.fps,
        duration_seconds=job.duration_seconds,
        video_width=job.video_width,
        video_height=job.video_height,
        total_bays=job.total_bays,
        final_occupied_count=job.final_occupied_count,
        final_vacant_count=job.final_vacant_count,
        final_unknown_count=job.final_unknown_count,
        final_occluded_count=job.final_occluded_count,
        total_state_transitions=job.total_state_transitions,
        has_annotated_video=has_video,
        has_timeline=has_timeline,
        has_manifest=has_manifest,
        bay_summary=job.bay_summary_json,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.post("/cameras/{camera_id}/occupancy/jobs", response_model=ParkingOccupancyJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_parking_occupancy_job(
    camera_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    manager: ParkingOccupancyJobManager = Depends(get_parking_job_manager),
    storage_root: Path = Depends(get_storage_root),
):
    """
    Submit a video file for gate-controlled parking occupancy evaluation and video annotation.
    Enforces fail-closed camera stability gate: returns HTTP 202; background job will set
    BLOCKED_BY_STABILITY_GATE if camera is not calibrated with an active verified layout and fresh STABLE assessment.
    """
    cam_res = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = cam_res.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    cfg = manager.config
    max_upload_bytes = cfg.execution.max_upload_bytes

    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded.")

    try:
        temp_video_path, fd = create_staged_upload_file(storage_root=storage_root)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create staged upload file: {e}")

    bytes_read = 0
    try:
        with os.fdopen(fd, "wb") as f_out:
            while chunk := await file.read(65536):
                bytes_read += len(chunk)
                if bytes_read > max_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Uploaded file exceeds maximum allowed size of {max_upload_bytes} bytes ({max_upload_bytes / (1024*1024):.1f}MB).",
                    )
                f_out.write(chunk)
            f_out.flush()
            os.fsync(f_out.fileno())
    except HTTPException:
        temp_video_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        temp_video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to stream upload: {e}")

    if bytes_read == 0:
        temp_video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty (0 bytes).")

    job_id = str(uuid.uuid4())
    job = ParkingOccupancyJob(
        id=job_id,
        camera_id=cam.id,
        site_id=cam.site_id,
        status="QUEUED",
        progress_pct=0.0,
        stage_message="Job queued for gate validation and evaluation",
        created_at=utc_now(),
    )
    try:
        db.add(job)
        await db.commit()
        await db.refresh(job)
    except Exception as e:
        temp_video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to persist job record: {e}")

    # Launch async job manager execution
    manager.submit_occupancy_job(
        job_id=job_id,
        camera_id=cam.id,
        site_id=cam.site_id,
        video_path=temp_video_path,
    )

    return _build_occupancy_job_response(job, storage_root=storage_root)


@router.get("/parking/jobs/{job_id}", response_model=ParkingOccupancyJobResponse)
async def get_parking_occupancy_job_detail(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    storage_root: Path = Depends(get_storage_root),
):
    """Retrieve status, metrics, and manifest summary of a parking occupancy job."""
    res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
    job = res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Parking occupancy job not found")
    return _build_occupancy_job_response(job, storage_root=storage_root)


@router.get("/cameras/{camera_id}/occupancy/jobs", response_model=List[ParkingOccupancyJobResponse])
async def list_camera_occupancy_jobs(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
    storage_root: Path = Depends(get_storage_root),
):
    """List historical parking occupancy jobs for a camera."""
    cam_res = await db.execute(select(Camera).where(Camera.id == camera_id))
    if not cam_res.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Camera not found")

    res = await db.execute(
        select(ParkingOccupancyJob)
        .where(ParkingOccupancyJob.camera_id == camera_id)
        .order_by(desc(ParkingOccupancyJob.created_at))
    )
    jobs = res.scalars().all()
    return [_build_occupancy_job_response(j, storage_root=storage_root) for j in jobs]


@router.post("/parking/jobs/{job_id}/cancel", response_model=ParkingOccupancyJobCancelResponse)
async def cancel_parking_occupancy_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    manager: ParkingOccupancyJobManager = Depends(get_parking_job_manager),
):
    """Cancel an active or queued parking occupancy job atomically through manager CAS."""
    res = await db.execute(select(ParkingOccupancyJob.id).where(ParkingOccupancyJob.id == job_id))
    if not res.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Parking occupancy job not found")

    result = await manager.request_job_cancellation(job_id)
    return ParkingOccupancyJobCancelResponse(
        job_id=result["job_id"],
        status=result["status"],
        cancelled=result["cancelled"],
        message=result["message"],
    )


@router.get("/parking/jobs/{job_id}/video")
async def get_parking_job_annotated_video(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    storage_root: Path = Depends(get_storage_root),
):
    """Stream the generated annotated MP4 video with strict authorization."""
    res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
    job = res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Parking occupancy job not found")

    if job.status != "COMPLETE":
        raise HTTPException(status_code=404, detail=f"Annotated video is only available for COMPLETE jobs (current status: '{job.status}').")

    if not job.output_video_sha256 or len(job.output_video_sha256) != 64:
        raise HTTPException(status_code=404, detail="Annotated video metadata SHA is missing or invalid.")

    # Validate provenance in manifest if present
    if job.manifest_json and job.manifest_json.get("output_video_sha256") != job.output_video_sha256:
        raise HTTPException(status_code=404, detail="Manifest provenance mismatch for annotated video.")

    safe_path = _resolve_safe_parking_artifact(job.id, "annotated.mp4", job.output_video_sha256, job.status, storage_root=storage_root)
    if not safe_path:
        raise HTTPException(status_code=404, detail="Annotated video file not available or failed integrity verification.")

    return FileResponse(
        path=safe_path,
        media_type="video/mp4",
        filename=f"annotated_parking_{job_id[:8]}.mp4",
    )


@router.get("/parking/jobs/{job_id}/manifest")
async def get_parking_job_manifest(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    storage_root: Path = Depends(get_storage_root),
):
    """Retrieve the full processing manifest JSON for a COMPLETE parking occupancy job."""
    res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
    job = res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Parking occupancy job not found")

    if job.status != "COMPLETE":
        raise HTTPException(status_code=404, detail=f"Manifest is only available for COMPLETE jobs (current status: '{job.status}').")

    if not job.manifest_json:
        raise HTTPException(status_code=404, detail="Processing manifest not available for this job.")

    manifest_sha = (job.manifest_json or {}).get("manifest_sha256")
    safe_path = _resolve_safe_parking_artifact(job.id, "processing_manifest.json", manifest_sha, job.status, storage_root=storage_root) if manifest_sha else None
    if safe_path and safe_path.is_file():
        return FileResponse(
            path=safe_path,
            media_type="application/json",
            filename=f"processing_manifest_{job_id[:8]}.json",
        )
    return job.manifest_json


@router.get("/parking/jobs/{job_id}/timeline")
async def get_parking_job_timeline(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    storage_root: Path = Depends(get_storage_root),
):
    """Download the state-transition timeline JSONL ledger with strict authorization."""
    res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
    job = res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Parking occupancy job not found")

    if job.status != "COMPLETE":
        raise HTTPException(status_code=404, detail=f"Timeline is only available for COMPLETE jobs (current status: '{job.status}').")

    manifest = job.manifest_json or {}
    timeline_sha = manifest.get("timeline_sha256") or (manifest.get("software_versions") or {}).get("timeline_sha256")
    if not timeline_sha or len(timeline_sha) != 64:
        raise HTTPException(status_code=404, detail="Timeline SHA not found in manifest or is invalid.")

    safe_path = _resolve_safe_parking_artifact(job.id, "occupancy_timeline.jsonl", timeline_sha, job.status, storage_root=storage_root)
    if not safe_path:
        raise HTTPException(status_code=404, detail="Timeline ledger file not available or failed integrity verification.")

    return FileResponse(
        path=safe_path,
        media_type="application/x-ndjson",
        filename=f"occupancy_timeline_{job_id[:8]}.jsonl",
    )


@router.get("/parking/jobs/{job_id}/summary")
async def get_parking_job_summary(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    storage_root: Path = Depends(get_storage_root),
):
    """Download the final occupancy summary JSON for a COMPLETE parking occupancy job."""
    res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
    job = res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Parking occupancy job not found")

    if job.status != "COMPLETE":
        raise HTTPException(status_code=404, detail=f"Summary is only available for COMPLETE jobs (current status: '{job.status}').")

    manifest = job.manifest_json or {}
    summary_sha = manifest.get("summary_sha256") or (manifest.get("software_versions") or {}).get("summary_sha256")
    if not summary_sha or len(summary_sha) != 64:
        raise HTTPException(status_code=404, detail="Summary SHA not found in manifest or is invalid.")

    safe_path = _resolve_safe_parking_artifact(job.id, "parking_summary.json", summary_sha, job.status, storage_root=storage_root)
    if not safe_path:
        raise HTTPException(status_code=404, detail="Summary JSON file not available or failed integrity verification.")

    return FileResponse(
        path=safe_path,
        media_type="application/json",
        filename=f"parking_summary_{job_id[:8]}.json",
    )


@router.delete("/parking/jobs/{job_id}", response_model=ParkingOccupancyJobDeleteResponse)
async def delete_parking_occupancy_job(
    job_id: str,
    req: ParkingOccupancyJobDeleteRequest,
    db: AsyncSession = Depends(get_db),
    storage_root: Path = Depends(get_storage_root),
):
    """
    Safely delete a parking occupancy job and purge its local artifacts with fail-closed retention controls.

    Requirements:
    - confirmation_acknowledged must be true
    - non-blank local_operator_label assertion
    - meaningful deletion_reason (>= 5 chars)
    - Terminal states only: COMPLETE, FAILED, CANCELLED, BLOCKED_BY_STABILITY_GATE
    - Guarded DB CAS to PURGING
    - Confined filesystem validation (no traversal or symlinks)
    - Atomic quarantine rename before purge; restored on DB commit failure
    - Minimal DELETED tombstone & append-only ParkingJobAuditEvent recorded in DB
    """
    request_time = datetime.now(timezone.utc)

    if not req.confirmation_acknowledged:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Explicit confirmation (confirmation_acknowledged=true) is required for deletion.",
        )

    op_label = (req.local_operator_label or "").strip()
    if not op_label:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Non-blank local_operator_label is required.",
        )

    reason = (req.deletion_reason or "").strip()
    if len(reason) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A meaningful deletion_reason (at least 5 characters) is required.",
        )

    # 1. Fetch current job for state check
    res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
    job = res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Parking occupancy job not found")

    terminal_statuses = {"COMPLETE", "FAILED", "CANCELLED", "BLOCKED_BY_STABILITY_GATE"}

    if job.status == "DELETED":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job {job_id} has already been deleted and purged.",
        )

    if job.status not in terminal_statuses:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot delete job in non-terminal state '{job.status}'. Deletion is strictly permitted only from terminal states ({', '.join(sorted(terminal_statuses))}).",
        )

    if req.expected_status and job.status != req.expected_status:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Concurrency mismatch: expected job status '{req.expected_status}', but current status is '{job.status}'.",
        )

    prior_status = job.status

    # 2. Guarded DB CAS: transition status from terminal state to PURGING
    from sqlalchemy import update
    conditions = [
        ParkingOccupancyJob.id == job_id,
        ParkingOccupancyJob.status.in_(list(terminal_statuses)),
    ]
    if req.expected_status:
        conditions.append(ParkingOccupancyJob.status == req.expected_status)

    cas_stmt = (
        update(ParkingOccupancyJob)
        .where(*conditions)
        .values(status="PURGING")
    )
    cas_res = await db.execute(cas_stmt)
    if cas_res.rowcount == 0:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Failed to reserve PURGING state: job status changed concurrently.",
        )
    await db.flush()

    # Capture artifact hashes before disk purge
    artifact_hashes = {
        "input_video_sha256": job.input_video_sha256,
        "output_video_sha256": job.output_video_sha256,
        "reference_image_sha256": job.reference_image_sha256,
        "layout_canonical_sha256": job.layout_canonical_sha256,
        "manifest_sha256": (job.manifest_json or {}).get("manifest_sha256"),
        "summary_sha256": (job.manifest_json or {}).get("summary_sha256"),
        "timeline_sha256": (job.manifest_json or {}).get("timeline_sha256"),
    }

    # 3. Path confinement & symlink validation before any filesystem operations
    media_root = storage_root.resolve()
    if media_root.is_symlink():
        await db.rollback()
        raise HTTPException(status_code=500, detail="Storage root directory cannot be a symlink.")

    jobs_root = (media_root / "parking_jobs").resolve()
    if jobs_root.is_symlink():
        await db.rollback()
        raise HTTPException(status_code=500, detail="Parking jobs root directory cannot be a symlink.")

    job_dir = jobs_root / job_id
    if ".." in job_dir.parts:
        await db.rollback()
        raise HTTPException(status_code=400, detail="Invalid job path.")

    # Check symlinks down to jobs_root
    curr = job_dir
    while curr != jobs_root and curr != curr.parent:
        if curr.is_symlink():
            await db.rollback()
            raise HTTPException(status_code=400, detail="Symlinked path component detected in job directory hierarchy.")
        curr = curr.parent

    # 4. Atomic quarantine rename & strict rmtree
    quarantine_dir = jobs_root / f".quarantine_{job_id}_{uuid.uuid4().hex[:8]}"
    quarantined = False
    purge_result = "NOOP"

    try:
        if job_dir.exists():
            if job_dir.is_symlink():
                raise RuntimeError("Job directory is a symlink.")
            job_dir.rename(quarantine_dir)
            quarantined = True

            import shutil
            shutil.rmtree(quarantine_dir)
            quarantined = False
            purge_result = "SUCCESS"

        # 5. DB transition to DELETED tombstone & record audit event
        job.status = "DELETED"
        job.output_video_path = None
        job.timeline_jsonl_path = None
        job.stage_message = f"Purged by {op_label}: {reason}"

        audit_event = ParkingJobAuditEvent(
            id=str(uuid.uuid4()),
            job_id=job.id,
            event_type="JOB_DELETED_AND_PURGED",
            operator_identity_assertion=op_label,
            deletion_reason=reason,
            prior_status=prior_status,
            resulting_status="DELETED",
            artifact_hashes=artifact_hashes,
            purge_result=purge_result,
            request_time=request_time,
            completed_time=datetime.now(timezone.utc),
        )
        db.add(audit_event)
        await db.commit()

    except Exception as e:
        await db.rollback()
        # Restore quarantined directory if DB commit or rmtree failed
        if quarantined and quarantine_dir.exists() and not job_dir.exists():
            try:
                quarantine_dir.rename(job_dir)
            except Exception as restore_err:
                logger.error(f"Failed to restore quarantined directory {quarantine_dir} to {job_dir}: {restore_err}")
        logger.exception(f"Failed during deletion of job {job_id}")
        raise HTTPException(status_code=500, detail=f"Job deletion failed: {str(e)}")

    completed_time = datetime.now(timezone.utc)
    return ParkingOccupancyJobDeleteResponse(
        deleted=True,
        job_id=job_id,
        prior_status=prior_status,
        resulting_status="DELETED",
        operator_identity_assertion=op_label,
        deletion_reason=reason,
        purged_at=completed_time,
        message=f"Job {job_id} artifacts successfully purged and job record marked DELETED.",
    )


# ============================================================================
# Phase 3A: Safe Usable Capacity & Hazard Association Endpoints
# ============================================================================

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
