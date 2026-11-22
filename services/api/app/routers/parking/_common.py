"""Shared imports, constants, dependencies and the single parking ``APIRouter``.

Split from the original ``routers/parking.py`` without behaviour changes; the
sub-resource modules register their routes on this router in the original order.
"""

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

logger = logging.getLogger("services.api.app.routers.parking")  # original module name, kept stable

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

