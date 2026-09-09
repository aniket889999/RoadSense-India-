"""Pydantic schemas for RoadSense SiteOps parking layout and geometry endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


def _strip_and_require_nonblank(v: Any, field_name: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{field_name} must be a non-blank string.")
    return v.strip()


def _strip_optional_nonblank(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    cleaned = v.strip()
    return cleaned if cleaned else None


class PointSchema(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized X coordinate in [0.0, 1.0]")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized Y coordinate in [0.0, 1.0]")


SpaceTypeLiteral = Literal["STANDARD", "ACCESSIBLE", "EV_CHARGING", "LOADING", "EMERGENCY", "OTHER"]
LayoutStatusLiteral = Literal["DRAFT", "PENDING_REVIEW", "VERIFIED", "SUPERSEDED", "INVALIDATED"]
CalibrationStatusLiteral = Literal["NOT_CONFIGURED", "PENDING_REVIEW", "VERIFIED", "INVALIDATED"]


class ParkingSpaceSchema(BaseModel):
    id: Optional[str] = None
    operator_label: str = Field(..., min_length=1, max_length=64)
    space_type: SpaceTypeLiteral = Field(default="STANDARD")
    polygon_normalized: List[PointSchema] = Field(..., min_length=3)
    active: bool = True

    @field_validator("operator_label")
    @classmethod
    def validate_operator_label(cls, v: str) -> str:
        return _strip_and_require_nonblank(v, "operator_label")


class ApproachZoneSchema(BaseModel):
    id: Optional[str] = None
    parking_space_id: str
    polygon_normalized: List[PointSchema] = Field(..., min_length=3)

    @field_validator("parking_space_id")
    @classmethod
    def validate_parking_space_id(cls, v: str) -> str:
        return _strip_and_require_nonblank(v, "parking_space_id")


# --- Site Schemas ---
class SiteCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: Optional[str] = None
    timezone: str = Field(default="UTC", max_length=64)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return _strip_and_require_nonblank(v, "name")

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        return _strip_and_require_nonblank(v, "timezone")

    @field_validator("description")
    @classmethod
    def validate_desc(cls, v: Optional[str]) -> Optional[str]:
        return _strip_optional_nonblank(v)


class SiteResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    timezone: str
    active: bool
    created_at: datetime
    updated_at: datetime
    cameras_count: int = 0


# --- Camera Schemas ---
class CameraCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: Optional[str] = None
    camera_position_description: Optional[str] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return _strip_and_require_nonblank(v, "name")

    @field_validator("description", "camera_position_description")
    @classmethod
    def validate_optional_text(cls, v: Optional[str]) -> Optional[str]:
        return _strip_optional_nonblank(v)


class CameraResponse(BaseModel):
    id: str
    site_id: str
    name: str
    description: Optional[str] = None
    reference_image_path: Optional[str] = None
    reference_image_sha256: Optional[str] = None
    reference_width: Optional[int] = None
    reference_height: Optional[int] = None
    calibration_status: CalibrationStatusLiteral
    camera_position_description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    active_verified_layout_id: Optional[str] = None


class CameraInvalidateCalibrationRequest(BaseModel):
    local_operator_label: str = Field(..., min_length=1, max_length=128)
    invalidation_reason: str = Field(..., min_length=3)
    note: Optional[str] = None

    @field_validator("local_operator_label")
    @classmethod
    def validate_op(cls, v: str) -> str:
        return _strip_and_require_nonblank(v, "local_operator_label")

    @field_validator("invalidation_reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        return _strip_and_require_nonblank(v, "invalidation_reason")

    @field_validator("note")
    @classmethod
    def validate_note(cls, v: Optional[str]) -> Optional[str]:
        return _strip_optional_nonblank(v)


# --- Layout Schemas ---
class LayoutCreateRequest(BaseModel):
    parking_spaces: List[ParkingSpaceSchema] = Field(default_factory=list)
    approach_zones: Optional[List[ApproachZoneSchema]] = Field(default_factory=list)


class LayoutUpdateRequest(BaseModel):
    parking_spaces: List[ParkingSpaceSchema] = Field(default_factory=list)
    approach_zones: Optional[List[ApproachZoneSchema]] = Field(default_factory=list)


class LayoutValidationErrorDict(BaseModel):
    rule_id: str
    message: str
    space_id: Optional[str] = None
    space_label: Optional[str] = None


class LayoutValidationResponse(BaseModel):
    is_valid: bool
    errors: List[LayoutValidationErrorDict] = Field(default_factory=list)
    spaces_count: int
    approach_zones_count: int


class LayoutSubmitRequest(BaseModel):
    local_operator_label: str = Field(..., min_length=1, max_length=128, description="Non-blank local operator assertion")
    note: Optional[str] = None

    @field_validator("local_operator_label")
    @classmethod
    def validate_op(cls, v: str) -> str:
        return _strip_and_require_nonblank(v, "local_operator_label")

    @field_validator("note")
    @classmethod
    def validate_note(cls, v: Optional[str]) -> Optional[str]:
        return _strip_optional_nonblank(v)


class LayoutVerifyRequest(BaseModel):
    local_operator_label: str = Field(..., min_length=1, max_length=128, description="Local operator identifier")
    confirmation_acknowledged: bool = Field(..., description="Explicit acknowledgement of manual polygon verification")
    note: Optional[str] = None

    @field_validator("local_operator_label")
    @classmethod
    def validate_op(cls, v: str) -> str:
        return _strip_and_require_nonblank(v, "local_operator_label")

    @field_validator("note")
    @classmethod
    def validate_note(cls, v: Optional[str]) -> Optional[str]:
        return _strip_optional_nonblank(v)


class LayoutInvalidateRequest(BaseModel):
    local_operator_label: str = Field(..., min_length=1, max_length=128)
    invalidation_reason: str = Field(..., min_length=3)
    note: Optional[str] = None

    @field_validator("local_operator_label")
    @classmethod
    def validate_op(cls, v: str) -> str:
        return _strip_and_require_nonblank(v, "local_operator_label")

    @field_validator("invalidation_reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        return _strip_and_require_nonblank(v, "invalidation_reason")

    @field_validator("note")
    @classmethod
    def validate_note(cls, v: Optional[str]) -> Optional[str]:
        return _strip_optional_nonblank(v)


class AuditEventResponse(BaseModel):
    id: str
    layout_revision_id: str
    event_type: str
    prior_status: Optional[str] = None
    new_status: str
    local_operator_label: Optional[str] = None
    note: Optional[str] = None
    created_at: datetime


class LayoutRevisionResponse(BaseModel):
    id: str
    camera_id: str
    revision_number: int
    status: LayoutStatusLiteral
    canonical_sha256: Optional[str] = None
    reference_image_sha256: Optional[str] = None
    reference_width: Optional[int] = None
    reference_height: Optional[int] = None
    created_at: datetime
    submitted_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    invalidated_at: Optional[datetime] = None
    invalidation_reason: Optional[str] = None
    parking_spaces: List[ParkingSpaceSchema] = Field(default_factory=list)
    approach_zones: List[ApproachZoneSchema] = Field(default_factory=list)
    audit_events: List[AuditEventResponse] = Field(default_factory=list)


StabilityDecisionLiteral = Literal["STABLE", "UNSTABLE", "INSUFFICIENT_EVIDENCE", "ERROR"]
OperationalGateLiteral = Literal["ALLOWED", "BLOCKED"]
StabilityStatusLiteral = Literal["QUEUED", "VALIDATING", "ANALYZING", "COMPLETE", "FAILED", "CANCELLED"]


class SampleMeasurementSchema(BaseModel):
    sample_index: int
    timestamp_seconds: float
    frame_index: int
    matched_features: int
    inlier_count: int
    inlier_ratio: float
    translation_px_x: float
    translation_px_y: float
    translation_magnitude_px: float
    translation_normalized: float
    scale_factor: float
    scale_change: float
    rotation_degrees: float
    perspective_distortion: float
    reprojection_error: float
    decision: StabilityDecisionLiteral
    rejection_reasons: List[str] = Field(default_factory=list)


class StabilityAssessmentResponse(BaseModel):
    id: str
    camera_id: str
    layout_revision_id: Optional[str] = None
    layout_canonical_sha256: Optional[str] = None
    reference_image_sha256: str
    video_sha256: Optional[str] = None
    status: StabilityStatusLiteral = "COMPLETE"
    progress_pct: float = 100.0
    stage_message: Optional[str] = None
    failure_code: Optional[str] = None
    failure_message: Optional[str] = None
    config_version: Optional[str] = None
    config_sha256: Optional[str] = None
    algorithm_version: Optional[str] = None
    opencv_version: Optional[str] = None
    thresholds_snapshot: Optional[dict[str, Any]] = None
    sample_measurements: List[SampleMeasurementSchema] = Field(default_factory=list)
    aggregate_decision: Optional[StabilityDecisionLiteral] = None
    operational_gate: Optional[OperationalGateLiteral] = None
    gate_reasons: List[str] = Field(default_factory=list)
    summary_metrics: Optional[dict[str, Any]] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    operator_acknowledged_at: Optional[datetime] = None
    operator_label: Optional[str] = None
    operator_note: Optional[str] = None


class CameraStabilityAuditResponse(BaseModel):
    id: str
    assessment_id: str
    camera_id: str
    event_type: str
    operator_identity: str
    explicit_reason: str
    note: Optional[str] = None
    previous_gate_state: Optional[str] = None
    resulting_gate_state: str
    previous_calibration_status: Optional[str] = None
    resulting_calibration_status: str
    assessment_sha256: Optional[str] = None
    config_sha256: Optional[str] = None
    reference_image_sha256: str
    layout_canonical_sha256: Optional[str] = None
    created_at: datetime


class CameraOperationalGateResponse(BaseModel):
    camera_id: str
    operational_gate: OperationalGateLiteral
    gate_reasons: List[str]
    aggregate_decision: Optional[StabilityDecisionLiteral] = None
    assessment_id: Optional[str] = None
    reference_image_sha256: Optional[str] = None
    layout_canonical_sha256: Optional[str] = None
    created_at: Optional[datetime] = None
    is_fresh: bool = False


class AcknowledgeStabilityRequest(BaseModel):
    local_operator_label: str = Field(..., min_length=1, max_length=128)
    note: Optional[str] = None
    trigger_calibration_invalidation: bool = False
    confirm_invalidation: bool = False
    invalidation_reason: Optional[str] = None

    @field_validator("local_operator_label", mode="after")
    @classmethod
    def validate_operator(cls, v: str) -> str:
        return _strip_and_require_nonblank(v, "local_operator_label")

    @field_validator("invalidation_reason", mode="after")
    @classmethod
    def validate_invalidation_reason(cls, v: Optional[str], values: Any) -> Optional[str]:
        if v is not None:
            cleaned = v.strip()
            return cleaned if cleaned else None
        return None


class StabilityAssessmentCancelResponse(BaseModel):
    assessment_id: str
    status: str
    cancelled: bool
    message: str


class ParkingOccupancyJobResponse(BaseModel):
    id: str
    camera_id: str
    site_id: str
    layout_revision_id: Optional[str] = None
    stability_assessment_id: Optional[str] = None

    status: str  # QUEUED, VALIDATING, DETECTING, TRACKING, CLASSIFYING_OCCUPANCY, RENDERING, ENCODING, COMPLETE, FAILED, CANCELLED, BLOCKED_BY_STABILITY_GATE
    progress_pct: float
    stage_message: Optional[str] = None
    failure_code: Optional[str] = None
    failure_message: Optional[str] = None

    gate_decision: Optional[str] = None
    gate_reasons: Optional[List[str]] = None

    input_video_sha256: Optional[str] = None
    output_video_sha256: Optional[str] = None
    reference_image_sha256: Optional[str] = None
    layout_canonical_sha256: Optional[str] = None
    detector_checkpoint_sha256: Optional[str] = None
    occupancy_config_sha256: Optional[str] = None

    total_frames: int = 0
    processed_frames: int = 0
    fps: float = 0.0
    duration_seconds: float = 0.0
    video_width: int = 0
    video_height: int = 0

    total_bays: int = 0
    final_occupied_count: int = 0
    final_vacant_count: int = 0
    final_unknown_count: int = 0
    final_occluded_count: int = 0
    total_state_transitions: int = 0

    has_annotated_video: bool = False
    has_timeline: bool = False
    has_manifest: bool = False
    bay_summary: Optional[Dict[str, Any]] = None

    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ParkingOccupancyJobCancelResponse(BaseModel):
    job_id: str
    status: str
    cancelled: bool
    message: str


class ParkingOccupancyJobDeleteRequest(BaseModel):
    """
    Fail-closed deletion and retention audit request.
    Note: local_operator_label is an operator identity assertion, not an authenticated SSO identity.
    """
    confirmation_acknowledged: bool = Field(..., description="Explicit confirmation of destructive deletion")
    local_operator_label: str = Field(..., min_length=1, max_length=128, description="Operator identity assertion")
    deletion_reason: str = Field(..., min_length=5, description="Meaningful audit reason for job and artifact purge")
    expected_status: Optional[str] = Field(None, description="Expected terminal status for concurrency CAS protection")


class ParkingOccupancyJobDeleteResponse(BaseModel):
    deleted: bool
    job_id: str
    prior_status: str
    resulting_status: str
    operator_identity_assertion: str
    deletion_reason: str
    purged_at: datetime
    message: str
