"""Pydantic schemas for RoadSense SiteOps parking layout and geometry endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class PointSchema(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized X coordinate in [0.0, 1.0]")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized Y coordinate in [0.0, 1.0]")


class ParkingSpaceSchema(BaseModel):
    id: Optional[str] = None
    operator_label: str = Field(..., min_length=1, max_length=64)
    space_type: str = Field(default="STANDARD", max_length=32)
    polygon_normalized: List[PointSchema] = Field(..., min_length=3)
    active: bool = True


class ApproachZoneSchema(BaseModel):
    id: Optional[str] = None
    parking_space_id: str
    polygon_normalized: List[PointSchema] = Field(..., min_length=3)


# --- Site Schemas ---
class SiteCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: Optional[str] = None
    timezone: str = Field(default="UTC", max_length=64)


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


class CameraResponse(BaseModel):
    id: str
    site_id: str
    name: str
    description: Optional[str] = None
    reference_image_path: Optional[str] = None
    reference_image_sha256: Optional[str] = None
    reference_width: Optional[int] = None
    reference_height: Optional[int] = None
    calibration_status: str
    camera_position_description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    active_verified_layout_id: Optional[str] = None


class CameraInvalidateCalibrationRequest(BaseModel):
    local_operator_label: str = Field(..., min_length=1, max_length=128)
    invalidation_reason: str = Field(..., min_length=3)
    note: Optional[str] = None


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
    local_operator_label: Optional[str] = None
    note: Optional[str] = None


class LayoutVerifyRequest(BaseModel):
    local_operator_label: str = Field(..., min_length=1, max_length=128, description="Local operator identifier")
    confirmation_acknowledged: bool = Field(..., description="Explicit acknowledgement of manual polygon verification")
    note: Optional[str] = None


class LayoutInvalidateRequest(BaseModel):
    local_operator_label: str = Field(..., min_length=1, max_length=128)
    invalidation_reason: str = Field(..., min_length=3)
    note: Optional[str] = None


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
    status: str
    canonical_sha256: Optional[str] = None
    created_at: datetime
    submitted_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    invalidated_at: Optional[datetime] = None
    invalidation_reason: Optional[str] = None
    parking_spaces: List[ParkingSpaceSchema] = Field(default_factory=list)
    approach_zones: List[ApproachZoneSchema] = Field(default_factory=list)
    audit_events: List[AuditEventResponse] = Field(default_factory=list)
