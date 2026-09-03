"""Pydantic schemas for DriveSession."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    mode: str
    source_filename: str
    source_hash: str
    processing_state: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    source_duration_seconds: Optional[float] = None
    source_fps: Optional[float] = None
    source_width: Optional[int] = None
    source_height: Optional[int] = None
    total_source_frames: Optional[int] = None
    sampled_frames_count: Optional[int] = None
    frames_with_detections: Optional[int] = None
    total_detections_count: Optional[int] = None
    processing_duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    media_metadata: Optional[dict[str, Any]] = None
    model_provenance: Optional[dict[str, Any]] = None
    route_telemetry: Optional[list[dict[str, Any]]] = None


class SessionProcessRequest(BaseModel):
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    sampling_fps: float = Field(5.0, gt=0.0, le=30.0)
    max_frames: int = Field(150, gt=0, le=1000)
    window_start_seconds: float = Field(0.0, ge=0.0)
    window_duration_seconds: float = Field(30.0, gt=0.0)
    apply_privacy_mask: bool = Field(False)


class SessionProgressEvent(BaseModel):
    session_id: str
    stage: str # validating, decoding, detecting, tracking, fusing_events, encoding, complete, failed, cancelled
    processed_frames: int = 0
    total_frames: int = 0
    percentage: float = 0.0
    current_fps: Optional[float] = None
    message: Optional[str] = None
    detections_found: int = 0


class SessionDeleteRequest(BaseModel):
    delete_source_media: bool = True
    delete_artifacts: bool = True
    delete_database_record: bool = True
