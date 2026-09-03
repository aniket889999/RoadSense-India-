"""Pydantic schemas for RawDetection and RoadEvent."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class RawDetectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    frame_index: int
    timestamp_seconds: float
    confidence: float
    class_id: int = 0
    track_id: Optional[int] = None
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    road_event_id: Optional[str] = None
    created_at: datetime


class RoadEventReviewRequest(BaseModel):
    action: str = Field(..., pattern="^(CONFIRM|REJECT|NEEDS_REVISIT|SPLIT|MERGE)$")
    reviewer_note: Optional[str] = Field(None, max_length=1000)


class ReviewActionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    event_id: str
    action: str
    previous_status: str
    new_status: str
    reviewer_note: Optional[str] = None
    created_at: datetime


class RoadEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    first_seen_seconds: float
    last_seen_seconds: float
    first_frame_index: int
    last_frame_index: int
    track_id: Optional[int] = None
    representative_detection_id: Optional[str] = None
    representative_confidence: float
    representative_bbox: dict[str, float]
    support_count: int
    evidence_crop_path: Optional[str] = None
    review_status: str
    reviewer_note: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    created_at: datetime
    review_actions: List[ReviewActionResponse] = []
