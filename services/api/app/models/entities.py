"""SQLAlchemy models for RoadSense India Operations Dashboard."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from services.api.app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Device(Base):
    __tablename__ = "devices"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=generate_uuid)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class DriveSession(Base):
    __tablename__ = "drive_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=generate_uuid)
    mode: Mapped[str] = mapped_column(String(32), default="upload", nullable=False) # upload | live
    source_filename: Mapped[str] = mapped_column(String(256), nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    processing_state: Mapped[str] = mapped_column(String(32), default="queued", nullable=False) # queued, validating, sampling, model_loading, processing, rendering, complete, failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Video Telemetry & Metrics
    source_duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source_fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_source_frames: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sampled_frames_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    frames_with_detections: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_detections_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    processing_duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Model Provenance snapshot (JSON)
    model_provenance: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Optional Route telemetry (JSON array of [lat, lon, timestamp] if present)
    route_telemetry: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSON, nullable=True)

    # Relationships
    detections: Mapped[List[RawDetection]] = relationship("RawDetection", back_populates="session", cascade="all, delete-orphan")
    road_events: Mapped[List[RoadEvent]] = relationship("RoadEvent", back_populates="session", cascade="all, delete-orphan")
    artifacts: Mapped[List[Artifact]] = relationship("Artifact", back_populates="session", cascade="all, delete-orphan")


class RawDetection(Base):
    __tablename__ = "raw_detections"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=generate_uuid)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("drive_sessions.id", ondelete="CASCADE"), nullable=False, index=True)

    frame_index: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    timestamp_seconds: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    class_id: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    x_min: Mapped[float] = mapped_column(Float, nullable=False)
    y_min: Mapped[float] = mapped_column(Float, nullable=False)
    x_max: Mapped[float] = mapped_column(Float, nullable=False)
    y_max: Mapped[float] = mapped_column(Float, nullable=False)

    road_event_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("road_events.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    session: Mapped[DriveSession] = relationship("DriveSession", back_populates="detections")
    road_event: Mapped[Optional[RoadEvent]] = relationship("RoadEvent", back_populates="detections")


class RoadEvent(Base):
    __tablename__ = "road_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=generate_uuid)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("drive_sessions.id", ondelete="CASCADE"), nullable=False, index=True)

    first_seen_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    last_seen_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    first_frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    last_frame_index: Mapped[int] = mapped_column(Integer, nullable=False)

    representative_detection_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    representative_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    representative_bbox: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False) # {x_min, y_min, x_max, y_max}
    support_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # Review status: PENDING_REVIEW, CONFIRMED, REJECTED, NEEDS_REVISIT
    review_status: Mapped[str] = mapped_column(String(32), default="PENDING_REVIEW", nullable=False, index=True)
    reviewer_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # GPS coordinates if present
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    session: Mapped[DriveSession] = relationship("DriveSession", back_populates="road_events")
    detections: Mapped[List[RawDetection]] = relationship("RawDetection", back_populates="road_event")
    review_actions: Mapped[List[ReviewAction]] = relationship("ReviewAction", back_populates="road_event", cascade="all, delete-orphan")


class ReviewAction(Base):
    __tablename__ = "review_actions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=generate_uuid)
    event_id: Mapped[str] = mapped_column(String(64), ForeignKey("road_events.id", ondelete="CASCADE"), nullable=False, index=True)

    action: Mapped[str] = mapped_column(String(32), nullable=False) # CONFIRM, REJECT, REVISIT, SPLIT, MERGE
    previous_status: Mapped[str] = mapped_column(String(32), nullable=False)
    new_status: Mapped[str] = mapped_column(String(32), nullable=False)
    reviewer_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    road_event: Mapped[RoadEvent] = relationship("RoadEvent", back_populates="review_actions")


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=generate_uuid)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("drive_sessions.id", ondelete="CASCADE"), nullable=False, index=True)

    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False) # raw_video, annotated_video, report_zip, detections_csv, metadata_json
    relative_path: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    session: Mapped[DriveSession] = relationship("DriveSession", back_populates="artifacts")
