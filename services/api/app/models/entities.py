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
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    JSON,
    text,
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
    processing_state: Mapped[str] = mapped_column(String(32), default="queued", nullable=False) # queued, validating, decoding, detecting, tracking, fusing_events, encoding, complete, failed, cancelled
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

    # Full ffprobe media metadata (JSON)
    media_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

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
    track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)

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

    track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    representative_detection_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    representative_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    representative_bbox: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False) # {x_min, y_min, x_max, y_max}
    support_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    evidence_crop_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

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

    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False) # raw_video, annotated_video, report_zip, detections_csv, metadata_json, evidence_crop
    relative_path: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    session: Mapped[DriveSession] = relationship("DriveSession", back_populates="artifacts")


# ============================================================================
# RoadSense SiteOps: Parking Domain Entities (Phase 1)
# ============================================================================

class Site(Base):
    """Physical facility campus entity."""
    __tablename__ = "sites"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timezone: Mapped[str] = mapped_column(String(64), default="UTC", nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    cameras: Mapped[List[Camera]] = relationship("Camera", back_populates="site", cascade="all, delete-orphan")


class Camera(Base):
    """Fixed CCTV camera sensor covering a parking zone."""
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=generate_uuid)
    site_id: Mapped[str] = mapped_column(String(64), ForeignKey("sites.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    reference_image_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    reference_image_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reference_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reference_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    calibration_status: Mapped[str] = mapped_column(String(32), default="NOT_CONFIGURED", nullable=False) # NOT_CONFIGURED, PENDING_REVIEW, VERIFIED, INVALIDATED
    camera_position_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    site: Mapped[Site] = relationship("Site", back_populates="cameras")
    layout_revisions: Mapped[List[ParkingLayoutRevision]] = relationship("ParkingLayoutRevision", back_populates="camera", cascade="all, delete-orphan")
    stability_assessments: Mapped[List["CameraStabilityAssessment"]] = relationship("CameraStabilityAssessment", back_populates="camera", cascade="all, delete-orphan")
    occupancy_jobs: Mapped[List["ParkingOccupancyJob"]] = relationship("ParkingOccupancyJob", back_populates="camera", cascade="all, delete-orphan")


class ParkingLayoutRevision(Base):
    """Versioned parking layout revision for a camera."""
    __tablename__ = "parking_layout_revisions"

    __table_args__ = (
        UniqueConstraint("camera_id", "revision_number", name="uq_parking_layout_camera_revision"),
        Index(
            "uq_one_verified_layout_per_camera",
            "camera_id",
            unique=True,
            postgresql_where=text("status = 'VERIFIED'"),
            sqlite_where=text("status = 'VERIFIED'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=generate_uuid)
    camera_id: Mapped[str] = mapped_column(String(64), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(32), default="DRAFT", nullable=False) # DRAFT, PENDING_REVIEW, VERIFIED, SUPERSEDED, INVALIDATED
    canonical_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Reference Image Snapshot (immutable per layout revision)
    reference_image_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reference_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reference_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    submitted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    invalidated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    invalidation_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    camera: Mapped[Camera] = relationship("Camera", back_populates="layout_revisions")
    parking_spaces: Mapped[List[ParkingSpace]] = relationship("ParkingSpace", back_populates="layout_revision", cascade="all, delete-orphan")
    audit_events: Mapped[List[LayoutAuditEvent]] = relationship("LayoutAuditEvent", back_populates="layout_revision", cascade="all, delete-orphan")


class ParkingSpace(Base):
    """Individual configured parking bay polygon."""
    __tablename__ = "parking_spaces"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=generate_uuid)
    layout_revision_id: Mapped[str] = mapped_column(String(64), ForeignKey("parking_layout_revisions.id", ondelete="CASCADE"), nullable=False, index=True)
    operator_label: Mapped[str] = mapped_column(String(64), nullable=False)
    space_type: Mapped[str] = mapped_column(String(32), default="STANDARD", nullable=False)
    polygon_normalized: Mapped[list[dict[str, float]]] = mapped_column(JSON, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    layout_revision: Mapped[ParkingLayoutRevision] = relationship("ParkingLayoutRevision", back_populates="parking_spaces")
    approach_zone: Mapped[Optional[ApproachZone]] = relationship("ApproachZone", back_populates="parking_space", uselist=False, cascade="all, delete-orphan")


class ApproachZone(Base):
    """Access / ingress zone polygon associated with a parking bay."""
    __tablename__ = "approach_zones"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=generate_uuid)
    parking_space_id: Mapped[str] = mapped_column(String(64), ForeignKey("parking_spaces.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    polygon_normalized: Mapped[list[dict[str, float]]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    parking_space: Mapped[ParkingSpace] = relationship("ParkingSpace", back_populates="approach_zone")


class LayoutAuditEvent(Base):
    """Append-only audit history log for parking layout lifecycle transitions."""
    __tablename__ = "layout_audit_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=generate_uuid)
    layout_revision_id: Mapped[str] = mapped_column(String(64), ForeignKey("parking_layout_revisions.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False) # CREATED, UPDATED, SUBMITTED, VERIFIED, INVALIDATED, SUPERSEDED
    prior_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    new_status: Mapped[str] = mapped_column(String(32), nullable=False)
    local_operator_label: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    layout_revision: Mapped[ParkingLayoutRevision] = relationship("ParkingLayoutRevision", back_populates="audit_events")


class CameraStabilityAssessment(Base):
    """Immutable operational camera stability assessment record."""
    __tablename__ = "camera_stability_assessments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=generate_uuid)
    camera_id: Mapped[str] = mapped_column(String(64), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    layout_revision_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("parking_layout_revisions.id", ondelete="SET NULL"), nullable=True)
    layout_canonical_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    status: Mapped[str] = mapped_column(String(32), default="COMPLETE", nullable=False) # QUEUED, VALIDATING, ANALYZING, COMPLETE, FAILED, CANCELLED
    progress_pct: Mapped[float] = mapped_column(Float, default=100.0, nullable=False)
    stage_message: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    failure_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    failure_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    config_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    config_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    reference_image_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    video_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    algorithm_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    opencv_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    thresholds_snapshot: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    sample_measurements: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSON, nullable=True)

    aggregate_decision: Mapped[Optional[str]] = mapped_column(String(32), nullable=True) # STABLE, UNSTABLE, INSUFFICIENT_EVIDENCE, ERROR
    operational_gate: Mapped[Optional[str]] = mapped_column(String(32), nullable=True) # ALLOWED, BLOCKED
    gate_reasons: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    summary_metrics: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Legacy fields preserved as nullable
    operator_acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    operator_label: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    operator_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    camera: Mapped[Camera] = relationship("Camera", back_populates="stability_assessments")
    layout_revision: Mapped[Optional[ParkingLayoutRevision]] = relationship("ParkingLayoutRevision")
    audit_events: Mapped[list[CameraStabilityAuditEvent]] = relationship("CameraStabilityAuditEvent", back_populates="assessment", cascade="all, delete-orphan")


class CameraStabilityAuditEvent(Base):
    """Append-only audit event log for camera stability assessments and calibration state transitions."""
    __tablename__ = "camera_stability_audit_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=generate_uuid)
    assessment_id: Mapped[str] = mapped_column(String(64), ForeignKey("camera_stability_assessments.id", ondelete="CASCADE"), nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String(64), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False) # OPERATOR_ACKNOWLEDGED, CALIBRATION_INVALIDATED
    operator_identity: Mapped[str] = mapped_column(String(128), nullable=False)
    explicit_reason: Mapped[str] = mapped_column(Text, nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    previous_gate_state: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    resulting_gate_state: Mapped[str] = mapped_column(String(32), nullable=False)
    previous_calibration_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    resulting_calibration_status: Mapped[str] = mapped_column(String(32), nullable=False)
    assessment_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    config_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reference_image_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    layout_canonical_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    assessment: Mapped[CameraStabilityAssessment] = relationship("CameraStabilityAssessment", back_populates="audit_events")


class ParkingOccupancyJob(Base):
    """Asynchronous background processing record for gated parking occupancy and video annotation."""
    __tablename__ = "parking_occupancy_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=generate_uuid)
    camera_id: Mapped[str] = mapped_column(String(64), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    site_id: Mapped[str] = mapped_column(String(64), ForeignKey("sites.id", ondelete="CASCADE"), nullable=False, index=True)
    layout_revision_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("parking_layout_revisions.id", ondelete="SET NULL"), nullable=True)
    stability_assessment_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("camera_stability_assessments.id", ondelete="SET NULL"), nullable=True)

    status: Mapped[str] = mapped_column(String(32), default="QUEUED", nullable=False)  # QUEUED, VALIDATING, DETECTING, TRACKING, CLASSIFYING_OCCUPANCY, RENDERING, ENCODING, COMPLETE, FAILED, CANCELLED, BLOCKED_BY_STABILITY_GATE
    progress_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    stage_message: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    failure_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    failure_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    gate_decision: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # ALLOWED, BLOCKED
    gate_reasons: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)

    input_video_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    output_video_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reference_image_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    layout_canonical_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    detector_checkpoint_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    occupancy_config_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    total_frames: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processed_frames: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    fps: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    video_width: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    video_height: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    total_bays: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    final_occupied_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    final_vacant_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    final_unknown_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    final_occluded_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_state_transitions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    output_video_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    timeline_jsonl_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    manifest_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    bay_summary_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    camera: Mapped[Camera] = relationship("Camera", back_populates="occupancy_jobs")
    audit_events: Mapped[list["ParkingJobAuditEvent"]] = relationship("ParkingJobAuditEvent", back_populates="job", cascade="all, delete-orphan")


class ParkingJobAuditEvent(Base):
    """Append-only audit event log for parking occupancy jobs, retentions, and safe purges."""
    __tablename__ = "parking_job_audit_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=generate_uuid)
    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("parking_occupancy_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)  # JOB_PURGED, DELETED_TOMBSTONE, CANCELLATION_REQUESTED
    operator_identity_assertion: Mapped[str] = mapped_column(String(128), nullable=False)
    deletion_reason: Mapped[str] = mapped_column(Text, nullable=False)
    prior_status: Mapped[str] = mapped_column(String(32), nullable=False)
    resulting_status: Mapped[str] = mapped_column(String(32), nullable=False)
    artifact_hashes: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    purge_result: Mapped[str] = mapped_column(String(64), nullable=False)  # SUCCESS, PARTIAL, NOOP
    request_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    job: Mapped[ParkingOccupancyJob] = relationship("ParkingOccupancyJob", back_populates="audit_events")
