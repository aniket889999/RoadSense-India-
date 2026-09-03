"""Video processing service: Bounded sampling, ML inference, and Road Event fusion."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional
import cv2
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.app.core.config import settings
from services.api.app.core.logging import logger
from services.api.app.db.session import async_session_factory
from services.api.app.models.entities import (
    Artifact,
    DriveSession,
    RawDetection,
    RoadEvent,
)
from services.api.app.schemas.session import (
    SessionProcessRequest,
    SessionProgressEvent,
)
from services.api.app.services.event_fusion import (
    DetectionCandidate,
    cluster_detections_into_road_events,
)
from services.api.app.services.live_manager import ws_manager
from services.api.app.services.model_loader import get_verified_model
from src.ml.drive_review import build_drive_review_plan, run_verified_drive_review
from src.ml.experimental_inference import ExperimentalInferenceSettings
from src.ml.model_provenance import REPO_ROOT


def _compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


async def process_video_session(session_id: str, request: SessionProcessRequest) -> None:
    """Asynchronous background worker to process an uploaded drive session."""
    session_dir = settings.session_dir / session_id
    raw_video_path = session_dir / "raw_video.mp4"

    start_time = time.monotonic()

    async with async_session_factory() as db:
        stmt = select(DriveSession).where(DriveSession.id == session_id)
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()
        if not session:
            logger.error("Session not found for processing", extra={"session_id": session_id})
            return

        try:
            # 1. Validating Stage
            session.processing_state = "validating"
            await db.commit()
            await ws_manager.broadcast_progress(
                session_id,
                SessionProgressEvent(
                    session_id=session_id,
                    stage="validating",
                    message="Validating video container and stream properties...",
                )
            )

            if not raw_video_path.exists() or raw_video_path.is_symlink():
                raise FileNotFoundError("Uploaded video file missing or invalid.")

            cap = cv2.VideoCapture(str(raw_video_path.resolve()))
            if not cap.isOpened():
                raise ValueError("Could not open video file.")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            source_fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if width <= 0 or height <= 0 or frame_count <= 0 or source_fps <= 0:
                raise ValueError("Video metadata is invalid or incomplete.")

            session.source_width = width
            session.source_height = height
            session.source_fps = source_fps
            session.total_source_frames = frame_count
            session.source_duration_seconds = frame_count / source_fps
            await db.commit()

            # 2. Sampling Stage
            session.processing_state = "sampling"
            await db.commit()
            await ws_manager.broadcast_progress(
                session_id,
                SessionProgressEvent(
                    session_id=session_id,
                    stage="sampling",
                    message=f"Generating bounded sampling plan ({request.sampling_fps} FPS, max {request.max_frames} frames)...",
                )
            )

            plan = build_drive_review_plan(
                frame_count=frame_count,
                source_fps=source_fps,
                window_start_seconds=request.window_start_seconds,
                window_duration_seconds=request.window_duration_seconds,
                sampling_fps=request.sampling_fps,
                max_frames=request.max_frames,
            )

            session.sampled_frames_count = plan.sampled_frame_count
            await db.commit()

            # 3. Model Loading Stage (Fail-Closed Provenance Verification)
            session.processing_state = "model_loading"
            await db.commit()
            await ws_manager.broadcast_progress(
                session_id,
                SessionProgressEvent(
                    session_id=session_id,
                    stage="model_loading",
                    message="Verifying local model provenance and loading neural network...",
                )
            )

            model, model_info = get_verified_model()
            inf_settings = ExperimentalInferenceSettings(
                device=settings.ROADSENSE_DEVICE,
                confidence_threshold=request.confidence_threshold,
                iou_threshold=request.iou_threshold,
                image_size=640,
                max_detections_per_frame=20,
                output_fps=5.0,
            )

            # 4. Processing & Rendering Stage
            session.processing_state = "processing"
            await db.commit()
            await ws_manager.broadcast_progress(
                session_id,
                SessionProgressEvent(
                    session_id=session_id,
                    stage="processing",
                    total_frames=plan.sampled_frame_count,
                    message="Running local model inference across sampled frames...",
                )
            )

            video_bytes = raw_video_path.read_bytes()
            input_hash = _compute_file_sha256(raw_video_path)

            # Execute drive review inference
            inference_result = run_verified_drive_review(
                video_bytes=video_bytes,
                video_filename=session.source_filename,
                plan=plan,
                model=model,
                model_info=model_info,
                settings=inf_settings,
                input_video_sha256=input_hash,
            )

            # 5. Save Artifacts (Annotated Video & Report Zip)
            session.processing_state = "rendering"
            await db.commit()
            await ws_manager.broadcast_progress(
                session_id,
                SessionProgressEvent(
                    session_id=session_id,
                    stage="rendering",
                    processed_frames=plan.sampled_frame_count,
                    total_frames=plan.sampled_frame_count,
                    percentage=90.0,
                    message="Rendering annotated replay video and compiling field dossier...",
                )
            )

            report_zip_path = session_dir / "report.zip"
            report_zip_path.write_bytes(inference_result.report_zip)

            # Extract annotated MP4 from report zip for direct web streaming
            import zipfile
            annotated_mp4_path = session_dir / "annotated_replay.mp4"
            with zipfile.ZipFile(report_zip_path, "r") as z:
                if "annotated_experimental_predictions.mp4" in z.namelist():
                    annotated_mp4_path.write_bytes(z.read("annotated_experimental_predictions.mp4"))

            # Record Artifact entities
            raw_art = Artifact(
                session_id=session_id,
                artifact_type="raw_video",
                relative_path=f"sessions/{session_id}/raw_video.mp4",
                sha256=input_hash,
                file_size_bytes=raw_video_path.stat().st_size,
            )
            zip_art = Artifact(
                session_id=session_id,
                artifact_type="report_zip",
                relative_path=f"sessions/{session_id}/report.zip",
                sha256=_compute_file_sha256(report_zip_path),
                file_size_bytes=report_zip_path.stat().st_size,
            )
            annotated_art = Artifact(
                session_id=session_id,
                artifact_type="annotated_video",
                relative_path=f"sessions/{session_id}/annotated_replay.mp4",
                sha256=_compute_file_sha256(annotated_mp4_path),
                file_size_bytes=annotated_mp4_path.stat().st_size,
            )
            db.add_all([raw_art, zip_art, annotated_art])

            # 6. Record Raw Detections
            detection_candidates: List[DetectionCandidate] = []
            db_detections: List[RawDetection] = []

            for det in inference_result.detections:
                raw_det = RawDetection(
                    session_id=session_id,
                    frame_index=det.frame_index,
                    timestamp_seconds=det.timestamp_seconds,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    x_min=det.x_min,
                    y_min=det.y_min,
                    x_max=det.x_max,
                    y_max=det.y_max,
                )
                db_detections.append(raw_det)
                db.add(raw_det)

            await db.flush() # Populate generated IDs

            for raw_det in db_detections:
                detection_candidates.append(
                    DetectionCandidate(
                        id=raw_det.id,
                        frame_index=raw_det.frame_index,
                        timestamp_seconds=raw_det.timestamp_seconds,
                        confidence=raw_det.confidence,
                        x_min=raw_det.x_min,
                        y_min=raw_det.y_min,
                        x_max=raw_det.x_max,
                        y_max=raw_det.y_max,
                    )
                )

            # 7. Event Fusion (Temporal & Spatial Clustering into Road Events)
            proposed_events = cluster_detections_into_road_events(
                detection_candidates,
                max_time_gap_seconds=1.5,
                min_iou=0.15,
                max_centroid_distance_px=80.0,
            )

            for prop in proposed_events:
                road_event = RoadEvent(
                    session_id=session_id,
                    first_seen_seconds=prop.first_seen_seconds,
                    last_seen_seconds=prop.last_seen_seconds,
                    first_frame_index=prop.first_frame_index,
                    last_frame_index=prop.last_frame_index,
                    representative_detection_id=prop.representative_detection_id,
                    representative_confidence=prop.representative_confidence,
                    representative_bbox=prop.representative_bbox,
                    support_count=prop.support_count,
                    review_status="PENDING_REVIEW",
                )
                db.add(road_event)
                await db.flush()

                # Associate raw detections with the parent RoadEvent
                det_ids = {d.id for d in prop.detections if d.id}
                for raw_det in db_detections:
                    if raw_det.id in det_ids:
                        raw_det.road_event_id = road_event.id

            # 8. Complete Session
            elapsed = time.monotonic() - start_time
            session.processing_state = "complete"
            session.completed_at = datetime.now(timezone.utc)
            session.frames_with_detections = inference_result.frames_with_detections
            session.total_detections_count = len(inference_result.detections)
            session.processing_duration_seconds = elapsed
            session.model_provenance = {
                "run_id": model_info.run_id,
                "checkpoint_sha256": model_info.checkpoint_sha256,
                "git_sha": model_info.git_sha,
                "device": settings.ROADSENSE_DEVICE,
                "confidence_threshold": request.confidence_threshold,
                "iou_threshold": request.iou_threshold,
            }

            await db.commit()

            await ws_manager.broadcast_progress(
                session_id,
                SessionProgressEvent(
                    session_id=session_id,
                    stage="complete",
                    processed_frames=plan.sampled_frame_count,
                    total_frames=plan.sampled_frame_count,
                    percentage=100.0,
                    detections_found=len(inference_result.detections),
                    message="Drive session processing complete. Ready for operator review.",
                )
            )

        except Exception as exc:
            logger.exception("Error processing drive session", extra={"session_id": session_id})
            session.processing_state = "failed"
            session.error_message = str(exc)
            await db.commit()

            await ws_manager.broadcast_progress(
                session_id,
                SessionProgressEvent(
                    session_id=session_id,
                    stage="failed",
                    message=f"Processing failed: {exc}",
                )
            )
