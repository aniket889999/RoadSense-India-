"""Video processing service: Bounded streaming, YOLOv8n inference, ByteTrack association, and FFmpeg encoding."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
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
    fuse_tracks_into_road_events,
)
from services.api.app.services.live_manager import ws_manager
from services.api.app.services.model_loader import get_verified_model
from src.media.encoder import encode_frames_to_mp4
from src.media.frame_pipeline import (
    FrameProcessingOptions,
    extract_evidence_crop,
    render_detection_overlays,
    stream_video_frames,
)
from src.media.inspector import MediaMetadata, inspect_media_file
from src.tracking.bytetrack_adapter import (
    ByteTrackAdapter,
    load_bytetrack_config,
)


# Global in-memory set of cancelled session IDs
CANCELLED_SESSIONS: set[str] = set()


def cancel_session_processing(session_id: str) -> None:
    """Request graceful cancellation of a running video processing job."""
    CANCELLED_SESSIONS.add(session_id)


def is_session_cancelled(session_id: str) -> bool:
    return session_id in CANCELLED_SESSIONS


def _compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


async def process_video_session(session_id: str, request: SessionProcessRequest) -> None:
    """Asynchronous background pipeline to inspect, track, fuse, and encode drive sessions."""
    session_dir = settings.session_dir / session_id
    raw_video_path = session_dir / "raw_video.mp4"
    evidence_dir = session_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    CANCELLED_SESSIONS.discard(session_id)

    async with async_session_factory() as db:
        stmt = select(DriveSession).where(DriveSession.id == session_id)
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()
        if not session:
            logger.error("Session not found for processing", extra={"session_id": session_id})
            return

        try:
            # 1. Validating Stage with FFprobe
            session.processing_state = "validating"
            await db.commit()
            await ws_manager.broadcast_progress(
                session_id,
                SessionProgressEvent(
                    session_id=session_id,
                    stage="validating",
                    message="Validating media container, rotation, and codecs with ffprobe...",
                ),
            )

            media_meta = inspect_media_file(
                raw_video_path,
                base_dir=settings.session_dir,
            )

            session.source_width = media_meta.width
            session.source_height = media_meta.height
            session.source_fps = media_meta.avg_fps
            session.total_source_frames = media_meta.frame_count
            session.source_duration_seconds = media_meta.duration_seconds
            session.media_metadata = media_meta.to_dict()
            await db.commit()

            # 2. Model & Tracker Setup
            model, model_info = get_verified_model()
            tracker_cfg_path = Path("configs/tracking/bytetrack_default.yaml")
            tracker_cfg = load_bytetrack_config(tracker_cfg_path)
            tracker = ByteTrackAdapter(tracker_cfg)

            frame_opts = FrameProcessingOptions(
                target_fps=request.sampling_fps,
                max_duration_seconds=request.window_duration_seconds,
                max_frames=request.max_frames,
                start_seconds=request.window_start_seconds,
                apply_privacy_mask=request.apply_privacy_mask,
                confidence_display_threshold=request.confidence_threshold,
            )

            # 3. Decoding, Detecting & Tracking Stream
            session.processing_state = "detecting"
            await db.commit()

            annotated_frames: List[np.ndarray] = []
            all_raw_detections: List[Dict[str, Any]] = []
            total_detections_count = 0
            frames_with_detections = 0
            processed_frames_count = 0

            for frame_idx, timestamp_sec, frame_bgr in stream_video_frames(raw_video_path, media_meta, frame_opts):
                if is_session_cancelled(session_id):
                    session.processing_state = "cancelled"
                    session.error_message = "Processing cancelled by operator."
                    await db.commit()
                    await ws_manager.broadcast_progress(
                        session_id,
                        SessionProgressEvent(
                            session_id=session_id,
                            stage="cancelled",
                            message="Processing cancelled by operator.",
                        ),
                    )
                    return

                processed_frames_count += 1

                # Execute YOLO model inference on frame copy
                results = model.predict(
                    source=frame_bgr.copy(),
                    conf=request.confidence_threshold,
                    iou=request.iou_threshold,
                    imgsz=640,
                    device=settings.ROADSENSE_DEVICE,
                    verbose=False,
                )

                frame_dets: List[Dict[str, Any]] = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        coords = box.xyxy[0].tolist()
                        conf = float(box.conf[0].item())
                        cls_id = int(box.cls[0].item())

                        det_record = {
                            "frame_index": frame_idx,
                            "timestamp_seconds": timestamp_sec,
                            "confidence": conf,
                            "class_id": cls_id,
                            "x_min": float(coords[0]),
                            "y_min": float(coords[1]),
                            "x_max": float(coords[2]),
                            "y_max": float(coords[3]),
                        }
                        frame_dets.append(det_record)

                if frame_dets:
                    frames_with_detections += 1
                    total_detections_count += len(frame_dets)

                # ByteTrack association
                observations = tracker.update(
                    frame_number=frame_idx,
                    timestamp_seconds=timestamp_sec,
                    detections=frame_dets,
                    session_id=session_id,
                    model_sha256=model_info.checkpoint_sha256,
                    frame_shape=frame_bgr.shape[:2],
                )

                # Link track IDs back to detections
                for det in frame_dets:
                    # Match observation with detection
                    matched_obs = next(
                        (o for o in observations if abs(o.confidence - det["confidence"]) < 1e-4),
                        None,
                    )
                    if matched_obs:
                        det["track_id"] = matched_obs.track_id
                    else:
                        det["track_id"] = None
                    all_raw_detections.append(det)

                # Render green overlays
                annotated = render_detection_overlays(
                    frame_bgr,
                    frame_dets,
                    frame_index=frame_idx,
                    timestamp_seconds=timestamp_sec,
                    confidence_threshold=request.confidence_threshold,
                    apply_privacy_mask=request.apply_privacy_mask,
                )
                annotated_frames.append(annotated)

                # Periodic progress broadcast
                if processed_frames_count % 5 == 0:
                    pct = min(90.0, (processed_frames_count / max(1, request.max_frames)) * 90.0)
                    await ws_manager.broadcast_progress(
                        session_id,
                        SessionProgressEvent(
                            session_id=session_id,
                            stage="detecting",
                            processed_frames=processed_frames_count,
                            total_frames=request.max_frames,
                            percentage=pct,
                            detections_found=total_detections_count,
                            message=f"Processed frame {processed_frames_count} ({total_detections_count} candidates tracked)...",
                        ),
                    )
                    await asyncio.sleep(0)  # Yield to event loop

            session.sampled_frames_count = processed_frames_count
            session.frames_with_detections = frames_with_detections
            session.total_detections_count = total_detections_count
            await db.commit()

            # 4. Road Event Fusion
            session.processing_state = "fusing_events"
            await db.commit()
            await ws_manager.broadcast_progress(
                session_id,
                SessionProgressEvent(
                    session_id=session_id,
                    stage="fusing_events",
                    processed_frames=processed_frames_count,
                    total_frames=processed_frames_count,
                    percentage=92.0,
                    message="Consolidating ByteTrack tracks into Road Event review queue...",
                ),
            )

            # Persist raw detections in DB
            db_detections: List[RawDetection] = []
            for d in all_raw_detections:
                raw_det = RawDetection(
                    session_id=session_id,
                    frame_index=d["frame_index"],
                    timestamp_seconds=d["timestamp_seconds"],
                    confidence=d["confidence"],
                    class_id=d["class_id"],
                    track_id=d.get("track_id"),
                    x_min=d["x_min"],
                    y_min=d["y_min"],
                    x_max=d["x_max"],
                    y_max=d["y_max"],
                )
                db.add(raw_det)
                db_detections.append(raw_det)

            await db.flush()

            # Convert stable tracks and detections to Road Events
            stable_tracks = tracker.get_stable_tracks()
            if stable_tracks:
                proposed_events = fuse_tracks_into_road_events(
                    stable_tracks,
                    min_confidence=request.confidence_threshold,
                )
            else:
                detection_candidates = [
                    DetectionCandidate(
                        id=rd.id,
                        frame_index=rd.frame_index,
                        timestamp_seconds=rd.timestamp_seconds,
                        confidence=rd.confidence,
                        x_min=rd.x_min,
                        y_min=rd.y_min,
                        x_max=rd.x_max,
                        y_max=rd.y_max,
                        track_id=rd.track_id,
                    )
                    for rd in db_detections
                ]
                proposed_events = cluster_detections_into_road_events(
                    detection_candidates,
                    max_time_gap_seconds=1.5,
                    min_iou=0.15,
                )

            for prop in proposed_events:
                # Extract representative crop if frame is available
                evidence_crop_rel_path = None
                if annotated_frames and 0 <= prop.first_frame_index < len(annotated_frames):
                    crop_frame = annotated_frames[min(prop.first_frame_index, len(annotated_frames) - 1)]
                    rep_box = (
                        prop.representative_bbox["x_min"],
                        prop.representative_bbox["y_min"],
                        prop.representative_bbox["x_max"],
                        prop.representative_bbox["y_max"],
                    )
                    crop_bytes = extract_evidence_crop(crop_frame, rep_box)
                    if crop_bytes:
                        crop_filename = f"event_{prop.track_id or prop.first_frame_index}.jpg"
                        crop_dest = evidence_dir / crop_filename
                        crop_dest.write_bytes(crop_bytes)
                        evidence_crop_rel_path = f"sessions/{session_id}/evidence/{crop_filename}"

                road_event = RoadEvent(
                    session_id=session_id,
                    first_seen_seconds=prop.first_seen_seconds,
                    last_seen_seconds=prop.last_seen_seconds,
                    first_frame_index=prop.first_frame_index,
                    last_frame_index=prop.last_frame_index,
                    track_id=prop.track_id,
                    representative_detection_id=prop.representative_detection_id,
                    representative_confidence=prop.representative_confidence,
                    representative_bbox=prop.representative_bbox,
                    support_count=prop.support_count,
                    evidence_crop_path=evidence_crop_rel_path,
                    review_status="PENDING_REVIEW",
                )
                db.add(road_event)
                await db.flush()

                # Link raw detections
                for raw_det in db_detections:
                    if prop.track_id is not None and raw_det.track_id == prop.track_id:
                        raw_det.road_event_id = road_event.id

            # 5. FFmpeg Encoding Stage
            session.processing_state = "encoding"
            await db.commit()
            await ws_manager.broadcast_progress(
                session_id,
                SessionProgressEvent(
                    session_id=session_id,
                    stage="encoding",
                    processed_frames=processed_frames_count,
                    total_frames=processed_frames_count,
                    percentage=95.0,
                    message="Encoding H.264/yuv420p browser-compatible MP4 with FFmpeg...",
                ),
            )

            annotated_mp4_path = session_dir / "annotated_replay.mp4"
            if annotated_frames:
                encoding_res = encode_frames_to_mp4(
                    annotated_frames,
                    annotated_mp4_path,
                    fps=request.sampling_fps,
                    source_audio_path=raw_video_path if media_meta.has_audio else None,
                )
            else:
                annotated_mp4_path.write_bytes(b"")
                encoding_res = None

            # Create report zip bundle
            report_zip_path = session_dir / "report.zip"
            import zipfile
            with zipfile.ZipFile(report_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                if annotated_mp4_path.is_file() and annotated_mp4_path.stat().st_size > 0:
                    z.write(annotated_mp4_path, arcname="annotated_replay.mp4")
                summary_data = {
                    "session_id": session_id,
                    "filename": session.source_filename,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "total_detections": total_detections_count,
                    "road_events_count": len(proposed_events),
                    "model_sha256": model_info.checkpoint_sha256,
                    "tracker_sha256": tracker_cfg.config_sha256,
                }
                z.writestr("session_summary.json", json.dumps(summary_data, indent=2))

            # Record Artifacts
            raw_art = Artifact(
                session_id=session_id,
                artifact_type="raw_video",
                relative_path=f"sessions/{session_id}/raw_video.mp4",
                sha256=_compute_file_sha256(raw_video_path),
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
                sha256=_compute_file_sha256(annotated_mp4_path) if annotated_mp4_path.exists() else "",
                file_size_bytes=annotated_mp4_path.stat().st_size if annotated_mp4_path.exists() else 0,
            )
            db.add_all([raw_art, zip_art, annotated_art])

            # 6. Complete Session
            elapsed = time.monotonic() - start_time
            session.processing_state = "complete"
            session.completed_at = datetime.now(timezone.utc)
            session.processing_duration_seconds = elapsed
            session.model_provenance = {
                "run_id": model_info.run_id,
                "checkpoint_sha256": model_info.checkpoint_sha256,
                "tracker_sha256": tracker_cfg.config_sha256,
                "git_sha": model_info.git_sha,
                "device": settings.ROADSENSE_DEVICE,
                "confidence_threshold": request.confidence_threshold,
                "iou_threshold": request.iou_threshold,
                "privacy_masked": request.apply_privacy_mask,
            }

            await db.commit()

            await ws_manager.broadcast_progress(
                session_id,
                SessionProgressEvent(
                    session_id=session_id,
                    stage="complete",
                    processed_frames=processed_frames_count,
                    total_frames=processed_frames_count,
                    percentage=100.0,
                    detections_found=total_detections_count,
                    message=f"Processing complete: {total_detections_count} candidate observations grouped into {len(proposed_events)} road event(s).",
                ),
            )

        except Exception as exc:
            logger.exception("Error in video processing pipeline", extra={"session_id": session_id})
            session.processing_state = "failed"
            session.error_message = str(exc)
            await db.commit()

            await ws_manager.broadcast_progress(
                session_id,
                SessionProgressEvent(
                    session_id=session_id,
                    stage="failed",
                    message=f"Processing failed: {exc}",
                ),
            )
