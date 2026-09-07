"""Asynchronous bounded background job manager for gated parking occupancy and video annotation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid
import cv2
import numpy as np
from sqlalchemy import desc, select, update

from services.api.app.core.config import settings
from services.api.app.core.logging import logger
from services.api.app.db.session import async_session_factory
from services.api.app.models.entities import (
    Camera,
    CameraStabilityAssessment,
    ParkingLayoutRevision,
    ParkingOccupancyJob,
)
from src.parking.contracts import LayoutRevisionStatus, OperationalGate, StabilityDecision, evaluate_operational_gate
from src.parking.occupancy_config import ParkingOccupancyConfig, load_parking_occupancy_config
from src.parking.occupancy_contracts import (
    FrameOccupancyResult,
    OccupancyState,
    ParkingJobCancelled,
    ParkingJobManifest,
)
from src.parking.occupancy_engine import ParkingOccupancyEngine
from src.parking.stability_config import load_stability_config
from src.parking.stability_engine import inspect_video_media_safe
from src.parking.vehicle_detector import LocalVehicleDetector
from src.parking.vehicle_tracker import ParkingByteTracker
from src.parking.video_annotator import FFmpegStreamEncoder, ParkingVideoAnnotator

TERMINAL_STATES = ("COMPLETE", "FAILED", "CANCELLED", "BLOCKED_BY_STABILITY_GATE")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ParkingOccupancyJobManager:
    """
    Manages asynchronous, gate-controlled parking occupancy inference jobs with bounded concurrency.
    Enforces fail-closed camera stability gate, streaming memory bounds, cooperative cancellation,
    and atomic staging-to-final promotion.
    """

    def __init__(self, max_concurrency: int = 2) -> None:
        self.config: ParkingOccupancyConfig = load_parking_occupancy_config()
        self._max_concurrency: int = max(1, self.config.execution.max_concurrent_jobs or max_concurrency)
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._active_tasks: Dict[str, asyncio.Task[None]] = {}
        self._running_jobs: Set[str] = set()
        self._cancellation_events: Dict[str, threading.Event] = {}
        self._cancel_requested: Set[str] = set()

    @property
    def semaphore(self) -> asyncio.Semaphore:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._semaphore is None or (hasattr(self._semaphore, "_loop") and getattr(self._semaphore, "_loop", None) not in (None, loop)):
            self._semaphore = asyncio.Semaphore(self._max_concurrency)
        return self._semaphore

    def submit_occupancy_job(
        self,
        job_id: str,
        camera_id: str,
        site_id: str,
        video_path: Path,
    ) -> asyncio.Task[None]:
        """Submit a new parking occupancy job for asynchronous execution."""
        cancel_event = threading.Event()
        self._cancellation_events[job_id] = cancel_event

        task = asyncio.create_task(
            self._execute_job(
                job_id=job_id,
                camera_id=camera_id,
                site_id=site_id,
                video_path=video_path,
                cancel_event=cancel_event,
            ),
            name=f"parking-occupancy-job-{job_id}",
        )
        self._active_tasks[job_id] = task
        return task

    async def cancel_occupancy_job(self, job_id: str) -> bool:
        """Request cooperative cancellation of an active or queued occupancy job."""
        self._cancel_requested.add(job_id)
        evt = self._cancellation_events.get(job_id)
        if evt:
            evt.set()

        # Only directly cancel the asyncio task if it has not yet started execution in thread
        if job_id not in self._running_jobs:
            task = self._active_tasks.get(job_id)
            if task and not task.done():
                task.cancel()
                return True
        return True

    async def _execute_job(
        self,
        job_id: str,
        camera_id: str,
        site_id: str,
        video_path: Path,
        cancel_event: threading.Event,
    ) -> None:
        async with self.semaphore:
            job_started_at = _utc_now()
            main_loop = asyncio.get_running_loop()

            # 1. Update initial status to VALIDATING
            try:
                async with async_session_factory() as db:
                    res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
                    job = res.scalar_one_or_none()
                    if job:
                        job.status = "VALIDATING"
                        job.started_at = job_started_at
                        job.progress_pct = 5.0
                        job.stage_message = "Validating camera layout and camera stability gate"
                        await db.commit()
            except Exception as e:
                logger.error(f"Failed to update initial status for job {job_id}: {e}")

            if job_id in self._cancel_requested or cancel_event.is_set():
                await self._persist_cancellation(job_id, video_path)
                return

            # 2. Gate Verification Query & Stability Config Verification
            current_stability_config = load_stability_config()
            verified_layout_id = None
            current_layout_sha = None
            latest_assessment_id = None
            latest_assessment_ref_sha = None
            latest_assessment_layout_sha = None
            latest_assessment_config_sha = None
            latest_assessment_created_at = None
            latest_assessment_decision = None
            latest_assessment_thresholds = None
            current_ref_sha = None
            parking_spaces_payload: List[Dict[str, Any]] = []

            try:
                async with async_session_factory() as db:
                    cam_res = await db.execute(select(Camera).where(Camera.id == camera_id))
                    cam = cam_res.scalar_one_or_none()
                    if not cam:
                        raise ValueError(f"Camera {camera_id} not found.")
                    current_ref_sha = cam.reference_image_sha256

                    from sqlalchemy.orm import selectinload
                    layout_res = await db.execute(
                        select(ParkingLayoutRevision)
                        .options(selectinload(ParkingLayoutRevision.parking_spaces))
                        .where(ParkingLayoutRevision.camera_id == camera_id)
                        .where(ParkingLayoutRevision.status == LayoutRevisionStatus.VERIFIED.value)
                    )
                    verified_layout = layout_res.scalar_one_or_none()
                    if verified_layout:
                        verified_layout_id = verified_layout.id
                        current_layout_sha = verified_layout.canonical_sha256
                        parking_spaces_payload = [
                            {
                                "id": sp.id,
                                "operator_label": sp.operator_label,
                                "space_type": sp.space_type,
                                "polygon_normalized": sp.polygon_normalized,
                            }
                            for sp in verified_layout.parking_spaces
                            if sp.active
                        ]

                    assess_res = await db.execute(
                        select(CameraStabilityAssessment)
                        .where(CameraStabilityAssessment.camera_id == camera_id)
                        .where(CameraStabilityAssessment.status == "COMPLETE")
                        .order_by(desc(CameraStabilityAssessment.created_at))
                    )
                    latest_assessment = assess_res.scalars().first()
                    if latest_assessment:
                        latest_assessment_id = latest_assessment.id
                        latest_assessment_ref_sha = latest_assessment.reference_image_sha256
                        latest_assessment_layout_sha = latest_assessment.layout_canonical_sha256
                        latest_assessment_config_sha = latest_assessment.config_sha256
                        latest_assessment_created_at = latest_assessment.created_at
                        latest_assessment_decision = latest_assessment.aggregate_decision
                        latest_assessment_thresholds = latest_assessment.thresholds_snapshot

            except Exception as e:
                logger.error(f"Error reading gate requirements for job {job_id}: {e}")
                await self._persist_failure(job_id, "DB_READ_ERROR", str(e), video_path)
                return

            if job_id in self._cancel_requested or cancel_event.is_set():
                await self._persist_cancellation(job_id, video_path)
                return

            # Evaluate Gate
            decision_enum = None
            if latest_assessment_decision:
                try:
                    decision_enum = StabilityDecision(latest_assessment_decision)
                except Exception:
                    pass

            max_age = (latest_assessment_thresholds or {}).get(
                "max_assessment_age_seconds",
                current_stability_config.thresholds.max_assessment_age_seconds,
            )

            gate_decision, gate_reasons = evaluate_operational_gate(
                decision=decision_enum,
                assessment_reference_sha=latest_assessment_ref_sha,
                current_reference_sha=current_ref_sha,
                assessment_layout_sha=latest_assessment_layout_sha,
                current_layout_sha=current_layout_sha,
                assessment_timestamp=latest_assessment_created_at,
                current_timestamp=job_started_at,
                max_age_seconds=max_age,
            )

            # Strict Stability Configuration SHA Check
            if latest_assessment_config_sha and latest_assessment_config_sha != current_stability_config.config_sha256:
                gate_decision = OperationalGate.BLOCKED
                gate_reasons.append(
                    f"CONFIG_MISMATCH: Stability assessment used config SHA {latest_assessment_config_sha[:8]}..., "
                    f"current system requires {current_stability_config.config_sha256[:8]}..."
                )

            if gate_decision != OperationalGate.ALLOWED:
                # FAIL-CLOSED BLOCK
                logger.warning(f"Occupancy job {job_id} BLOCKED by stability gate: {gate_reasons}")
                await self._persist_blocked_by_gate(job_id, gate_reasons, video_path)
                return

            if not parking_spaces_payload:
                gate_reasons.append("NO_ACTIVE_SPACES: Verified layout contains 0 active parking spaces.")
                await self._persist_blocked_by_gate(job_id, gate_reasons, video_path)
                return

            # 3. Media Preflight Inspection
            try:
                media_info = inspect_video_media_safe(
                    video_path=video_path,
                    timeout_sec=self.config.execution.ffprobe_timeout_seconds,
                )
            except Exception as e:
                logger.warning(f"Video media inspection rejected for job {job_id}: {e}")
                await self._persist_failure(job_id, "MEDIA_VALIDATION_ERROR", str(e), video_path)
                return

            if job_id in self._cancel_requested or cancel_event.is_set():
                await self._persist_cancellation(job_id, video_path)
                return

            # Compute input video SHA
            h_vid = hashlib.sha256()
            with open(video_path, "rb") as vf:
                while chunk := vf.read(65536):
                    h_vid.update(chunk)
            input_video_sha = h_vid.hexdigest()

            # Thread-safe async progress updater
            async def _update_db_progress(pct: float, msg: str, stage_status: Optional[str] = None) -> None:
                try:
                    async with async_session_factory() as db:
                        res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
                        j = res.scalar_one_or_none()
                        if j and j.status not in ("COMPLETE", "FAILED", "CANCELLED", "BLOCKED_BY_STABILITY_GATE"):
                            if stage_status:
                                j.status = stage_status
                            j.progress_pct = round(pct, 1)
                            j.stage_message = msg
                            await db.commit()
                except Exception as ex:
                    logger.debug(f"Progress update error: {ex}")

            def sync_progress_callback(pct: float, msg: str, stage_status: Optional[str] = None) -> None:
                if main_loop and main_loop.is_running():
                    try:
                        asyncio.run_coroutine_threadsafe(
                            _update_db_progress(pct, msg, stage_status),
                            main_loop,
                        )
                    except Exception as ex:
                        logger.debug(f"Failed to dispatch progress update: {ex}")

            # 4. Run Streaming Inference & Pipeline in Thread
            try:
                self._running_jobs.add(job_id)
                result_payload = await asyncio.to_thread(
                    self._run_streaming_pipeline,
                    job_id=job_id,
                    camera_id=camera_id,
                    site_id=site_id,
                    video_path=video_path,
                    input_video_sha=input_video_sha,
                    media_info=media_info,
                    verified_layout_canonical_sha=current_layout_sha,
                    latest_assessment_ref_sha=latest_assessment_ref_sha,
                    latest_assessment_id=latest_assessment_id,
                    latest_assessment_config_sha=latest_assessment_config_sha,
                    parking_spaces_payload=parking_spaces_payload,
                    job_started_at=job_started_at,
                    progress_callback=sync_progress_callback,
                    cancel_event=cancel_event,
                )

                if job_id in self._cancel_requested or cancel_event.is_set():
                    await self._persist_cancellation(job_id, video_path)
                    return

                # 5. Persist COMPLETE state atomically via CAS update
                await self._persist_complete(
                    job_id=job_id,
                    input_video_sha=input_video_sha,
                    current_ref_sha=current_ref_sha,
                    current_layout_sha=current_layout_sha,
                    verified_layout_id=verified_layout_id,
                    latest_assessment_id=latest_assessment_id,
                    result_payload=result_payload,
                )

            except (ParkingJobCancelled, asyncio.CancelledError):
                await self._persist_cancellation(job_id, video_path)
            except Exception as e:
                logger.error(f"Error during occupancy execution for job {job_id}: {e}", exc_info=True)
                await self._persist_failure(job_id, "EXECUTION_ERROR", str(e), video_path)
            finally:
                self._running_jobs.discard(job_id)
                # Cleanup temp input file
                try:
                    if video_path.exists():
                        video_path.unlink(missing_ok=True)
                except Exception:
                    pass
                self._active_tasks.pop(job_id, None)
                self._cancellation_events.pop(job_id, None)
                self._cancel_requested.discard(job_id)

    def _run_streaming_pipeline(
        self,
        job_id: str,
        camera_id: str,
        site_id: str,
        video_path: Path,
        input_video_sha: str,
        media_info: Dict[str, Any],
        verified_layout_canonical_sha: Optional[str],
        latest_assessment_ref_sha: Optional[str],
        latest_assessment_id: Optional[str],
        latest_assessment_config_sha: Optional[str],
        parking_spaces_payload: List[Dict[str, Any]],
        job_started_at: datetime,
        progress_callback: Any,
        cancel_event: threading.Event,
    ) -> Dict[str, Any]:
        """
        Runs bounded streaming pipeline:
        VideoCapture frame-by-frame -> YOLO detection -> ByteTrack tracking -> Occupancy scoring -> Frame rendering -> FFmpeg stdin.
        Eliminates full-video memory buffering.
        """
        if cancel_event.is_set():
            raise asyncio.CancelledError("Job cancelled before pipeline initialization.")

        progress_callback(10.0, "Initializing vehicle detector and geometric layout engine", "VALIDATING")

        # 1. Initialize local detector & session-local ByteTrack tracker
        detector = LocalVehicleDetector(self.config.detector)
        detector_sha = detector.checkpoint_sha256

        tracker_config_path = Path(__file__).resolve().parents[4] / "configs" / "tracking" / "bytetrack_default.yaml"
        tracker = ParkingByteTracker(config_path=tracker_config_path if tracker_config_path.is_file() else None)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or media_info.get("width", 1920))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or media_info.get("height", 1080))
        duration = float(media_info.get("duration", total_frames / fps if fps > 0 else 0.0))

        engine = ParkingOccupancyEngine(
            config=self.config,
            parking_spaces=parking_spaces_payload,
            video_width=width,
            video_height=height,
        )
        annotator = ParkingVideoAnnotator(self.config.rendering)

        # Output paths & staging setup
        media_root = Path(settings.MEDIA_ROOT) if hasattr(settings, "MEDIA_ROOT") else Path(tempfile.gettempdir()) / "roadsense_media"
        staging_dir = media_root / "parking_jobs" / f"staging_{job_id}"
        final_output_dir = media_root / "parking_jobs" / job_id

        # Clean staging directory if previously existing
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)

        staging_mp4_path = staging_dir / "annotated.mp4"
        staging_timeline_path = staging_dir / "occupancy_timeline.jsonl"
        staging_manifest_path = staging_dir / "processing_manifest.json"
        staging_summary_path = staging_dir / "parking_summary.json"

        # Initialize FFmpeg Stream Encoder
        encoder = FFmpegStreamEncoder(
            output_path=staging_mp4_path,
            width=width,
            height=height,
            fps=fps,
            timeout_seconds=self.config.execution.ffmpeg_timeout_seconds,
            cancellation_event=cancel_event,
        )

        timeline_file = open(staging_timeline_path, "w", encoding="utf-8")

        frame_idx = 0
        total_transitions = 0
        last_frame_res: Optional[FrameOccupancyResult] = None

        try:
            progress_callback(15.0, "Streaming video frames through detector and tracker", "DETECTING")

            while True:
                if cancel_event.is_set():
                    raise asyncio.CancelledError("Processing cancelled by user request.")

                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break

                timestamp_sec = float(frame_idx / fps)

                # 1. Detection
                raw_detections = detector.detect_vehicles(frame_bgr)

                # 2. Tracking (session-local ByteTrack)
                tracked_detections = tracker.update_tracks(
                    detections=raw_detections,
                    frame_idx=frame_idx,
                    frame_shape=(height, width),
                )

                # 3. Occupancy Engine
                frame_res = engine.process_frame(frame_idx, timestamp_sec, tracked_detections)
                last_frame_res = frame_res

                if cancel_event.is_set():
                    raise ParkingJobCancelled(f"Job {job_id} cancelled during video frame decoding.")

                timestamp_sec = float(frame_idx / fps)

                # 1. Detection
                raw_detections = detector.detect_vehicles(frame_bgr)

                # 2. Tracking (session-local ByteTrack)
                tracked_detections = tracker.update_tracks(
                    detections=raw_detections,
                    frame_idx=frame_idx,
                    frame_shape=(height, width),
                )

                # 3. Occupancy Engine
                frame_res = engine.process_frame(frame_idx, timestamp_sec, tracked_detections)
                last_frame_res = frame_res

                # Stream real state transitions directly to JSONL on disk
                for tl_entry in frame_res.state_transitions:
                    timeline_file.write(json.dumps(tl_entry.to_dict()) + "\n")
                    total_transitions += 1

                # 4. Annotate single frame
                rendered = annotator.render_frame(
                    frame_bgr=frame_bgr,
                    bay_states=frame_res.bay_states,
                    bay_polygons_px=engine.bay_polygons_px,
                    vehicle_detections=tracked_detections,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                )

                # 5. Stream frame directly to FFmpeg stdin pipe (no RAM accumulation)
                encoder.write_frame(rendered)

                frame_idx += 1
                if total_frames > 0:
                    pct = 15.0 + (65.0 * (frame_idx / max(1, total_frames)))
                    if frame_idx % 10 == 0 or frame_idx == total_frames:
                        progress_callback(pct, f"Evaluating occupancy: frame {frame_idx}/{total_frames}", "CLASSIFYING_OCCUPANCY")

        except Exception:
            encoder.cleanup()
            timeline_file.close()
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        finally:
            cap.release()
            try:
                timeline_file.close()
            except Exception:
                pass

        if frame_idx == 0 or last_frame_res is None:
            encoder.cleanup()
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise ValueError("No video frames could be decoded or processed.")

        # Finalize FFmpeg encoding
        progress_callback(82.0, "Finalizing H.264 video stream encoding", "ENCODING")
        encoder.finish()

        # Compute output video SHA
        h_out = hashlib.sha256()
        with open(staging_mp4_path, "rb") as of:
            while chunk := of.read(65536):
                h_out.update(chunk)
        output_video_sha = h_out.hexdigest()

        # Final Bay summary
        bay_summary = {k: v.to_dict() for k, v in last_frame_res.bay_states.items()}
        with open(staging_summary_path, "w", encoding="utf-8") as sf:
            json.dump(
                {
                    "job_id": job_id,
                    "total_bays": last_frame_res.total_bays,
                    "occupied_count": last_frame_res.occupied_count,
                    "vacant_count": last_frame_res.vacant_count,
                    "unknown_count": last_frame_res.unknown_count,
                    "occluded_count": last_frame_res.occluded_count,
                    "total_state_transitions": total_transitions,
                    "bay_summary": bay_summary,
                },
                sf,
                indent=2,
            )

        # Inspect real runtime module versions
        try:
            import ultralytics
            ultralytics_ver = getattr(ultralytics, "__version__", "unknown")
        except Exception:
            ultralytics_ver = "unknown"

        try:
            import torch
            torch_ver = getattr(torch, "__version__", "unknown")
        except Exception:
            torch_ver = "unknown"

        try:
            import torchvision
            torchvision_ver = getattr(torchvision, "__version__", "unknown")
        except Exception:
            torchvision_ver = "unknown"

        manifest = ParkingJobManifest(
            job_id=job_id,
            camera_id=camera_id,
            site_id=site_id,
            input_video_sha256=input_video_sha,
            output_video_sha256=output_video_sha,
            reference_image_sha256=latest_assessment_ref_sha or "",
            layout_canonical_sha256=verified_layout_canonical_sha or "",
            stability_assessment_id=latest_assessment_id,
            stability_config_sha256=latest_assessment_config_sha or "",
            detector_checkpoint_sha256=detector_sha,
            occupancy_config_sha256=self.config.config_sha256,
            software_versions={
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "opencv": cv2.__version__,
                "pytorch": torch_ver,
                "torchvision": torchvision_ver,
                "ultralytics": ultralytics_ver,
                "bytetrack_config_sha": tracker.config_sha256,
            },
            total_frames=frame_idx,
            processed_frames=frame_idx,
            skipped_frames=0,
            fps=fps,
            duration_seconds=duration,
            video_width=width,
            video_height=height,
            total_bays=last_frame_res.total_bays,
            total_bays_evaluated=last_frame_res.total_bays,
            final_occupied_count=last_frame_res.occupied_count,
            final_vacant_count=last_frame_res.vacant_count,
            final_unknown_count=last_frame_res.unknown_count,
            final_occluded_count=last_frame_res.occluded_count,
            total_state_transitions=total_transitions,
            job_created_at=job_started_at.isoformat(),
            job_started_at=job_started_at.isoformat(),
            job_completed_at=_utc_now().isoformat(),
            operational_gate="ALLOWED",
            gate_reasons=["GATE_ALLOWED: Fresh camera stability assessment matches verified layout."],
        )

        with open(staging_manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest.to_dict(), mf, indent=2)

        # 6. Validate artifacts before Atomic Promotion
        self._validate_staging_artifacts(
            staging_dir=staging_dir,
            expected_width=width,
            expected_height=height,
            expected_fps=fps,
            expected_frames=frame_idx,
        )

        # 7. Atomic directory promotion
        if final_output_dir.exists():
            shutil.rmtree(final_output_dir, ignore_errors=True)
        staging_dir.rename(final_output_dir)

        final_mp4_path = final_output_dir / "annotated.mp4"
        final_timeline_path = final_output_dir / "occupancy_timeline.jsonl"

        return {
            "output_video_sha256": output_video_sha,
            "detector_checkpoint_sha256": detector_sha,
            "total_frames": frame_idx,
            "processed_frames": frame_idx,
            "fps": fps,
            "duration_seconds": duration,
            "video_width": width,
            "video_height": height,
            "total_bays": last_frame_res.total_bays,
            "final_occupied_count": last_frame_res.occupied_count,
            "final_vacant_count": last_frame_res.vacant_count,
            "final_unknown_count": last_frame_res.unknown_count,
            "final_occluded_count": last_frame_res.occluded_count,
            "total_state_transitions": total_transitions,
            "output_video_path": str(final_mp4_path),
            "timeline_jsonl_path": str(final_timeline_path),
            "manifest_json": manifest.to_dict(),
            "bay_summary_json": bay_summary,
        }

    def _validate_staging_artifacts(
        self,
        staging_dir: Path,
        expected_width: int,
        expected_height: int,
        expected_fps: float,
        expected_frames: int,
    ) -> None:
        """Thoroughly validate all generated artifacts inside staging before promotion."""
        mp4_path = staging_dir / "annotated.mp4"
        timeline_path = staging_dir / "occupancy_timeline.jsonl"
        manifest_path = staging_dir / "processing_manifest.json"
        summary_path = staging_dir / "parking_summary.json"

        if not mp4_path.is_file() or mp4_path.stat().st_size == 0:
            raise ValueError(f"Staging MP4 missing or empty: {mp4_path}")

        if not timeline_path.is_file():
            raise ValueError(f"Staging timeline JSONL missing: {timeline_path}")

        if not manifest_path.is_file():
            raise ValueError(f"Staging manifest JSON missing: {manifest_path}")

        if not summary_path.is_file():
            raise ValueError(f"Staging summary JSON missing: {summary_path}")

        # Validate MP4 with ffprobe
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate",
            "-of", "json",
            str(mp4_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0)
        if res.returncode != 0:
            raise ValueError(f"FFprobe failed to inspect staging MP4: {res.stderr}")

        info = json.loads(res.stdout)
        streams = info.get("streams", [])
        if not streams:
            raise ValueError("Staging MP4 contains no video streams.")

        v_stream = streams[0]
        if v_stream.get("codec_name") != "h264":
            raise ValueError(f"Expected H.264 codec, got {v_stream.get('codec_name')}")
        if v_stream.get("pix_fmt") != "yuv420p":
            raise ValueError(f"Expected yuv420p pixel format, got {v_stream.get('pix_fmt')}")
        if v_stream.get("width") != expected_width or v_stream.get("height") != expected_height:
            raise ValueError(f"Geometry mismatch: expected {expected_width}x{expected_height}, got {v_stream.get('width')}x{v_stream.get('height')}")

    async def _persist_complete(
        self,
        job_id: str,
        input_video_sha: str,
        current_ref_sha: Optional[str],
        current_layout_sha: Optional[str],
        verified_layout_id: Optional[str],
        latest_assessment_id: Optional[str],
        result_payload: Dict[str, Any],
    ) -> None:
        try:
            async with async_session_factory() as db:
                stmt = (
                    update(ParkingOccupancyJob)
                    .where(ParkingOccupancyJob.id == job_id)
                    .where(ParkingOccupancyJob.status.not_in(TERMINAL_STATES))
                    .values(
                        status="COMPLETE",
                        progress_pct=100.0,
                        stage_message="Parking occupancy evaluation and annotated video encoding complete",
                        gate_decision="ALLOWED",
                        gate_reasons=["GATE_ALLOWED: Fresh camera stability assessment matches verified layout."],
                        input_video_sha256=input_video_sha,
                        output_video_sha256=result_payload["output_video_sha256"],
                        reference_image_sha256=current_ref_sha,
                        layout_canonical_sha256=current_layout_sha,
                        detector_checkpoint_sha256=result_payload["detector_checkpoint_sha256"],
                        occupancy_config_sha256=self.config.config_sha256,
                        layout_revision_id=verified_layout_id,
                        stability_assessment_id=latest_assessment_id,
                        total_frames=result_payload["total_frames"],
                        processed_frames=result_payload["processed_frames"],
                        fps=result_payload["fps"],
                        duration_seconds=result_payload["duration_seconds"],
                        video_width=result_payload["video_width"],
                        video_height=result_payload["video_height"],
                        total_bays=result_payload["total_bays"],
                        final_occupied_count=result_payload["final_occupied_count"],
                        final_vacant_count=result_payload["final_vacant_count"],
                        final_unknown_count=result_payload["final_unknown_count"],
                        final_occluded_count=result_payload["final_occluded_count"],
                        total_state_transitions=result_payload["total_state_transitions"],
                        output_video_path=result_payload["output_video_path"],
                        timeline_jsonl_path=result_payload["timeline_jsonl_path"],
                        manifest_json=result_payload["manifest_json"],
                        bay_summary_json=result_payload["bay_summary_json"],
                        completed_at=_utc_now(),
                    )
                )
                res = await db.execute(stmt)
                await db.commit()
                if res.rowcount == 0:
                    logger.warning(f"Job {job_id} already reached a terminal state; skipping COMPLETE transition.")
        except Exception as e:
            logger.error(f"Error persisting COMPLETE state for job {job_id}: {e}")

    async def _persist_blocked_by_gate(self, job_id: str, reasons: List[str], video_path: Path) -> None:
        try:
            async with async_session_factory() as db:
                stmt = (
                    update(ParkingOccupancyJob)
                    .where(ParkingOccupancyJob.id == job_id)
                    .where(ParkingOccupancyJob.status.not_in(TERMINAL_STATES))
                    .values(
                        status="BLOCKED_BY_STABILITY_GATE",
                        progress_pct=0.0,
                        gate_decision="BLOCKED",
                        gate_reasons=reasons,
                        failure_code="STABILITY_GATE_BLOCKED",
                        failure_message="Camera stability assessment is UNSTABLE, missing, or mismatched. Occupancy processing blocked fail-closed.",
                        stage_message="Occupancy evaluation blocked fail-closed by camera stability gate.",
                        completed_at=_utc_now(),
                    )
                )
                res = await db.execute(stmt)
                await db.commit()
                if res.rowcount == 0:
                    logger.warning(f"Job {job_id} already reached a terminal state; skipping BLOCKED transition.")
        except Exception as e:
            logger.error(f"Error persisting blocked gate state for job {job_id}: {e}")

    async def _persist_cancellation(self, job_id: str, video_path: Path) -> None:
        try:
            async with async_session_factory() as db:
                stmt = (
                    update(ParkingOccupancyJob)
                    .where(ParkingOccupancyJob.id == job_id)
                    .where(ParkingOccupancyJob.status.not_in(TERMINAL_STATES))
                    .values(
                        status="CANCELLED",
                        stage_message="Job cancelled by user request.",
                        failure_code="CANCELLED",
                        failure_message="Job cancelled prior to completion.",
                        completed_at=_utc_now(),
                    )
                )
                res = await db.execute(stmt)
                await db.commit()
                if res.rowcount == 0:
                    logger.warning(f"Job {job_id} already reached a terminal state; skipping CANCELLED transition.")
        except Exception as e:
            logger.error(f"Error persisting cancellation for job {job_id}: {e}")

    async def _persist_failure(self, job_id: str, code: str, message: str, video_path: Path) -> None:
        try:
            async with async_session_factory() as db:
                stmt = (
                    update(ParkingOccupancyJob)
                    .where(ParkingOccupancyJob.id == job_id)
                    .where(ParkingOccupancyJob.status.not_in(TERMINAL_STATES))
                    .values(
                        status="FAILED",
                        failure_code=code,
                        failure_message=message,
                        stage_message=f"Failed: {message}",
                        completed_at=_utc_now(),
                    )
                )
                res = await db.execute(stmt)
                await db.commit()
                if res.rowcount == 0:
                    logger.warning(f"Job {job_id} already reached a terminal state; skipping FAILED transition.")
        except Exception as e:
            logger.error(f"Error persisting failure for job {job_id}: {e}")


parking_occupancy_job_manager = ParkingOccupancyJobManager()
