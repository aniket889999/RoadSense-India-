"""Asynchronous bounded background job manager for gated parking occupancy and video annotation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Set
import uuid
import cv2
import numpy as np
from sqlalchemy import desc, select

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
    ParkingJobManifest,
)
from src.parking.occupancy_engine import ParkingOccupancyEngine
from src.parking.stability_engine import inspect_video_media_safe
from src.parking.vehicle_detector import LocalVehicleDetector
from src.parking.video_annotator import ParkingVideoAnnotator, encode_frames_to_mp4


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ParkingOccupancyJobManager:
    """
    Manages asynchronous, gate-controlled parking occupancy inference jobs with bounded concurrency.
    """

    def __init__(self, max_concurrency: int = 2) -> None:
        self.config: ParkingOccupancyConfig = load_parking_occupancy_config()
        self._max_concurrency: int = max(1, self.config.execution.max_concurrent_jobs or max_concurrency)
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(self._max_concurrency)
        self._active_tasks: Dict[str, asyncio.Task[None]] = {}
        self._cancel_requested: Set[str] = set()

    def submit_occupancy_job(
        self,
        job_id: str,
        camera_id: str,
        site_id: str,
        video_path: Path,
    ) -> asyncio.Task[None]:
        """Submit a new parking occupancy job for asynchronous execution."""
        task = asyncio.create_task(
            self._execute_job(
                job_id=job_id,
                camera_id=camera_id,
                site_id=site_id,
                video_path=video_path,
            ),
            name=f"parking-occupancy-job-{job_id}",
        )
        self._active_tasks[job_id] = task
        return task

    async def cancel_occupancy_job(self, job_id: str) -> bool:
        """Request cancellation of an active or queued occupancy job."""
        self._cancel_requested.add(job_id)
        task = self._active_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    async def _execute_job(
        self,
        job_id: str,
        camera_id: str,
        site_id: str,
        video_path: Path,
    ) -> None:
        async with self._semaphore:
            job_started_at = _utc_now()
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

            if job_id in self._cancel_requested:
                await self._persist_cancellation(job_id, video_path)
                return

            # 2. Gate Verification Query
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
                        ]

                    ass_res = await db.execute(
                        select(CameraStabilityAssessment)
                        .where(CameraStabilityAssessment.camera_id == camera_id)
                        .where(CameraStabilityAssessment.status == "COMPLETE")
                        .order_by(desc(CameraStabilityAssessment.created_at))
                        .limit(1)
                    )
                    latest_assessment = ass_res.scalar_one_or_none()
                    if latest_assessment:
                        latest_assessment_id = latest_assessment.id
                        latest_assessment_ref_sha = latest_assessment.reference_image_sha256
                        latest_assessment_layout_sha = latest_assessment.layout_canonical_sha256
                        latest_assessment_config_sha = latest_assessment.config_sha256
                        latest_assessment_created_at = latest_assessment.created_at
                        latest_assessment_decision = latest_assessment.aggregate_decision
                        latest_assessment_thresholds = latest_assessment.thresholds_snapshot

            except Exception as e:
                logger.error(f"Database query error validating gate for job {job_id}: {e}")
                await self._persist_failure(job_id, "DB_ERROR", str(e), video_path)
                return

            # Fail-closed operational gate evaluation
            decision_enum = None
            if latest_assessment_decision:
                try:
                    decision_enum = StabilityDecision(latest_assessment_decision)
                except Exception:
                    pass

            now = _utc_now()
            max_age = (latest_assessment_thresholds or {}).get("max_assessment_age_seconds", 86400)

            gate, gate_reasons = evaluate_operational_gate(
                decision=decision_enum,
                assessment_reference_sha=latest_assessment_ref_sha,
                current_reference_sha=current_ref_sha,
                assessment_layout_sha=latest_assessment_layout_sha,
                current_layout_sha=current_layout_sha,
                assessment_timestamp=latest_assessment_created_at,
                current_timestamp=now,
                max_age_seconds=max_age,
            )

            # FAIL-CLOSED CHECK: Gate must be ALLOWED
            if gate != OperationalGate.ALLOWED or not verified_layout_id:
                all_reasons = list(gate_reasons)
                if not verified_layout_id:
                    all_reasons.insert(0, "NO_ACTIVE_VERIFIED_LAYOUT: Camera has no verified parking layout.")

                logger.warning(f"Job {job_id} BLOCKED by stability gate: {all_reasons}")
                await self._persist_blocked_by_gate(
                    job_id=job_id,
                    gate_reasons=all_reasons,
                    video_path=video_path,
                    layout_revision_id=verified_layout_id,
                    stability_assessment_id=latest_assessment_id,
                    layout_canonical_sha256=current_layout_sha,
                    reference_image_sha256=current_ref_sha,
                )
                return

            # 3. Media inspection & Hash
            try:
                media_info = inspect_video_media_safe(
                    video_path=video_path,
                    timeout_sec=self.config.execution.ffprobe_timeout_seconds,
                )
            except Exception as e:
                logger.error(f"Media validation error for job {job_id}: {e}")
                await self._persist_failure(job_id, "INVALID_MEDIA", str(e), video_path)
                return

            # Compute video SHA-256
            h_vid = hashlib.sha256()
            with open(video_path, "rb") as vf:
                while chunk := vf.read(65536):
                    h_vid.update(chunk)
            input_video_sha = h_vid.hexdigest()

            # Execute pipeline in worker thread
            def sync_progress_callback(pct: float, msg: str) -> None:
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self._update_progress_db(job_id, pct, msg),
                            loop,
                        )
                except Exception:
                    pass

            try:
                result_payload = await asyncio.to_thread(
                    self._run_synchronous_pipeline,
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
                )

                if job_id in self._cancel_requested:
                    await self._persist_cancellation(job_id, video_path)
                    return

                # 5. Persist COMPLETE state
                async with async_session_factory() as db:
                    res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
                    job = res.scalar_one_or_none()
                    if job:
                        job.status = "COMPLETE"
                        job.progress_pct = 100.0
                        job.stage_message = "Parking occupancy evaluation and annotated video encoding complete"
                        job.gate_decision = "ALLOWED"
                        job.gate_reasons = ["GATE_ALLOWED: Fresh camera stability assessment matches verified layout."]
                        job.input_video_sha256 = input_video_sha
                        job.output_video_sha256 = result_payload["output_video_sha256"]
                        job.reference_image_sha256 = current_ref_sha
                        job.layout_canonical_sha256 = current_layout_sha
                        job.detector_checkpoint_sha256 = result_payload["detector_checkpoint_sha256"]
                        job.occupancy_config_sha256 = self.config.config_sha256
                        job.layout_revision_id = verified_layout_id
                        job.stability_assessment_id = latest_assessment_id
                        job.total_frames = result_payload["total_frames"]
                        job.processed_frames = result_payload["processed_frames"]
                        job.fps = result_payload["fps"]
                        job.duration_seconds = result_payload["duration_seconds"]
                        job.video_width = result_payload["video_width"]
                        job.video_height = result_payload["video_height"]
                        job.total_bays = result_payload["total_bays"]
                        job.final_occupied_count = result_payload["final_occupied_count"]
                        job.final_vacant_count = result_payload["final_vacant_count"]
                        job.final_unknown_count = result_payload["final_unknown_count"]
                        job.final_occluded_count = result_payload["final_occluded_count"]
                        job.total_state_transitions = result_payload["total_state_transitions"]
                        job.output_video_path = result_payload["output_video_path"]
                        job.timeline_jsonl_path = result_payload["timeline_jsonl_path"]
                        job.manifest_json = result_payload["manifest_json"]
                        job.bay_summary_json = result_payload["bay_summary_json"]
                        job.completed_at = _utc_now()
                        await db.commit()


            except asyncio.CancelledError:
                await self._persist_cancellation(job_id, video_path)
            except Exception as e:
                logger.error(f"Error during occupancy execution for job {job_id}: {e}", exc_info=True)
                await self._persist_failure(job_id, "EXECUTION_ERROR", str(e), video_path)
            finally:
                # Cleanup temp input file
                try:
                    if video_path.exists():
                        video_path.unlink(missing_ok=True)
                except Exception:
                    pass
                self._active_tasks.pop(job_id, None)
                self._cancel_requested.discard(job_id)

    def _run_synchronous_pipeline(
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
    ) -> Dict[str, Any]:
        """Runs the CPU/GPU pipeline: frame extraction -> YOLO detection -> occupancy scoring -> overlay rendering -> FFmpeg."""
        progress_callback(10.0, "Initializing vehicle detector and geometric layout engine")

        detector = LocalVehicleDetector(self.config.detector)
        detector_sha = detector.checkpoint_sha256

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

        # Output paths
        media_root = Path(settings.MEDIA_ROOT) if hasattr(settings, "MEDIA_ROOT") else Path(tempfile.gettempdir()) / "roadsense_media"
        output_dir = media_root / "parking_jobs" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        annotated_mp4_path = output_dir / "annotated.mp4"
        timeline_path = output_dir / "occupancy_timeline.jsonl"
        manifest_path = output_dir / "processing_manifest.json"
        summary_path = output_dir / "parking_summary.json"

        frame_results: List[FrameOccupancyResult] = []
        rendered_frames: List[np.ndarray] = []

        try:
            frame_idx = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break

                timestamp_sec = float(frame_idx / fps)

                # 1. Detection
                detections = detector.detect_vehicles(frame_bgr)

                # 2. Occupancy Engine
                frame_res = engine.process_frame(frame_idx, timestamp_sec, detections)
                frame_results.append(frame_res)

                # 3. Annotate frame
                rendered = annotator.render_frame(
                    frame_bgr=frame_bgr,
                    bay_states=frame_res.bay_states,
                    bay_polygons_px=engine.bay_polygons_px,
                    vehicle_detections=detections,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                )
                rendered_frames.append(rendered)

                frame_idx += 1
                if total_frames > 0:
                    pct = 15.0 + (65.0 * (frame_idx / max(1, total_frames)))
                    if frame_idx % 10 == 0 or frame_idx == total_frames:
                        progress_callback(pct, f"Evaluating occupancy: frame {frame_idx}/{total_frames}")

        finally:
            cap.release()

        processed_count = len(rendered_frames)
        if processed_count == 0:
            raise ValueError("No video frames could be decoded or processed.")

        # Encode MP4 via FFmpeg
        progress_callback(82.0, "Encoding annotated video to H.264 MP4")
        encode_frames_to_mp4(
            frame_generator=iter(rendered_frames),
            output_path=annotated_mp4_path,
            fps=fps,
            width=width,
            height=height,
            total_frames=processed_count,
            timeout_seconds=self.config.execution.ffmpeg_timeout_seconds,
        )

        progress_callback(95.0, "Generating processing manifest and timeline ledger")

        # Compute output video SHA
        h_out = hashlib.sha256()
        with open(annotated_mp4_path, "rb") as of:
            while chunk := of.read(65536):
                h_out.update(chunk)
        output_video_sha = h_out.hexdigest()

        # Write timeline JSONL
        with open(timeline_path, "w", encoding="utf-8") as tf:
            for t in engine.timeline:
                tf.write(json.dumps(t.to_dict()) + "\n")

        # Final frame stats
        final_frame_res = frame_results[-1]
        final_bay_states = {k: v.to_dict() for k, v in final_frame_res.bay_states.items()}

        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(
                {
                    "job_id": job_id,
                    "total_bays": final_frame_res.total_bays,
                    "occupied_count": final_frame_res.occupied_count,
                    "vacant_count": final_frame_res.vacant_count,
                    "unknown_count": final_frame_res.unknown_count,
                    "occluded_count": final_frame_res.occluded_count,
                    "bay_states": final_bay_states,
                    "total_timeline_events": len(engine.timeline),
                },
                sf,
                indent=2,
            )

        job_completed_at = _utc_now()

        manifest = ParkingJobManifest(
            job_id=job_id,
            camera_id=camera_id,
            site_id=site_id,
            input_video_sha256=input_video_sha,
            output_video_sha256=output_video_sha,
            reference_image_sha256=latest_assessment_ref_sha or "",
            layout_canonical_sha256=verified_layout_canonical_sha or "",
            stability_assessment_id=latest_assessment_id or "",
            stability_config_sha256=latest_assessment_config_sha or "",
            detector_checkpoint_sha256=detector_sha,
            occupancy_config_sha256=self.config.config_sha256,
            algorithm_version=self.config.algorithm_version,
            opencv_version=cv2.__version__,
            ultralytics_version="8.3.0",
            total_frames=total_frames,
            processed_frames=processed_count,
            fps=fps,
            duration_seconds=duration,
            video_width=width,
            video_height=height,
            total_bays_evaluated=final_frame_res.total_bays,
            final_occupied_count=final_frame_res.occupied_count,
            final_vacant_count=final_frame_res.vacant_count,
            final_unknown_count=final_frame_res.unknown_count,
            final_occluded_count=final_frame_res.occluded_count,
            total_state_transitions=len(engine.timeline),
            job_started_at=job_started_at.isoformat(),
            job_completed_at=job_completed_at.isoformat(),
            errors_or_warnings=[],
        )

        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest.to_dict(), mf, indent=2)

        return {
            "output_video_sha256": output_video_sha,
            "detector_checkpoint_sha256": detector_sha,
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "fps": fps,
            "duration_seconds": duration,
            "video_width": width,
            "video_height": height,
            "total_bays": final_frame_res.total_bays,
            "final_occupied_count": final_frame_res.occupied_count,
            "final_vacant_count": final_frame_res.vacant_count,
            "final_unknown_count": final_frame_res.unknown_count,
            "final_occluded_count": final_frame_res.occluded_count,
            "total_state_transitions": len(engine.timeline),
            "output_video_path": str(annotated_mp4_path),
            "timeline_jsonl_path": str(timeline_path),
            "manifest_json": manifest.to_dict(),
            "bay_summary_json": final_bay_states,
        }

    async def _update_progress_db(self, job_id: str, pct: float, msg: str) -> None:
        try:
            async with async_session_factory() as db:
                res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
                job = res.scalar_one_or_none()
                if job and job.status not in ("COMPLETE", "FAILED", "CANCELLED", "BLOCKED_BY_STABILITY_GATE"):
                    if pct < 30.0:
                        job.status = "DETECTING"
                    elif pct < 80.0:
                        job.status = "CLASSIFYING_OCCUPANCY"
                    elif pct < 95.0:
                        job.status = "RENDERING"
                    else:
                        job.status = "ENCODING"
                    job.progress_pct = round(pct, 1)
                    job.stage_message = msg
                    await db.commit()
        except Exception:
            pass

    async def _persist_blocked_by_gate(
        self,
        job_id: str,
        gate_reasons: List[str],
        video_path: Path,
        layout_revision_id: Optional[str],
        stability_assessment_id: Optional[str],
        layout_canonical_sha256: Optional[str],
        reference_image_sha256: Optional[str],
    ) -> None:
        try:
            async with async_session_factory() as db:
                res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
                job = res.scalar_one_or_none()
                if job:
                    job.status = "BLOCKED_BY_STABILITY_GATE"
                    job.progress_pct = 100.0
                    job.stage_message = "Occupancy inference blocked: camera stability gate is not ALLOWED"
                    job.failure_code = "STABILITY_GATE_BLOCKED"
                    job.failure_message = "; ".join(gate_reasons)
                    job.gate_decision = "BLOCKED"
                    job.gate_reasons = gate_reasons
                    job.layout_revision_id = layout_revision_id
                    job.stability_assessment_id = stability_assessment_id
                    job.layout_canonical_sha256 = layout_canonical_sha256
                    job.reference_image_sha256 = reference_image_sha256
                    job.completed_at = _utc_now()
                    await db.commit()
        except Exception as e:
            logger.error(f"Failed to persist blocked state for job {job_id}: {e}")
        finally:
            try:
                if video_path.exists():
                    video_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._active_tasks.pop(job_id, None)

    async def _persist_failure(self, job_id: str, code: str, msg: str, video_path: Path) -> None:
        try:
            async with async_session_factory() as db:
                res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
                job = res.scalar_one_or_none()
                if job:
                    job.status = "FAILED"
                    job.progress_pct = 100.0
                    job.stage_message = "Parking occupancy job failed during processing"
                    job.failure_code = code
                    job.failure_message = msg
                    job.completed_at = _utc_now()
                    await db.commit()
        except Exception as e:
            logger.error(f"Failed to persist failure for job {job_id}: {e}")
        finally:
            try:
                if video_path.exists():
                    video_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._active_tasks.pop(job_id, None)

    async def _persist_cancellation(self, job_id: str, video_path: Path) -> None:
        try:
            async with async_session_factory() as db:
                res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
                job = res.scalar_one_or_none()
                if job:
                    job.status = "CANCELLED"
                    job.progress_pct = 100.0
                    job.stage_message = "Parking occupancy job cancelled by operator"
                    job.failure_code = "CANCELLED"
                    job.failure_message = "Job cancelled prior to completion."
                    job.completed_at = _utc_now()
                    await db.commit()
        except Exception as e:
            logger.error(f"Failed to persist cancellation for job {job_id}: {e}")
        finally:
            try:
                if video_path.exists():
                    video_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._active_tasks.pop(job_id, None)


# Singleton instance
parking_occupancy_job_manager = ParkingOccupancyJobManager()
