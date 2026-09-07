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
        self._detector_override: Optional[Any] = None

    def set_detector_override(self, detector: Optional[Any]) -> None:
        """Inject a test double vehicle detector for deterministic automated tests and validation."""
        self._detector_override = detector

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

    async def request_job_cancellation(self, job_id: str) -> Dict[str, Any]:
        """
        Atomically request cancellation of an active or queued occupancy job using database CAS.
        PUBLISHING and every terminal state are strictly excluded.
        Sets cooperative cancellation event only if the CAS succeeds.
        """
        cancellable_statuses = (
            "QUEUED",
            "PENDING",
            "VALIDATING",
            "DETECTING",
            "TRACKING",
            "CLASSIFYING_OCCUPANCY",
            "RENDERING",
            "ENCODING",
            "RUNNING",
        )
        try:
            async with async_session_factory() as db:
                stmt = (
                    update(ParkingOccupancyJob)
                    .where(ParkingOccupancyJob.id == job_id)
                    .where(ParkingOccupancyJob.status.in_(cancellable_statuses))
                    .values(
                        status="CANCELLED",
                        progress_pct=100.0,
                        stage_message="Job cancelled by operator request",
                        failure_code="CANCELLED",
                        failure_message="Job cancelled by operator request.",
                        completed_at=_utc_now(),
                    )
                )
                res = await db.execute(stmt)
                await db.commit()

                if res.rowcount > 0:
                    self._cancel_requested.add(job_id)
                    evt = self._cancellation_events.get(job_id)
                    if evt:
                        evt.set()
                    return {
                        "job_id": job_id,
                        "status": "CANCELLED",
                        "cancelled": True,
                        "message": "Parking occupancy job cancelled successfully.",
                    }

                # CAS failed: fetch current state from DB
                check_res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
                job = check_res.scalar_one_or_none()
                if not job:
                    return {
                        "job_id": job_id,
                        "status": "NOT_FOUND",
                        "cancelled": False,
                        "message": "Parking occupancy job not found.",
                    }

                return {
                    "job_id": job_id,
                    "status": job.status,
                    "cancelled": False,
                    "message": f"Job is in non-cancellable state '{job.status}'.",
                }
        except Exception as e:
            logger.error(f"Error executing cancellation CAS for job {job_id}: {e}")
            raise

    async def cancel_occupancy_job(self, job_id: str) -> bool:
        """Cooperative cancellation alias delegating to atomic request_job_cancellation."""
        res = await self.request_job_cancellation(job_id)
        return bool(res.get("cancelled", False))

    async def _execute_job(
        self,
        job_id: str,
        camera_id: str,
        site_id: str,
        video_path: Path,
        cancel_event: threading.Event,
    ) -> None:
        try:
            if job_id in self._cancel_requested or cancel_event.is_set():
                await self._persist_cancellation(job_id, video_path)
                return

            async with self.semaphore:
                if job_id in self._cancel_requested or cancel_event.is_set():
                    await self._persist_cancellation(job_id, video_path)
                    return

                main_loop = asyncio.get_running_loop()

                # 1. Update initial status to VALIDATING atomically via CAS (only QUEUED or PENDING -> VALIDATING)
                startup_started_at = _utc_now()
                startup_cas_ok = False
                try:
                    async with async_session_factory() as db:
                        stmt = (
                            update(ParkingOccupancyJob)
                            .where(ParkingOccupancyJob.id == job_id)
                            .where(ParkingOccupancyJob.status.in_(("QUEUED", "PENDING")))
                            .values(
                                status="VALIDATING",
                                started_at=startup_started_at,
                                progress_pct=5.0,
                                stage_message="Validating camera layout and camera stability gate",
                            )
                        )
                        res = await db.execute(stmt)
                        await db.commit()
                        startup_cas_ok = res.rowcount > 0
                except Exception as e:
                    logger.error(f"Failed to execute startup CAS for job {job_id}: {e}")
                    await self._persist_failure(job_id, "STARTUP_CAS_ERROR", str(e), video_path)
                    return

                if not startup_cas_ok:
                    # CAS failed: reload current state from DB
                    try:
                        async with async_session_factory() as db:
                            res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
                            current_job = res.scalar_one_or_none()
                    except Exception as e:
                        logger.error(f"Failed to reload job {job_id} after failed startup CAS: {e}")
                        current_job = None

                    if current_job is None:
                        logger.warning(f"Occupancy job {job_id} missing during startup validation.")
                        return

                    if current_job.status == "CANCELLED":
                        logger.info(f"Occupancy job {job_id} was cancelled before startup validation.")
                        return

                    if current_job.status in TERMINAL_STATES or current_job.status == "PUBLISHING":
                        logger.info(f"Occupancy job {job_id} already in state '{current_job.status}'; aborting startup without overwriting.")
                        return

                    logger.warning(f"Occupancy job {job_id} in unexpected state '{current_job.status}' during startup; aborting.")
                    return

                job_started_at = startup_started_at

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

                import math
                raw_max_age = (latest_assessment_thresholds or {}).get(
                    "max_assessment_age_seconds",
                    current_stability_config.thresholds.max_assessment_age_seconds,
                )
                if isinstance(raw_max_age, bool) or not isinstance(raw_max_age, (int, float)) or not math.isfinite(raw_max_age) or raw_max_age <= 0:
                    logger.warning(f"Occupancy job {job_id} BLOCKED: max_assessment_age_seconds must be a finite positive number, got {raw_max_age}")
                    await self._persist_blocked_by_gate(
                        job_id,
                        [f"INVALID_THRESHOLD: max_assessment_age_seconds must be a finite positive number, got {raw_max_age}"],
                        video_path,
                    )
                    return

                max_age = float(raw_max_age)

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

                # Strict Stability Configuration SHA Check: missing config_sha256 must block
                if not latest_assessment_config_sha:
                    gate_decision = OperationalGate.BLOCKED
                    gate_reasons.append("MISSING_CONFIG_SHA: Camera stability assessment record is missing config_sha256.")
                elif latest_assessment_config_sha != current_stability_config.config_sha256:
                    gate_decision = OperationalGate.BLOCKED
                    gate_reasons.append(
                        f"CONFIG_MISMATCH: Stability assessment used config SHA {latest_assessment_config_sha[:8]}..., "
                        f"current system requires {current_stability_config.config_sha256[:8]}..."
                    )

                if gate_decision != OperationalGate.ALLOWED:
                    logger.warning(f"Occupancy job {job_id} BLOCKED by stability gate: {gate_reasons}")
                    await self._persist_blocked_by_gate(job_id, gate_reasons, video_path)
                    return

                if not parking_spaces_payload:
                    gate_reasons.append("NO_ACTIVE_SPACES: Verified layout contains 0 active parking spaces.")
                    await self._persist_blocked_by_gate(job_id, gate_reasons, video_path)
                    return

                # Capture reservation snapshot for exact provenance re-validation
                reserved_ref_sha = current_ref_sha
                reserved_layout_id = verified_layout_id
                reserved_layout_sha = current_layout_sha
                reserved_assessment_id = latest_assessment_id
                reserved_assessment_decision = latest_assessment_decision
                reserved_assessment_timestamp = latest_assessment_created_at
                reserved_assessment_config_sha = latest_assessment_config_sha
                reserved_stability_config_sha = current_stability_config.config_sha256

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
                        if job_id in self._cancel_requested or cancel_event.is_set():
                            await self._persist_cancellation(job_id, video_path)
                            return
                        h_vid.update(chunk)
                input_video_sha = h_vid.hexdigest()

                # 4. Provenance Re-Validation immediately prior to model / tracker loading
                reval_stability_cfg = load_stability_config()
                try:
                    async with async_session_factory() as db:
                        c_res = await db.execute(select(Camera).where(Camera.id == camera_id))
                        c_now = c_res.scalar_one_or_none()
                        if not c_now or c_now.reference_image_sha256 != reserved_ref_sha:
                            await self._persist_blocked_by_gate(
                                job_id,
                                [f"GATE_PROVENANCE_MUTATION: Camera reference SHA changed immediately prior to model construction"],
                                video_path,
                            )
                            return

                        from sqlalchemy.orm import selectinload
                        l_res = await db.execute(
                            select(ParkingLayoutRevision)
                            .options(selectinload(ParkingLayoutRevision.parking_spaces))
                            .where(ParkingLayoutRevision.camera_id == camera_id)
                            .where(ParkingLayoutRevision.status == LayoutRevisionStatus.VERIFIED.value)
                        )
                        l_now = l_res.scalar_one_or_none()
                        if not l_now or l_now.id != reserved_layout_id or l_now.canonical_sha256 != reserved_layout_sha:
                            await self._persist_blocked_by_gate(
                                job_id,
                                [f"GATE_PROVENANCE_MUTATION: Verified layout changed immediately prior to model construction"],
                                video_path,
                            )
                            return

                        a_res = await db.execute(
                            select(CameraStabilityAssessment)
                            .where(CameraStabilityAssessment.camera_id == camera_id)
                            .where(CameraStabilityAssessment.status == "COMPLETE")
                            .order_by(desc(CameraStabilityAssessment.created_at))
                        )
                        a_now = a_res.scalars().first()
                        if not a_now or a_now.id != reserved_assessment_id:
                            await self._persist_blocked_by_gate(
                                job_id,
                                [f"GATE_PROVENANCE_MUTATION: Latest stability assessment changed immediately prior to model construction"],
                                video_path,
                            )
                            return

                        if (
                            a_now.aggregate_decision != reserved_assessment_decision
                            or a_now.created_at != reserved_assessment_timestamp
                            or a_now.config_sha256 != reserved_assessment_config_sha
                            or not a_now.config_sha256
                        ):
                            await self._persist_blocked_by_gate(
                                job_id,
                                [f"GATE_PROVENANCE_MUTATION: Stability assessment provenance mutated immediately prior to model construction"],
                                video_path,
                            )
                            return

                        if (
                            reval_stability_cfg.config_sha256 != reserved_stability_config_sha
                            or a_now.config_sha256 != reval_stability_cfg.config_sha256
                        ):
                            await self._persist_blocked_by_gate(
                                job_id,
                                [f"GATE_PROVENANCE_MUTATION: Stability configuration mutated immediately prior to model construction"],
                                video_path,
                            )
                            return

                except Exception as e:
                    logger.error(f"Error during provenance re-validation for job {job_id}: {e}")
                    await self._persist_failure(job_id, "PROVENANCE_REVALIDATION_ERROR", str(e), video_path)
                    return

                if job_id in self._cancel_requested or cancel_event.is_set():
                    await self._persist_cancellation(job_id, video_path)
                    return

                # Thread-safe async progress updater
                async def _update_db_progress(pct: float, msg: str, stage_status: Optional[str] = None) -> None:
                    try:
                        async with async_session_factory() as db:
                            values_dict: Dict[str, Any] = {
                                "progress_pct": round(pct, 1),
                                "stage_message": msg,
                            }
                            if stage_status:
                                values_dict["status"] = stage_status
                            stmt = (
                                update(ParkingOccupancyJob)
                                .where(ParkingOccupancyJob.id == job_id)
                                .where(ParkingOccupancyJob.status.not_in(TERMINAL_STATES + ("PUBLISHING",)))
                                .values(**values_dict)
                            )
                            await db.execute(stmt)
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

                # 5. Run Streaming Inference & Pipeline in Thread
                self._running_jobs.add(job_id)
                bundle: Dict[str, Any] = {}
                try:
                    bundle = await asyncio.to_thread(
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
                finally:
                    self._running_jobs.discard(job_id)

                if job_id in self._cancel_requested or cancel_event.is_set():
                    if bundle.get("staging_dir") and Path(bundle["staging_dir"]).exists():
                        shutil.rmtree(Path(bundle["staging_dir"]), ignore_errors=True)
                    await self._persist_cancellation(job_id, video_path)
                    return

                # 6. Transactional Publication State Machine:
                # RUNNING -> ARTIFACTS_VALIDATED -> PUBLISHING -> COMPLETE
                reserved = await self._reserve_publishing_state(job_id)
                if not reserved:
                    if bundle.get("staging_dir") and Path(bundle["staging_dir"]).exists():
                        shutil.rmtree(Path(bundle["staging_dir"]), ignore_errors=True)
                    raise ParkingJobCancelled(f"Publication reservation aborted for job {job_id}.")

                staging_dir = Path(bundle["staging_dir"])
                final_output_dir = Path(bundle["final_output_dir"])

                if final_output_dir.exists():
                    shutil.rmtree(staging_dir, ignore_errors=True)
                    raise FileExistsError(
                        f"Target directory {final_output_dir} exists; promotion aborted to prevent destruction."
                    )

                staging_dir.rename(final_output_dir)

                # Persist COMPLETE state from PUBLISHING
                try:
                    await self._persist_complete(
                        job_id=job_id,
                        input_video_sha=input_video_sha,
                        current_ref_sha=current_ref_sha,
                        current_layout_sha=current_layout_sha,
                        verified_layout_id=verified_layout_id,
                        latest_assessment_id=latest_assessment_id,
                        result_payload=bundle,
                    )
                except Exception as ex:
                    # Rollback / quarantine final directory so no failed/non-complete job exposes artifacts
                    shutil.rmtree(final_output_dir, ignore_errors=True)
                    raise ex

        except (ParkingJobCancelled, asyncio.CancelledError):
            logger.info(f"Job {job_id} cancelled.")
            await self._persist_cancellation(job_id, video_path)
        except Exception as e:
            logger.error(f"Error during occupancy execution for job {job_id}: {e}", exc_info=True)
            await self._persist_failure(job_id, "EXECUTION_ERROR", str(e), video_path)
        finally:
            self._running_jobs.discard(job_id)
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
            raise ParkingJobCancelled("Job cancelled before pipeline initialization.")

        progress_callback(10.0, "Initializing vehicle detector and geometric layout engine", "VALIDATING")

        # 1. Output paths & staging setup beneath configured parking jobs root
        media_root = Path(settings.MEDIA_ROOT) if hasattr(settings, "MEDIA_ROOT") else Path(tempfile.gettempdir()) / "roadsense_media"

        curr = media_root
        while curr != curr.parent:
            if curr.is_symlink():
                raise ValueError(f"Symlink found in media root path: {curr}")
            curr = curr.parent

        jobs_root = media_root / "parking_jobs"
        curr = jobs_root
        while curr != curr.parent:
            if curr.is_symlink():
                raise ValueError(f"Symlink found in jobs root path: {curr}")
            curr = curr.parent

        jobs_root.mkdir(parents=True, exist_ok=True)
        resolved_jobs_root = jobs_root.resolve()

        final_output_dir = jobs_root / job_id
        if ".." in final_output_dir.parts:
            raise ValueError(f"Path traversal ('..') prohibited in final output path: {final_output_dir}")

        resolved_final = final_output_dir.resolve()
        try:
            resolved_final.relative_to(resolved_jobs_root)
        except ValueError:
            raise ValueError(f"Final output directory {resolved_final} escapes jobs root {resolved_jobs_root}")

        if final_output_dir.exists():
            raise FileExistsError(
                f"Final output directory already exists for job {job_id}: {final_output_dir}. "
                "Publication must fail closed to prevent destructive overwrites."
            )

        staging_id = f"staging_{job_id}_{uuid.uuid4().hex}"
        staging_dir = jobs_root / staging_id
        if ".." in staging_dir.parts:
            raise ValueError(f"Path traversal in staging path: {staging_dir}")

        resolved_staging = staging_dir.resolve()
        try:
            resolved_staging.relative_to(resolved_jobs_root)
        except ValueError:
            raise ValueError(f"Staging directory {resolved_staging} escapes jobs root {resolved_jobs_root}")

        staging_dir.mkdir(parents=False, exist_ok=False)

        staging_mp4_path = staging_dir / "annotated.mp4"
        staging_timeline_path = staging_dir / "occupancy_timeline.jsonl"
        staging_manifest_path = staging_dir / "processing_manifest.json"
        staging_summary_path = staging_dir / "parking_summary.json"

        cap = None
        timeline_file = None
        encoder = None

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or media_info.get("width", 1920))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or media_info.get("height", 1080))
            duration = float(media_info.get("duration", total_frames / fps if fps > 0 else 0.0))

            if cancel_event.is_set():
                raise ParkingJobCancelled("Job cancelled before detector and tracker initialization.")

            # 2. Initialize local detector & session-local ByteTrack tracker with validated runtime FPS
            detector = self._detector_override or LocalVehicleDetector(self.config.detector)
            detector_sha = getattr(detector, "checkpoint_sha256", "0" * 64)

            tracker_config_path = Path(__file__).resolve().parents[4] / "configs" / "tracking" / "bytetrack_default.yaml"
            tracker = ParkingByteTracker(
                config_path=tracker_config_path if tracker_config_path.is_file() else None,
                fps=fps,
            )

            engine = ParkingOccupancyEngine(
                config=self.config,
                parking_spaces=parking_spaces_payload,
                video_width=width,
                video_height=height,
            )
            annotator = ParkingVideoAnnotator(self.config.rendering)

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

            progress_callback(15.0, "Streaming video frames through detector and tracker", "DETECTING")

            while True:
                if cancel_event.is_set():
                    raise ParkingJobCancelled(f"Job {job_id} cancelled by user request.")

                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break

                timestamp_sec = float(frame_idx / fps)

                # 1. Detection
                raw_detections = detector.detect_vehicles(frame_bgr)

                # 2. Tracking (session-local ByteTrack)
                tracker_result = tracker.update_tracks(
                    detections=raw_detections,
                    frame_idx=frame_idx,
                    frame_shape=(height, width),
                )

                # 3. Occupancy Engine
                frame_res = engine.process_frame(frame_idx, timestamp_sec, tracker_result)
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
                    vehicle_detections=tracker_result.detections,
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

            if frame_idx == 0 or last_frame_res is None:
                raise ValueError("No video frames could be decoded or processed.")

            # Finalize timeline file before hashing
            timeline_file.close()
            timeline_file = None

            # Finalize FFmpeg encoding
            progress_callback(82.0, "Finalizing H.264 video stream encoding", "ENCODING")
            encoder.finish()
            encoder = None

            # Compute output video SHA
            h_out = hashlib.sha256()
            with open(staging_mp4_path, "rb") as of:
                while chunk := of.read(65536):
                    h_out.update(chunk)
            output_video_sha = h_out.hexdigest()

            # Compute timeline SHA
            h_tl = hashlib.sha256()
            with open(staging_timeline_path, "rb") as tlf:
                while chunk := tlf.read(65536):
                    h_tl.update(chunk)
            timeline_sha = h_tl.hexdigest()

            # Final Bay summary
            bay_summary = {k: v.to_dict() for k, v in last_frame_res.bay_states.items()}
            summary_payload = {
                "job_id": job_id,
                "total_bays": last_frame_res.total_bays,
                "occupied_count": last_frame_res.occupied_count,
                "vacant_count": last_frame_res.vacant_count,
                "unknown_count": last_frame_res.unknown_count,
                "occluded_count": last_frame_res.occluded_count,
                "total_state_transitions": total_transitions,
                "bay_summary": bay_summary,
            }
            with open(staging_summary_path, "w", encoding="utf-8") as sf:
                json.dump(summary_payload, sf, indent=2)

            # Compute summary SHA
            h_sm = hashlib.sha256()
            with open(staging_summary_path, "rb") as smf:
                while chunk := smf.read(65536):
                    h_sm.update(chunk)
            summary_sha = h_sm.hexdigest()

            def _get_tool_version(cmd: List[str]) -> str:
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
                    line = res.stdout.splitlines()[0] if res.stdout else (res.stderr.splitlines()[0] if res.stderr else "unknown")
                    return line.strip()
                except Exception:
                    return "unknown"

            ffmpeg_ver = _get_tool_version(["ffmpeg", "-version"])
            ffprobe_ver = _get_tool_version(["ffprobe", "-version"])

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

            det_cfg = getattr(detector, "config", None)
            device_str = str(getattr(det_cfg, "device", None) or "cpu")

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
                    "ffmpeg": ffmpeg_ver,
                    "ffprobe": ffprobe_ver,
                    "compute_device": device_str,
                    "bytetrack_config_sha": tracker.config_sha256,
                    "bytetrack_frame_rate": tracker.fps,
                    "timeline_sha256": timeline_sha,
                    "summary_sha256": summary_sha,
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
                timeline_sha256=timeline_sha,
                summary_sha256=summary_sha,
                operational_gate="ALLOWED",
                gate_reasons=["GATE_ALLOWED: Fresh camera stability assessment matches verified layout."],
            )

            with open(staging_manifest_path, "w", encoding="utf-8") as mf:
                json.dump(manifest.to_dict(), mf, indent=2)

            # 6. Validate artifacts inside staging (recomputing hashes and verifying provenance)
            self._validate_staging_artifacts(
                staging_dir=staging_dir,
                expected_job_id=job_id,
                expected_camera_id=camera_id,
                expected_site_id=site_id,
                expected_layout_sha=verified_layout_canonical_sha or "",
                expected_assessment_id=latest_assessment_id,
                expected_stability_config_sha=latest_assessment_config_sha or "",
                expected_occupancy_config_sha=self.config.config_sha256,
                expected_width=width,
                expected_height=height,
                expected_fps=fps,
                expected_frames=frame_idx,
                expected_output_sha=output_video_sha,
                expected_timeline_sha=timeline_sha,
                expected_summary_sha=summary_sha,
            )

            final_mp4_path = final_output_dir / "annotated.mp4"
            final_timeline_path = final_output_dir / "occupancy_timeline.jsonl"

            return {
                "staging_dir": staging_dir,
                "final_output_dir": final_output_dir,
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

        except BaseException:
            if encoder is not None:
                try:
                    encoder.cleanup()
                except Exception:
                    pass
            if timeline_file is not None and not timeline_file.closed:
                try:
                    timeline_file.close()
                except Exception:
                    pass
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if staging_dir.exists():
                try:
                    shutil.rmtree(staging_dir, ignore_errors=True)
                except Exception:
                    pass
            raise
        finally:
            if timeline_file is not None and not timeline_file.closed:
                try:
                    timeline_file.close()
                except Exception:
                    pass
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def _validate_staging_artifacts(
        self,
        staging_dir: Path,
        expected_job_id: str,
        expected_camera_id: str,
        expected_site_id: str,
        expected_layout_sha: str,
        expected_assessment_id: Optional[str],
        expected_stability_config_sha: str,
        expected_occupancy_config_sha: str,
        expected_width: int,
        expected_height: int,
        expected_fps: float,
        expected_frames: int,
        expected_output_sha: str,
        expected_timeline_sha: str,
        expected_summary_sha: str,
    ) -> None:
        """Thoroughly validate all generated artifacts inside staging before promotion."""
        import re
        sha256_re = re.compile(r"^[0-9a-f]{64}$")

        mp4_path = staging_dir / "annotated.mp4"
        timeline_path = staging_dir / "occupancy_timeline.jsonl"
        manifest_path = staging_dir / "processing_manifest.json"
        summary_path = staging_dir / "parking_summary.json"

        for p in (mp4_path, timeline_path, manifest_path, summary_path):
            if not p.is_file() or p.is_symlink():
                raise ValueError(f"Staging artifact missing or is symlink: {p}")
            if p.stat().st_size == 0:
                raise ValueError(f"Staging artifact is empty (0 bytes): {p}")

        # Independently recompute SHA-256 for all artifacts from disk
        def _recompute_sha(p: Path) -> str:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()

        recomputed_mp4_sha = _recompute_sha(mp4_path)
        recomputed_timeline_sha = _recompute_sha(timeline_path)
        recomputed_summary_sha = _recompute_sha(summary_path)

        if recomputed_mp4_sha != expected_output_sha:
            raise ValueError(f"Recomputed MP4 SHA ({recomputed_mp4_sha}) does not match expected ({expected_output_sha})")
        if recomputed_timeline_sha != expected_timeline_sha:
            raise ValueError(f"Recomputed timeline SHA ({recomputed_timeline_sha}) does not match expected ({expected_timeline_sha})")
        if recomputed_summary_sha != expected_summary_sha:
            raise ValueError(f"Recomputed summary SHA ({recomputed_summary_sha}) does not match expected ({expected_summary_sha})")

        # 1. Validate MP4 with ffprobe
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "stream=codec_type,codec_name,pix_fmt,width,height,r_frame_rate,nb_frames,duration",
            "-show_entries", "format=duration,nb_streams",
            "-of", "json",
            str(mp4_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0)
        if res.returncode != 0:
            raise ValueError(f"FFprobe failed to inspect staging MP4: {res.stderr}")

        info = json.loads(res.stdout)
        streams = info.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        if len(video_streams) != 1:
            raise ValueError(f"Expected exactly 1 video stream, got {len(video_streams)}")

        v_stream = video_streams[0]
        if v_stream.get("codec_name") != "h264":
            raise ValueError(f"Expected H.264 video codec, got {v_stream.get('codec_name')}")
        if v_stream.get("pix_fmt") != "yuv420p":
            raise ValueError(f"Expected yuv420p pixel format, got {v_stream.get('pix_fmt')}")
        if int(v_stream.get("width", 0)) != expected_width or int(v_stream.get("height", 0)) != expected_height:
            raise ValueError(f"Geometry mismatch: expected {expected_width}x{expected_height}, got {v_stream.get('width')}x{v_stream.get('height')}")

        # FPS tolerance check
        r_fps_str = str(v_stream.get("r_frame_rate", "30/1"))
        try:
            num, den = map(float, r_fps_str.split("/"))
            actual_fps = num / den if den > 0 else 0.0
            if abs(actual_fps - expected_fps) > 1.0:
                raise ValueError(f"FPS deviation beyond tolerance: expected {expected_fps}, got {actual_fps}")
        except ValueError as e:
            if "FPS deviation" in str(e):
                raise

        # Frame count tolerance check
        nb_frames_str = v_stream.get("nb_frames")
        if nb_frames_str and nb_frames_str.isdigit():
            actual_nb = int(nb_frames_str)
            if abs(actual_nb - expected_frames) > 2:
                raise ValueError(f"Frame count deviation beyond tolerance: expected {expected_frames}, got {actual_nb}")

        # 2. Validate Timeline JSONL line by line
        timeline_line_count = 0
        with open(timeline_path, "r", encoding="utf-8") as tlf:
            for line_idx, line in enumerate(tlf, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception as e:
                    raise ValueError(f"Timeline JSONL line {line_idx} is malformed: {e}")

                req_fields = ["bay_id", "previous_state", "new_state", "trigger_reason", "frame_index", "timestamp_seconds"]
                for f in req_fields:
                    if f not in entry:
                        raise ValueError(f"Timeline line {line_idx} missing required field '{f}'")

                if entry["previous_state"] == entry["new_state"]:
                    raise ValueError(
                        f"Timeline line {line_idx} has identical previous and new states: "
                        f"'{entry['previous_state']}' -> '{entry['new_state']}'"
                    )
                timeline_line_count += 1

        # 3. Validate Summary JSON
        try:
            with open(summary_path, "r", encoding="utf-8") as sf:
                summary_data = json.load(sf)
        except Exception as e:
            raise ValueError(f"Parking summary JSON is malformed: {e}")

        total_bays = summary_data.get("total_bays", 0)
        occ = summary_data.get("occupied_count", 0)
        vac = summary_data.get("vacant_count", 0)
        unk = summary_data.get("unknown_count", 0)
        occl = summary_data.get("occluded_count", 0)
        if (occ + vac + unk + occl) != total_bays:
            raise ValueError(f"Summary counter mismatch: {occ}+{vac}+{unk}+{occl} != {total_bays}")
        if summary_data.get("total_state_transitions") != timeline_line_count:
            raise ValueError(f"Summary state transitions count ({summary_data.get('total_state_transitions')}) does not match timeline line count ({timeline_line_count})")

        # 4. Validate Manifest JSON and verify SHA-256 hashes against recomputed hashes
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest_data = json.load(mf)
        except Exception as e:
            raise ValueError(f"Processing manifest JSON is malformed: {e}")

        for h_key in (
            "input_video_sha256", "output_video_sha256",
            "detector_checkpoint_sha256", "occupancy_config_sha256",
            "stability_config_sha256", "timeline_sha256", "summary_sha256",
        ):
            h_val = str(manifest_data.get(h_key, "")).strip()
            if not sha256_re.match(h_val):
                raise ValueError(f"Manifest field '{h_key}' is not a valid 64-character lowercase hex SHA-256: '{h_val}'")

        if manifest_data["output_video_sha256"] != recomputed_mp4_sha:
            raise ValueError(f"Manifest output video SHA mismatch: {manifest_data['output_video_sha256']} != {recomputed_mp4_sha}")
        if manifest_data["timeline_sha256"] != recomputed_timeline_sha:
            raise ValueError(f"Manifest timeline SHA mismatch: {manifest_data['timeline_sha256']} != {recomputed_timeline_sha}")
        if manifest_data["summary_sha256"] != recomputed_summary_sha:
            raise ValueError(f"Manifest summary SHA mismatch: {manifest_data['summary_sha256']} != {recomputed_summary_sha}")

        # Provenance verification
        if manifest_data.get("job_id") != expected_job_id:
            raise ValueError(f"Manifest provenance mismatch: job_id {manifest_data.get('job_id')} != {expected_job_id}")
        if manifest_data.get("camera_id") != expected_camera_id:
            raise ValueError(f"Manifest provenance mismatch: camera_id {manifest_data.get('camera_id')} != {expected_camera_id}")
        if manifest_data.get("site_id") != expected_site_id:
            raise ValueError(f"Manifest provenance mismatch: site_id {manifest_data.get('site_id')} != {expected_site_id}")
        if manifest_data.get("layout_canonical_sha256") != expected_layout_sha:
            raise ValueError(f"Manifest provenance mismatch: layout SHA {manifest_data.get('layout_canonical_sha256')} != {expected_layout_sha}")
        if manifest_data.get("stability_assessment_id") != expected_assessment_id:
            raise ValueError(f"Manifest provenance mismatch: assessment ID {manifest_data.get('stability_assessment_id')} != {expected_assessment_id}")
        if manifest_data.get("stability_config_sha256") != expected_stability_config_sha:
            raise ValueError(f"Manifest provenance mismatch: stability config SHA {manifest_data.get('stability_config_sha256')} != {expected_stability_config_sha}")
        if manifest_data.get("occupancy_config_sha256") != expected_occupancy_config_sha:
            raise ValueError(f"Manifest provenance mismatch: occupancy config SHA {manifest_data.get('occupancy_config_sha256')} != {expected_occupancy_config_sha}")

    async def _reserve_publishing_state(self, job_id: str) -> bool:
        """Atomically reserve the PUBLISHING state in the database using a guarded CAS update."""
        try:
            async with async_session_factory() as db:
                stmt = (
                    update(ParkingOccupancyJob)
                    .where(ParkingOccupancyJob.id == job_id)
                    .where(ParkingOccupancyJob.status.not_in(TERMINAL_STATES))
                    .where(ParkingOccupancyJob.status != "PUBLISHING")
                    .values(
                        status="PUBLISHING",
                        progress_pct=95.0,
                        stage_message="Promoting validated parking artifacts transactionally",
                    )
                )
                res = await db.execute(stmt)
                await db.commit()
                return res.rowcount > 0
        except Exception as e:
            logger.error(f"Error reserving PUBLISHING state for job {job_id}: {e}")
            return False

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
                    .where(ParkingOccupancyJob.status == "PUBLISHING")
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
                    raise RuntimeError(f"Job {job_id} was not in PUBLISHING state during final COMPLETE transition.")
        except Exception as e:
            logger.error(f"Error persisting COMPLETE state for job {job_id}: {e}")
            raise

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
                    .where(ParkingOccupancyJob.status.not_in(TERMINAL_STATES + ("PUBLISHING", "COMPLETE")))
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
                    logger.warning(f"Job {job_id} already reached a terminal/publishing state; skipping CANCELLED transition.")
        except Exception as e:
            logger.error(f"Error persisting cancellation for job {job_id}: {e}")

    async def _persist_failure(self, job_id: str, code: str, message: str, video_path: Path) -> None:
        try:
            async with async_session_factory() as db:
                stmt = (
                    update(ParkingOccupancyJob)
                    .where(ParkingOccupancyJob.id == job_id)
                    .where(ParkingOccupancyJob.status.not_in(TERMINAL_STATES + ("COMPLETE",)))
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
