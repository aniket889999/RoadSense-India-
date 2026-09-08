"""Asynchronous Background Job Manager for Camera Stability Assessments."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set
import uuid

from sqlalchemy import select
from services.api.app.core.logging import logger
from services.api.app.db.session import async_session_factory
from services.api.app.models.entities import CameraStabilityAssessment
from src.parking.contracts import OperationalGate, StabilityDecision
from src.parking.stability_config import StabilityConfig, load_stability_config
from src.parking.stability_engine import evaluate_video_camera_stability


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StabilityAssessmentJobManager:
    """Manages bounded concurrency, background execution, progress tracking, and cancellation for stability assessments."""

    def __init__(self, max_concurrency: int = 2, db_session_factory: Optional[Any] = None) -> None:
        self.config: StabilityConfig = load_stability_config()
        self._max_concurrency: int = max(1, self.config.execution.max_concurrent_jobs or max_concurrency)
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(self._max_concurrency)
        self._active_tasks: Dict[str, asyncio.Task[None]] = {}
        self._cancel_requested: Set[str] = set()
        self._db_session_factory = db_session_factory

    def _get_db_session(self):
        factory = self._db_session_factory or async_session_factory
        return factory()

    def submit_assessment_job(
        self,
        assessment_id: str,
        video_path: Path,
        camera_id: str,
        reference_image_bytes: bytes,
        expected_reference_sha256: str,
        active_layout_canonical_sha256: Optional[str],
    ) -> asyncio.Task[None]:
        """Submit a new stability assessment job for asynchronous execution."""
        task = asyncio.create_task(
            self._execute_job(
                assessment_id=assessment_id,
                video_path=video_path,
                camera_id=camera_id,
                reference_image_bytes=reference_image_bytes,
                expected_reference_sha256=expected_reference_sha256,
                active_layout_canonical_sha256=active_layout_canonical_sha256,
            ),
            name=f"stability-job-{assessment_id}",
        )
        self._active_tasks[assessment_id] = task
        return task

    async def cancel_assessment_job(self, assessment_id: str) -> bool:
        """Request cancellation of an in-progress or queued stability assessment job."""
        self._cancel_requested.add(assessment_id)
        task = self._active_tasks.get(assessment_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    async def _execute_job(
        self,
        assessment_id: str,
        video_path: Path,
        camera_id: str,
        reference_image_bytes: bytes,
        expected_reference_sha256: str,
        active_layout_canonical_sha256: Optional[str],
    ) -> None:
        async with self._semaphore:
            # 1. Update DB to VALIDATING / STARTED
            try:
                async with self._get_db_session() as db:
                    res = await db.execute(select(CameraStabilityAssessment).where(CameraStabilityAssessment.id == assessment_id))
                    assessment = res.scalar_one_or_none()
                    if assessment:
                        assessment.status = "VALIDATING"
                        assessment.started_at = _utc_now()
                        assessment.progress_pct = 10.0
                        assessment.stage_message = "Validating media container & metadata"
                        await db.commit()
            except Exception as e:
                logger.error(f"Failed to update initial status for assessment {assessment_id}: {e}")

            if assessment_id in self._cancel_requested:
                await self._persist_cancellation(assessment_id, video_path)
                return

            # 2. Progress callback adapter
            last_progress_pct = 10.0

            def sync_progress_callback(pct: float, msg: str) -> None:
                nonlocal last_progress_pct
                last_progress_pct = pct
                # Fire and forget async DB progress update if loop is running
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self._update_progress_db(assessment_id, pct, msg),
                            loop,
                        )
                except Exception:
                    pass

            # 3. Execute CPU-bound evaluation in worker thread
            try:
                result = await asyncio.to_thread(
                    evaluate_video_camera_stability,
                    video_path=video_path,
                    reference_image_bytes=reference_image_bytes,
                    expected_reference_sha256=expected_reference_sha256,
                    active_layout_canonical_sha256=active_layout_canonical_sha256,
                    config=self.config,
                    progress_callback=sync_progress_callback,
                )

                if assessment_id in self._cancel_requested:
                    await self._persist_cancellation(assessment_id, video_path)
                    return

                # 4. Persist COMPLETE result
                async with self._get_db_session() as db:
                    res = await db.execute(select(CameraStabilityAssessment).where(CameraStabilityAssessment.id == assessment_id))
                    assessment = res.scalar_one_or_none()
                    if assessment:
                        assessment.status = "COMPLETE"
                        assessment.progress_pct = 100.0
                        assessment.stage_message = "Camera stability assessment complete"
                        assessment.video_sha256 = result.video_sha256
                        assessment.algorithm_version = result.algorithm_version
                        assessment.opencv_version = result.opencv_version
                        assessment.config_version = result.config_version
                        assessment.config_sha256 = result.config_sha256
                        assessment.thresholds_snapshot = result.thresholds_snapshot
                        assessment.sample_measurements = [s.to_dict() for s in result.samples]
                        assessment.aggregate_decision = result.aggregate_decision.value
                        assessment.operational_gate = result.operational_gate.value
                        assessment.gate_reasons = result.gate_reasons
                        assessment.summary_metrics = result.summary_metrics
                        assessment.completed_at = _utc_now()
                        await db.commit()

            except asyncio.CancelledError:
                await self._persist_cancellation(assessment_id, video_path)
            except Exception as e:
                logger.error(f"Error during stability evaluation for assessment {assessment_id}: {e}", exc_info=True)
                sanitized_msg = str(e)
                async with self._get_db_session() as db:
                    res = await db.execute(select(CameraStabilityAssessment).where(CameraStabilityAssessment.id == assessment_id))
                    assessment = res.scalar_one_or_none()
                    if assessment:
                        assessment.status = "FAILED"
                        assessment.progress_pct = 100.0
                        assessment.stage_message = "Assessment failed during processing"
                        assessment.failure_code = "PROCESSING_ERROR"
                        assessment.failure_message = sanitized_msg
                        assessment.aggregate_decision = StabilityDecision.ERROR.value
                        assessment.operational_gate = OperationalGate.BLOCKED.value
                        assessment.gate_reasons = [f"ASSESSMENT_ERROR: {sanitized_msg}"]
                        assessment.completed_at = _utc_now()
                        await db.commit()
            finally:
                # Clean up temp file safely
                try:
                    if video_path.exists():
                        video_path.unlink(missing_ok=True)
                except Exception:
                    pass
                self._active_tasks.pop(assessment_id, None)
                self._cancel_requested.discard(assessment_id)

    async def _update_progress_db(self, assessment_id: str, pct: float, msg: str) -> None:
        try:
            async with self._get_db_session() as db:
                res = await db.execute(select(CameraStabilityAssessment).where(CameraStabilityAssessment.id == assessment_id))
                assessment = res.scalar_one_or_none()
                if assessment and assessment.status not in ("COMPLETE", "FAILED", "CANCELLED"):
                    assessment.status = "ANALYZING"
                    assessment.progress_pct = round(pct, 1)
                    assessment.stage_message = msg
                    await db.commit()
        except Exception:
            pass

    async def _persist_cancellation(self, assessment_id: str, video_path: Path) -> None:
        try:
            async with self._get_db_session() as db:
                res = await db.execute(select(CameraStabilityAssessment).where(CameraStabilityAssessment.id == assessment_id))
                assessment = res.scalar_one_or_none()
                if assessment:
                    assessment.status = "CANCELLED"
                    assessment.progress_pct = 100.0
                    assessment.stage_message = "Assessment cancelled by operator"
                    assessment.failure_code = "CANCELLED"
                    assessment.failure_message = "Assessment cancelled prior to completion."
                    assessment.aggregate_decision = StabilityDecision.ERROR.value
                    assessment.operational_gate = OperationalGate.BLOCKED.value
                    assessment.gate_reasons = ["CANCELLED: Operator cancelled stability assessment."]
                    assessment.completed_at = _utc_now()
                    await db.commit()
        except Exception as e:
            logger.error(f"Failed to persist cancellation for {assessment_id}: {e}")
        finally:
            try:
                if video_path.exists():
                    video_path.unlink(missing_ok=True)
            except Exception:
                pass


# Global singleton instance
stability_job_manager = StabilityAssessmentJobManager()
