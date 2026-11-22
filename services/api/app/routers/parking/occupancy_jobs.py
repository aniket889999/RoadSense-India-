"""Parking API: occupancy jobs, artifacts, cancellation and retention deletion.

Moved verbatim from the original ``routers/parking.py`` (no behaviour change).
"""

from __future__ import annotations

from services.api.app.routers.parking._common import *  # noqa: F401,F403  (shared imports, models, schemas, router)


def _resolve_safe_parking_artifact(
    job_id: str,
    artifact_filename: str,
    expected_sha256: Optional[str] = None,
    job_status: Optional[str] = "COMPLETE",
    storage_root: Optional[Path] = None,
) -> Optional[Path]:
    """
    Resolves and verifies an artifact path with fail-closed security guarantees:
    - Enforces allowed filename whitelist
    - Requires job status to be COMPLETE
    - Validates lexical confinement under media_root/parking_jobs/{job_id}
    - Rejects any symlinks in path components
    - Verifies byte-level SHA-256 hash match against database provenance
    """
    import re
    if not job_id or not re.match(r"^[a-f0-9\-]{36}$", job_id):
        return None

    ALLOWED_ARTIFACTS = {
        "annotated.mp4",
        "occupancy_timeline.jsonl",
        "processing_manifest.json",
        "parking_summary.json",
    }
    if artifact_filename not in ALLOWED_ARTIFACTS:
        return None

    # Strict authorization: require job status to be exactly COMPLETE
    if job_status != "COMPLETE":
        return None

    if not expected_sha256 or len(expected_sha256) != 64:
        return None

    raw_root = storage_root if storage_root is not None else get_storage_root()
    if raw_root.is_symlink() or os.path.islink(str(raw_root)):
        return None

    media_root = raw_root.resolve()
    jobs_root = (media_root / "parking_jobs").resolve()
    if jobs_root.is_symlink() or os.path.islink(str(jobs_root)):
        return None

    candidate = jobs_root / job_id / artifact_filename
    if ".." in candidate.parts:
        return None

    curr = candidate
    while curr != jobs_root and curr != curr.parent:
        if curr.is_symlink() or os.path.islink(str(curr)):
            return None
        curr = curr.parent

    resolved_jobs_root = jobs_root
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_jobs_root)
    except ValueError:
        return None

    if not resolved.exists() or not resolved.is_file() or resolved.is_symlink():
        return None

    # Recompute file SHA on disk and verify match
    h = hashlib.sha256()
    with open(resolved, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    if h.hexdigest().lower() != expected_sha256.lower():
        logger.warning(f"Artifact SHA-256 verification failed for {resolved}: expected {expected_sha256}, got {h.hexdigest()}")
        return None

    return resolved


def create_staged_upload_file(storage_root: Optional[Path] = None) -> Tuple[Path, int]:
    """
    Creates an exclusively locked staged upload file under configured upload_staging root.
    Rejects symlinks in raw path chain, enforces confinement, and uses a neutral staging suffix.
    """
    raw_root = storage_root if storage_root is not None else get_storage_root()
    if raw_root.is_symlink() or os.path.islink(str(raw_root)):
        raise ValueError(f"Symlink found in media root path component: {raw_root}")

    media_root = raw_root.resolve()
    upload_staging_dir = media_root / "upload_staging"
    if upload_staging_dir.is_symlink() or os.path.islink(str(upload_staging_dir)):
        raise ValueError(f"Symlink found in upload staging path component: {upload_staging_dir}")

    upload_staging_dir.mkdir(parents=True, exist_ok=True)
    resolved_staging = upload_staging_dir.resolve()

    staging_file_name = f"upload_{uuid.uuid4().hex}.upload.tmp"
    staging_file_path = upload_staging_dir / staging_file_name

    if ".." in staging_file_path.parts:
        raise ValueError("Path traversal in staging file path")

    resolved_path = staging_file_path.resolve()
    try:
        resolved_path.relative_to(resolved_staging)
    except ValueError:
        raise ValueError("Staging file escapes upload staging root")

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    fd = os.open(str(staging_file_path), flags, 0o600)
    return staging_file_path, fd


def _build_occupancy_job_response(job: ParkingOccupancyJob, storage_root: Optional[Path] = None) -> ParkingOccupancyJobResponse:
    timeline_sha = (job.manifest_json or {}).get("timeline_sha256") or ((job.manifest_json or {}).get("software_versions") or {}).get("timeline_sha256")
    safe_video = _resolve_safe_parking_artifact(job.id, "annotated.mp4", job.output_video_sha256, job.status, storage_root=storage_root)
    safe_timeline = _resolve_safe_parking_artifact(job.id, "occupancy_timeline.jsonl", timeline_sha, job.status, storage_root=storage_root) if timeline_sha else None
    has_video = safe_video is not None
    has_timeline = safe_timeline is not None
    has_manifest = bool(job.manifest_json is not None and job.status == "COMPLETE")
    return ParkingOccupancyJobResponse(
        id=job.id,
        camera_id=job.camera_id,
        site_id=job.site_id,
        layout_revision_id=job.layout_revision_id,
        stability_assessment_id=job.stability_assessment_id,
        status=job.status,
        progress_pct=job.progress_pct,
        stage_message=job.stage_message,
        failure_code=job.failure_code,
        failure_message=job.failure_message,
        gate_decision=job.gate_decision,
        gate_reasons=job.gate_reasons,
        input_video_sha256=job.input_video_sha256,
        output_video_sha256=job.output_video_sha256,
        reference_image_sha256=job.reference_image_sha256,
        layout_canonical_sha256=job.layout_canonical_sha256,
        detector_checkpoint_sha256=job.detector_checkpoint_sha256,
        occupancy_config_sha256=job.occupancy_config_sha256,
        total_frames=job.total_frames,
        processed_frames=job.processed_frames,
        fps=job.fps,
        duration_seconds=job.duration_seconds,
        video_width=job.video_width,
        video_height=job.video_height,
        total_bays=job.total_bays,
        final_occupied_count=job.final_occupied_count,
        final_vacant_count=job.final_vacant_count,
        final_unknown_count=job.final_unknown_count,
        final_occluded_count=job.final_occluded_count,
        total_state_transitions=job.total_state_transitions,
        has_annotated_video=has_video,
        has_timeline=has_timeline,
        has_manifest=has_manifest,
        bay_summary=job.bay_summary_json,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.post("/cameras/{camera_id}/occupancy/jobs", response_model=ParkingOccupancyJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_parking_occupancy_job(
    camera_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    manager: ParkingOccupancyJobManager = Depends(get_parking_job_manager),
    storage_root: Path = Depends(get_storage_root),
):
    """
    Submit a video file for gate-controlled parking occupancy evaluation and video annotation.
    Enforces fail-closed camera stability gate: returns HTTP 202; background job will set
    BLOCKED_BY_STABILITY_GATE if camera is not calibrated with an active verified layout and fresh STABLE assessment.
    """
    cam_res = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = cam_res.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    cfg = manager.config
    max_upload_bytes = cfg.execution.max_upload_bytes

    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded.")

    try:
        temp_video_path, fd = create_staged_upload_file(storage_root=storage_root)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create staged upload file: {e}")

    bytes_read = 0
    try:
        with os.fdopen(fd, "wb") as f_out:
            while chunk := await file.read(65536):
                bytes_read += len(chunk)
                if bytes_read > max_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Uploaded file exceeds maximum allowed size of {max_upload_bytes} bytes ({max_upload_bytes / (1024*1024):.1f}MB).",
                    )
                f_out.write(chunk)
            f_out.flush()
            os.fsync(f_out.fileno())
    except HTTPException:
        temp_video_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        temp_video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to stream upload: {e}")

    if bytes_read == 0:
        temp_video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty (0 bytes).")

    job_id = str(uuid.uuid4())
    job = ParkingOccupancyJob(
        id=job_id,
        camera_id=cam.id,
        site_id=cam.site_id,
        status="QUEUED",
        progress_pct=0.0,
        stage_message="Job queued for gate validation and evaluation",
        created_at=utc_now(),
    )
    try:
        db.add(job)
        await db.commit()
        await db.refresh(job)
    except Exception as e:
        temp_video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to persist job record: {e}")

    # Launch async job manager execution
    manager.submit_occupancy_job(
        job_id=job_id,
        camera_id=cam.id,
        site_id=cam.site_id,
        video_path=temp_video_path,
    )

    return _build_occupancy_job_response(job, storage_root=storage_root)


@router.get("/parking/jobs/{job_id}", response_model=ParkingOccupancyJobResponse)
async def get_parking_occupancy_job_detail(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    storage_root: Path = Depends(get_storage_root),
):
    """Retrieve status, metrics, and manifest summary of a parking occupancy job."""
    res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
    job = res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Parking occupancy job not found")
    return _build_occupancy_job_response(job, storage_root=storage_root)


@router.get("/cameras/{camera_id}/occupancy/jobs", response_model=List[ParkingOccupancyJobResponse])
async def list_camera_occupancy_jobs(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
    storage_root: Path = Depends(get_storage_root),
):
    """List historical parking occupancy jobs for a camera."""
    cam_res = await db.execute(select(Camera).where(Camera.id == camera_id))
    if not cam_res.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Camera not found")

    res = await db.execute(
        select(ParkingOccupancyJob)
        .where(ParkingOccupancyJob.camera_id == camera_id)
        .order_by(desc(ParkingOccupancyJob.created_at))
    )
    jobs = res.scalars().all()
    return [_build_occupancy_job_response(j, storage_root=storage_root) for j in jobs]


@router.post("/parking/jobs/{job_id}/cancel", response_model=ParkingOccupancyJobCancelResponse)
async def cancel_parking_occupancy_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    manager: ParkingOccupancyJobManager = Depends(get_parking_job_manager),
):
    """Cancel an active or queued parking occupancy job atomically through manager CAS."""
    res = await db.execute(select(ParkingOccupancyJob.id).where(ParkingOccupancyJob.id == job_id))
    if not res.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Parking occupancy job not found")

    result = await manager.request_job_cancellation(job_id)
    return ParkingOccupancyJobCancelResponse(
        job_id=result["job_id"],
        status=result["status"],
        cancelled=result["cancelled"],
        message=result["message"],
    )


@router.get("/parking/jobs/{job_id}/video")
async def get_parking_job_annotated_video(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    storage_root: Path = Depends(get_storage_root),
):
    """Stream the generated annotated MP4 video with strict authorization."""
    res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
    job = res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Parking occupancy job not found")

    if job.status != "COMPLETE":
        raise HTTPException(status_code=404, detail=f"Annotated video is only available for COMPLETE jobs (current status: '{job.status}').")

    if not job.output_video_sha256 or len(job.output_video_sha256) != 64:
        raise HTTPException(status_code=404, detail="Annotated video metadata SHA is missing or invalid.")

    # Validate provenance in manifest if present
    if job.manifest_json and job.manifest_json.get("output_video_sha256") != job.output_video_sha256:
        raise HTTPException(status_code=404, detail="Manifest provenance mismatch for annotated video.")

    safe_path = _resolve_safe_parking_artifact(job.id, "annotated.mp4", job.output_video_sha256, job.status, storage_root=storage_root)
    if not safe_path:
        raise HTTPException(status_code=404, detail="Annotated video file not available or failed integrity verification.")

    return FileResponse(
        path=safe_path,
        media_type="video/mp4",
        filename=f"annotated_parking_{job_id[:8]}.mp4",
    )


@router.get("/parking/jobs/{job_id}/manifest")
async def get_parking_job_manifest(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    storage_root: Path = Depends(get_storage_root),
):
    """Retrieve the full processing manifest JSON for a COMPLETE parking occupancy job."""
    res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
    job = res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Parking occupancy job not found")

    if job.status != "COMPLETE":
        raise HTTPException(status_code=404, detail=f"Manifest is only available for COMPLETE jobs (current status: '{job.status}').")

    if not job.manifest_json:
        raise HTTPException(status_code=404, detail="Processing manifest not available for this job.")

    manifest_sha = (job.manifest_json or {}).get("manifest_sha256")
    safe_path = _resolve_safe_parking_artifact(job.id, "processing_manifest.json", manifest_sha, job.status, storage_root=storage_root) if manifest_sha else None
    if safe_path and safe_path.is_file():
        return FileResponse(
            path=safe_path,
            media_type="application/json",
            filename=f"processing_manifest_{job_id[:8]}.json",
        )
    return job.manifest_json


@router.get("/parking/jobs/{job_id}/timeline")
async def get_parking_job_timeline(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    storage_root: Path = Depends(get_storage_root),
):
    """Download the state-transition timeline JSONL ledger with strict authorization."""
    res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
    job = res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Parking occupancy job not found")

    if job.status != "COMPLETE":
        raise HTTPException(status_code=404, detail=f"Timeline is only available for COMPLETE jobs (current status: '{job.status}').")

    manifest = job.manifest_json or {}
    timeline_sha = manifest.get("timeline_sha256") or (manifest.get("software_versions") or {}).get("timeline_sha256")
    if not timeline_sha or len(timeline_sha) != 64:
        raise HTTPException(status_code=404, detail="Timeline SHA not found in manifest or is invalid.")

    safe_path = _resolve_safe_parking_artifact(job.id, "occupancy_timeline.jsonl", timeline_sha, job.status, storage_root=storage_root)
    if not safe_path:
        raise HTTPException(status_code=404, detail="Timeline ledger file not available or failed integrity verification.")

    return FileResponse(
        path=safe_path,
        media_type="application/x-ndjson",
        filename=f"occupancy_timeline_{job_id[:8]}.jsonl",
    )


@router.get("/parking/jobs/{job_id}/summary")
async def get_parking_job_summary(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    storage_root: Path = Depends(get_storage_root),
):
    """Download the final occupancy summary JSON for a COMPLETE parking occupancy job."""
    res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
    job = res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Parking occupancy job not found")

    if job.status != "COMPLETE":
        raise HTTPException(status_code=404, detail=f"Summary is only available for COMPLETE jobs (current status: '{job.status}').")

    manifest = job.manifest_json or {}
    summary_sha = manifest.get("summary_sha256") or (manifest.get("software_versions") or {}).get("summary_sha256")
    if not summary_sha or len(summary_sha) != 64:
        raise HTTPException(status_code=404, detail="Summary SHA not found in manifest or is invalid.")

    safe_path = _resolve_safe_parking_artifact(job.id, "parking_summary.json", summary_sha, job.status, storage_root=storage_root)
    if not safe_path:
        raise HTTPException(status_code=404, detail="Summary JSON file not available or failed integrity verification.")

    return FileResponse(
        path=safe_path,
        media_type="application/json",
        filename=f"parking_summary_{job_id[:8]}.json",
    )


@router.delete("/parking/jobs/{job_id}", response_model=ParkingOccupancyJobDeleteResponse)
async def delete_parking_occupancy_job(
    job_id: str,
    req: ParkingOccupancyJobDeleteRequest,
    db: AsyncSession = Depends(get_db),
    storage_root: Path = Depends(get_storage_root),
):
    """
    Safely delete a parking occupancy job and purge its local artifacts with fail-closed retention controls.

    Requirements:
    - confirmation_acknowledged must be true
    - non-blank local_operator_label assertion
    - meaningful deletion_reason (>= 5 chars)
    - Terminal states only: COMPLETE, FAILED, CANCELLED, BLOCKED_BY_STABILITY_GATE
    - Guarded DB CAS to PURGING
    - Confined filesystem validation (no traversal or symlinks)
    - Atomic quarantine rename before purge; restored on DB commit failure
    - Minimal DELETED tombstone & append-only ParkingJobAuditEvent recorded in DB
    """
    request_time = datetime.now(timezone.utc)

    if not req.confirmation_acknowledged:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Explicit confirmation (confirmation_acknowledged=true) is required for deletion.",
        )

    op_label = (req.local_operator_label or "").strip()
    if not op_label:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Non-blank local_operator_label is required.",
        )

    reason = (req.deletion_reason or "").strip()
    if len(reason) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A meaningful deletion_reason (at least 5 characters) is required.",
        )

    # 1. Fetch current job for state check
    res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.id == job_id))
    job = res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Parking occupancy job not found")

    terminal_statuses = {"COMPLETE", "FAILED", "CANCELLED", "BLOCKED_BY_STABILITY_GATE"}

    if job.status == "DELETED":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job {job_id} has already been deleted and purged.",
        )

    if job.status not in terminal_statuses:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot delete job in non-terminal state '{job.status}'. Deletion is strictly permitted only from terminal states ({', '.join(sorted(terminal_statuses))}).",
        )

    if req.expected_status and job.status != req.expected_status:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Concurrency mismatch: expected job status '{req.expected_status}', but current status is '{job.status}'.",
        )

    prior_status = job.status

    # 2. Guarded DB CAS: transition status from terminal state to PURGING
    from sqlalchemy import update
    conditions = [
        ParkingOccupancyJob.id == job_id,
        ParkingOccupancyJob.status.in_(list(terminal_statuses)),
    ]
    if req.expected_status:
        conditions.append(ParkingOccupancyJob.status == req.expected_status)

    cas_stmt = (
        update(ParkingOccupancyJob)
        .where(*conditions)
        .values(status="PURGING")
    )
    cas_res = await db.execute(cas_stmt)
    if cas_res.rowcount == 0:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Failed to reserve PURGING state: job status changed concurrently.",
        )
    await db.flush()

    # Capture artifact hashes before disk purge
    artifact_hashes = {
        "input_video_sha256": job.input_video_sha256,
        "output_video_sha256": job.output_video_sha256,
        "reference_image_sha256": job.reference_image_sha256,
        "layout_canonical_sha256": job.layout_canonical_sha256,
        "manifest_sha256": (job.manifest_json or {}).get("manifest_sha256"),
        "summary_sha256": (job.manifest_json or {}).get("summary_sha256"),
        "timeline_sha256": (job.manifest_json or {}).get("timeline_sha256"),
    }

    # 3. Path confinement & symlink validation before any filesystem operations
    media_root = storage_root.resolve()
    if media_root.is_symlink():
        await db.rollback()
        raise HTTPException(status_code=500, detail="Storage root directory cannot be a symlink.")

    jobs_root = (media_root / "parking_jobs").resolve()
    if jobs_root.is_symlink():
        await db.rollback()
        raise HTTPException(status_code=500, detail="Parking jobs root directory cannot be a symlink.")

    job_dir = jobs_root / job_id
    if ".." in job_dir.parts:
        await db.rollback()
        raise HTTPException(status_code=400, detail="Invalid job path.")

    # Check symlinks down to jobs_root
    curr = job_dir
    while curr != jobs_root and curr != curr.parent:
        if curr.is_symlink():
            await db.rollback()
            raise HTTPException(status_code=400, detail="Symlinked path component detected in job directory hierarchy.")
        curr = curr.parent

    # 4. Atomic quarantine rename & strict rmtree
    quarantine_dir = jobs_root / f".quarantine_{job_id}_{uuid.uuid4().hex[:8]}"
    quarantined = False
    purge_result = "NOOP"

    try:
        if job_dir.exists():
            if job_dir.is_symlink():
                raise RuntimeError("Job directory is a symlink.")
            job_dir.rename(quarantine_dir)
            quarantined = True

            import shutil
            shutil.rmtree(quarantine_dir)
            quarantined = False
            purge_result = "SUCCESS"

        # 5. DB transition to DELETED tombstone & record audit event
        job.status = "DELETED"
        job.output_video_path = None
        job.timeline_jsonl_path = None
        job.stage_message = f"Purged by {op_label}: {reason}"

        audit_event = ParkingJobAuditEvent(
            id=str(uuid.uuid4()),
            job_id=job.id,
            event_type="JOB_DELETED_AND_PURGED",
            operator_identity_assertion=op_label,
            deletion_reason=reason,
            prior_status=prior_status,
            resulting_status="DELETED",
            artifact_hashes=artifact_hashes,
            purge_result=purge_result,
            request_time=request_time,
            completed_time=datetime.now(timezone.utc),
        )
        db.add(audit_event)
        await db.commit()

    except Exception as e:
        await db.rollback()
        # Restore quarantined directory if DB commit or rmtree failed
        if quarantined and quarantine_dir.exists() and not job_dir.exists():
            try:
                quarantine_dir.rename(job_dir)
            except Exception as restore_err:
                logger.error(f"Failed to restore quarantined directory {quarantine_dir} to {job_dir}: {restore_err}")
        logger.exception(f"Failed during deletion of job {job_id}")
        raise HTTPException(status_code=500, detail=f"Job deletion failed: {str(e)}")

    completed_time = datetime.now(timezone.utc)
    return ParkingOccupancyJobDeleteResponse(
        deleted=True,
        job_id=job_id,
        prior_status=prior_status,
        resulting_status="DELETED",
        operator_identity_assertion=op_label,
        deletion_reason=reason,
        purged_at=completed_time,
        message=f"Job {job_id} artifacts successfully purged and job record marked DELETED.",
    )


# ============================================================================
# Phase 3A: Safe Usable Capacity & Hazard Association Endpoints
# ============================================================================

