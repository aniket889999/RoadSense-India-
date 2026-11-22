"""Parking API: camera stability assessments, operational gate and audit events.

Moved verbatim from the original ``routers/parking.py`` (no behaviour change).
"""

from __future__ import annotations

from services.api.app.routers.parking._common import *  # noqa: F401,F403  (shared imports, models, schemas, router)


def _build_stability_response(a: CameraStabilityAssessment) -> StabilityAssessmentResponse:
    samples_out: List[SampleMeasurementSchema] = []
    if a.sample_measurements:
        for s in a.sample_measurements:
            samples_out.append(
                SampleMeasurementSchema(
                    sample_index=int(s["sample_index"]),
                    timestamp_seconds=float(s["timestamp_seconds"]),
                    frame_index=int(s["frame_index"]),
                    matched_features=int(s["matched_features"]),
                    inlier_count=int(s["inlier_count"]),
                    inlier_ratio=float(s["inlier_ratio"]),
                    translation_px_x=float(s["translation_px_x"]),
                    translation_px_y=float(s["translation_px_y"]),
                    translation_magnitude_px=float(s["translation_magnitude_px"]),
                    translation_normalized=float(s["translation_normalized"]),
                    scale_factor=float(s["scale_factor"]),
                    scale_change=float(s["scale_change"]),
                    rotation_degrees=float(s["rotation_degrees"]),
                    perspective_distortion=float(s["perspective_distortion"]),
                    reprojection_error=float(s["reprojection_error"]),
                    decision=s["decision"],
                    rejection_reasons=s.get("rejection_reasons", []),
                )
            )

    return StabilityAssessmentResponse(
        id=a.id,
        camera_id=a.camera_id,
        layout_revision_id=a.layout_revision_id,
        layout_canonical_sha256=a.layout_canonical_sha256,
        reference_image_sha256=a.reference_image_sha256,
        video_sha256=a.video_sha256,
        status=a.status or "COMPLETE",
        progress_pct=a.progress_pct if a.progress_pct is not None else 100.0,
        stage_message=a.stage_message,
        failure_code=a.failure_code,
        failure_message=a.failure_message,
        config_version=a.config_version,
        config_sha256=a.config_sha256,
        algorithm_version=a.algorithm_version,
        opencv_version=a.opencv_version,
        thresholds_snapshot=a.thresholds_snapshot,
        sample_measurements=samples_out,
        aggregate_decision=a.aggregate_decision,
        operational_gate=a.operational_gate,
        gate_reasons=a.gate_reasons or [],
        summary_metrics=a.summary_metrics,
        created_at=a.created_at,
        started_at=a.started_at,
        completed_at=a.completed_at,
        operator_acknowledged_at=a.operator_acknowledged_at,
        operator_label=a.operator_label,
        operator_note=a.operator_note,
    )


# ============================================================================
# Camera Stability Endpoints
# ============================================================================

@router.post("/cameras/{camera_id}/stability/assess", response_model=StabilityAssessmentResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_camera_stability_assessment(
    camera_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    stab_manager: StabilityAssessmentJobManager = Depends(get_stability_job_manager),
    storage_root: Path = Depends(get_storage_root),
):
    """
    Execute an asynchronous bounded camera stability assessment against the camera's reference image.
    Accepts multipart video upload only, returns HTTP 202 with job progress tracking.
    """
    cam_res = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = cam_res.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not cam.reference_image_path or not cam.reference_image_sha256:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera has no reference image uploaded. Stability assessment requires a verified reference frame.",
        )
    ref_file_path = validate_served_file_path(cam.reference_image_path, storage_root)
    if not ref_file_path.exists():
        raise HTTPException(status_code=404, detail="Reference image file missing from storage.")

    with open(ref_file_path, "rb") as rf:
        ref_bytes = rf.read()

    # Find active verified layout if present
    layout_res = await db.execute(
        select(ParkingLayoutRevision)
        .where(ParkingLayoutRevision.camera_id == camera_id)
        .where(ParkingLayoutRevision.status == LayoutRevisionStatus.VERIFIED.value)
    )
    verified_layout = layout_res.scalar_one_or_none()
    active_layout_id = verified_layout.id if verified_layout else None
    active_layout_sha = verified_layout.canonical_sha256 if verified_layout else None

    # Stream video to secure staging path
    camera_dir = storage_root / "staging" / camera_id
    camera_dir.mkdir(parents=True, exist_ok=True)
    assessment_id = str(uuid.uuid4())
    temp_video_path = camera_dir / f"assess_stream_{assessment_id}.mp4"

    cfg = stab_manager.config
    MAX_VIDEO_BYTES = cfg.intake.max_upload_bytes
    total_bytes = 0

    try:
        with open(temp_video_path, "wb") as vf:
            while chunk := await file.read(65536):
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_BYTES:
                    temp_video_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Uploaded video exceeds configured limit of {MAX_VIDEO_BYTES // (1024*1024)} MB.",
                    )
                vf.write(chunk)

        if total_bytes == 0:
            temp_video_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded video file is empty (0 bytes).",
            )

        assessment = CameraStabilityAssessment(
            id=assessment_id,
            camera_id=camera_id,
            layout_revision_id=active_layout_id,
            layout_canonical_sha256=active_layout_sha,
            reference_image_sha256=cam.reference_image_sha256,
            video_sha256=None,
            status="QUEUED",
            progress_pct=0.0,
            stage_message="Queued for stability evaluation",
            config_version=cfg.version,
            config_sha256=cfg.config_sha256,
            algorithm_version=cfg.algorithm_version,
            opencv_version=None,
            thresholds_snapshot=cfg.thresholds_snapshot,
            sample_measurements=[],
            aggregate_decision=None,
            operational_gate=OperationalGate.BLOCKED.value,
            gate_reasons=["PROCESSING: Assessment job is currently running."],
            summary_metrics=None,
            created_at=utc_now(),
        )
        db.add(assessment)
        await db.commit()
        await db.refresh(assessment)

        # Launch background evaluation worker
        stab_manager.submit_assessment_job(
            assessment_id=assessment_id,
            video_path=temp_video_path,
            camera_id=camera_id,
            reference_image_bytes=ref_bytes,
            expected_reference_sha256=cam.reference_image_sha256,
            active_layout_canonical_sha256=active_layout_sha,
        )

        return _build_stability_response(assessment)

    except HTTPException:
        temp_video_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        temp_video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/stability/assessments/{assessment_id}/cancel", response_model=StabilityAssessmentCancelResponse)
async def cancel_stability_assessment(
    assessment_id: str,
    db: AsyncSession = Depends(get_db),
    stab_manager: StabilityAssessmentJobManager = Depends(get_stability_job_manager),
):
    """Safely cancel an in-progress or queued stability assessment."""
    res = await db.execute(select(CameraStabilityAssessment).where(CameraStabilityAssessment.id == assessment_id))
    a = res.scalar_one_or_none()
    if not a:
        raise HTTPException(status_code=404, detail="Stability assessment not found")

    if a.status in ("COMPLETE", "FAILED", "CANCELLED"):
        return StabilityAssessmentCancelResponse(
            assessment_id=assessment_id,
            status=a.status,
            cancelled=False,
            message=f"Assessment is already in terminal state '{a.status}'.",
        )

    await stab_manager.cancel_assessment_job(assessment_id)
    a.status = "CANCELLED"
    a.progress_pct = 100.0
    a.stage_message = "Assessment cancelled by operator"
    a.failure_code = "CANCELLED"
    a.failure_message = "Assessment cancelled by operator request."
    a.aggregate_decision = StabilityDecision.ERROR.value
    a.operational_gate = OperationalGate.BLOCKED.value
    a.gate_reasons = ["CANCELLED: Assessment cancelled by operator."]
    a.completed_at = utc_now()
    await db.commit()

    return StabilityAssessmentCancelResponse(
        assessment_id=assessment_id,
        status="CANCELLED",
        cancelled=True,
        message="Stability assessment cancelled successfully.",
    )


@router.get("/cameras/{camera_id}/stability/gate", response_model=CameraOperationalGateResponse)
async def get_camera_operational_gate(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
    stab_manager: StabilityAssessmentJobManager = Depends(get_stability_job_manager),
):
    """
    Retrieve the current fail-closed operational inference gate for a camera.
    Inference is ALLOWED only if a fresh STABLE assessment matches both reference and layout SHAs.
    """
    cam_res = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = cam_res.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    layout_res = await db.execute(
        select(ParkingLayoutRevision)
        .where(ParkingLayoutRevision.camera_id == camera_id)
        .where(ParkingLayoutRevision.status == LayoutRevisionStatus.VERIFIED.value)
    )
    verified_layout = layout_res.scalar_one_or_none()
    current_layout_sha = verified_layout.canonical_sha256 if verified_layout else None

    # Find latest completed assessment for camera
    assess_res = await db.execute(
        select(CameraStabilityAssessment)
        .where(CameraStabilityAssessment.camera_id == camera_id)
        .where(CameraStabilityAssessment.status == "COMPLETE")
        .order_by(desc(CameraStabilityAssessment.created_at))
    )
    latest_assessment = assess_res.scalars().first()

    if not latest_assessment:
        gate, reasons = evaluate_operational_gate(
            decision=None,
            assessment_reference_sha=None,
            current_reference_sha=cam.reference_image_sha256,
            assessment_layout_sha=None,
            current_layout_sha=current_layout_sha,
        )
        return CameraOperationalGateResponse(
            camera_id=camera_id,
            operational_gate=gate.value,
            gate_reasons=reasons,
            aggregate_decision=None,
            assessment_id=None,
            reference_image_sha256=cam.reference_image_sha256,
            layout_canonical_sha256=current_layout_sha,
            created_at=None,
            is_fresh=False,
        )

    decision_enum = None
    try:
        decision_enum = StabilityDecision(latest_assessment.aggregate_decision)
    except Exception:
        pass

    now = utc_now()
    cfg = stab_manager.config
    max_age = (latest_assessment.thresholds_snapshot or {}).get("max_assessment_age_seconds", cfg.thresholds.max_assessment_age_seconds)

    gate, reasons = evaluate_operational_gate(
        decision=decision_enum,
        assessment_reference_sha=latest_assessment.reference_image_sha256,
        current_reference_sha=cam.reference_image_sha256,
        assessment_layout_sha=latest_assessment.layout_canonical_sha256,
        current_layout_sha=current_layout_sha,
        assessment_timestamp=latest_assessment.created_at,
        current_timestamp=now,
        max_age_seconds=max_age,
    )

    t_ass = latest_assessment.created_at if latest_assessment.created_at.tzinfo else latest_assessment.created_at.replace(tzinfo=timezone.utc)
    age_seconds = (now - t_ass).total_seconds()
    is_fresh = (0.0 <= age_seconds <= max_age) and (gate == OperationalGate.ALLOWED)

    return CameraOperationalGateResponse(
        camera_id=camera_id,
        operational_gate=gate.value,
        gate_reasons=reasons,
        aggregate_decision=latest_assessment.aggregate_decision,
        assessment_id=latest_assessment.id,
        reference_image_sha256=cam.reference_image_sha256,
        layout_canonical_sha256=current_layout_sha,
        created_at=latest_assessment.created_at,
        is_fresh=is_fresh,
    )


@router.get("/cameras/{camera_id}/stability/assessments", response_model=List[StabilityAssessmentResponse])
async def list_camera_stability_assessments(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
):
    """List historical camera stability assessments ordered by creation time."""
    cam_res = await db.execute(select(Camera).where(Camera.id == camera_id))
    if not cam_res.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Camera not found")

    res = await db.execute(
        select(CameraStabilityAssessment)
        .where(CameraStabilityAssessment.camera_id == camera_id)
        .order_by(desc(CameraStabilityAssessment.created_at))
    )
    assessments = res.scalars().all()
    return [_build_stability_response(a) for a in assessments]


@router.get("/stability/assessments/{assessment_id}", response_model=StabilityAssessmentResponse)
async def get_stability_assessment_detail(
    assessment_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve full stability assessment report including per-sample measurements."""
    res = await db.execute(select(CameraStabilityAssessment).where(CameraStabilityAssessment.id == assessment_id))
    a = res.scalar_one_or_none()
    if not a:
        raise HTTPException(status_code=404, detail="Stability assessment not found")
    return _build_stability_response(a)


@router.post("/stability/assessments/{assessment_id}/acknowledge", response_model=CameraStabilityAuditResponse, status_code=status.HTTP_201_CREATED)
async def acknowledge_stability_assessment(
    assessment_id: str,
    payload: AcknowledgeStabilityRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Operator acknowledgement of a stability assessment outcome.
    Appends an immutable audit event and optionally triggers audited calibration invalidation for UNSTABLE assessments.
    """
    res = await db.execute(
        select(CameraStabilityAssessment)
        .where(CameraStabilityAssessment.id == assessment_id)
    )
    a = res.scalar_one_or_none()
    if not a:
        raise HTTPException(status_code=404, detail="Stability assessment not found")

    if a.status != "COMPLETE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot acknowledge assessment in non-terminal status '{a.status}'.",
        )

    cam_res = await db.execute(select(Camera).where(Camera.id == a.camera_id))
    cam = cam_res.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Associated camera not found")

    # Get active verified layout if present
    layout_res = await db.execute(
        select(ParkingLayoutRevision)
        .where(ParkingLayoutRevision.camera_id == a.camera_id)
        .where(ParkingLayoutRevision.status == LayoutRevisionStatus.VERIFIED.value)
    )
    verified_layouts = layout_res.scalars().all()
    current_layout = verified_layouts[0] if verified_layouts else None
    current_layout_sha = current_layout.canonical_sha256 if current_layout else None

    prev_gate = a.operational_gate or OperationalGate.BLOCKED.value
    prev_calib = cam.calibration_status
    resulting_gate = prev_gate
    resulting_calib = prev_calib
    event_type = "OPERATOR_ACKNOWLEDGED"

    if payload.trigger_calibration_invalidation:
        if not payload.confirm_invalidation:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Explicit confirmation (confirm_invalidation=True) is required to invalidate camera calibration.",
            )
        if not payload.invalidation_reason or not payload.invalidation_reason.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A non-blank invalidation reason is required to invalidate camera calibration.",
            )
        if a.aggregate_decision != StabilityDecision.UNSTABLE.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Calibration invalidation via this endpoint is strictly allowed only for UNSTABLE assessments; current assessment is {a.aggregate_decision}.",
            )
        if a.reference_image_sha256 != cam.reference_image_sha256:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Assessment reference image SHA does not match the camera's current reference image SHA.",
            )
        if current_layout_sha and a.layout_canonical_sha256 != current_layout_sha:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Assessment layout canonical SHA does not match the active verified layout SHA.",
            )

        # Staleness check
        max_age = (a.thresholds_snapshot or {}).get("max_assessment_age_seconds", 86400)
        t_ass = a.created_at if a.created_at.tzinfo else a.created_at.replace(tzinfo=timezone.utc)
        age = (utc_now() - t_ass).total_seconds()
        if age > max_age:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot invalidate calibration from an expired assessment ({int(age)}s old > {max_age}s max age).",
            )

        # Invalidate camera calibration
        cam.calibration_status = CalibrationStatus.INVALIDATED.value
        cam.updated_at = utc_now()
        resulting_calib = CalibrationStatus.INVALIDATED.value
        resulting_gate = OperationalGate.BLOCKED.value
        event_type = "CALIBRATION_INVALIDATED"

        for rev in verified_layouts:
            rev.status = LayoutRevisionStatus.INVALIDATED.value
            rev.invalidated_at = utc_now()
            rev.invalidation_reason = f"Camera stability assessment UNSTABLE: {payload.invalidation_reason.strip()}"
            audit = LayoutAuditEvent(
                id=str(uuid.uuid4()),
                layout_revision_id=rev.id,
                event_type="CALIBRATION_INVALIDATED",
                prior_status=LayoutRevisionStatus.VERIFIED.value,
                new_status=LayoutRevisionStatus.INVALIDATED.value,
                local_operator_label=payload.local_operator_label,
                note=payload.note or f"Stability assessment {assessment_id} flagged UNSTABLE.",
                created_at=utc_now(),
            )
            db.add(audit)

    # Compute assessment summary SHA
    summary_str = f"{a.id}:{a.video_sha256}:{a.reference_image_sha256}:{a.aggregate_decision}:{a.operational_gate}"
    ass_sha = hashlib.sha256(summary_str.encode("utf-8")).hexdigest()

    audit_event = CameraStabilityAuditEvent(
        id=str(uuid.uuid4()),
        assessment_id=a.id,
        camera_id=cam.id,
        event_type=event_type,
        operator_identity=payload.local_operator_label,
        explicit_reason=payload.invalidation_reason.strip() if payload.invalidation_reason else (payload.note or "Operator acknowledged stability assessment outcome."),
        note=payload.note,
        previous_gate_state=prev_gate,
        resulting_gate_state=resulting_gate,
        previous_calibration_status=prev_calib,
        resulting_calibration_status=resulting_calib,
        assessment_sha256=ass_sha,
        config_sha256=a.config_sha256,
        reference_image_sha256=cam.reference_image_sha256,
        layout_canonical_sha256=current_layout_sha,
        created_at=utc_now(),
    )
    db.add(audit_event)

    await db.commit()
    await db.refresh(audit_event)

    return CameraStabilityAuditResponse(
        id=audit_event.id,
        assessment_id=audit_event.assessment_id,
        camera_id=audit_event.camera_id,
        event_type=audit_event.event_type,
        operator_identity=audit_event.operator_identity,
        explicit_reason=audit_event.explicit_reason,
        note=audit_event.note,
        previous_gate_state=audit_event.previous_gate_state,
        resulting_gate_state=audit_event.resulting_gate_state,
        previous_calibration_status=audit_event.previous_calibration_status,
        resulting_calibration_status=audit_event.resulting_calibration_status,
        assessment_sha256=audit_event.assessment_sha256,
        config_sha256=audit_event.config_sha256,
        reference_image_sha256=audit_event.reference_image_sha256,
        layout_canonical_sha256=audit_event.layout_canonical_sha256,
        created_at=audit_event.created_at,
    )


@router.get("/cameras/{camera_id}/stability/audit-events", response_model=List[CameraStabilityAuditResponse])
async def list_camera_stability_audit_events(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
):
    """List append-only camera stability audit and acknowledgement events."""
    cam_res = await db.execute(select(Camera).where(Camera.id == camera_id))
    if not cam_res.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Camera not found")

    res = await db.execute(
        select(CameraStabilityAuditEvent)
        .where(CameraStabilityAuditEvent.camera_id == camera_id)
        .order_by(desc(CameraStabilityAuditEvent.created_at))
    )
    events = res.scalars().all()
    return [
        CameraStabilityAuditResponse(
            id=e.id,
            assessment_id=e.assessment_id,
            camera_id=e.camera_id,
            event_type=e.event_type,
            operator_identity=e.operator_identity,
            explicit_reason=e.explicit_reason,
            note=e.note,
            previous_gate_state=e.previous_gate_state,
            resulting_gate_state=e.resulting_gate_state,
            previous_calibration_status=e.previous_calibration_status,
            resulting_calibration_status=e.resulting_calibration_status,
            assessment_sha256=e.assessment_sha256,
            config_sha256=e.config_sha256,
            reference_image_sha256=e.reference_image_sha256,
            layout_canonical_sha256=e.layout_canonical_sha256,
            created_at=e.created_at,
        )
        for e in events
    ]


# ============================================================================
# Phase 2B: Gate-Controlled Parking Occupancy & Video Inference Endpoints
# ============================================================================

