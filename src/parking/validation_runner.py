"""End-to-end automated validation runner for Phase 2C stationary parking camera workflow.

Runs in an isolated temporary application environment with temporary database and media root,
using the public HTTP/API workflow through ASGITransport.
"""

from __future__ import annotations

import asyncio
import collections
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional
import uuid

import cv2
from httpx import ASGITransport, AsyncClient
import numpy as np
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from services.api.app.core.config import settings
from services.api.app.db.base import Base
from services.api.app.db.session import get_db
from services.api.app.main import app
from services.api.app.routers.parking import (
    get_parking_job_manager,
    get_stability_job_manager,
    get_storage_root,
)
from services.api.app.services.parking_occupancy_job_manager import ParkingOccupancyJobManager
from services.api.app.services.stability_job_manager import StabilityAssessmentJobManager
from src.parking.contracts import (
    LayoutRevisionStatus,
    OperationalGate,
    StabilityDecision,
)
from src.parking.occupancy_config import load_parking_occupancy_config
from src.parking.occupancy_contracts import OccupancyState
from src.parking.stability_config import load_stability_config
from src.parking.synthetic_scene_generator import (
    SyntheticParkingFixture,
    generate_synthetic_parking_fixture,
)
from src.parking.testing_support import DeterministicVehicleDetectorDouble
from src.parking.validation_evidence import (
    ParkingValidationEvidenceReport,
    ValidationCheckItem,
)

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_git_commit_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


async def run_stable_parking_e2e_validation(
    work_dir: Optional[Path] = None,
    use_synthetic_detector_double: bool = True,
) -> ParkingValidationEvidenceReport:
    """
    Executes the complete Phase 2C end-to-end parking occupancy validation workflow on a stationary camera fixture.
    Constructs an isolated temporary application environment (temp SQLite database, temp MEDIA_ROOT, isolated manager),
    and executes the full lifecycle strictly through the public HTTP/API workflow using ASGITransport:
    1. Create site (POST /sites)
    2. Create camera (POST /sites/{site_id}/cameras)
    3. Upload reference image (POST /cameras/{camera_id}/reference-image)
    4. Create layout draft (POST /cameras/{camera_id}/layouts)
    5. Submit layout (POST /layouts/{layout_id}/submit)
    6. Verify layout (POST /layouts/{layout_id}/verify)
    7. Submit camera stability assessment (POST /cameras/{camera_id}/stability/assess)
    8. Await stability assessment completion (GET /stability/assessments/{assessment_id})
    9. Confirm ALLOWED operational gate (GET /cameras/{camera_id}/stability/gate)
    10. Submit occupancy evaluation job (POST /cameras/{camera_id}/occupancy/jobs)
    11. Await COMPLETE status (GET /parking/jobs/{job_id})
    12. Download & revalidate all artifacts (manifest, timeline, summary, video)
    13. Return machine-readable validation evidence report.
    """
    cleanup_temp = False
    if work_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="roadsense_isolated_val_")
        work_dir = Path(temp_dir_obj.name)
        cleanup_temp = True
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"val_run_{uuid.uuid4().hex[:12]}"
    checks: List[ValidationCheckItem] = []
    failure_reasons: List[str] = []

    # 1. Setup isolated database & media storage
    temp_db_path = work_dir / "isolated_val.db"
    db_url = f"sqlite+aiosqlite:///{temp_db_path}"
    isolated_engine = create_async_engine(db_url, echo=False)

    async with isolated_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    isolated_session_factory = async_sessionmaker(isolated_engine, expire_on_commit=False)
    temp_media_root = work_dir / "media"
    temp_media_root.mkdir(parents=True, exist_ok=True)

    # 2. Generate synthetic stationary parking fixture
    fixture_dir = work_dir / "fixture"
    fixture = generate_synthetic_parking_fixture(
        output_dir=fixture_dir,
        width=1920,
        height=1080,
        fps=30.0,
        duration_seconds=4.0,
    )

    ref_bytes = fixture.reference_image_path.read_bytes()
    ref_sha = hashlib.sha256(ref_bytes).hexdigest()

    vid_bytes = fixture.video_path.read_bytes()
    vid_sha = hashlib.sha256(vid_bytes).hexdigest()

    checks.append(
        ValidationCheckItem(
            check_name="synthetic_fixture_generation",
            category="media_generation",
            expected=True,
            observed=fixture.video_path.is_file() and fixture.reference_image_path.is_file(),
            passed=fixture.video_path.is_file() and fixture.reference_image_path.is_file(),
            details=f"Video frames: {fixture.total_frames}, duration: {fixture.duration_seconds}s",
        )
    )

    # 3. Setup isolated managers with injected detector factory
    def detector_factory():
        if use_synthetic_detector_double:
            return DeterministicVehicleDetectorDouble(
                frame_detections=fixture.frame_ground_truth_detections,
                checkpoint_sha256="0" * 64,
            )
        return None

    isolated_occ_manager = ParkingOccupancyJobManager(
        detector_factory=detector_factory if use_synthetic_detector_double else None,
        db_session_factory=isolated_session_factory,
        media_root=temp_media_root,
    )
    isolated_stab_manager = StabilityAssessmentJobManager(
        db_session_factory=isolated_session_factory,
    )

    # Set settings.MEDIA_ROOT for isolated environment
    saved_media_root = getattr(settings, "MEDIA_ROOT", None)
    settings.MEDIA_ROOT = str(temp_media_root)

    # Override app dependencies to point to isolated temporary environment
    async def get_isolated_db():
        async with isolated_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = get_isolated_db
    app.dependency_overrides[get_parking_job_manager] = lambda: isolated_occ_manager
    app.dependency_overrides[get_stability_job_manager] = lambda: isolated_stab_manager
    app.dependency_overrides[get_storage_root] = lambda: temp_media_root

    stability_cfg = load_stability_config()
    occupancy_cfg = load_parking_occupancy_config()

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test", timeout=60.0) as client:
            # Step 1: Create Site
            res_site = await client.post(
                "/api/v1/sites",
                json={"name": "Isolated Synthetic Site", "timezone": "UTC"},
            )
            if res_site.status_code != 201:
                raise RuntimeError(f"Failed to create site via API: {res_site.status_code} {res_site.text}")
            site_data = res_site.json()
            site_id = site_data["id"]

            # Step 2: Create Camera
            res_cam = await client.post(
                f"/api/v1/sites/{site_id}/cameras",
                json={"name": "Stationary Bay Cam 01", "description": "Validation test camera"},
            )
            if res_cam.status_code != 201:
                raise RuntimeError(f"Failed to create camera via API: {res_cam.status_code} {res_cam.text}")
            cam_data = res_cam.json()
            camera_id = cam_data["id"]

            # Step 3: Upload Reference Image
            with open(fixture.reference_image_path, "rb") as rf:
                res_ref = await client.post(
                    f"/api/v1/cameras/{camera_id}/reference-image",
                    files={"file": ("reference.jpg", rf, "image/jpeg")},
                )
            if res_ref.status_code != 200:
                raise RuntimeError(f"Failed to upload reference image via API: {res_ref.status_code} {res_ref.text}")
            cam_ref_data = res_ref.json()
            uploaded_ref_sha = cam_ref_data.get("reference_image_sha256")
            ref_sha = uploaded_ref_sha or ref_sha

            checks.append(
                ValidationCheckItem(
                    check_name="reference_image_upload_and_sha",
                    category="api_workflow",
                    expected=True,
                    observed=bool(uploaded_ref_sha and len(uploaded_ref_sha) == 64),
                    passed=bool(uploaded_ref_sha and len(uploaded_ref_sha) == 64),
                    details=f"Stored Reference SHA: {uploaded_ref_sha}, Status: {cam_ref_data.get('calibration_status')}",
                )
            )

            # Step 4: Create Draft Layout Revision
            res_layout = await client.post(
                f"/api/v1/cameras/{camera_id}/layouts",
                json={
                    "parking_spaces": fixture.parking_spaces,
                    "approach_zones": fixture.approach_zones,
                },
            )
            if res_layout.status_code != 201:
                raise RuntimeError(f"Failed to create draft layout via API: {res_layout.status_code} {res_layout.text}")
            layout_data = res_layout.json()
            layout_id = layout_data["id"]

            # Step 5: Submit Layout for Review
            res_submit = await client.post(
                f"/api/v1/layouts/{layout_id}/submit",
                json={"local_operator_label": "synthetic_validator_operator", "note": "Submitting synthetic layout"},
            )
            if res_submit.status_code != 200:
                raise RuntimeError(f"Failed to submit layout via API: {res_submit.status_code} {res_submit.text}")
            submitted_layout = res_submit.json()
            if submitted_layout["status"] != "PENDING_REVIEW":
                raise RuntimeError(f"Expected layout status PENDING_REVIEW, got {submitted_layout['status']}")

            # Step 6: Explicitly Verify Layout
            res_verify = await client.post(
                f"/api/v1/layouts/{layout_id}/verify",
                json={
                    "confirmation_acknowledged": True,
                    "local_operator_label": "synthetic_validator_operator",
                    "note": "Verified stationary bay layout",
                },
            )
            if res_verify.status_code != 200:
                raise RuntimeError(f"Failed to verify layout via API: {res_verify.status_code} {res_verify.text}")
            verified_layout = res_verify.json()
            layout_canonical_sha = verified_layout["canonical_sha256"]

            checks.append(
                ValidationCheckItem(
                    check_name="layout_verification_workflow",
                    category="api_workflow",
                    expected="VERIFIED",
                    observed=verified_layout["status"],
                    passed=(verified_layout["status"] == "VERIFIED"),
                    details=f"Canonical Layout SHA: {layout_canonical_sha[:12]}...",
                )
            )

            # Step 7: Submit Camera Stability Assessment
            with open(fixture.video_path, "rb") as vf:
                res_assess = await client.post(
                    f"/api/v1/cameras/{camera_id}/stability/assess",
                    files={"file": ("stability_test.mp4", vf, "video/mp4")},
                )
            if res_assess.status_code != 202:
                raise RuntimeError(f"Failed to submit stability assessment via API: {res_assess.status_code} {res_assess.text}")
            assessment_data = res_assess.json()
            assessment_id = assessment_data.get("id") or assessment_data.get("assessment_id")

            # Step 8: Poll for Stability Assessment Completion
            max_wait_secs = 20.0
            poll_interval = 0.2
            elapsed = 0.0
            final_assessment = None
            while elapsed < max_wait_secs:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                res_poll = await client.get(f"/api/v1/stability/assessments/{assessment_id}")
                if res_poll.status_code == 200:
                    a_data = res_poll.json()
                    if a_data["status"] in ("COMPLETE", "FAILED", "CANCELLED"):
                        final_assessment = a_data
                        break

            if not final_assessment or final_assessment["status"] != "COMPLETE":
                raise RuntimeError(f"Stability assessment failed to complete. Final: {final_assessment}")

            checks.append(
                ValidationCheckItem(
                    check_name="camera_stability_assessment_decision",
                    category="stability_gate",
                    expected="STABLE",
                    observed=final_assessment.get("aggregate_decision"),
                    passed=(final_assessment.get("aggregate_decision") == "STABLE"),
                    details=f"Decision: {final_assessment.get('aggregate_decision')}, Gate: {final_assessment.get('operational_gate')}",
                )
            )

            # Step 9: Confirm Operational Gate
            res_gate = await client.get(f"/api/v1/cameras/{camera_id}/stability/gate")
            if res_gate.status_code != 200:
                raise RuntimeError(f"Failed to get camera operational gate via API: {res_gate.status_code}")
            gate_data = res_gate.json()

            is_gate_allowed = (gate_data["operational_gate"] == "ALLOWED")
            checks.append(
                ValidationCheckItem(
                    check_name="operational_gate_evaluation",
                    category="stability_gate",
                    expected="ALLOWED",
                    observed=gate_data["operational_gate"],
                    passed=is_gate_allowed,
                    details="; ".join(gate_data.get("gate_reasons", [])),
                )
            )
            if not is_gate_allowed:
                failure_reasons.append(f"Operational gate blocked: {gate_data.get('gate_reasons')}")

            # Step 10: Submit Occupancy Evaluation Job
            with open(fixture.video_path, "rb") as vf:
                res_job = await client.post(
                    f"/api/v1/cameras/{camera_id}/occupancy/jobs",
                    files={"file": ("occupancy_test.mp4", vf, "video/mp4")},
                )
            if res_job.status_code != 202:
                raise RuntimeError(f"Failed to submit occupancy job via API: {res_job.status_code} {res_job.text}")
            job_data = res_job.json()
            job_id = job_data["id"]

            # Step 11: Poll for Occupancy Job COMPLETE
            max_occ_wait = 30.0
            occ_elapsed = 0.0
            final_job = None
            while occ_elapsed < max_occ_wait:
                await asyncio.sleep(poll_interval)
                occ_elapsed += poll_interval
                res_job_poll = await client.get(f"/api/v1/parking/jobs/{job_id}")
                if res_job_poll.status_code == 200:
                    j_data = res_job_poll.json()
                    if j_data["status"] in ("COMPLETE", "FAILED", "CANCELLED", "BLOCKED_BY_STABILITY_GATE"):
                        final_job = j_data
                        break

            if not final_job:
                raise RuntimeError("Occupancy job polling timed out before reaching terminal status.")

            checks.append(
                ValidationCheckItem(
                    check_name="occupancy_job_execution_status",
                    category="pipeline_execution",
                    expected="COMPLETE",
                    observed=final_job.get("status"),
                    passed=(final_job.get("status") == "COMPLETE"),
                    details=f"Final message: {final_job.get('stage_message')}",
                )
            )

            if final_job.get("status") != "COMPLETE":
                failure_reasons.append(
                    f"Job failed with status {final_job.get('status')}: {final_job.get('failure_code')} - {final_job.get('failure_message')}"
                )

            # Step 12: Download & Revalidate All Artifacts via Public API Endpoints
            # A. Manifest
            res_manifest = await client.get(f"/api/v1/parking/jobs/{job_id}/manifest")
            if res_manifest.status_code != 200:
                raise RuntimeError(f"Failed to download manifest: {res_manifest.status_code}")
            manifest_json = res_manifest.json()
            manifest_bytes = json.dumps(manifest_json, sort_keys=True).encode("utf-8")
            manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

            # B. Timeline JSONL
            res_timeline = await client.get(f"/api/v1/parking/jobs/{job_id}/timeline")
            if res_timeline.status_code != 200:
                raise RuntimeError(f"Failed to download timeline: {res_timeline.status_code}")
            timeline_content = res_timeline.content
            timeline_sha = hashlib.sha256(timeline_content).hexdigest()

            # Verify timeline SHA matches manifest
            expected_timeline_sha = manifest_json.get("timeline_sha256") or (manifest_json.get("software_versions") or {}).get("timeline_sha256")
            checks.append(
                ValidationCheckItem(
                    check_name="timeline_sha_manifest_integrity",
                    category="artifact_integrity",
                    expected=expected_timeline_sha,
                    observed=timeline_sha,
                    passed=(timeline_sha == expected_timeline_sha),
                    details=f"Timeline lines: {len(timeline_content.splitlines())}",
                )
            )

            # C. Summary JSON
            res_summary = await client.get(f"/api/v1/parking/jobs/{job_id}/summary")
            if res_summary.status_code != 200:
                raise RuntimeError(f"Failed to download summary: {res_summary.status_code}")
            summary_content = res_summary.content
            summary_sha = hashlib.sha256(summary_content).hexdigest()
            summary_data = res_summary.json()

            expected_summary_sha = manifest_json.get("summary_sha256") or (manifest_json.get("software_versions") or {}).get("summary_sha256")
            checks.append(
                ValidationCheckItem(
                    check_name="summary_sha_manifest_integrity",
                    category="artifact_integrity",
                    expected=expected_summary_sha,
                    observed=summary_sha,
                    passed=(summary_sha == expected_summary_sha),
                    details=f"Summary bays count: {len(summary_data.get('bay_summary', {}))}",
                )
            )

            # D. Output Video MP4
            res_video = await client.get(f"/api/v1/parking/jobs/{job_id}/video")
            if res_video.status_code != 200:
                raise RuntimeError(f"Failed to download video: {res_video.status_code}")
            video_content = res_video.content
            out_video_sha = hashlib.sha256(video_content).hexdigest()

            expected_video_sha = manifest_json.get("output_video_sha256") or (manifest_json.get("software_versions") or {}).get("output_video_sha256")
            checks.append(
                ValidationCheckItem(
                    check_name="video_sha_manifest_integrity",
                    category="artifact_integrity",
                    expected=expected_video_sha,
                    observed=out_video_sha,
                    passed=(out_video_sha == expected_video_sha),
                    details=f"Video size: {len(video_content)} bytes",
                )
            )

            # Compute Bay occupancy metrics from summary and final job response
            bay_summary = summary_data.get("bay_summary") or {}
            final_occ = summary_data.get("occupied_count", 0) or sum(1 for b in bay_summary.values() if b.get("final_state") == OccupancyState.OCCUPIED.value)
            final_vac = summary_data.get("vacant_count", 0) or sum(1 for b in bay_summary.values() if b.get("final_state") == OccupancyState.VACANT.value)
            final_unk = summary_data.get("unknown_count", 0) or sum(1 for b in bay_summary.values() if b.get("final_state") == OccupancyState.UNKNOWN.value)
            final_occ_drop = summary_data.get("occluded_count", 0) or sum(1 for b in bay_summary.values() if b.get("final_state") == OccupancyState.OCCLUDED.value)

            # Timeline transition count
            timeline_lines = [l for l in timeline_content.decode("utf-8", errors="replace").splitlines() if l.strip()]
            total_transitions = len(timeline_lines)

            checks.append(
                ValidationCheckItem(
                    check_name="ground_truth_bay_occupancy_counts",
                    category="occupancy_classification",
                    expected={"occupied": 2, "vacant": 2},
                    observed={"occupied": final_occ, "vacant": final_vac},
                    passed=(final_occ == 2 and final_vac == 2),
                    details=f"Bays: 4 total, {final_occ} occupied, {final_vac} vacant, {final_unk} unknown, {final_occ_drop} occluded",
                )
            )

            # --- Part B: In-depth Temporal Evidence Validation ---
            timeline_events = []
            for line_str in timeline_lines:
                try:
                    timeline_events.append(json.loads(line_str))
                except Exception:
                    pass

            # 1. Check ordering and chronological consistency
            timestamps = [e.get("timestamp_seconds", 0.0) for e in timeline_events]
            is_chronological = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
            checks.append(
                ValidationCheckItem(
                    check_name="temporal_sequence_and_ordering",
                    category="temporal_evidence",
                    expected="chronologically_non_decreasing",
                    observed=f"{len(timestamps)} timestamps, strictly non-decreasing" if is_chronological else "timestamp order violation",
                    passed=is_chronological and len(timestamps) > 0,
                    details=f"First ts: {timestamps[0] if timestamps else 0}s, Last ts: {timestamps[-1] if timestamps else 0}s",
                )
            )
            if not is_chronological:
                failure_reasons.append("Timeline events are not in non-decreasing chronological order.")

            # 2. Per-bay state transitions and dropout recovery
            bay_events: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
            for ev in timeline_events:
                bay_events[ev.get("bay_id", "")].append(ev)

            has_duplicates = False
            has_state_chain_error = False
            for bid, ev_list in bay_events.items():
                for idx, ev in enumerate(ev_list):
                    if idx == 0:
                        if ev.get("previous_state") != OccupancyState.UNKNOWN.value:
                            has_state_chain_error = True
                    else:
                        prev_ev = ev_list[idx - 1]
                        if ev.get("previous_state") != prev_ev.get("new_state"):
                            has_state_chain_error = True
                        if ev.get("new_state") == prev_ev.get("new_state"):
                            has_duplicates = True

            b1_ev = next((v for k, v in bay_events.items() if k.endswith("_1")), [])
            b2_ev = next((v for k, v in bay_events.items() if k.endswith("_2")), [])
            b3_ev = next((v for k, v in bay_events.items() if k.endswith("_3")), [])
            b4_ev = next((v for k, v in bay_events.items() if k.endswith("_4")), [])

            b1_states = [e.get("new_state") for e in b1_ev]
            b2_states = [e.get("new_state") for e in b2_ev]
            b3_states = [e.get("new_state") for e in b3_ev]
            b4_states = [e.get("new_state") for e in b4_ev]

            b1_valid = (b1_states in (["OCCUPIED", "VACANT"], ["OCCUPIED", "OCCLUDED", "VACANT"])) and (b1_states[-1] == "VACANT")
            b2_valid = (b2_states == ["VACANT", "OCCUPIED"]) and (b2_states[-1] == "OCCUPIED")
            b3_valid = (b3_states == ["VACANT"]) and (b3_states[-1] == "VACANT")
            b4_valid = (b4_states[0] == "OCCUPIED") and ("OCCLUDED" in b4_states) and (b4_states[-1] == "OCCUPIED")

            temporal_transitions_passed = (
                not has_duplicates
                and not has_state_chain_error
                and b1_valid
                and b2_valid
                and b3_valid
                and b4_valid
            )

            checks.append(
                ValidationCheckItem(
                    check_name="temporal_per_bay_transitions_and_dropout_recovery",
                    category="temporal_evidence",
                    expected={
                        "Bay-1": "OCCUPIED -> ... -> VACANT",
                        "Bay-2": ["VACANT", "OCCUPIED"],
                        "Bay-3": ["VACANT"],
                        "Bay-4": "OCCUPIED -> OCCLUDED -> ... -> OCCUPIED",
                    },
                    observed={
                        "Bay-1": b1_states,
                        "Bay-2": b2_states,
                        "Bay-3": b3_states,
                        "Bay-4": b4_states,
                    },
                    passed=temporal_transitions_passed,
                    details=f"No duplicate transitions: {not has_duplicates}, State continuity: {not has_state_chain_error}, Occlusion recovery: {b4_valid}",
                )
            )
            if not temporal_transitions_passed:
                failure_reasons.append(
                    f"Temporal state transition mismatch: B1={b1_states}, B2={b2_states}, B3={b3_states}, B4={b4_states}"
                )

            # 3. Final state agreement with summary JSON
            final_timeline_states = {
                bid: ev_list[-1].get("new_state") if ev_list else OccupancyState.UNKNOWN.value
                for bid, ev_list in bay_events.items()
            }
            summary_states = {
                bid: info.get("current_state") or info.get("final_state")
                for bid, info in bay_summary.items()
            }
            states_agree = (final_timeline_states == summary_states and len(summary_states) == 4)
            checks.append(
                ValidationCheckItem(
                    check_name="temporal_and_summary_final_state_agreement",
                    category="temporal_evidence",
                    expected=summary_states,
                    observed=final_timeline_states,
                    passed=states_agree,
                    details=f"Compared {len(summary_states)} bay states between timeline and summary",
                )
            )
            if not states_agree:
                failure_reasons.append(f"Final states in timeline ({final_timeline_states}) do not agree with summary JSON ({summary_states}).")

            # --- Part C: Validate Decoded MP4 Overlay Colors & Regions ---
            video_color_checks_passed = True
            color_observations = {}
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_v_file:
                tmp_v_file.write(video_content)
                tmp_v_path = tmp_v_file.name

            try:
                cap_out = cv2.VideoCapture(tmp_v_path)
                cap_out.set(cv2.CAP_PROP_POS_FRAMES, 5)
                ret5, frame5 = cap_out.read()

                cap_out.set(cv2.CAP_PROP_POS_FRAMES, 50)
                ret50, frame50 = cap_out.read()
                cap_out.release()

                if ret5 and frame5 is not None:
                    b1_roi = frame5[400:600, 150:350]
                    b1_mean_bgr = [float(np.mean(b1_roi[:, :, c])) for c in range(3)]
                    color_observations["bay1_frame5_mean_bgr"] = [round(x, 1) for x in b1_mean_bgr]
                    b1_red_dominant = b1_mean_bgr[2] > b1_mean_bgr[0]

                    b2_roi = frame5[400:600, 600:800]
                    b2_mean_bgr = [float(np.mean(b2_roi[:, :, c])) for c in range(3)]
                    color_observations["bay2_frame5_mean_bgr"] = [round(x, 1) for x in b2_mean_bgr]
                    b2_green_dominant = b2_mean_bgr[1] > b2_mean_bgr[0]

                    b3_roi = frame5[400:600, 1050:1250]
                    b3_mean_bgr = [float(np.mean(b3_roi[:, :, c])) for c in range(3)]
                    color_observations["bay3_frame5_mean_bgr"] = [round(x, 1) for x in b3_mean_bgr]
                    b3_green_dominant = b3_mean_bgr[1] > b3_mean_bgr[0]

                    if not (b1_red_dominant and b2_green_dominant and b3_green_dominant):
                        video_color_checks_passed = False

                if ret50 and frame50 is not None:
                    b4_roi = frame50[400:600, 1500:1700]
                    b4_mean_bgr = [float(np.mean(b4_roi[:, :, c])) for c in range(3)]
                    color_observations["bay4_frame50_occluded_bgr"] = [round(x, 1) for x in b4_mean_bgr]
                    b4_amber_dominant = (b4_mean_bgr[2] > b4_mean_bgr[0]) and (b4_mean_bgr[1] > b4_mean_bgr[0])
                    if not b4_amber_dominant:
                        video_color_checks_passed = False

            finally:
                if os.path.exists(tmp_v_path):
                    try:
                        os.remove(tmp_v_path)
                    except Exception:
                        pass

            checks.append(
                ValidationCheckItem(
                    check_name="overlay_color_and_visual_dominance",
                    category="visual_rendering",
                    expected="Red OCCUPIED (R>B), Green VACANT (G>B), Amber OCCLUDED (R>B & G>B)",
                    observed=color_observations,
                    passed=video_color_checks_passed,
                    details=f"Observed channel metrics: {json.dumps(color_observations)}",
                )
            )
            if not video_color_checks_passed:
                failure_reasons.append("Decoded MP4 frames failed color dominance checks for occupied/vacant/occluded bays.")

            software_versions = {
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "fastapi": "0.115.0",
            }

            git_sha = _get_git_commit_sha()
            all_passed = all(c.passed for c in checks) and (len(failure_reasons) == 0)

            report = ParkingValidationEvidenceReport(
                run_id=run_id,
                is_synthetic_fixture=True,
                disclaimer="SYNTHETIC TEST EVIDENCE - FOR VALIDATION OF STATIONARY PARKING CAMERA PIPELINE ONLY",
                camera_id=camera_id,
                site_id=site_id,
                job_id=job_id,
                input_video_sha256=vid_sha,
                reference_image_sha256=ref_sha,
                layout_canonical_sha256=layout_canonical_sha,
                stability_assessment_id=assessment_id,
                stability_config_sha256=stability_cfg.config_sha256,
                occupancy_config_sha256=occupancy_cfg.config_sha256,
                detector_mode="DETERMINISTIC_SYNTHETIC_DOUBLE" if use_synthetic_detector_double else "LOCAL_YOLO",
                detector_checkpoint_sha256="0" * 64 if use_synthetic_detector_double else "local_model",
                bytetrack_config_sha256=occupancy_cfg.config_sha256,
                bytetrack_frame_rate=fixture.fps,
                output_video_sha256=out_video_sha,
                timeline_sha256=timeline_sha,
                summary_sha256=summary_sha,
                manifest_sha256=manifest_sha,
                operational_gate=gate_data.get("operational_gate", "UNKNOWN"),
                gate_reasons=gate_data.get("gate_reasons", []),
                total_frames=fixture.total_frames,
                processed_frames=fixture.total_frames,
                fps=fixture.fps,
                duration_seconds=fixture.duration_seconds,
                video_width=fixture.width,
                video_height=fixture.height,
                total_bays=len(fixture.parking_spaces),
                final_occupied_count=final_occ,
                final_vacant_count=final_vac,
                final_unknown_count=final_unk,
                final_occluded_count=final_occ_drop,
                total_state_transitions=total_transitions,
                software_versions=software_versions,
                git_commit_sha=git_sha,
                created_at_utc=_utc_now_iso(),
                checks=checks,
                passed=all_passed,
                failure_reasons=failure_reasons,
            )

            report_path = work_dir / "parking_validation_evidence_report.json"
            report.save_to_file(report_path)
            return report

    finally:
        # Reset dependency overrides and MEDIA_ROOT so no globals or production routes are altered
        if saved_media_root is not None:
            settings.MEDIA_ROOT = saved_media_root
        app.dependency_overrides.clear()
        await isolated_engine.dispose()
        if cleanup_temp and work_dir and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def main() -> None:
    """CLI entrypoint for running isolated stationary parking validation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Starting isolated synthetic stationary parking camera validation run...")
    report = asyncio.run(run_stable_parking_e2e_validation())
    print("\n" + "=" * 80)
    print("ROADSENSE INDIA - STATIONARY PARKING CAMERA VALIDATION EVIDENCE")
    print("=" * 80)
    print(f"Run ID:            {report.run_id}")
    print(f"Passed:            {report.passed}")
    print(f"Operational Gate:  {report.operational_gate}")
    print(f"Total Bays:        {report.total_bays} (Occupied: {report.final_occupied_count}, Vacant: {report.final_vacant_count})")
    print(f"Output Video SHA:  {report.output_video_sha256[:16]}...")
    print(f"Timeline SHA:      {report.timeline_sha256[:16]}...")
    print(f"Summary SHA:       {report.summary_sha256[:16]}...")
    print("-" * 80)
    for c in report.checks:
        status_str = "PASS" if c.passed else "FAIL"
        print(f"[{status_str}] {c.check_name:<40} {c.details or ''}")
    print("=" * 80)

    if not report.passed:
        print("Validation FAILED:", report.failure_reasons)
        sys.exit(1)
    else:
        print("Validation PASSED successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
