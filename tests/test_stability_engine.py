"""Comprehensive unit and synthetic tests for the Camera Stability Assessment Engine.

Tests:
1. Synthetic frame comparisons (identical, noise, translation, rotation, scale, perspective, featureless)
2. Homography decomposition accuracy
3. Operational gate fail-closed rules
4. Alembic migration upgrade/downgrade cycle
5. Real local parking video read-only negative test
"""

import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Tuple

import cv2
import numpy as np
import pytest

from src.parking.contracts import (
    OperationalGate,
    StabilityDecision,
    StabilityThresholds,
    evaluate_operational_gate,
)
from src.parking.stability_engine import (
    assess_frame_pair,
    compute_reprojection_error,
    decompose_homography,
    evaluate_video_camera_stability,
    inspect_video_media_safe,
)


def _create_rich_feature_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a high-contrast textured image with many detectable corners/features."""
    img = np.zeros((height, width), dtype=np.uint8)
    # Checkerboard base
    cb_size = 32
    for y in range(0, height, cb_size):
        for x in range(0, width, cb_size):
            if ((x // cb_size) + (y // cb_size)) % 2 == 0:
                img[y : y + cb_size, x : x + cb_size] = 200

    # Overlay geometric shapes and text
    cv2.circle(img, (width // 4, height // 4), 40, 50, -1)
    cv2.rectangle(img, (width // 2, height // 3), (width // 2 + 100, height // 3 + 80), 80, -1)
    cv2.circle(img, (3 * width // 4, 3 * height // 4), 60, 255, 3)
    cv2.putText(img, "ROADSENSE CALIBRATION GRID", (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(img, "FIXED ROI TARGET AREA", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    return img


def test_homography_decomposition_accuracy():
    """Verify decompose_homography recovers known translation, rotation, and scale."""
    w, h = 1920, 1080
    known_tx, known_ty = 15.0, -10.0
    known_angle_deg = 3.0
    known_scale = 1.05

    rad = math.radians(known_angle_deg)
    cos_a = math.cos(rad) * known_scale
    sin_a = math.sin(rad) * known_scale

    H = np.array([
        [cos_a, -sin_a, known_tx],
        [sin_a,  cos_a, known_ty],
        [0.0,    0.0,   1.0],
    ], dtype=np.float64)

    tx, ty, trans_norm, scale, rot_deg, persp = decompose_homography(H, w, h)

    assert abs(tx - known_tx) < 1e-4
    assert abs(ty - known_ty) < 1e-4
    assert abs(scale - known_scale) < 1e-4
    assert abs(rot_deg - known_angle_deg) < 1e-4
    assert persp < 1e-6


def test_identical_frames_yield_stable():
    """Identical reference and sample frames must be classified STABLE with 0 displacement."""
    img = _create_rich_feature_image(800, 600)
    t = StabilityThresholds()

    meas = assess_frame_pair(img, img.copy(), t)

    assert meas.decision == StabilityDecision.STABLE
    assert meas.inlier_count >= t.min_inliers
    assert meas.translation_magnitude_px < 0.5
    assert abs(meas.rotation_degrees) < 0.2
    assert abs(meas.scale_change) < 0.01
    assert len(meas.rejection_reasons) == 0


def test_minor_gaussian_noise_yields_stable():
    """Slight sensor noise or compression artifacts within threshold must remain STABLE."""
    img = _create_rich_feature_image(800, 600)
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    t = StabilityThresholds()
    meas = assess_frame_pair(img, noisy_img, t)

    assert meas.decision == StabilityDecision.STABLE
    assert meas.translation_magnitude_px < t.max_translation_px


def test_excessive_translation_yields_unstable():
    """Camera horizontal shift of 20px exceeds threshold (8px) and yields UNSTABLE."""
    img = _create_rich_feature_image(800, 600)
    M = np.float32([[1, 0, 20.0], [0, 1, 0.0]])
    shifted_img = cv2.warpAffine(img, M, (800, 600))

    t = StabilityThresholds(max_translation_px=8.0)
    meas = assess_frame_pair(img, shifted_img, t)

    assert meas.decision == StabilityDecision.UNSTABLE
    assert any("EXCESSIVE_TRANSLATION_PX" in r for r in meas.rejection_reasons)
    assert abs(meas.translation_px_x - 20.0) < 2.0


def test_excessive_rotation_yields_unstable():
    """Camera roll rotation of 3.5 degrees exceeds threshold (1.0 deg) and yields UNSTABLE."""
    img = _create_rich_feature_image(800, 600)
    center = (400, 300)
    M = cv2.getRotationMatrix2D(center, 3.5, 1.0)
    rotated_img = cv2.warpAffine(img, M, (800, 600))

    t = StabilityThresholds(max_rotation_deg=1.0)
    meas = assess_frame_pair(img, rotated_img, t)

    assert meas.decision == StabilityDecision.UNSTABLE
    assert any("EXCESSIVE_ROTATION" in r for r in meas.rejection_reasons)
    assert abs(abs(meas.rotation_degrees) - 3.5) < 0.5


def test_excessive_zoom_scale_yields_unstable():
    """Camera zoom in of 10% exceeds threshold (3%) and yields UNSTABLE."""
    img = _create_rich_feature_image(800, 600)
    center = (400, 300)
    M = cv2.getRotationMatrix2D(center, 0.0, 1.10)
    zoomed_img = cv2.warpAffine(img, M, (800, 600))

    t = StabilityThresholds(max_scale_change=0.03)
    meas = assess_frame_pair(img, zoomed_img, t)

    assert meas.decision == StabilityDecision.UNSTABLE
    assert any("EXCESSIVE_SCALE_CHANGE" in r for r in meas.rejection_reasons)
    assert abs(meas.scale_factor - 1.10) < 0.05


def test_perspective_keystone_warp_yields_unstable():
    """Perspective warp (e.g. camera tilt) yields UNSTABLE."""
    img = _create_rich_feature_image(800, 600)
    pts1 = np.float32([[0, 0], [800, 0], [0, 600], [800, 600]])
    pts2 = np.float32([[40, 20], [760, 5], [10, 590], [790, 580]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (800, 600))

    t = StabilityThresholds()
    meas = assess_frame_pair(img, warped, t)

    assert meas.decision == StabilityDecision.UNSTABLE


def test_blank_featureless_frame_yields_insufficient_evidence():
    """Featureless uniform black or white frame must return INSUFFICIENT_EVIDENCE."""
    ref_img = _create_rich_feature_image(800, 600)
    blank_img = np.zeros((600, 800), dtype=np.uint8)

    t = StabilityThresholds()
    meas = assess_frame_pair(ref_img, blank_img, t)

    assert meas.decision == StabilityDecision.INSUFFICIENT_EVIDENCE
    assert any("INSUFFICIENT" in r or "LOW_MATCH" in r for r in meas.rejection_reasons)


def test_operational_gate_derivation():
    """Test pure evaluate_operational_gate logic."""
    ref_sha = "a" * 64
    layout_sha = "b" * 64

    # 1. Perfect STABLE assessment with matching SHAs -> ALLOWED
    gate, reasons = evaluate_operational_gate(
        decision=StabilityDecision.STABLE,
        assessment_reference_sha=ref_sha,
        current_reference_sha=ref_sha,
        assessment_layout_sha=layout_sha,
        current_layout_sha=layout_sha,
    )
    assert gate == OperationalGate.ALLOWED
    assert len(reasons) == 1

    # 2. UNSTABLE assessment -> BLOCKED
    gate, reasons = evaluate_operational_gate(
        decision=StabilityDecision.UNSTABLE,
        assessment_reference_sha=ref_sha,
        current_reference_sha=ref_sha,
        assessment_layout_sha=layout_sha,
        current_layout_sha=layout_sha,
    )
    assert gate == OperationalGate.BLOCKED
    assert any("UNSTABLE" in r for r in reasons)

    # 3. INSUFFICIENT_EVIDENCE -> BLOCKED
    gate, reasons = evaluate_operational_gate(
        decision=StabilityDecision.INSUFFICIENT_EVIDENCE,
        assessment_reference_sha=ref_sha,
        current_reference_sha=ref_sha,
        assessment_layout_sha=layout_sha,
        current_layout_sha=layout_sha,
    )
    assert gate == OperationalGate.BLOCKED

    # 4. Reference SHA mismatch -> BLOCKED
    gate, reasons = evaluate_operational_gate(
        decision=StabilityDecision.STABLE,
        assessment_reference_sha=ref_sha,
        current_reference_sha="different_sha" * 4,
        assessment_layout_sha=layout_sha,
        current_layout_sha=layout_sha,
    )
    assert gate == OperationalGate.BLOCKED
    assert any("STALE_REFERENCE_SHA" in r for r in reasons)

    # 5. Layout SHA mismatch -> BLOCKED
    gate, reasons = evaluate_operational_gate(
        decision=StabilityDecision.STABLE,
        assessment_reference_sha=ref_sha,
        current_reference_sha=ref_sha,
        assessment_layout_sha=layout_sha,
        current_layout_sha="new_layout_sha" * 4,
    )
    assert gate == OperationalGate.BLOCKED
    assert any("STALE_LAYOUT_SHA" in r for r in reasons)

    # 6. None decision -> BLOCKED
    gate, reasons = evaluate_operational_gate(
        decision=None,
        assessment_reference_sha=None,
        current_reference_sha=ref_sha,
        assessment_layout_sha=None,
        current_layout_sha=layout_sha,
    )
    assert gate == OperationalGate.BLOCKED


def test_synthetic_video_stability_assessment(tmp_path):
    """Generate short synthetic videos and evaluate with evaluate_video_camera_stability."""
    # 1. Create reference image
    ref_img = _create_rich_feature_image(640, 480)
    _, ref_buf = cv2.imencode(".jpg", ref_img)
    ref_bytes = ref_buf.tobytes()
    ref_sha = hashlib.sha256(ref_bytes).hexdigest()

    # 2. Generate stationary synthetic video
    stationary_video_path = tmp_path / "stationary.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(stationary_video_path), fourcc, 30.0, (640, 480), isColor=False)
    for _ in range(60):  # 2 seconds
        out.write(ref_img)
    out.release()

    res_stationary = evaluate_video_camera_stability(
        video_path=stationary_video_path,
        reference_image_bytes=ref_bytes,
        expected_reference_sha256=ref_sha,
        active_layout_canonical_sha256="layout_sha_123",
        sample_count=3,
    )
    assert res_stationary.aggregate_decision == StabilityDecision.STABLE
    assert res_stationary.operational_gate == OperationalGate.ALLOWED
    assert res_stationary.summary_metrics["max_translation_magnitude_px"] < 1.0

    # 3. Generate panning/moving synthetic video
    moving_video_path = tmp_path / "moving.mp4"
    out_moving = cv2.VideoWriter(str(moving_video_path), fourcc, 30.0, (640, 480), isColor=False)
    for i in range(60):
        # Shift 0.5px per frame = 30px shift by frame 60
        shift_x = float(i * 0.5)
        M = np.float32([[1, 0, shift_x], [0, 1, 0]])
        frame = cv2.warpAffine(ref_img, M, (640, 480))
        out_moving.write(frame)
    out_moving.release()

    res_moving = evaluate_video_camera_stability(
        video_path=moving_video_path,
        reference_image_bytes=ref_bytes,
        expected_reference_sha256=ref_sha,
        active_layout_canonical_sha256="layout_sha_123",
        sample_count=3,
    )
    assert res_moving.aggregate_decision == StabilityDecision.UNSTABLE
    assert res_moving.operational_gate == OperationalGate.BLOCKED


def test_alembic_stability_migration_upgrade_downgrade():
    """Verify Alembic migration upgrade to head, downgrade one revision, upgrade to head with stability table."""
    from alembic import command
    from alembic.config import Config

    tmp_dir = tempfile.mkdtemp(prefix="roadsense_stability_alembic_")
    db_path = Path(tmp_dir) / "stability_test.db"
    db_url = f"sqlite:///{db_path}"

    alembic_ini_path = Path(__file__).resolve().parent.parent / "services" / "api" / "alembic.ini"
    alembic_cfg = Config(str(alembic_ini_path))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    try:
        # Upgrade to head (applies f6a7b8c9d0e1)
        command.upgrade(alembic_cfg, "head")

        # Downgrade one revision
        command.downgrade(alembic_cfg, "-1")

        # Re-upgrade to head
        command.upgrade(alembic_cfg, "head")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_real_parking_video_read_only_negative_check():
    """
    Read-only integration test on actual local test video.

    Expectation: Camera drifts and pans, so it must be classified as UNSTABLE
    and operational gate must be BLOCKED.
    """
    video_path = Path("/Users/aniket/Downloads/PARKING LOT TEST.mp4")
    ref_path = Path("/Users/aniket/Downloads/parking_reference.jpg")

    if not video_path.exists() or not ref_path.exists():
        pytest.skip("Local media files not present at expected paths.")

    with open(ref_path, "rb") as rf:
        ref_bytes = rf.read()
    ref_sha = hashlib.sha256(ref_bytes).hexdigest()

    result = evaluate_video_camera_stability(
        video_path=video_path,
        reference_image_bytes=ref_bytes,
        expected_reference_sha256=ref_sha,
        active_layout_canonical_sha256="test_layout_sha",
        sample_count=5,
    )

    # Must be UNSTABLE due to camera translation (>20px drift)
    assert result.aggregate_decision == StabilityDecision.UNSTABLE
    assert result.operational_gate == OperationalGate.BLOCKED
    assert result.summary_metrics["max_translation_magnitude_px"] > 10.0
    assert result.summary_metrics["unstable_samples_count"] > 0
