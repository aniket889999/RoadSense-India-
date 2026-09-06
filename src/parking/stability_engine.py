"""OpenCV-based Camera Stability Assessment Engine for RoadSense SiteOps.

Evaluates whether physical camera geometry remains stationary compared to a verified
reference image using ORB feature detection, RANSAC homography, and mathematical
motion decomposition (translation, rotation, scale, perspective distortion, reprojection error).
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from src.parking.contracts import OperationalGate, StabilityDecision, StabilityThresholds, evaluate_operational_gate
from src.parking.stability_config import StabilityConfig, StabilityIntakeConfig, load_stability_config

# Module metadata provenance
STABILITY_ALGORITHM_VERSION = "2.0.0-orb-ransac-decomposed"


@dataclass
class SampleMeasurement:
    """Individual sampled video frame stability metrics."""
    sample_index: int
    timestamp_seconds: float
    frame_index: int
    matched_features: int
    inlier_count: int
    inlier_ratio: float
    translation_px_x: float
    translation_px_y: float
    translation_magnitude_px: float
    translation_normalized: float
    scale_factor: float
    scale_change: float
    rotation_degrees: float
    perspective_distortion: float
    reprojection_error: float
    decision: StabilityDecision
    rejection_reasons: List[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


@dataclass
class StabilityAssessmentResult:
    """Complete multi-sample video stability assessment outcome."""
    video_sha256: str
    reference_image_sha256: str
    layout_canonical_sha256: Optional[str]
    algorithm_version: str
    opencv_version: str
    config_version: str
    config_sha256: str
    thresholds_snapshot: dict[str, Any]
    samples: List[SampleMeasurement]
    summary_metrics: dict[str, Any]
    aggregate_decision: StabilityDecision
    operational_gate: OperationalGate
    gate_reasons: List[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_sha256": self.video_sha256,
            "reference_image_sha256": self.reference_image_sha256,
            "layout_canonical_sha256": self.layout_canonical_sha256,
            "algorithm_version": self.algorithm_version,
            "opencv_version": self.opencv_version,
            "config_version": self.config_version,
            "config_sha256": self.config_sha256,
            "thresholds_snapshot": self.thresholds_snapshot,
            "samples": [s.to_dict() for s in self.samples],
            "summary_metrics": self.summary_metrics,
            "aggregate_decision": self.aggregate_decision.value,
            "operational_gate": self.operational_gate.value,
            "gate_reasons": self.gate_reasons,
        }


def decompose_homography(H: np.ndarray, img_width: int, img_height: int) -> Tuple[float, float, float, float, float, float]:
    """
    Decompose a 3x3 homography matrix into geometric transform components.

    Returns:
      (tx_px, ty_px, translation_normalized, scale_factor, rotation_deg, perspective_distortion)
    """
    # Normalize H so H[2, 2] == 1.0
    if abs(H[2, 2]) > 1e-9:
        H_norm = H / H[2, 2]
    else:
        H_norm = H

    tx = float(H_norm[0, 2])
    ty = float(H_norm[1, 2])

    diag_dim = math.hypot(img_width, img_height) if (img_width > 0 and img_height > 0) else 1.0
    translation_norm = math.hypot(tx, ty) / diag_dim

    # Scale factor from affine approximation components
    scale_x = math.hypot(H_norm[0, 0], H_norm[1, 0])
    scale_y = math.hypot(H_norm[0, 1], H_norm[1, 1])
    scale_factor = (scale_x + scale_y) / 2.0

    # Rotation angle from first column
    rotation_rad = math.atan2(float(H_norm[1, 0]), float(H_norm[0, 0]))
    rotation_deg = math.degrees(rotation_rad)

    # Perspective distortion magnitude
    perspective = math.hypot(float(H_norm[2, 0]), float(H_norm[2, 1]))

    return tx, ty, translation_norm, scale_factor, rotation_deg, perspective


def compute_reprojection_error(H: np.ndarray, pts_src: np.ndarray, pts_dst: np.ndarray, inlier_mask: np.ndarray) -> float:
    """Compute mean Euclidean reprojection error across RANSAC inliers."""
    if inlier_mask is None or len(pts_src) == 0:
        return 0.0

    inlier_indices = np.where(inlier_mask.ravel() == 1)[0]
    if len(inlier_indices) == 0:
        return 0.0

    src_inliers = pts_src[inlier_indices]
    dst_inliers = pts_dst[inlier_indices]

    # Project src points through H
    projected = cv2.perspectiveTransform(src_inliers, H)
    errors = np.linalg.norm(projected - dst_inliers, axis=2)
    return float(np.mean(errors))


def assess_frame_pair(
    ref_gray: np.ndarray,
    sample_gray: np.ndarray,
    thresholds: StabilityThresholds,
    sample_idx: int = 0,
    timestamp_sec: float = 0.0,
    frame_idx: int = 0,
) -> SampleMeasurement:
    """
    Compare a single sampled grayscale frame against the reference frame using ORB & RANSAC.
    """
    h, w = ref_gray.shape[:2]
    rejection_reasons: List[str] = []

    # ORB feature detector
    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=15)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    kp_sample, des_sample = orb.detectAndCompute(sample_gray, None)

    if des_ref is None or des_sample is None or len(kp_ref) < thresholds.min_matches or len(kp_sample) < thresholds.min_matches:
        return SampleMeasurement(
            sample_index=sample_idx,
            timestamp_seconds=timestamp_sec,
            frame_index=frame_idx,
            matched_features=0 if (des_ref is None or des_sample is None) else min(len(kp_ref), len(kp_sample)),
            inlier_count=0,
            inlier_ratio=0.0,
            translation_px_x=0.0,
            translation_px_y=0.0,
            translation_magnitude_px=0.0,
            translation_normalized=0.0,
            scale_factor=1.0,
            scale_change=0.0,
            rotation_degrees=0.0,
            perspective_distortion=0.0,
            reprojection_error=0.0,
            decision=StabilityDecision.INSUFFICIENT_EVIDENCE,
            rejection_reasons=["INSUFFICIENT_FEATURES: Too few keypoints detected in frame pair."],
        )

    # Cross-check BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_ref, des_sample)

    if len(matches) < thresholds.min_matches:
        return SampleMeasurement(
            sample_index=sample_idx,
            timestamp_seconds=timestamp_sec,
            frame_index=frame_idx,
            matched_features=len(matches),
            inlier_count=0,
            inlier_ratio=0.0,
            translation_px_x=0.0,
            translation_px_y=0.0,
            translation_magnitude_px=0.0,
            translation_normalized=0.0,
            scale_factor=1.0,
            scale_change=0.0,
            rotation_degrees=0.0,
            perspective_distortion=0.0,
            reprojection_error=0.0,
            decision=StabilityDecision.INSUFFICIENT_EVIDENCE,
            rejection_reasons=[f"LOW_MATCH_COUNT: Only {len(matches)} matches found (minimum {thresholds.min_matches})."],
        )

    pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_sample = np.float32([kp_sample[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_ref, pts_sample, cv2.RANSAC, 3.0)

    if H is None or mask is None:
        return SampleMeasurement(
            sample_index=sample_idx,
            timestamp_seconds=timestamp_sec,
            frame_index=frame_idx,
            matched_features=len(matches),
            inlier_count=0,
            inlier_ratio=0.0,
            translation_px_x=0.0,
            translation_px_y=0.0,
            translation_magnitude_px=0.0,
            translation_normalized=0.0,
            scale_factor=1.0,
            scale_change=0.0,
            rotation_degrees=0.0,
            perspective_distortion=0.0,
            reprojection_error=0.0,
            decision=StabilityDecision.INSUFFICIENT_EVIDENCE,
            rejection_reasons=["HOMOGRAPHY_ESTIMATION_FAILED: RANSAC failed to find consensus model."],
        )

    inliers = int(mask.sum())
    inlier_ratio = float(inliers / len(matches)) if len(matches) > 0 else 0.0

    if inliers < thresholds.min_inliers or inlier_ratio < thresholds.min_inlier_ratio:
        return SampleMeasurement(
            sample_index=sample_idx,
            timestamp_seconds=timestamp_sec,
            frame_index=frame_idx,
            matched_features=len(matches),
            inlier_count=inliers,
            inlier_ratio=inlier_ratio,
            translation_px_x=0.0,
            translation_px_y=0.0,
            translation_magnitude_px=0.0,
            translation_normalized=0.0,
            scale_factor=1.0,
            scale_change=0.0,
            rotation_degrees=0.0,
            perspective_distortion=0.0,
            reprojection_error=0.0,
            decision=StabilityDecision.INSUFFICIENT_EVIDENCE,
            rejection_reasons=[
                f"INSUFFICIENT_INLIERS: Found {inliers} inliers ({inlier_ratio:.2%}), below minimum requirement."
            ],
        )

    tx, ty, trans_norm, scale_factor, rot_deg, perspective = decompose_homography(H, w, h)
    trans_mag = math.hypot(tx, ty)
    scale_change = abs(scale_factor - 1.0)
    reproj_err = compute_reprojection_error(H, pts_ref, pts_sample, mask)

    # Threshold checks
    if trans_mag > thresholds.max_translation_px:
        rejection_reasons.append(
            f"EXCESSIVE_TRANSLATION_PX: {trans_mag:.2f}px displacement exceeds threshold {thresholds.max_translation_px}px."
        )
    if trans_norm > thresholds.max_translation_norm:
        rejection_reasons.append(
            f"EXCESSIVE_TRANSLATION_NORM: {trans_norm:.4f} normalized exceeds threshold {thresholds.max_translation_norm}."
        )
    if abs(rot_deg) > thresholds.max_rotation_deg:
        rejection_reasons.append(
            f"EXCESSIVE_ROTATION: {abs(rot_deg):.2f}° exceeds threshold {thresholds.max_rotation_deg}°."
        )
    if scale_change > thresholds.max_scale_change:
        rejection_reasons.append(
            f"EXCESSIVE_SCALE_CHANGE: {scale_change:.2%} zoom/scale change exceeds threshold {thresholds.max_scale_change:.2%}."
        )
    if perspective > thresholds.max_perspective_distortion:
        rejection_reasons.append(
            f"EXCESSIVE_PERSPECTIVE: {perspective:.6f} perspective distortion exceeds threshold {thresholds.max_perspective_distortion}."
        )
    if reproj_err > thresholds.max_reprojection_error:
        rejection_reasons.append(
            f"EXCESSIVE_REPROJECTION_ERROR: {reproj_err:.2f}px reprojection error exceeds threshold {thresholds.max_reprojection_error}px."
        )

    decision = StabilityDecision.UNSTABLE if rejection_reasons else StabilityDecision.STABLE

    return SampleMeasurement(
        sample_index=sample_idx,
        timestamp_seconds=timestamp_sec,
        frame_index=frame_idx,
        matched_features=len(matches),
        inlier_count=inliers,
        inlier_ratio=inlier_ratio,
        translation_px_x=tx,
        translation_px_y=ty,
        translation_magnitude_px=trans_mag,
        translation_normalized=trans_norm,
        scale_factor=scale_factor,
        scale_change=scale_change,
        rotation_degrees=rot_deg,
        perspective_distortion=perspective,
        reprojection_error=reproj_err,
        decision=decision,
        rejection_reasons=rejection_reasons,
    )


def inspect_video_media_safe(
    video_path: Path,
    intake_config: Optional[StabilityIntakeConfig] = None,
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    """
    Inspect media file using ffprobe argument array without shell=True.
    Validates file size, non-emptiness, codec, resolution, duration, and stream properties.
    """
    intake = intake_config or StabilityIntakeConfig()

    if video_path.is_symlink():
        raise ValueError("Symlink video paths are not permitted for security reasons.")

    if not video_path.exists() or not video_path.is_file():
        raise ValueError(f"Video path does not exist or is not a regular file: {video_path}")

    file_size = video_path.stat().st_size
    if file_size == 0:
        raise ValueError("Uploaded media file is empty (0 bytes).")
    if file_size > intake.max_upload_bytes:
        raise ValueError(
            f"Video file size ({file_size} bytes / {file_size / (1024*1024):.1f}MB) exceeds configured limit "
            f"({intake.max_upload_bytes} bytes / {intake.max_upload_bytes / (1024*1024):.1f}MB)."
        )

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration,size,nb_streams",
        "-show_entries", "stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames",
        "-of", "json",
        str(video_path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=timeout_sec,
            check=True,
        )
        data = json.loads(proc.stdout)
    except subprocess.TimeoutExpired as te:
        raise ValueError(f"ffprobe timed out after {timeout_sec}s inspecting media: {te}")
    except Exception as e:
        raise ValueError(f"ffprobe failed to inspect media: {e}")

    streams = data.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        raise ValueError("Media file contains no valid video streams.")
    if len(video_streams) > 1:
        raise ValueError(f"Media file contains {len(video_streams)} video streams; expected exactly 1.")

    v_stream = video_streams[0]
    codec = str(v_stream.get("codec_name", "unknown")).lower()
    if codec not in intake.allowed_codecs:
        raise ValueError(
            f"Unsupported video codec '{codec}'. Allowed codecs are: {', '.join(intake.allowed_codecs)}."
        )

    width = int(v_stream.get("width", 0))
    height = int(v_stream.get("height", 0))
    if width < intake.min_width or width > intake.max_width or height < intake.min_height or height > intake.max_height:
        raise ValueError(
            f"Video resolution ({width}x{height}) is outside allowed bounds "
            f"([{intake.min_width}..{intake.max_width}] x [{intake.min_height}..{intake.max_height}])."
        )

    fmt = data.get("format", {})
    try:
        duration = float(fmt.get("duration", 0.0))
    except (ValueError, TypeError):
        duration = 0.0

    if duration <= 0.0:
        raise ValueError("Could not determine a valid positive video duration from container metadata.")
    if duration > intake.max_duration_seconds:
        raise ValueError(
            f"Video duration ({duration:.2f}s) exceeds max allowed assessment duration ({intake.max_duration_seconds}s)."
        )

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "codec": codec,
        "nb_frames": int(v_stream.get("nb_frames", 0)) if v_stream.get("nb_frames") else None,
        "file_size": file_size,
    }


def evaluate_video_camera_stability(
    video_path: Path,
    reference_image_bytes: bytes,
    expected_reference_sha256: str,
    active_layout_canonical_sha256: Optional[str] = None,
    config: Optional[StabilityConfig] = None,
    sample_count: Optional[int] = None,
    progress_callback: Optional[Any] = None,
) -> StabilityAssessmentResult:
    """
    Execute complete fail-closed camera stability assessment against a video file using versioned configuration.
    """
    cfg = config or load_stability_config()
    t = cfg.thresholds
    intake = cfg.intake

    if progress_callback:
        progress_callback(10.0, "Validating video media container and metadata")

    # Inspect media safely
    media_info = inspect_video_media_safe(
        video_path=video_path,
        intake_config=intake,
        timeout_sec=cfg.execution.ffprobe_timeout_seconds,
    )

    if progress_callback:
        progress_callback(20.0, "Decoding reference image and initializing feature detector")

    # Verify reference image bytes
    computed_ref_sha = hashlib.sha256(reference_image_bytes).hexdigest()
    ref_arr = np.frombuffer(reference_image_bytes, dtype=np.uint8)
    ref_img = cv2.imdecode(ref_arr, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        raise ValueError("Failed to decode reference image bytes into grayscale image.")

    # Hash video file in 64KB chunks
    h_vid = hashlib.sha256()
    with open(video_path, "rb") as vf:
        while chunk := vf.read(65536):
            h_vid.update(chunk)
    video_sha = h_vid.hexdigest()

    # Open video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file with OpenCV: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if total_frames <= 0:
        total_frames = int(media_info["duration"] * fps)

    effective_sample_count = sample_count if sample_count is not None else intake.sample_count
    sample_count_val = max(1, min(20, effective_sample_count))
    if total_frames <= sample_count_val:
        sample_indices = list(range(total_frames))
    else:
        step = total_frames / (sample_count_val + 1)
        sample_indices = [int(step * (i + 1)) for i in range(sample_count_val)]

    samples: List[SampleMeasurement] = []

    try:
        for idx, target_frame_idx in enumerate(sample_indices):
            if progress_callback:
                pct = 20.0 + (70.0 * (idx + 1) / max(1, len(sample_indices)))
                progress_callback(pct, f"Evaluating stability sample {idx + 1}/{len(sample_indices)} (frame {target_frame_idx})")

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame = cap.read()
            timestamp_sec = float(target_frame_idx / fps)

            if not ret or frame is None:
                samples.append(
                    SampleMeasurement(
                        sample_index=idx,
                        timestamp_seconds=timestamp_sec,
                        frame_index=target_frame_idx,
                        matched_features=0,
                        inlier_count=0,
                        inlier_ratio=0.0,
                        translation_px_x=0.0,
                        translation_px_y=0.0,
                        translation_magnitude_px=0.0,
                        translation_normalized=0.0,
                        scale_factor=1.0,
                        scale_change=0.0,
                        rotation_degrees=0.0,
                        perspective_distortion=0.0,
                        reprojection_error=0.0,
                        decision=StabilityDecision.ERROR,
                        rejection_reasons=[f"FRAME_DECODE_FAILED: Frame {target_frame_idx} could not be read."],
                    )
                )
                continue

            sample_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            meas = assess_frame_pair(ref_img, sample_gray, t, idx, timestamp_sec, target_frame_idx)
            samples.append(meas)
    finally:
        cap.release()

    # Conservative aggregation:
    # 1. Any UNSTABLE sample -> UNSTABLE
    # 2. Any ERROR sample -> ERROR
    # 3. Any INSUFFICIENT_EVIDENCE sample -> INSUFFICIENT_EVIDENCE
    # 4. Otherwise -> STABLE
    has_unstable = any(s.decision == StabilityDecision.UNSTABLE for s in samples)
    has_error = any(s.decision == StabilityDecision.ERROR for s in samples)
    has_insufficient = any(s.decision == StabilityDecision.INSUFFICIENT_EVIDENCE for s in samples)

    if has_unstable:
        aggregate_decision = StabilityDecision.UNSTABLE
    elif has_error:
        aggregate_decision = StabilityDecision.ERROR
    elif has_insufficient:
        aggregate_decision = StabilityDecision.INSUFFICIENT_EVIDENCE
    else:
        aggregate_decision = StabilityDecision.STABLE

    # Compute summary metrics
    summary_metrics = {
        "max_translation_magnitude_px": max((s.translation_magnitude_px for s in samples), default=0.0),
        "max_translation_normalized": max((s.translation_normalized for s in samples), default=0.0),
        "max_scale_change": max((s.scale_change for s in samples), default=0.0),
        "max_rotation_degrees": max((abs(s.rotation_degrees) for s in samples), default=0.0),
        "max_perspective_distortion": max((s.perspective_distortion for s in samples), default=0.0),
        "min_inlier_ratio": min((s.inlier_ratio for s in samples if s.inlier_ratio > 0), default=0.0),
        "max_reprojection_error": max((s.reprojection_error for s in samples), default=0.0),
        "total_samples": len(samples),
        "stable_samples_count": sum(1 for s in samples if s.decision == StabilityDecision.STABLE),
        "unstable_samples_count": sum(1 for s in samples if s.decision == StabilityDecision.UNSTABLE),
        "insufficient_samples_count": sum(1 for s in samples if s.decision == StabilityDecision.INSUFFICIENT_EVIDENCE),
    }

    if progress_callback:
        progress_callback(95.0, "Aggregating geometric metrics and evaluating operational gate")

    # Evaluate operational gate
    gate, gate_reasons = evaluate_operational_gate(
        decision=aggregate_decision,
        assessment_reference_sha=computed_ref_sha,
        current_reference_sha=expected_reference_sha256,
        assessment_layout_sha=active_layout_canonical_sha256,
        current_layout_sha=active_layout_canonical_sha256,
        max_age_seconds=t.max_assessment_age_seconds,
    )

    if progress_callback:
        progress_callback(100.0, "Camera stability assessment complete")

    return StabilityAssessmentResult(
        video_sha256=video_sha,
        reference_image_sha256=computed_ref_sha,
        layout_canonical_sha256=active_layout_canonical_sha256,
        algorithm_version=cfg.algorithm_version,
        opencv_version=cv2.__version__,
        config_version=cfg.version,
        config_sha256=cfg.config_sha256,
        thresholds_snapshot=cfg.thresholds_snapshot,
        samples=samples,
        summary_metrics=summary_metrics,
        aggregate_decision=aggregate_decision,
        operational_gate=gate,
        gate_reasons=gate_reasons,
    )
