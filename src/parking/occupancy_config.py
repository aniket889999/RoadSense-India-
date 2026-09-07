"""Strict loader and validator for versioned parking occupancy configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass(frozen=True)
class DetectorConfig:
    """Validated parameters for local vehicle detection."""
    model_path: str
    expected_model_sha256: str = ""
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    allowed_classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])
    class_names: Dict[int, str] = field(default_factory=lambda: {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"})
    device: str = "cpu"


@dataclass(frozen=True)
class TrackingConfig:
    """Validated parameters for ByteTrack temporal tracking."""
    enabled: bool
    track_high_thresh: float
    track_low_thresh: float
    new_track_thresh: float
    track_buffer: int
    match_thresh: float


@dataclass(frozen=True)
class ScoringConfig:
    """Validated parameters for geometric polygon overlap and occupancy scoring."""
    min_bay_coverage_ratio: float
    min_vehicle_overlap_ratio: float
    high_coverage_threshold: float
    min_detection_confidence: float
    require_center_or_coverage: bool


@dataclass(frozen=True)
class TemporalConfig:
    """Validated parameters for temporal hysteresis and anti-flicker state machine."""
    min_frames_occupied: int
    min_frames_vacant: int
    dropout_tolerance_frames: int


@dataclass(frozen=True)
class RenderConfig:
    """Visual style definitions for annotated video output."""
    vacant_color_bgr: List[int]
    vacant_fill_alpha: float
    occupied_color_bgr: List[int]
    occupied_fill_alpha: float
    unknown_color_bgr: List[int]
    unknown_fill_alpha: float
    occluded_color_bgr: List[int]
    occluded_fill_alpha: float
    vehicle_box_bgr: List[int]
    draw_vehicle_boxes: bool
    draw_track_ids: bool
    draw_bay_labels: bool
    draw_summary_hud: bool


@dataclass(frozen=True)
class ExecutionConfig:
    """Runtime limits and execution bounds."""
    max_concurrent_jobs: int
    max_upload_bytes: int
    max_duration_seconds: float
    ffprobe_timeout_seconds: float
    ffmpeg_timeout_seconds: float


@dataclass(frozen=True)
class ParkingOccupancyConfig:
    """Validated immutable root parking occupancy configuration."""
    version: str
    algorithm_version: str
    config_sha256: str
    detector: DetectorConfig
    tracking: TrackingConfig
    scoring: ScoringConfig
    temporal: TemporalConfig
    rendering: RenderConfig
    execution: ExecutionConfig
    snapshot: Dict[str, Any] = field(default_factory=dict)


def _validate_float_field(val: Any, name: str, min_val: float = 0.0, max_val: float = 1.0) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"Config field '{name}' must be a float, got {type(val).__name__} ({val})")
    f_val = float(val)
    if not math.isfinite(f_val):
        raise ValueError(f"Config field '{name}' must be finite, got {f_val}")
    if f_val < min_val or f_val > max_val:
        raise ValueError(f"Config field '{name}' ({f_val}) must be in range [{min_val}, {max_val}]")
    return f_val


def _validate_int_field(val: Any, name: str, min_val: int = 0, max_val: int = 1000000000) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"Config field '{name}' must be an integer, got {type(val).__name__} ({val})")
    if val < min_val or val > max_val:
        raise ValueError(f"Config field '{name}' ({val}) must be in range [{min_val}, {max_val}]")
    return int(val)


def _validate_color_bgr(val: Any, name: str) -> List[int]:
    if not isinstance(val, (list, tuple)) or len(val) != 3:
        raise ValueError(f"Config color '{name}' must be a 3-element list [B, G, R], got {val}")
    res = []
    for i, c in enumerate(val):
        if isinstance(c, bool) or not isinstance(c, int) or c < 0 or c > 255:
            raise ValueError(f"Config color '{name}[{i}]' must be integer in [0, 255], got {c}")
        res.append(int(c))
    return res


def load_parking_occupancy_config(config_path: Optional[Path | str] = None) -> ParkingOccupancyConfig:
    """
    Load and strictly validate versioned parking occupancy configuration YAML.
    Computes canonical SHA-256 checksum over raw file content.
    """
    if config_path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        config_path = base_dir / "configs" / "parking" / "parking_occupancy_v1.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.is_file() or config_path.is_symlink():
        raise FileNotFoundError(f"Parking occupancy config file missing or invalid: {config_path}")

    raw_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    raw_text = raw_bytes.decode("utf-8")
    data = yaml.safe_load(raw_text)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML content in config file {config_path}: expected dictionary root.")

    version = str(data.get("version", "")).strip()
    if not version:
        raise ValueError("Config missing required 'version' string.")

    algorithm_version = str(data.get("algorithm_version", "")).strip()
    if not algorithm_version:
        raise ValueError("Config missing required 'algorithm_version' string.")

    # Detector
    det = data.get("detector", {})
    if not isinstance(det, dict):
        raise ValueError("Config 'detector' section must be a dictionary.")
    model_path = str(det.get("model_path", "")).strip()
    if not model_path:
        raise ValueError("Detector missing 'model_path'.")
    expected_model_sha256 = str(det.get("expected_model_sha256", "")).strip()
    if not expected_model_sha256 or len(expected_model_sha256) != 64:
        raise ValueError("Detector missing valid 64-char 'expected_model_sha256'.")

    det_conf = _validate_float_field(det.get("confidence_threshold"), "detector.confidence_threshold", 0.05, 1.0)
    det_iou = _validate_float_field(det.get("iou_threshold"), "detector.iou_threshold", 0.05, 1.0)
    raw_classes = det.get("allowed_classes", [])
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ValueError("Detector 'allowed_classes' must be a non-empty list of integer class IDs.")
    allowed_classes = [_validate_int_field(c, "detector.allowed_classes[]", 0, 1000) for c in raw_classes]
    raw_names = det.get("class_names", {})
    if not isinstance(raw_names, dict):
        raise ValueError("Detector 'class_names' must be a dictionary.")
    class_names = {int(k): str(v).strip() for k, v in raw_names.items()}
    device = str(det.get("device", "cpu")).strip()

    detector_cfg = DetectorConfig(
        model_path=model_path,
        expected_model_sha256=expected_model_sha256,
        confidence_threshold=det_conf,
        iou_threshold=det_iou,
        allowed_classes=allowed_classes,
        class_names=class_names,
        device=device,
    )

    # Tracking
    trk = data.get("tracking", {})
    if not isinstance(trk, dict):
        raise ValueError("Config 'tracking' section must be a dictionary.")
    trk_enabled = bool(trk.get("enabled", True))
    trk_high = _validate_float_field(trk.get("track_high_thresh"), "tracking.track_high_thresh", 0.05, 1.0)
    trk_low = _validate_float_field(trk.get("track_low_thresh"), "tracking.track_low_thresh", 0.01, trk_high)
    trk_new = _validate_float_field(trk.get("new_track_thresh"), "tracking.new_track_thresh", trk_low, 1.0)
    trk_buf = _validate_int_field(trk.get("track_buffer"), "tracking.track_buffer", 1, 300)
    trk_match = _validate_float_field(trk.get("match_thresh"), "tracking.match_thresh", 0.1, 1.0)

    tracking_cfg = TrackingConfig(
        enabled=trk_enabled,
        track_high_thresh=trk_high,
        track_low_thresh=trk_low,
        new_track_thresh=trk_new,
        track_buffer=trk_buf,
        match_thresh=trk_match,
    )

    # Scoring
    sc = data.get("scoring", {})
    if not isinstance(sc, dict):
        raise ValueError("Config 'scoring' section must be a dictionary.")
    min_bay_cov = _validate_float_field(sc.get("min_bay_coverage_ratio"), "scoring.min_bay_coverage_ratio", 0.05, 1.0)
    min_veh_ov = _validate_float_field(sc.get("min_vehicle_overlap_ratio"), "scoring.min_vehicle_overlap_ratio", 0.05, 1.0)
    high_cov = _validate_float_field(sc.get("high_coverage_threshold"), "scoring.high_coverage_threshold", min_bay_cov, 1.0)
    min_det_conf = _validate_float_field(sc.get("min_detection_confidence"), "scoring.min_detection_confidence", 0.05, 1.0)
    req_center = bool(sc.get("require_center_or_coverage", True))

    scoring_cfg = ScoringConfig(
        min_bay_coverage_ratio=min_bay_cov,
        min_vehicle_overlap_ratio=min_veh_ov,
        high_coverage_threshold=high_cov,
        min_detection_confidence=min_det_conf,
        require_center_or_coverage=req_center,
    )

    # Temporal
    tmp = data.get("temporal", {})
    if not isinstance(tmp, dict):
        raise ValueError("Config 'temporal' section must be a dictionary.")
    min_occ = _validate_int_field(tmp.get("min_frames_occupied"), "temporal.min_frames_occupied", 1, 60)
    min_vac = _validate_int_field(tmp.get("min_frames_vacant"), "temporal.min_frames_vacant", 1, 60)
    drop_tol = _validate_int_field(tmp.get("dropout_tolerance_frames"), "temporal.dropout_tolerance_frames", 0, 15)

    temporal_cfg = TemporalConfig(
        min_frames_occupied=min_occ,
        min_frames_vacant=min_vac,
        dropout_tolerance_frames=drop_tol,
    )

    # Rendering
    ren = data.get("rendering", {})
    if not isinstance(ren, dict):
        raise ValueError("Config 'rendering' section must be a dictionary.")
    vac_col = _validate_color_bgr(ren.get("vacant_color_bgr"), "rendering.vacant_color_bgr")
    vac_alpha = _validate_float_field(ren.get("vacant_fill_alpha"), "rendering.vacant_fill_alpha", 0.0, 1.0)
    occ_col = _validate_color_bgr(ren.get("occupied_color_bgr"), "rendering.occupied_color_bgr")
    occ_alpha = _validate_float_field(ren.get("occupied_fill_alpha"), "rendering.occupied_fill_alpha", 0.0, 1.0)
    unk_col = _validate_color_bgr(ren.get("unknown_color_bgr"), "rendering.unknown_color_bgr")
    unk_alpha = _validate_float_field(ren.get("unknown_fill_alpha"), "rendering.unknown_fill_alpha", 0.0, 1.0)
    occld_col = _validate_color_bgr(ren.get("occluded_color_bgr"), "rendering.occluded_color_bgr")
    occld_alpha = _validate_float_field(ren.get("occluded_fill_alpha"), "rendering.occluded_fill_alpha", 0.0, 1.0)
    veh_col = _validate_color_bgr(ren.get("vehicle_box_bgr"), "rendering.vehicle_box_bgr")

    render_cfg = RenderConfig(
        vacant_color_bgr=vac_col,
        vacant_fill_alpha=vac_alpha,
        occupied_color_bgr=occ_col,
        occupied_fill_alpha=occ_alpha,
        unknown_color_bgr=unk_col,
        unknown_fill_alpha=unk_alpha,
        occluded_color_bgr=occld_col,
        occluded_fill_alpha=occld_alpha,
        vehicle_box_bgr=veh_col,
        draw_vehicle_boxes=bool(ren.get("draw_vehicle_boxes", True)),
        draw_track_ids=bool(ren.get("draw_track_ids", True)),
        draw_bay_labels=bool(ren.get("draw_bay_labels", True)),
        draw_summary_hud=bool(ren.get("draw_summary_hud", True)),
    )

    # Execution
    exe = data.get("execution", {})
    if not isinstance(exe, dict):
        raise ValueError("Config 'execution' section must be a dictionary.")
    max_jobs = _validate_int_field(exe.get("max_concurrent_jobs"), "execution.max_concurrent_jobs", 1, 10)
    max_upload = _validate_int_field(exe.get("max_upload_bytes"), "execution.max_upload_bytes", 1024 * 1024, 1024 * 1024 * 1024)
    max_dur = _validate_float_field(exe.get("max_duration_seconds"), "execution.max_duration_seconds", 5.0, 3600.0)
    ffprobe_to = _validate_float_field(exe.get("ffprobe_timeout_seconds"), "execution.ffprobe_timeout_seconds", 1.0, 300.0)
    ffmpeg_to = _validate_float_field(exe.get("ffmpeg_timeout_seconds"), "execution.ffmpeg_timeout_seconds", 5.0, 1800.0)

    execution_cfg = ExecutionConfig(
        max_concurrent_jobs=max_jobs,
        max_upload_bytes=max_upload,
        max_duration_seconds=max_dur,
        ffprobe_timeout_seconds=ffprobe_to,
        ffmpeg_timeout_seconds=ffmpeg_to,
    )

    # Create immutable snapshot dict
    snapshot = {
        "version": version,
        "algorithm_version": algorithm_version,
        "detector": {
            "model_path": detector_cfg.model_path,
            "expected_model_sha256": detector_cfg.expected_model_sha256,
            "confidence_threshold": detector_cfg.confidence_threshold,
            "iou_threshold": detector_cfg.iou_threshold,
            "allowed_classes": detector_cfg.allowed_classes,
        },
        "scoring": {
            "min_bay_coverage_ratio": scoring_cfg.min_bay_coverage_ratio,
            "min_vehicle_overlap_ratio": scoring_cfg.min_vehicle_overlap_ratio,
            "high_coverage_threshold": scoring_cfg.high_coverage_threshold,
            "min_detection_confidence": scoring_cfg.min_detection_confidence,
            "require_center_or_coverage": scoring_cfg.require_center_or_coverage,
        },
        "temporal": {
            "min_frames_occupied": temporal_cfg.min_frames_occupied,
            "min_frames_vacant": temporal_cfg.min_frames_vacant,
            "dropout_tolerance_frames": temporal_cfg.dropout_tolerance_frames,
        },
        "config_sha256": config_sha256,
    }

    return ParkingOccupancyConfig(
        version=version,
        algorithm_version=algorithm_version,
        config_sha256=config_sha256,
        detector=detector_cfg,
        tracking=tracking_cfg,
        scoring=scoring_cfg,
        temporal=temporal_cfg,
        rendering=render_cfg,
        execution=execution_cfg,
        snapshot=snapshot,
    )
