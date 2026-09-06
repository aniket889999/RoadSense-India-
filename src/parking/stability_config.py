"""Configuration loader and schema validator for Camera Stability Assessment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from src.parking.contracts import StabilityThresholds

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "stability" / "camera_stability_v1.yaml"


@dataclass(frozen=True)
class StabilityIntakeConfig:
    """Configured limits for video upload and container validation."""
    max_upload_bytes: int = 209715200 # 200 MB
    max_duration_seconds: float = 300.0
    min_width: int = 64
    max_width: int = 4096
    min_height: int = 64
    max_height: int = 4096
    sample_count: int = 5
    allowed_codecs: tuple[str, ...] = ("h264", "hevc", "h265", "vp8", "vp9", "av1", "mjpeg", "mpeg4")


@dataclass(frozen=True)
class StabilityExecutionConfig:
    """Runtime concurrency and timeout parameters for stability assessment jobs."""
    max_concurrent_jobs: int = 2
    ffprobe_timeout_seconds: float = 10.0


@dataclass(frozen=True)
class StabilityConfig:
    """Complete versioned and verified stability assessment configuration."""
    version: str
    algorithm_version: str
    config_sha256: str
    thresholds: StabilityThresholds
    intake: StabilityIntakeConfig
    execution: StabilityExecutionConfig
    thresholds_snapshot: Dict[str, Any]


def _check_strict_number(val: Any, field_name: str, allow_float: bool = True, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """Strict numeric validation rejecting booleans, non-finite values, and out-of-range numbers."""
    if isinstance(val, bool):
        raise ValueError(f"Field '{field_name}' cannot be a boolean; expected number.")
    if not isinstance(val, (int, float)):
        raise ValueError(f"Field '{field_name}' must be numeric; got {type(val).__name__}.")
    if not allow_float and isinstance(val, float) and not val.is_integer():
        raise ValueError(f"Field '{field_name}' must be an integer; got float {val}.")
    fval = float(val)
    if not math.isfinite(fval):
        raise ValueError(f"Field '{field_name}' must be finite; got {val}.")
    if min_val is not None and fval < min_val:
        raise ValueError(f"Field '{field_name}' value {val} is below minimum allowed ({min_val}).")
    if max_val is not None and fval > max_val:
        raise ValueError(f"Field '{field_name}' value {val} is above maximum allowed ({max_val}).")
    return fval


def _check_strict_string(val: Any, field_name: str) -> str:
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"Field '{field_name}' must be a non-empty string.")
    return val.strip()


def validate_stability_config_dict(raw: Dict[str, Any], raw_sha256: str) -> StabilityConfig:
    """Validate a raw configuration dictionary strictly against the stability schema."""
    if not isinstance(raw, dict):
        raise ValueError(f"Root configuration must be a dictionary, got {type(raw).__name__}")

    allowed_top_keys = {"version", "algorithm_version", "thresholds", "intake", "execution"}
    extra_top = set(raw.keys()) - allowed_top_keys
    if extra_top:
        raise ValueError(f"Unknown configuration keys found: {extra_top}")

    missing_top = allowed_top_keys - set(raw.keys())
    if missing_top:
        raise ValueError(f"Missing required configuration keys: {missing_top}")

    version = _check_strict_string(raw["version"], "version")
    algorithm_version = _check_strict_string(raw["algorithm_version"], "algorithm_version")

    # Thresholds validation
    t_raw = raw["thresholds"]
    if not isinstance(t_raw, dict):
        raise ValueError("Field 'thresholds' must be a dictionary.")

    allowed_t_keys = {
        "max_translation_px",
        "max_translation_norm",
        "max_rotation_deg",
        "max_scale_change",
        "max_perspective_distortion",
        "min_matches",
        "min_inliers",
        "min_inlier_ratio",
        "max_reprojection_error",
        "max_assessment_age_seconds",
    }
    extra_t = set(t_raw.keys()) - allowed_t_keys
    if extra_t:
        raise ValueError(f"Unknown keys in 'thresholds': {extra_t}")
    missing_t = allowed_t_keys - set(t_raw.keys())
    if missing_t:
        raise ValueError(f"Missing required keys in 'thresholds': {missing_t}")

    max_trans_px = _check_strict_number(t_raw["max_translation_px"], "thresholds.max_translation_px", min_val=0.1, max_val=1000.0)
    max_trans_norm = _check_strict_number(t_raw["max_translation_norm"], "thresholds.max_translation_norm", min_val=0.0001, max_val=1.0)
    max_rot_deg = _check_strict_number(t_raw["max_rotation_deg"], "thresholds.max_rotation_deg", min_val=0.01, max_val=180.0)
    max_scale = _check_strict_number(t_raw["max_scale_change"], "thresholds.max_scale_change", min_val=0.001, max_val=1.0)
    max_persp = _check_strict_number(t_raw["max_perspective_distortion"], "thresholds.max_perspective_distortion", min_val=0.000001, max_val=1.0)
    min_matches = int(_check_strict_number(t_raw["min_matches"], "thresholds.min_matches", allow_float=False, min_val=4, max_val=10000))
    min_inliers = int(_check_strict_number(t_raw["min_inliers"], "thresholds.min_inliers", allow_float=False, min_val=4, max_val=min_matches))
    min_inlier_ratio = _check_strict_number(t_raw["min_inlier_ratio"], "thresholds.min_inlier_ratio", min_val=0.01, max_val=1.0)
    max_reproj = _check_strict_number(t_raw["max_reprojection_error"], "thresholds.max_reprojection_error", min_val=0.1, max_val=100.0)
    max_age_sec = int(_check_strict_number(t_raw["max_assessment_age_seconds"], "thresholds.max_assessment_age_seconds", allow_float=False, min_val=60, max_val=31536000))

    thresholds = StabilityThresholds(
        max_translation_px=max_trans_px,
        max_translation_norm=max_trans_norm,
        max_rotation_deg=max_rot_deg,
        max_scale_change=max_scale,
        max_perspective_distortion=max_persp,
        min_matches=min_matches,
        min_inliers=min_inliers,
        min_inlier_ratio=min_inlier_ratio,
        max_reprojection_error=max_reproj,
        max_assessment_age_seconds=max_age_sec,
    )

    # Intake validation
    i_raw = raw["intake"]
    if not isinstance(i_raw, dict):
        raise ValueError("Field 'intake' must be a dictionary.")

    allowed_i_keys = {
        "max_upload_bytes",
        "max_duration_seconds",
        "min_width",
        "max_width",
        "min_height",
        "max_height",
        "sample_count",
        "allowed_codecs",
    }
    extra_i = set(i_raw.keys()) - allowed_i_keys
    if extra_i:
        raise ValueError(f"Unknown keys in 'intake': {extra_i}")
    missing_i = allowed_i_keys - set(i_raw.keys())
    if missing_i:
        raise ValueError(f"Missing required keys in 'intake': {missing_i}")

    max_bytes = int(_check_strict_number(i_raw["max_upload_bytes"], "intake.max_upload_bytes", allow_float=False, min_val=1024, max_val=1073741824))
    max_duration = _check_strict_number(i_raw["max_duration_seconds"], "intake.max_duration_seconds", min_val=1.0, max_val=3600.0)
    min_w = int(_check_strict_number(i_raw["min_width"], "intake.min_width", allow_float=False, min_val=16, max_val=8192))
    max_w = int(_check_strict_number(i_raw["max_width"], "intake.max_width", allow_float=False, min_val=min_w, max_val=8192))
    min_h = int(_check_strict_number(i_raw["min_height"], "intake.min_height", allow_float=False, min_val=16, max_val=8192))
    max_h = int(_check_strict_number(i_raw["max_height"], "intake.max_height", allow_float=False, min_val=min_h, max_val=8192))
    sample_count = int(_check_strict_number(i_raw["sample_count"], "intake.sample_count", allow_float=False, min_val=1, max_val=20))

    raw_codecs = i_raw["allowed_codecs"]
    if not isinstance(raw_codecs, (list, tuple)) or not raw_codecs:
        raise ValueError("Field 'intake.allowed_codecs' must be a non-empty list of codec names.")
    allowed_codecs = tuple(_check_strict_string(c, "intake.allowed_codecs item").lower() for c in raw_codecs)

    intake = StabilityIntakeConfig(
        max_upload_bytes=max_bytes,
        max_duration_seconds=max_duration,
        min_width=min_w,
        max_width=max_w,
        min_height=min_h,
        max_height=max_h,
        sample_count=sample_count,
        allowed_codecs=allowed_codecs,
    )

    # Execution validation
    e_raw = raw["execution"]
    if not isinstance(e_raw, dict):
        raise ValueError("Field 'execution' must be a dictionary.")

    allowed_e_keys = {"max_concurrent_jobs", "ffprobe_timeout_seconds"}
    extra_e = set(e_raw.keys()) - allowed_e_keys
    if extra_e:
        raise ValueError(f"Unknown keys in 'execution': {extra_e}")
    missing_e = allowed_e_keys - set(e_raw.keys())
    if missing_e:
        raise ValueError(f"Missing required keys in 'execution': {missing_e}")

    max_concurrent = int(_check_strict_number(e_raw["max_concurrent_jobs"], "execution.max_concurrent_jobs", allow_float=False, min_val=1, max_val=16))
    ffprobe_timeout = _check_strict_number(e_raw["ffprobe_timeout_seconds"], "execution.ffprobe_timeout_seconds", min_val=1.0, max_val=120.0)

    execution = StabilityExecutionConfig(
        max_concurrent_jobs=max_concurrent,
        ffprobe_timeout_seconds=ffprobe_timeout,
    )

    thresholds_snapshot = asdict(thresholds)
    thresholds_snapshot["config_version"] = version
    thresholds_snapshot["algorithm_version"] = algorithm_version

    return StabilityConfig(
        version=version,
        algorithm_version=algorithm_version,
        config_sha256=raw_sha256,
        thresholds=thresholds,
        intake=intake,
        execution=execution,
        thresholds_snapshot=thresholds_snapshot,
    )


def load_stability_config(config_path: Optional[Path] = None) -> StabilityConfig:
    """Load and strictly validate a stability YAML configuration file."""
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Stability configuration file not found at: {path}")

    raw_bytes = path.read_bytes()
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()

    try:
        data = yaml.safe_load(raw_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to parse YAML from {path}: {e}")

    return validate_stability_config_dict(data, raw_sha)
