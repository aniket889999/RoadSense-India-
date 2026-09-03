"""Orchestration for explicitly experimental, local frozen-baseline inference.

The functions here compose a provenance-verified checkpoint, a loaded local
model, and the sampled-video renderer.  They intentionally have no concept of
manual incidents, repair priority, traffic, or safety decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Callable, Mapping, Sequence

from src.ml.local_model_runtime import predict_yolo_frame
from src.ml.model_provenance import FrozenModelInfo
from src.ml.video_inference import ExperimentalInferenceResult, run_sampled_video_inference


@dataclass(frozen=True)
class ExperimentalInferenceSettings:
    """Explicit runtime controls for raw model suggestions on sampled frames."""

    device: str
    image_size: int
    confidence_threshold: float
    iou_threshold: float
    max_detections_per_frame: int
    output_fps: float

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("Experimental inference device must be non-empty.")
        if isinstance(self.image_size, bool) or not isinstance(self.image_size, int) or self.image_size <= 0:
            raise ValueError("Experimental inference image_size must be a positive integer.")
        if (
            isinstance(self.max_detections_per_frame, bool)
            or not isinstance(self.max_detections_per_frame, int)
            or self.max_detections_per_frame <= 0
        ):
            raise ValueError("Experimental inference max_detections_per_frame must be a positive integer.")
        for field_name, value, minimum, maximum in (
            ("confidence_threshold", self.confidence_threshold, 0.0, 1.0),
            ("iou_threshold", self.iou_threshold, 0.0, 1.0),
            ("output_fps", self.output_fps, 0.0, None),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Experimental inference {field_name} must be a finite number.")
            numeric_value = float(value)
            if maximum is None:
                valid = numeric_value > minimum
            else:
                valid = minimum <= numeric_value <= maximum
            if not valid:
                if maximum is None:
                    raise ValueError(f"Experimental inference {field_name} must be greater than {minimum}.")
                raise ValueError(
                    f"Experimental inference {field_name} must be between {minimum} and {maximum}."
                )


def build_public_inference_metadata(
    model_info: FrozenModelInfo,
    settings: ExperimentalInferenceSettings,
) -> Mapping[str, Any]:
    """Return portable provenance without local user or artifact paths."""

    return {
        "model_source": "local_frozen_baseline",
        "baseline_run_id": model_info.run_id,
        "checkpoint_sha256": model_info.checkpoint_sha256,
        "model_metadata_sha256": model_info.model_metadata_sha256,
        "training_git_sha": model_info.git_sha,
        "dataset_fingerprint": model_info.dataset_fingerprint,
        "task": model_info.task,
        "class_mapping": dict(model_info.class_mapping),
        "device": settings.device,
        "image_size": settings.image_size,
        "iou_threshold": settings.iou_threshold,
        "max_detections_per_frame": settings.max_detections_per_frame,
        "local_only": True,
        "held_out_test_reused": False,
    }


def build_verified_frame_predictor(
    model: Any,
    settings: ExperimentalInferenceSettings,
) -> Callable[[Any], list[dict[str, float | int]]]:
    """Build a raw-prediction callback that invokes only ``model.predict``."""

    def predict_frame(frame: Any) -> list[dict[str, float | int]]:
        return predict_yolo_frame(
            model,
            frame,
            device=settings.device,
            image_size=settings.image_size,
            confidence_threshold=settings.confidence_threshold,
            iou_threshold=settings.iou_threshold,
            max_detections_per_frame=settings.max_detections_per_frame,
        )

    return predict_frame


def run_verified_sampled_video_inference(
    *,
    video_bytes: bytes,
    video_filename: str,
    sampled_frame_indices: Sequence[int],
    model: Any,
    model_info: FrozenModelInfo,
    settings: ExperimentalInferenceSettings,
    input_video_sha256: str | None = None,
    render_mode: str = "experimental",
) -> ExperimentalInferenceResult:
    """Render raw, unverified local-model suggestions for an existing sample plan.

    ``render_mode`` changes presentation only.  It never changes the model,
    sample plan, confidence threshold, or the meaning of a raw detection.
    """

    metadata = dict(build_public_inference_metadata(model_info, settings))
    if input_video_sha256 is not None:
        if not isinstance(input_video_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", input_video_sha256):
            raise ValueError("input_video_sha256 must be a lowercase 64-character SHA-256 digest when supplied.")
        metadata["input_video_sha256"] = input_video_sha256

    return run_sampled_video_inference(
        video_bytes,
        video_filename,
        sampled_frame_indices,
        build_verified_frame_predictor(model, settings),
        confidence_threshold=settings.confidence_threshold,
        output_fps=settings.output_fps,
        model_metadata=metadata,
        render_mode=render_mode,
    )
