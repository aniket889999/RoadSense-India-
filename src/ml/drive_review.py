"""Bounded, upload-backed Drive Review planning for RoadSense India.

This module creates an explicit contiguous source-frame plan for a short
uploaded-video replay.  It does not access a camera, training data, a held-out
test split, or any model framework.  The caller supplies the already-verified
local model and may render only raw, unverified suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any

from src.ml.experimental_inference import (
    ExperimentalInferenceSettings,
    run_verified_sampled_video_inference,
)
from src.ml.model_provenance import FrozenModelInfo
from src.ml.video_inference import ExperimentalInferenceResult


@dataclass(frozen=True)
class DriveReviewPlan:
    """A bounded, strictly increasing source-frame plan for one replay window."""

    frame_indices: tuple[int, ...]
    source_fps: float
    source_duration_seconds: float
    window_start_seconds: float
    window_end_seconds: float
    requested_sampling_fps: float
    max_frames: int

    @property
    def sampled_frame_count(self) -> int:
        return len(self.frame_indices)


def build_drive_review_plan(
    *,
    frame_count: int,
    source_fps: float,
    window_start_seconds: float,
    window_duration_seconds: float,
    sampling_fps: float,
    max_frames: int,
) -> DriveReviewPlan:
    """Build a safe contiguous review plan without decoding video frames.

    The plan is always bounded by ``max_frames`` and never includes a frame
    outside the requested source window.  If the requested cadence would
    exceed the cap, points are distributed across the same window rather than
    silently expanding it or processing every source frame.
    """

    if isinstance(frame_count, bool) or not isinstance(frame_count, Integral) or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer.")
    if isinstance(max_frames, bool) or not isinstance(max_frames, Integral) or max_frames <= 0:
        raise ValueError("max_frames must be a positive integer.")

    fps = _positive_finite(source_fps, "source_fps")
    start_seconds = _non_negative_finite(window_start_seconds, "window_start_seconds")
    duration_seconds = _positive_finite(window_duration_seconds, "window_duration_seconds")
    requested_sampling_fps = _positive_finite(sampling_fps, "sampling_fps")

    source_duration = int(frame_count) / fps
    if start_seconds >= source_duration:
        raise ValueError("window_start_seconds must be inside the uploaded video duration.")

    start_frame = min(int(frame_count) - 1, int(math.floor(start_seconds * fps)))
    requested_end_seconds = min(source_duration, start_seconds + duration_seconds)
    end_exclusive = min(int(frame_count), max(start_frame + 1, int(math.ceil(requested_end_seconds * fps))))
    end_frame = end_exclusive - 1

    # Sampling at a rate higher than the source does not duplicate frames.
    stride = max(1, int(round(fps / requested_sampling_fps)))
    candidates = list(range(start_frame, end_exclusive, stride))
    if candidates[-1] != end_frame:
        candidates.append(end_frame)

    selected = _bounded_even_selection(candidates, int(max_frames))
    return DriveReviewPlan(
        frame_indices=tuple(selected),
        source_fps=fps,
        source_duration_seconds=source_duration,
        window_start_seconds=start_frame / fps,
        window_end_seconds=(end_frame + 1) / fps,
        requested_sampling_fps=requested_sampling_fps,
        max_frames=int(max_frames),
    )


def run_verified_drive_review(
    *,
    video_bytes: bytes,
    video_filename: str,
    plan: DriveReviewPlan,
    model: Any,
    model_info: FrozenModelInfo,
    settings: ExperimentalInferenceSettings,
    input_video_sha256: str | None = None,
) -> ExperimentalInferenceResult:
    """Render only the caller's bounded Drive Review plan.

    The ``drive_review`` render mode adds green circular markers for raw model
    suggestions.  It remains an experimental sampled playback and cannot
    create manual incidents or human-verified potholes.
    """

    if not isinstance(plan, DriveReviewPlan) or not plan.frame_indices:
        raise ValueError("Drive Review requires a non-empty DriveReviewPlan.")
    return run_verified_sampled_video_inference(
        video_bytes=video_bytes,
        video_filename=video_filename,
        sampled_frame_indices=plan.frame_indices,
        model=model,
        model_info=model_info,
        settings=settings,
        input_video_sha256=input_video_sha256,
        render_mode="drive_review",
    )


def _positive_finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a positive finite number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field_name} must be a positive finite number.")
    return result


def _non_negative_finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a non-negative finite number.")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field_name} must be a non-negative finite number.")
    return result


def _bounded_even_selection(candidates: list[int], max_frames: int) -> list[int]:
    if not candidates:
        raise ValueError("Drive Review sampling produced no frame indices.")
    if len(candidates) <= max_frames:
        return candidates
    if max_frames == 1:
        return [candidates[0]]
    last_index = len(candidates) - 1
    return [candidates[(position * last_index) // (max_frames - 1)] for position in range(max_frames)]
