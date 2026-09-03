"""Experimental sampled-video inference rendering.

This module deliberately has no dependency on a particular model framework.  A
caller supplies a small ``predict_frame(frame)`` callback, which makes the
video/reporting behaviour independently testable and keeps model loading at
the application boundary.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from io import BytesIO, StringIO
from numbers import Integral
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

import cv2


PredictionMapping = Mapping[str, Any]
FramePredictor = Callable[[Any], Optional[Iterable[PredictionMapping]]]


@dataclass(frozen=True)
class ExperimentalDetection:
    """One raw model prediction for one source-video frame.

    These records intentionally are not incidents.  They have not been merged,
    tracked, or verified by a person.
    """

    frame_index: int
    timestamp_seconds: float
    class_id: int
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class ExperimentalInferenceResult:
    """The raw records and portable experimental-report archive."""

    detections: List[ExperimentalDetection]
    report_zip: bytes
    total_sampled_frames: int
    frames_with_detections: int


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def run_sampled_video_inference(
    video_bytes: bytes,
    video_filename: str,
    sampled_frame_indices: Sequence[int],
    predictor: FramePredictor,
    *,
    confidence_threshold: float = 0.25,
    output_fps: float = 5.0,
    model_metadata: Optional[Mapping[str, Any]] = None,
    render_mode: str = "experimental",
) -> ExperimentalInferenceResult:
    """Run a predictor only on caller-supplied frame indices and package results.

    ``sampled_frame_indices`` is treated as the authoritative sampling plan: it
    is never expanded, resampled, or replaced with an inferred cadence.  The
    callback receives the native OpenCV BGR frame and may return an iterable of
    plain mappings.  Each valid mapping must contain ``class_id``,
    ``confidence``, and either ``x_min/y_min/x_max/y_max`` or
    ``x1/y1/x2/y2`` (or a four-value ``xyxy`` sequence).  Invalid predictions
    are filtered rather than converted or clamped.

    The archive contains raw per-frame model detections, a path-free metadata
    document, and an annotated MP4 containing exactly the supplied sampled
    frames.  It does not claim that any prediction is human-verified.
    """

    frame_indices = _validate_frame_indices(sampled_frame_indices)
    _validate_runtime_options(confidence_threshold, output_fps)
    _validate_render_mode(render_mode)

    if not callable(predictor):
        raise TypeError("predictor must be a callable accepting one OpenCV frame.")
    if model_metadata is not None and not isinstance(model_metadata, Mapping):
        raise TypeError("model_metadata must be a mapping when supplied.")

    if not isinstance(video_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("video_bytes must be bytes-like.")
    input_bytes = bytes(video_bytes)
    if not input_bytes:
        raise ValueError("Video input is empty.")

    source_path: Optional[str] = None
    output_path: Optional[str] = None
    capture = None
    writer = None

    try:
        suffix = _safe_video_suffix(video_filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as source_file:
            source_file.write(input_bytes)
            source_path = source_file.name

        capture = cv2.VideoCapture(source_path)
        if not capture.isOpened():
            raise ValueError("Unreadable video file.")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or frame_count <= 0 or not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("Video metadata is invalid or incomplete.")

        for frame_index in frame_indices:
            if frame_index >= frame_count:
                raise ValueError("A supplied sampled frame index is outside the video frame range.")

        fd, output_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, float(output_fps), (width, height))
        if writer is None or not writer.isOpened():
            if writer is not None:
                writer.release()
                writer = None
            raise RuntimeError("Failed to open VideoWriter for experimental prediction video.")

        detections: List[ExperimentalDetection] = []
        frames_with_detections = 0

        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Unable to read a supplied sampled frame from the video.")

            try:
                callback_result = predictor(frame)
            except Exception as exc:  # callback errors should identify the relevant frame
                raise RuntimeError(f"Prediction callback failed for sampled frame {frame_index}.") from exc

            frame_detections = _validated_frame_predictions(
                callback_result,
                frame_index=frame_index,
                timestamp_seconds=frame_index / source_fps,
                frame_width=width,
                frame_height=height,
                confidence_threshold=float(confidence_threshold),
            )
            if frame_detections:
                frames_with_detections += 1
                detections.extend(frame_detections)

            _draw_experimental_overlays(
                frame,
                frame_detections,
                frame_index=frame_index,
                timestamp_seconds=frame_index / source_fps,
                render_mode=render_mode,
            )
            writer.write(frame)

        writer.release()
        writer = None

        rendered_frame_count = _verify_rendered_video(output_path, expected_frames=len(frame_indices))
        archive = _build_report_zip(
            detections=detections,
            rendered_video_path=output_path,
            metadata={
                "analysis_mode": "experimental_sampled_video_inference",
                "render_mode": render_mode,
                "human_verification_status": "not_human_verified",
                "prediction_status": "experimental",
                "prediction_record_type": "raw_per_frame_detection",
                "class_mapping": {"0": "pothole"},
                "confidence_threshold": float(confidence_threshold),
                "output_video_fps": float(output_fps),
                "source_video": {
                    "width": width,
                    "height": height,
                    "fps": source_fps,
                    "frame_count": frame_count,
                },
                "sampled_frame_indices": frame_indices,
                "total_sampled_frames": len(frame_indices),
                "frames_with_detections": frames_with_detections,
                "total_detections": len(detections),
                "annotated_video_frame_count": rendered_frame_count,
                "model_metadata": _public_model_metadata(model_metadata or {}),
            },
        )

        return ExperimentalInferenceResult(
            detections=detections,
            report_zip=archive,
            total_sampled_frames=len(frame_indices),
            frames_with_detections=frames_with_detections,
        )
    finally:
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
        if source_path and os.path.exists(source_path):
            os.remove(source_path)


def _validate_frame_indices(sampled_frame_indices: Sequence[int]) -> List[int]:
    if isinstance(sampled_frame_indices, (str, bytes, bytearray)):
        raise TypeError("sampled_frame_indices must be a sequence of integers.")

    try:
        indices = list(sampled_frame_indices)
    except TypeError as exc:
        raise TypeError("sampled_frame_indices must be a sequence of integers.") from exc

    if not indices:
        raise ValueError("At least one sampled frame index is required.")

    seen = set()
    previous_index = -1
    for value in indices:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("Every sampled frame index must be a non-negative integer.")
        if value < 0:
            raise ValueError("Every sampled frame index must be a non-negative integer.")
        if value in seen:
            raise ValueError("Sampled frame indices must not contain duplicates.")
        if value <= previous_index:
            raise ValueError("Sampled frame indices must be strictly increasing.")
        seen.add(value)
        previous_index = int(value)

    return [int(value) for value in indices]


def _validate_runtime_options(confidence_threshold: float, output_fps: float) -> None:
    try:
        threshold = float(confidence_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_threshold must be a finite number between 0 and 1.") from exc
    if isinstance(confidence_threshold, bool) or not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence_threshold must be a finite number between 0 and 1.")

    try:
        rendered_fps = float(output_fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("output_fps must be a positive finite number.") from exc
    if isinstance(output_fps, bool) or not math.isfinite(rendered_fps) or rendered_fps <= 0:
        raise ValueError("output_fps must be a positive finite number.")


def _validate_render_mode(render_mode: str) -> None:
    if render_mode not in {"experimental", "drive_review"}:
        raise ValueError("render_mode must be exactly 'experimental' or 'drive_review'.")


def _safe_video_suffix(video_filename: str) -> str:
    name = os.path.basename(str(video_filename or ""))
    suffix = os.path.splitext(name)[1].lower()
    if suffix and re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
        return suffix
    return ".mp4"


def _as_prediction_mappings(callback_result: Any) -> Iterable[PredictionMapping]:
    if callback_result is None:
        return []
    if isinstance(callback_result, Mapping):
        return [callback_result]
    if isinstance(callback_result, (str, bytes, bytearray)):
        return []
    try:
        return iter(callback_result)
    except TypeError:
        return []


def _validated_frame_predictions(
    callback_result: Any,
    *,
    frame_index: int,
    timestamp_seconds: float,
    frame_width: int,
    frame_height: int,
    confidence_threshold: float,
) -> List[ExperimentalDetection]:
    valid: List[ExperimentalDetection] = []
    for candidate in _as_prediction_mappings(callback_result):
        if not isinstance(candidate, Mapping):
            continue

        class_id = _as_class_zero(candidate.get("class_id"))
        confidence = _as_finite_float(candidate.get("confidence"))
        coords = _extract_xyxy(candidate)
        if class_id is None or confidence is None or coords is None:
            continue
        if confidence < confidence_threshold or confidence < 0.0 or confidence > 1.0:
            continue

        x_min, y_min, x_max, y_max = coords
        if not (0.0 <= x_min < x_max <= frame_width and 0.0 <= y_min < y_max <= frame_height):
            continue

        valid.append(
            ExperimentalDetection(
                frame_index=frame_index,
                timestamp_seconds=float(timestamp_seconds),
                class_id=class_id,
                confidence=confidence,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )
        )
    return valid


def _as_class_zero(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value) or numeric_value != 0.0 or not numeric_value.is_integer():
        return None
    return 0


def _as_finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _extract_xyxy(prediction: PredictionMapping) -> Optional[tuple[float, float, float, float]]:
    if all(key in prediction for key in ("x_min", "y_min", "x_max", "y_max")):
        raw_coords = (
            prediction["x_min"],
            prediction["y_min"],
            prediction["x_max"],
            prediction["y_max"],
        )
    elif all(key in prediction for key in ("x1", "y1", "x2", "y2")):
        raw_coords = (prediction["x1"], prediction["y1"], prediction["x2"], prediction["y2"])
    elif all(key in prediction for key in ("xmin", "ymin", "xmax", "ymax")):
        raw_coords = (prediction["xmin"], prediction["ymin"], prediction["xmax"], prediction["ymax"])
    elif "xyxy" in prediction or "bbox" in prediction:
        raw_xyxy = prediction.get("xyxy", prediction.get("bbox"))
        if isinstance(raw_xyxy, (str, bytes, bytearray)):
            return None
        try:
            raw_coords = tuple(raw_xyxy)
        except TypeError:
            return None
        if len(raw_coords) != 4:
            return None
    else:
        return None

    values = tuple(_as_finite_float(value) for value in raw_coords)
    if any(value is None for value in values):
        return None
    return values  # type: ignore[return-value]


def _draw_experimental_overlays(
    frame: Any,
    detections: Sequence[ExperimentalDetection],
    *,
    frame_index: int,
    timestamp_seconds: float,
    render_mode: str = "experimental",
) -> None:
    _validate_render_mode(render_mode)
    height, width = frame.shape[:2]
    box_color = (0, 220, 0)
    text_color = (0, 255, 255)

    for detection in detections:
        x_min = max(0, min(width - 1, int(round(detection.x_min))))
        y_min = max(0, min(height - 1, int(round(detection.y_min))))
        x_max = max(0, min(width - 1, int(round(detection.x_max))))
        y_max = max(0, min(height - 1, int(round(detection.y_max))))
        if render_mode == "drive_review":
            center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
            radius = max(8, int(math.ceil(max(x_max - x_min, y_max - y_min) / 2.0)) + 5)
            cv2.circle(frame, center, radius, box_color, 2, cv2.LINE_AA)
            label = f"POTHOLE SUGGESTION {detection.confidence:.2f}"
        else:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
            label = f"pothole {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x_min, max(16, y_min - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA,
        )

    # These labels deliberately describe the output as experimental and not
    # human-verified, even when no detections are present in the frame.
    font_scale = max(0.25, min(0.55, width / 1200.0))
    line_step = max(14, int(32 * font_scale))
    overlay_height = min(height, line_step * 3 + 8)
    cv2.rectangle(frame, (0, 0), (width, overlay_height), (0, 0, 0), -1)
    title = (
        "DRIVE REVIEW · SAMPLED PLAYBACK"
        if render_mode == "drive_review"
        else "EXPERIMENTAL MODEL OUTPUT"
    )
    subtitle = (
        "GREEN CIRCLES = UNVERIFIED SUGGESTIONS"
        if render_mode == "drive_review"
        else "NOT HUMAN-VERIFIED"
    )
    cv2.putText(
        frame,
        title,
        (6, line_step),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        subtitle,
        (6, line_step * 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Frame: {frame_index} | Original time: {timestamp_seconds:.2f} s",
        (6, line_step * 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        1,
        cv2.LINE_AA,
    )


def _verify_rendered_video(rendered_video_path: str, *, expected_frames: int) -> int:
    if not os.path.isfile(rendered_video_path) or os.path.getsize(rendered_video_path) <= 0:
        raise RuntimeError("Generated experimental prediction MP4 is empty or missing.")

    verification_capture = cv2.VideoCapture(rendered_video_path)
    try:
        if not verification_capture.isOpened():
            raise RuntimeError("Generated experimental prediction MP4 is unreadable.")
        rendered_frame_count = int(verification_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        verification_capture.release()

    if rendered_frame_count != expected_frames:
        raise RuntimeError("Generated experimental prediction MP4 has an unexpected frame count.")
    return rendered_frame_count


def _build_report_zip(
    *,
    detections: Sequence[ExperimentalDetection],
    rendered_video_path: str,
    metadata: Mapping[str, Any],
) -> bytes:
    csv_buffer = StringIO()
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=[
            "frame_index",
            "timestamp_seconds",
            "class_id",
            "confidence",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for detection in detections:
        writer.writerow(asdict(detection))

    archive_buffer = BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model_predictions.csv", csv_buffer.getvalue())
        archive.writestr(
            "inference_metadata.json",
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False),
        )
        with open(rendered_video_path, "rb") as rendered_file:
            video_bytes = rendered_file.read()
        if not video_bytes:
            raise RuntimeError("Generated experimental prediction MP4 is empty or missing.")
        archive.writestr("annotated_experimental_predictions.mp4", video_bytes)

    return archive_buffer.getvalue()


def _public_model_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Emit only the fixed, portable frozen-baseline provenance contract.

    The report must never become a general-purpose metadata echo.  In
    particular, arbitrary keys, nested data, URI values, and local paths are
    intentionally discarded—even when a string looks like a URL—because an
    allowlist is easier to audit than trying to recognise every path encoding.
    """

    public: dict[str, Any] = {}

    if value.get("model_source") == "local_frozen_baseline":
        public["model_source"] = "local_frozen_baseline"

    run_id = value.get("baseline_run_id")
    if isinstance(run_id, str) and _RUN_ID_RE.fullmatch(run_id):
        public["baseline_run_id"] = run_id

    for field_name in (
        "checkpoint_sha256",
        "model_metadata_sha256",
        "dataset_fingerprint",
        "input_video_sha256",
    ):
        candidate = value.get(field_name)
        if isinstance(candidate, str) and _SHA256_RE.fullmatch(candidate):
            public[field_name] = candidate

    git_sha = value.get("training_git_sha")
    if isinstance(git_sha, str) and _GIT_SHA_RE.fullmatch(git_sha):
        public["training_git_sha"] = git_sha

    if value.get("task") == "detection":
        public["task"] = "detection"

    class_mapping = value.get("class_mapping")
    if isinstance(class_mapping, Mapping) and set(class_mapping.keys()) in ({0}, {"0"}):
        raw_label = class_mapping.get(0, class_mapping.get("0"))
        if raw_label == "pothole":
            public["class_mapping"] = {"0": "pothole"}

    device = value.get("device")
    if isinstance(device, str) and device in {"mps", "cpu", "cuda"}:
        public["device"] = device

    image_size = value.get("image_size")
    if isinstance(image_size, Integral) and not isinstance(image_size, bool) and image_size > 0:
        public["image_size"] = int(image_size)

    iou_threshold = value.get("iou_threshold")
    if (
        isinstance(iou_threshold, (int, float))
        and not isinstance(iou_threshold, bool)
        and math.isfinite(float(iou_threshold))
        and 0.0 <= float(iou_threshold) <= 1.0
    ):
        public["iou_threshold"] = float(iou_threshold)

    max_detections = value.get("max_detections_per_frame")
    if (
        isinstance(max_detections, Integral)
        and not isinstance(max_detections, bool)
        and max_detections > 0
    ):
        public["max_detections_per_frame"] = int(max_detections)

    for field_name in ("local_only", "held_out_test_reused"):
        candidate = value.get(field_name)
        if isinstance(candidate, bool):
            public[field_name] = candidate

    return public
