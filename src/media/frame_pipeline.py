"""Bounded OpenCV streaming frame pipeline and visual overlay engine.

Processes video frames sequentially through a memory-bounded streaming generator,
applying orientation correction, visual quality checks, neon-green target overlays,
evidence crops, and optional privacy masking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.media.inspector import MediaMetadata, inspect_media_file


@dataclass(frozen=True)
class FrameProcessingOptions:
    """Configurable controls for bounded frame streaming."""
    target_fps: float = 5.0
    max_duration_seconds: float = 120.0
    max_frames: int = 300
    start_seconds: float = 0.0
    apply_privacy_mask: bool = False
    confidence_display_threshold: float = 0.25


@dataclass
class ProcessedFrame:
    """A processed video frame ready for display, tracking, and encoding."""
    frame_index: int
    timestamp_seconds: float
    raw_frame_bgr: np.ndarray
    annotated_frame_bgr: np.ndarray
    detections: List[Dict[str, Any]]
    active_tracks: List[Any]


def rotate_frame(frame: np.ndarray, rotation_degrees: int) -> np.ndarray:
    """Apply lossless 90/180/270 degree rotation if indicated by container metadata."""
    if rotation_degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def stream_video_frames(
    video_path: Path | str,
    metadata: MediaMetadata,
    options: FrameProcessingOptions,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """Yield frames sequentially without loading the full video into memory.

    Yields:
        (frame_index, timestamp_seconds, oriented_bgr_frame)
    """
    cap = cv2.VideoCapture(str(Path(video_path).resolve()))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for streaming: {video_path}")

    try:
        source_fps = metadata.avg_fps if metadata.avg_fps > 0 else 30.0
        step = max(1, int(round(source_fps / options.target_fps)))
        start_frame = int(math.floor(options.start_seconds * source_fps))
        max_source_frame = int(math.ceil((options.start_seconds + options.max_duration_seconds) * source_fps))
        if metadata.frame_count:
            max_source_frame = min(max_source_frame, metadata.frame_count)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        yielded_count = 0

        while current_frame < max_source_frame and yielded_count < options.max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            oriented = rotate_frame(frame, metadata.rotation_degrees)
            timestamp = current_frame / source_fps

            yield (current_frame, timestamp, oriented)
            yielded_count += 1
            current_frame += step
    finally:
        cap.release()


def render_detection_overlays(
    frame: np.ndarray,
    detections: Sequence[Dict[str, Any]],
    *,
    frame_index: int,
    timestamp_seconds: float,
    confidence_threshold: float = 0.25,
    apply_privacy_mask: bool = False,
) -> np.ndarray:
    """Render green circular pothole suggestions, track IDs, and HUD banners on a frame copy."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    # 1. Optional Privacy Masking (e.g. blur top 12% windscreen region if requested)
    if apply_privacy_mask:
        mask_h = int(height * 0.12)
        top_roi = canvas[0:mask_h, 0:width]
        blurred = cv2.GaussianBlur(top_roi, (31, 31), 0)
        canvas[0:mask_h, 0:width] = blurred

    # Filter detections passing review threshold
    valid_dets = [
        d for d in detections
        if float(d.get("confidence", 0.0)) >= confidence_threshold
    ]

    # 2. Draw Target Markers
    for det in valid_dets:
        x_min = max(0, min(width - 1, int(round(float(det["x_min"])))))
        y_min = max(0, min(height - 1, int(round(float(det["y_min"])))))
        x_max = max(0, min(width - 1, int(round(float(det["x_max"])))))
        y_max = max(0, min(height - 1, int(round(float(det["y_max"])))))

        conf = float(det.get("confidence", 0.0))
        track_id = det.get("track_id")

        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        radius = max(12, int(math.ceil(max(x_max - x_min, y_max - y_min) / 2.0)) + 6)

        # Neon-green circle & crosshair
        cv2.circle(canvas, (center_x, center_y), radius, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas, (center_x, center_y), radius + 2, (0, 160, 0), 1, cv2.LINE_AA)

        cv2.line(canvas, (center_x - 4, center_y), (center_x + 4, center_y), (0, 255, 0), 1)
        cv2.line(canvas, (center_x, center_y - 4), (center_x, center_y + 4), (0, 255, 0), 1)

        # Label Pill
        track_tag = f"Track #{track_id} | " if track_id is not None else ""
        label = f"Pothole candidate | {track_tag}{conf * 100:.0f}%"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_x = max(6, min(width - text_w - 12, x_min))
        label_y = max(text_h + 8, y_min - 8)

        # Label background
        cv2.rectangle(
            canvas,
            (label_x - 3, label_y - text_h - 4),
            (label_x + text_w + 3, label_y + baseline),
            (10, 15, 20),
            -1,
        )
        cv2.rectangle(
            canvas,
            (label_x - 3, label_y - text_h - 4),
            (label_x + text_w + 3, label_y + baseline),
            (0, 220, 0),
            1,
        )
        cv2.putText(canvas, label, (label_x, label_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    # 3. Top HUD Banner
    hud_h = 44
    cv2.rectangle(canvas, (0, 0), (width, hud_h), (11, 15, 20), -1)
    cv2.line(canvas, (0, hud_h), (width, hud_h), (31, 46, 64), 1)

    title = "ROADSENSE INDIA · MEDIA INTELLIGENCE"
    subtitle = "AI SUGGESTION — NOT HUMAN VERIFIED"
    time_info = f"Frame {frame_index} | {timestamp_seconds:.2f}s | {len(valid_dets)} target(s)"

    cv2.putText(canvas, title, (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1, cv2.LINE_AA)

    (tw, _), _ = cv2.getTextSize(time_info, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(canvas, time_info, (max(10, width - tw - 12), 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (148, 163, 184), 1, cv2.LINE_AA)

    return canvas


def extract_evidence_crop(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    padding_factor: float = 0.25,
) -> bytes:
    """Extract a padded high-resolution JPEG crop of a detected pothole."""
    h, w = frame.shape[:2]
    x_min, y_min, x_max, y_max = bbox

    box_w = x_max - x_min
    box_h = y_max - y_min

    pad_x = box_w * padding_factor
    pad_y = box_h * padding_factor

    crop_x1 = max(0, int(round(x_min - pad_x)))
    crop_y1 = max(0, int(round(y_min - pad_y)))
    crop_x2 = min(w, int(round(x_max + pad_x)))
    crop_y2 = min(h, int(round(y_max + pad_y)))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, min(100, w), min(100, h)

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    success, buffer = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not success:
        return b""
    return buffer.tobytes()
