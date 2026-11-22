"""Bounded, advisory crop measurements; not a calibrated readability classifier."""
from __future__ import annotations

import math

import cv2
import numpy as np

from src.anpr.contracts import PlateImageQuality, QualityFlag
from src.anpr.preprocessing import validate_bgr_image


def _observed_line_tilt(gray: np.ndarray) -> float | None:
    height, width = gray.shape
    if min(height, width) < 24:
        return None
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                            minLineLength=max(20, width // 5), maxLineGap=max(3, width // 30))
    if lines is None:
        return None
    measurements = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4)[:256]:
        dx, dy = int(x2) - int(x1), int(y2) - int(y1)
        if dx < 0:
            dx, dy = -dx, -dy
        angle = math.degrees(math.atan2(dy, dx))
        if abs(angle) <= 35:
            measurements.append((angle, math.hypot(dx, dy)))
    if len(measurements) < 2 or sum(length for _, length in measurements) < width * 0.7:
        return None
    measurements.sort()
    half = sum(length for _, length in measurements) / 2
    cumulative = 0.0
    median = 0.0
    for angle, length in measurements:
        cumulative += length
        if cumulative >= half:
            median = angle
            break
    coherent = sum(length for angle, length in measurements if abs(angle - median) <= 4)
    if coherent < 0.7 * sum(length for _, length in measurements):
        return None
    return round(median, 1)


def assess_plate_quality(crop: np.ndarray) -> PlateImageQuality:
    """Measure source resolution, detail, illumination and supported 2D edge tilt.

    Thresholds are review cues, not probabilities or proof of blur, camera pose,
    or OCR correctness. This never changes pixels passed to the recognizer.
    """
    validate_bgr_image(crop)
    height, width = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = min(64 / height, 320 / width, 3.0)
    bounded = cv2.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    sharpness = float(cv2.Laplacian(bounded, cv2.CV_64F).var())
    brightness = float(gray.mean())
    clipped = float(np.mean(gray >= 250))
    tilt = _observed_line_tilt(bounded)
    flags = []
    if width < 80 or height < 24:
        flags.append(QualityFlag.SMALL_CROP)
    if sharpness < 35:
        flags.append(QualityFlag.LOW_SHARPNESS)
    if brightness < 45:
        flags.append(QualityFlag.DARK_CROP)
    if clipped > 0.65:
        flags.append(QualityFlag.CLIPPED_HIGHLIGHTS)
    if tilt is not None and abs(tilt) >= 8:
        flags.append(QualityFlag.TILT_OBSERVED)
    return PlateImageQuality(crop_width=width, crop_height=height,
                             sharpness_score=round(sharpness, 2), mean_brightness=round(brightness, 2),
                             clipped_highlight_fraction=round(clipped, 4),
                             observed_line_tilt_degrees=tilt, warnings=flags)
