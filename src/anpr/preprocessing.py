"""Bounded OpenCV crop preprocessing for ANPR OCR.

This module preserves the original plate crop and generates a bounded set of
grayscale / contrast-normalized OpenCV views without inventing corners,
repairing characters, or claiming recovered detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from src.anpr.contracts import PlateBox

MIN_CROP_DIMENSION_PX = 8
MIN_OCR_HEIGHT_PX = 48
MAX_OCR_HEIGHT_PX = 160
MAX_OCR_WIDTH_PX = 640
MAX_UPSCALE_FACTOR = 3.0


@dataclass(frozen=True)
class PreprocessedPlateCrop:
    """Preserved original BGR crop alongside bounded OpenCV views for OCR."""

    original_bgr: np.ndarray
    resized_bgr: np.ndarray
    grayscale: np.ndarray
    enhanced_grayscale: np.ndarray
    scale_x: float
    scale_y: float


def validate_bgr_image(image_bgr: np.ndarray, *, label: str = "image_bgr") -> np.ndarray:
    """Validate a finite uint8 3-channel BGR image array."""
    if not isinstance(image_bgr, np.ndarray):
        raise ValueError(f"{label} must be a NumPy ndarray.")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"{label} must have shape (H, W, 3).")
    if image_bgr.dtype != np.uint8:
        raise ValueError(f"{label} must have dtype uint8.")
    height, width = image_bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"{label} must have positive width and height.")
    return image_bgr


def extract_plate_crop(
    image_bgr: np.ndarray,
    bbox: PlateBox,
    *,
    padding_ratio: float = 0.04,
) -> np.ndarray:
    """Extract an un-mutated copy of the detected plate region within image bounds."""
    validate_bgr_image(image_bgr, label="image_bgr")
    if not (0.0 <= padding_ratio <= 0.25):
        raise ValueError("padding_ratio must be between 0.0 and 0.25.")

    img_h, img_w = image_bgr.shape[:2]
    box_w = float(bbox.x2 - bbox.x1)
    box_h = float(bbox.y2 - bbox.y1)
    if box_w <= 0.0 or box_h <= 0.0:
        raise ValueError("Plate bounding box must have positive width and height.")

    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio

    x1 = max(0, min(img_w - 1, int(np.floor(bbox.x1 - pad_x))))
    y1 = max(0, min(img_h - 1, int(np.floor(bbox.y1 - pad_y))))
    x2 = max(x1 + 1, min(img_w, int(np.ceil(bbox.x2 + pad_x))))
    y2 = max(y1 + 1, min(img_h, int(np.ceil(bbox.y2 + pad_y))))

    crop = image_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise ValueError("Extracted plate crop is empty.")
    return crop


def _compute_bounded_size(width: int, height: int) -> Tuple[int, int, float, float]:
    scale = 1.0
    if height < MIN_OCR_HEIGHT_PX:
        scale = min(MAX_UPSCALE_FACTOR, MIN_OCR_HEIGHT_PX / float(height))
    elif height > MAX_OCR_HEIGHT_PX:
        scale = MAX_OCR_HEIGHT_PX / float(height)

    if width * scale > MAX_OCR_WIDTH_PX:
        scale = min(scale, MAX_OCR_WIDTH_PX / float(width))

    target_w = max(1, int(round(width * scale)))
    target_h = max(1, int(round(height * scale)))
    return target_w, target_h, target_w / float(width), target_h / float(height)


def preprocess_plate_crop(plate_bgr: np.ndarray) -> PreprocessedPlateCrop:
    """Prepare bounded OpenCV OCR views while preserving the original crop.

    Performs only aspect-ratio-preserving bounded resize, grayscale conversion,
    mild bilateral denoising, and CLAHE local contrast normalization. Does not
    invent corners, warp perspective without corner evidence, or hallucinate
    missing character strokes.
    """
    validate_bgr_image(plate_bgr, label="plate_bgr")
    original = plate_bgr.copy()
    height, width = original.shape[:2]

    target_w, target_h, scale_x, scale_y = _compute_bounded_size(width, height)
    if (target_w, target_h) != (width, height):
        interpolation = cv2.INTER_CUBIC if target_h > height else cv2.INTER_AREA
        resized_bgr = cv2.resize(original, (target_w, target_h), interpolation=interpolation)
    else:
        resized_bgr = original.copy()

    grayscale = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

    if min(target_w, target_h) >= MIN_CROP_DIMENSION_PX:
        filtered = cv2.bilateralFilter(grayscale, d=5, sigmaColor=35, sigmaSpace=35)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(filtered)
    else:
        enhanced = grayscale.copy()

    return PreprocessedPlateCrop(
        original_bgr=original,
        resized_bgr=resized_bgr,
        grayscale=grayscale,
        enhanced_grayscale=enhanced,
        scale_x=scale_x,
        scale_y=scale_y,
    )
