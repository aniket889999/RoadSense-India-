"""Locate one/two text lines from visible character components, without warping pixels."""
from __future__ import annotations

import cv2
import numpy as np


def plate_text_regions(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return conservative text-line bounds, or [] when component evidence is weak.

    Both ink polarities are considered. Small bolts/side legends and large frame
    borders are excluded by geometry, never by an expected registration string.
    OCR always receives the original pixels inside these bounds.
    """
    height, width = image.shape[:2]
    if min(height, width) < 12:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    candidates = []
    for mask in (dark, cv2.bitwise_not(dark)):
        mask = mask.copy()
        border = max(1, round(height * 0.025))
        mask[:border] = mask[-border:] = 0
        mask[:, :border] = mask[:, -border:] = 0
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        boxes = []
        for x, y, w, h, area in stats[1:]:
            if (max(6, height * 0.18) <= h <= height * 0.90
                    and 1 <= w <= min(width * 0.32, h * 1.5)
                    and area >= max(4, width * height * 0.0008)):
                boxes.append((int(x), int(y), int(w), int(h)))
        if len(boxes) < 4 or len(boxes) > 24:
            continue
        median_h = float(np.median([b[3] for b in boxes]))
        boxes = [b for b in boxes if b[3] >= median_h * 0.55]
        if len(boxes) >= 4:
            candidates.append((sum(b[3] for b in boxes), boxes))
    if not candidates:
        return []
    boxes = max(candidates, key=lambda candidate: candidate[0])[1]
    boxes.sort(key=lambda b: b[1] + b[3] / 2)
    centers = [b[1] + b[3] / 2 for b in boxes]
    gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    split = int(np.argmax(gaps)) + 1
    groups = [boxes]
    if max(gaps) >= height * 0.18 and min(split, len(boxes) - split) >= 2:
        groups = [boxes[:split], boxes[split:]]
    regions = []
    for group in groups:
        x1, y1 = min(b[0] for b in group), min(b[1] for b in group)
        x2, y2 = max(b[0] + b[2] for b in group), max(b[1] + b[3] for b in group)
        pad = max(2, round((y2 - y1) * 0.12))
        regions.append((max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)))
    return regions
