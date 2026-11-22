"""Lightweight interfaces shared by ANPR orchestration and local model adapters.

These definitions must remain importable without OpenCV, PyTorch or EasyOCR.
Images passed to adapters are contiguous uint8 BGR arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence

from src.anpr.contracts import PlateBox

if TYPE_CHECKING:
    import numpy as np


class RuntimeUnavailable(RuntimeError):
    """A required local package, configuration or model asset is unavailable."""


@dataclass(frozen=True)
class PlateDetection:
    bbox: PlateBox
    confidence: float


@dataclass(frozen=True)
class OcrRead:
    raw_text: str
    confidence: float | None
    review_only: bool = False


class PlateDetector(Protocol):
    def detect(self, image_bgr: "np.ndarray") -> Sequence[PlateDetection]:
        """Return plate bounds in source-image pixels, not resized input pixels."""
        ...


class PlateRecognizer(Protocol):
    def recognize(self, plate_bgr: "np.ndarray") -> OcrRead:
        """Read a plate crop without silently fixing its characters."""
        ...


@dataclass(frozen=True)
class AnprRuntime:
    detector: PlateDetector
    recognizer: PlateRecognizer
    # Engine-specific acceptance threshold from a named policy; None keeps the service default.
    minimum_confidence: float | None = None
