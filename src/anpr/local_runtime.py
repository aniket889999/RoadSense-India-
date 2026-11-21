"""Compose the local plate-detector and EasyOCR recognizer runtime.

Importing this module performs no model loading, package downloads, or
filesystem scans. Call ``build_local_runtime()`` to construct the verified
local ``AnprRuntime``.
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Optional

from src.anpr.detector import UltralyticsPlateDetector
from src.anpr.ocr import (
    ENV_OCR_ENGINE,
    EasyOcrPlateRecognizer,
    RapidOcrPlateRecognizer,
    SUPPORTED_OCR_ENGINES,
)
from src.anpr.runtime_contracts import AnprRuntime, PlateRecognizer, RuntimeUnavailable


def build_local_runtime(
    *,
    weights_path: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    ocr_model_dir: Optional[str] = None,
    ocr_engine: Optional[str] = None,
    ocr_crop_view: Optional[str] = None,
    device: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> AnprRuntime:
    """Build the verified local ANPR runtime from repository-confined assets.

    Raises ``RuntimeUnavailable`` if any required environment setting, local
    weight file, SHA-256 digest, single-class ``license_plate`` schema, or
    optional dependency (Ultralytics / EasyOCR / RapidOCR) is missing or
    incompatible. Defaults to ``ocr_engine='easyocr'`` so the deployed
    application runtime remains unchanged unless explicitly configured.
    """
    raw_engine = os.environ.get(ENV_OCR_ENGINE, "easyocr") if ocr_engine is None else ocr_engine
    if not isinstance(raw_engine, str) or raw_engine.strip().lower() not in SUPPORTED_OCR_ENGINES:
        raise RuntimeUnavailable(
            f"Unsupported {ENV_OCR_ENGINE} {raw_engine!r}. Supported values: {SUPPORTED_OCR_ENGINES}."
        )
    normalized_engine = raw_engine.strip().lower()

    detector = UltralyticsPlateDetector(
        weights_path=weights_path,
        expected_sha256=expected_sha256,
        device=device,
        repo_root=repo_root,
    )
    recognizer: PlateRecognizer
    if normalized_engine == "rapidocr":
        recognizer = RapidOcrPlateRecognizer(
            model_dir=ocr_model_dir,
            repo_root=repo_root,
            crop_view=ocr_crop_view,
            review_only=True,
        )
    else:
        recognizer = EasyOcrPlateRecognizer(
            model_dir=ocr_model_dir,
            device=device,
            repo_root=repo_root,
            crop_view=ocr_crop_view,
        )
    return AnprRuntime(detector=detector, recognizer=recognizer)
