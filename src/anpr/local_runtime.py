"""Compose the local plate-detector and EasyOCR recognizer runtime.

Importing this module performs no model loading, package downloads, or
filesystem scans. Call ``build_local_runtime()`` to construct the verified
local ``AnprRuntime``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.anpr.detector import UltralyticsPlateDetector
from src.anpr.ocr import EasyOcrPlateRecognizer
from src.anpr.runtime_contracts import AnprRuntime


def build_local_runtime(
    *,
    weights_path: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    ocr_model_dir: Optional[str] = None,
    device: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> AnprRuntime:
    """Build the verified local ANPR runtime from repository-confined assets.

    Raises ``RuntimeUnavailable`` if any required environment setting, local
    weight file, SHA-256 digest, single-class ``license_plate`` schema, or
    optional dependency (Ultralytics / EasyOCR) is missing or incompatible.
    """
    detector = UltralyticsPlateDetector(
        weights_path=weights_path,
        expected_sha256=expected_sha256,
        device=device,
        repo_root=repo_root,
    )
    recognizer = EasyOcrPlateRecognizer(
        model_dir=ocr_model_dir,
        device=device,
        repo_root=repo_root,
    )
    return AnprRuntime(detector=detector, recognizer=recognizer)
