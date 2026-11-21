"""Local EasyOCR plate recognizer adapter with lazy imports and row-ordered assembly.

Importing this module does not import EasyOCR or Torch, load weights, or make
network requests. Recognition always constructs EasyOCR with
``download_enabled=False`` and requires local OCR weight files inside a
repository-relative directory under ``models/`` configured via
``ROADSENSE_ANPR_OCR_MODEL_DIR``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re
from pathlib import Path
import stat
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from src.anpr.detector import (
    ENV_ANPR_DEVICE,
    _compute_file_sha256,
    _validate_device,
    verify_confinement_under_models,
)
from src.anpr.preprocessing import preprocess_plate_crop, validate_bgr_image
from src.anpr.runtime_contracts import OcrRead, RuntimeUnavailable

ENV_OCR_MODEL_DIR = "ROADSENSE_ANPR_OCR_MODEL_DIR"
ENV_OCR_CROP_VIEW = "ROADSENSE_ANPR_OCR_CROP_VIEW"
ENV_OCR_RECOGNIZER_WEIGHTS = "ROADSENSE_ANPR_OCR_RECOGNIZER_WEIGHTS"
ENV_OCR_RECOGNIZER_SHA256 = "ROADSENSE_ANPR_OCR_RECOGNIZER_SHA256"
SUPPORTED_OCR_CROP_VIEWS = (
    "original_bgr",
    "resized_bgr",
    "grayscale",
    "enhanced_grayscale",
)
_OCR_WEIGHT_EXTENSIONS = {".pth", ".pt"}
_ROW_MERGE_HEIGHT_RATIO = 0.55


@dataclass(frozen=True)
class _TextSpan:
    text: str
    confidence: Optional[float]
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    center_y: float
    height: float


def _verify_local_ocr_assets(model_dir: Path) -> List[Path]:
    """Verify the confined OCR model directory contains at least one non-symlink weight file."""
    weight_files: List[Path] = []
    try:
        with os.scandir(model_dir) as entries_iter:
            entries = sorted(entries_iter, key=lambda e: e.name)
    except OSError as exc:
        raise RuntimeUnavailable(
            f"Unable to inspect OCR model directory {model_dir}: {exc}"
        ) from exc

    for entry in entries:
        entry_path = model_dir / entry.name
        try:
            info = os.lstat(entry_path)
        except OSError as exc:
            raise RuntimeUnavailable(
                f"Unable to inspect OCR asset {entry_path}: {exc}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeUnavailable(
                f"ROADSENSE_ANPR_OCR_MODEL_DIR contains a forbidden symlink asset: {entry_path}"
            )
        if stat.S_ISREG(info.st_mode) and entry_path.suffix.lower() in _OCR_WEIGHT_EXTENSIONS:
            if info.st_size <= 0:
                raise RuntimeUnavailable(
                    f"OCR weight file in ROADSENSE_ANPR_OCR_MODEL_DIR is empty: {entry_path}"
                )
            weight_files.append(entry_path)

    if not weight_files:
        raise RuntimeUnavailable(
            f"No local EasyOCR weight files (.pth/.pt) found in {model_dir}. "
            "Automatic model downloads are disabled."
        )
    return weight_files


def _parse_bbox_geometry(raw_bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) == 0:
        return None

    xs: List[float] = []
    ys: List[float] = []
    for pt in raw_bbox:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            return None
        try:
            px = float(pt[0])
            py = float(pt[1])
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(px) and math.isfinite(py)):
            return None
        xs.append(px)
        ys.append(py)

    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _normalize_span_confidence(raw_conf: Any) -> Optional[float]:
    if raw_conf is None or isinstance(raw_conf, bool):
        return None
    try:
        score = float(raw_conf)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or not (0.0 <= score <= 1.0):
        return None
    return score


def order_and_merge_ocr_detections(detections: Sequence[Any]) -> OcrRead:
    """Order EasyOCR detections into rows (top-to-bottom), then left-to-right.

    Supports both single-line and two-line license plates. Preserves raw OCR
    tokens separated by spaces within a row and newline across distinct rows,
    and reports a conservative aggregate engine score (the minimum across all
    non-empty text spans, or ``None`` if any non-empty span lacks a valid score).
    """
    spans: List[_TextSpan] = []

    for item in detections:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        raw_bbox = item[0]
        raw_text = item[1]
        raw_conf = item[2] if len(item) >= 3 else None

        if not isinstance(raw_text, str) or not raw_text:
            continue
        if not raw_text.strip():
            continue

        geom = _parse_bbox_geometry(raw_bbox)
        if geom is None:
            continue
        min_x, min_y, max_x, max_y = geom
        height = max(1.0, max_y - min_y)
        center_y = 0.5 * (min_y + max_y)
        conf = _normalize_span_confidence(raw_conf)

        spans.append(
            _TextSpan(
                text=raw_text,
                confidence=conf,
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                center_y=center_y,
                height=height,
            )
        )

    if not spans:
        return OcrRead(raw_text="", confidence=None)

    # Sort spans by vertical center first to group into rows
    spans_by_y = sorted(spans, key=lambda s: (s.center_y, s.min_x))
    rows: List[List[_TextSpan]] = []

    for span in spans_by_y:
        if not rows:
            rows.append([span])
            continue

        current_row = rows[-1]
        row_center_y = sum(s.center_y for s in current_row) / len(current_row)
        row_median_h = sorted(s.height for s in current_row)[len(current_row) // 2]
        threshold = max(row_median_h, span.height) * _ROW_MERGE_HEIGHT_RATIO

        if abs(span.center_y - row_center_y) <= threshold:
            current_row.append(span)
        else:
            rows.append([span])

    # Ensure rows are ordered top-to-bottom and tokens within each row left-to-right
    rows.sort(key=lambda r: sum(s.center_y for s in r) / len(r))
    row_strings: List[str] = []
    all_scores: List[Optional[float]] = []

    for row in rows:
        ordered_row = sorted(row, key=lambda s: (s.min_x, s.center_y))
        row_strings.append(" ".join(s.text for s in ordered_row))
        for s in ordered_row:
            all_scores.append(s.confidence)

    combined_text = "\n".join(row_strings)
    if len(combined_text) > 256:
        raise RuntimeUnavailable(
            "OCR output exceeds the 256-character contract limit; refusing to truncate evidence."
        )

    if not all_scores or any(score is None for score in all_scores):
        aggregate_confidence: Optional[float] = None
    else:
        valid_scores = [score for score in all_scores if score is not None]
        aggregate_confidence = float(min(valid_scores))

    return OcrRead(raw_text=combined_text, confidence=aggregate_confidence)


class EasyOcrPlateRecognizer:
    """EasyOCR plate text recognizer with offline model-directory enforcement."""

    def __init__(
        self,
        *,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        repo_root: Optional[Path] = None,
        languages: Sequence[str] = ("en",),
        crop_view: Optional[str] = None,
        recognizer_weights: Optional[str] = None,
        recognizer_sha256: Optional[str] = None,
    ) -> None:
        if not languages or not all(isinstance(lang, str) and lang.strip() for lang in languages):
            raise ValueError("languages must be a non-empty sequence of language codes.")

        raw_crop_view = (
            os.environ.get(ENV_OCR_CROP_VIEW, "original_bgr")
            if crop_view is None
            else crop_view
        )
        if not isinstance(raw_crop_view, str) or raw_crop_view.strip() not in SUPPORTED_OCR_CROP_VIEWS:
            raise ValueError(
                f"crop_view must be one of {SUPPORTED_OCR_CROP_VIEWS}, got {raw_crop_view!r}."
            )
        validated_crop_view = raw_crop_view.strip()

        raw_model_dir = (
            os.environ.get(ENV_OCR_MODEL_DIR, "") if model_dir is None else model_dir
        )
        raw_device = os.environ.get(ENV_ANPR_DEVICE, "cpu") if device is None else device

        verified_dir = verify_confinement_under_models(
            raw_model_dir,
            repo_root=repo_root,
            expect_directory=True,
            setting_name=ENV_OCR_MODEL_DIR,
        )
        weight_files = _verify_local_ocr_assets(verified_dir)
        validated_device = _validate_device(raw_device)

        candidate_path = None
        raw_candidate = os.environ.get(ENV_OCR_RECOGNIZER_WEIGHTS, "") if recognizer_weights is None else recognizer_weights
        candidate_sha = os.environ.get(ENV_OCR_RECOGNIZER_SHA256, "") if recognizer_sha256 is None else recognizer_sha256
        if raw_candidate or candidate_sha:
            if not raw_candidate or not re.fullmatch(r"[0-9a-f]{64}", candidate_sha):
                raise RuntimeUnavailable("Adapted OCR weights require a model path and exact lowercase SHA-256.")
            candidate_path = verify_confinement_under_models(
                raw_candidate, repo_root=repo_root, expect_directory=False,
                setting_name=ENV_OCR_RECOGNIZER_WEIGHTS,
            )
            if _compute_file_sha256(candidate_path) != candidate_sha:
                raise RuntimeUnavailable("Adapted OCR checkpoint SHA-256 mismatch.")

        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeUnavailable(
                "EasyOCR is not installed in the local Python environment."
            ) from exc

        use_gpu: bool | str = False if validated_device == "cpu" else validated_device
        try:
            reader = easyocr.Reader(
                list(languages),
                gpu=use_gpu,
                model_storage_directory=str(verified_dir),
                user_network_directory=str(verified_dir),
                download_enabled=False,
                verbose=False,
                # A float checkpoint cannot be restored into dynamically quantized layers.
                **({"quantize": False} if candidate_path is not None else {}),
            )
            if candidate_path is not None:
                import torch
                state = torch.load(str(candidate_path), map_location="cpu", weights_only=True)
                if not isinstance(state, dict):
                    raise ValueError("Adapted OCR checkpoint must contain only a state dictionary.")
                if not hasattr(reader.recognizer, "module"):
                    state = {key.removeprefix("module."): value for key, value in state.items()}
                reader.recognizer.load_state_dict(state, strict=True)
                reader.recognizer.eval()
        except Exception as exc:
            raise RuntimeUnavailable(
                f"Unable to initialize local EasyOCR reader from {verified_dir}: {exc}"
            ) from exc

        self.model_dir = verified_dir
        self.weight_files = tuple(weight_files) + ((candidate_path,) if candidate_path is not None else ())
        self.recognizer_checkpoint_sha256 = candidate_sha or None
        self.device = validated_device
        self.languages = tuple(languages)
        self.crop_view = validated_crop_view
        self._reader = reader

    def recognize(self, plate_bgr: np.ndarray) -> OcrRead:
        """Preprocess a plate crop with bounded OpenCV ops and run offline EasyOCR."""
        validate_bgr_image(plate_bgr, label="plate_bgr")
        preprocessed = preprocess_plate_crop(plate_bgr)
        primary_image = getattr(preprocessed, self.crop_view)

        try:
            detections = self._reader.readtext(
                primary_image,
                detail=1,
                paragraph=False,
            )
        except Exception as exc:
            raise RuntimeUnavailable(f"EasyOCR readtext failed: {exc}") from exc

        if not isinstance(detections, Sequence):
            raise RuntimeUnavailable("EasyOCR returned an invalid detections sequence.")

        read = order_and_merge_ocr_detections(detections)

        # If the primary view produced no text on a tiny/low-contrast crop, fall back once
        if not read.raw_text.strip():
            fallback_image = (
                preprocessed.enhanced_grayscale
                if self.crop_view != "enhanced_grayscale"
                else preprocessed.resized_bgr
            )
            try:
                fallback_detections = self._reader.readtext(
                    fallback_image,
                    detail=1,
                    paragraph=False,
                )
            except Exception as exc:
                raise RuntimeUnavailable(f"EasyOCR fallback readtext failed: {exc}") from exc
            if isinstance(fallback_detections, Sequence):
                fallback_read = order_and_merge_ocr_detections(fallback_detections)
                if fallback_read.raw_text.strip():
                    return fallback_read

        return read
