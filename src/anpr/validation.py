"""Deterministic normalization and a deliberately limited plate-format profile."""

from __future__ import annotations

import math
import re

from src.anpr.contracts import FormatStatus, OcrAssessment, ReadStatus, ReasonCode


# This is a shape profile, not a complete registry of Indian registration rules.
# BH, diplomatic, temporary and other unsupported layouts must remain reviewable.
_COMMON_INDIAN_SHAPE = re.compile(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}", re.ASCII)
_NORMALIZATION_TABLE = str.maketrans(
    "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", " \t\r\n\f\v-"
)


def normalize_plate_text(raw_text: str) -> str:
    """Normalize ASCII case/separators without guessing ambiguous characters.

    Retaining punctuation and non-ASCII characters prevents normalization from
    making unsupported OCR output appear to match the configured shape.
    """
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string.")
    if len(raw_text) > 256:
        raise ValueError("raw_text cannot exceed 256 characters.")
    return raw_text.translate(_NORMALIZATION_TABLE)


def check_plate_format(normalized_text: str) -> FormatStatus:
    """Check the IN_COMMON_V1 shape; no identity or registration lookup occurs."""
    if not isinstance(normalized_text, str):
        raise TypeError("normalized_text must be a string.")
    if not normalized_text:
        return FormatStatus.EMPTY
    if _COMMON_INDIAN_SHAPE.fullmatch(normalized_text):
        return FormatStatus.MATCHED
    return FormatStatus.NO_MATCH


def _validate_confidence(value: float, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{field_name} must be a finite number between 0 and 1.")
    return float(value)


def assess_ocr_read(
    raw_text: str,
    ocr_confidence: float | None,
    *,
    minimum_confidence: float = 0.75,
) -> OcrAssessment:
    """Preserve OCR evidence and derive a candidate/review decision.

    The default threshold is a starting policy to tune on validation data.
    An OCR engine score is not a calibrated probability of correct recognition.
    """
    threshold = _validate_confidence(minimum_confidence, "minimum_confidence")
    confidence = (
        None
        if ocr_confidence is None
        else _validate_confidence(ocr_confidence, "ocr_confidence")
    )
    normalized = normalize_plate_text(raw_text)
    format_status = check_plate_format(normalized)
    reasons: list[ReasonCode] = []

    if format_status == FormatStatus.EMPTY:
        reasons.append(ReasonCode.NO_TEXT)
        recognition_status = ReadStatus.UNREADABLE
    else:
        if confidence is None:
            reasons.append(ReasonCode.OCR_CONFIDENCE_MISSING)
        elif confidence < threshold:
            reasons.append(ReasonCode.LOW_OCR_CONFIDENCE)
        if format_status == FormatStatus.NO_MATCH:
            reasons.append(ReasonCode.FORMAT_NOT_SUPPORTED)
        recognition_status = ReadStatus.NEEDS_REVIEW if reasons else ReadStatus.READ

    return OcrAssessment(
        raw_text=raw_text,
        normalized_text=normalized,
        ocr_confidence=confidence,
        format_status=format_status,
        recognition_status=recognition_status,
        reason_codes=reasons,
    )
