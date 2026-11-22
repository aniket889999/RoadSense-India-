"""Deterministic normalization and a deliberately limited plate-format profile."""

from __future__ import annotations

import math
import re

from src.anpr.contracts import (
    FormatStatus, OcrAssessment, PlateFormatAssessment, ReadStatus, ReasonCode, TextCleanup,
)


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


# Indian high-security registration plates print this country code beside the
# registration; it is not part of the registration itself.
EMBLEM_TOKEN = "IND"
# Marks with no resemblance to a letter or digit (bolts, borders, emblem dots). Any
# other character, e.g. "!", "ß" or full-width letters, may be a misread character
# and is never removed.
REMOVABLE_MARKS = ".,:·•°"
_REMOVABLE = str.maketrans("", "", REMOVABLE_MARKS)


def derive_candidate_text(normalized_text: str) -> tuple[str | None, list[TextCleanup]]:
    """Derive a review candidate by explicit removals only; never substitutes characters.

    1. Remove the dot-like marks in ``REMOVABLE_MARKS`` (never letter/digit look-alikes).
    2. Remove a leading ``IND`` token only when the remainder matches IN_COMMON_V1.

    Returns ``(None, [])`` when nothing was removed or nothing would remain.
    """
    if not isinstance(normalized_text, str):
        raise TypeError("normalized_text must be a string.")
    steps: list[TextCleanup] = []
    text = normalized_text.translate(_REMOVABLE)
    if text != normalized_text:
        steps.append(TextCleanup.PUNCTUATION_REMOVED)
    remainder = text[len(EMBLEM_TOKEN):]
    if text.startswith(EMBLEM_TOKEN) and _COMMON_INDIAN_SHAPE.fullmatch(remainder):
        text = remainder
        steps.append(TextCleanup.LEADING_EMBLEM_REMOVED)
    if not steps or not text:
        return None, []
    return text, steps


# IN_PLATE_V2: state/UT codes in current use, plus legacy codes still seen on older
# vehicles (DN before the 2020 DNH-DD merger, OR before Odisha's OD, UA before
# Uttarakhand's UK, TS before Telangana reverted to TG in March 2024). Sources are
# recorded in docs/anpr/format_check_v2_results.md. A known code does not verify a
# registration.
STATE_UT_CODES = frozenset({
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "GA", "GJ", "HP", "HR", "JH", "JK",
    "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
    "TG", "TN", "TR", "UK", "UP", "WB",
})
LEGACY_STATE_CODES = frozenset({"DN", "OR", "TS", "UA"})
_BH_SERIES = re.compile(r"[0-9]{2}BH[0-9]{4}[A-Z]{2}", re.ASCII)  # MoRTH 2021: YY BH #### XX
_LETTER_FOR_DIGIT = {"0": "ODQ", "1": "IL", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
_DIGIT_FOR_LETTER = {letter: digit for digit, letters in _LETTER_FOR_DIGIT.items() for letter in letters}
MAX_LOOKALIKE_SWAPS = 2
MAX_SUGGESTIONS = 5


def _known_state(code: str) -> bool:
    return code in STATE_UT_CODES or code in LEGACY_STATE_CODES


def _pattern(text: str) -> str | None:
    if _COMMON_INDIAN_SHAPE.fullmatch(text):
        return "STATE_SERIES"
    if _BH_SERIES.fullmatch(text):
        return "BH_SERIES"
    return None


def lookalike_suggestions(text: str) -> list[str]:
    """Fewest-swap readings (at most two look-alike swaps) that fit a supported pattern.

    Suggestions are reviewer hints only; callers must never substitute them.
    """
    current_pattern = _pattern(text)
    if current_pattern == "BH_SERIES" or (current_pattern == "STATE_SERIES" and _known_state(text[:2])):
        return []
    frontier = {text}
    found: set[str] = set()
    for _ in range(MAX_LOOKALIKE_SWAPS):
        nxt = set()
        for current in frontier:
            for index, character in enumerate(current):
                for swap in _LETTER_FOR_DIGIT.get(character, "") + _DIGIT_FOR_LETTER.get(character, ""):
                    nxt.add(current[:index] + swap + current[index + 1:])
        for option in nxt:
            pattern = _pattern(option)
            if option != text and (pattern == "BH_SERIES" or (pattern == "STATE_SERIES" and _known_state(option[:2]))):
                found.add(option)
        if found:  # only the fewest-swap readings: a second swap is not offered when one suffices
            break
        frontier = nxt
    return sorted(found)[:MAX_SUGGESTIONS]


def assess_plate_format_v2(text: str, source: str) -> PlateFormatAssessment:
    pattern = _pattern(text)
    state_code = text[:2] if pattern == "STATE_SERIES" else None
    known = _known_state(state_code) if state_code is not None else None
    suggestions = lookalike_suggestions(text) if pattern is None or known is False else []
    from src.anpr.rto import lookup_origin  # local import: table loads lazily, offline

    origin = lookup_origin(text) if pattern == "STATE_SERIES" else None
    return PlateFormatAssessment(assessed_text_source=source, pattern=pattern, state_code=state_code,
                                 state_code_known=known, lookalike_suggestions=suggestions,
                                 registration_origin=origin)


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
    review_only: bool = False,
) -> OcrAssessment:
    """Preserve OCR evidence and derive a candidate/review decision.

    The default threshold is a starting policy to tune on validation data.
    An OCR engine score is not a calibrated probability of correct recognition.
    """
    threshold = _validate_confidence(minimum_confidence, "minimum_confidence")
    if not isinstance(review_only, bool):
        raise ValueError("review_only must be a boolean.")
    confidence = (
        None
        if ocr_confidence is None
        else _validate_confidence(ocr_confidence, "ocr_confidence")
    )
    normalized = normalize_plate_text(raw_text)
    candidate, cleanup = derive_candidate_text(normalized)
    format_status = check_plate_format(normalized)
    candidate_format = check_plate_format(candidate) if candidate is not None else None
    review_text = candidate if candidate is not None else normalized
    plate_format = (assess_plate_format_v2(review_text, "candidate" if candidate is not None else "normalized")
                    if review_text else None)
    reasons: list[ReasonCode] = []

    if format_status == FormatStatus.EMPTY:
        reasons.append(ReasonCode.NO_TEXT)
        recognition_status = ReadStatus.UNREADABLE
    else:
        if review_only:
            reasons.append(ReasonCode.OCR_REVIEW_ONLY)
        elif confidence is None:
            reasons.append(ReasonCode.OCR_CONFIDENCE_MISSING)
        elif confidence < threshold:
            reasons.append(ReasonCode.LOW_OCR_CONFIDENCE)
        if cleanup:
            reasons.append(ReasonCode.TEXT_CLEANED)
        if plate_format is not None and plate_format.state_code_known is False:
            reasons.append(ReasonCode.UNKNOWN_STATE_CODE)
        if format_status == FormatStatus.NO_MATCH:
            reasons.append(ReasonCode.FORMAT_NOT_SUPPORTED)
        recognition_status = ReadStatus.NEEDS_REVIEW if reasons else ReadStatus.READ

    return OcrAssessment(
        raw_text=raw_text,
        normalized_text=normalized,
        candidate_text=candidate,
        candidate_format_status=candidate_format,
        text_cleanup=cleanup,
        plate_format=plate_format,
        ocr_confidence=confidence,
        format_status=format_status,
        recognition_status=recognition_status,
        reason_codes=reasons,
    )
