"""Recompute the deterministic post-OCR text assessment of frozen predictions.

OCR output (``raw_text`` and score), boxes, detection scores and image quality are kept
exactly; only normalization, candidate cleanup, format checks and review reasons are
recomputed with the current policy. No model is loaded.
"""

from __future__ import annotations

from src.anpr.contracts import AnprImageResult, PlateResult, ReadStatus, ResultStatus
from src.anpr.validation import assess_ocr_read


def reassess_result(result: AnprImageResult, *, minimum_confidence: float, review_only: bool) -> AnprImageResult:
    plates = []
    for plate in result.plates:
        assessment = assess_ocr_read(plate.raw_text, plate.ocr_confidence,
                                     minimum_confidence=minimum_confidence, review_only=review_only)
        plates.append(PlateResult(**assessment.model_dump(), plate_id=plate.plate_id, bbox=plate.bbox,
                                  detection_confidence=plate.detection_confidence,
                                  image_quality=plate.image_quality))
    status = result.status
    if status not in (ResultStatus.UNAVAILABLE, ResultStatus.NO_PLATE):
        status = (ResultStatus.COMPLETE if all(p.recognition_status == ReadStatus.READ for p in plates)
                  else ResultStatus.NEEDS_REVIEW)
    return AnprImageResult(request_id=result.request_id, status=status, image_width=result.image_width,
                           image_height=result.image_height, plates=plates,
                           processing_ms=result.processing_ms, warnings=list(result.warnings))
