"""Explicit candidate cleanup: raw evidence preserved, never auto-accepted, never substitutes."""
import pytest
from pydantic import ValidationError

from src.anpr.contracts import (
    AnprImageResult, FormatStatus, OcrAssessment, ReadStatus, ReasonCode, TextCleanup,
)
from src.anpr.validation import assess_ocr_read, derive_candidate_text


@pytest.mark.parametrize("normalized, candidate, steps", [
    ("MH12AB1234", None, []),                                   # nothing to clean
    ("MH12.AB1234", "MH12AB1234", [TextCleanup.PUNCTUATION_REMOVED]),
    ("INDMH12AB1234", "MH12AB1234", [TextCleanup.LEADING_EMBLEM_REMOVED]),
    ("IND.MH12AB1234", "MH12AB1234", [TextCleanup.PUNCTUATION_REMOVED, TextCleanup.LEADING_EMBLEM_REMOVED]),
    ("INDMH12AB12", None, []),                 # remainder unsupported -> emblem kept, nothing removed
    ("MH12AB1234IND", None, []),               # only a LEADING token is ever removed
    ("...", None, []),                         # nothing would remain
    ("MH12AB12O4", None, []),                  # look-alike characters are never substituted
    ("MH12AB1234!", None, []),                 # "!" may be a misread I/1: never removed
    ("MH12Aß1234", None, []),                  # non-ASCII letters may be misreads: never removed
    ("MH12AB1234°", "MH12AB1234", [TextCleanup.PUNCTUATION_REMOVED]),
])
def test_derive_candidate_text(normalized, candidate, steps):
    assert derive_candidate_text(normalized) == (candidate, steps)


def test_cleaned_read_preserves_evidence_and_is_never_read_even_with_high_confidence():
    result = assess_ocr_read("IND MH12·AB1234", 0.99)
    assert result.raw_text == "IND MH12·AB1234"
    assert result.normalized_text == "INDMH12·AB1234"
    assert result.candidate_text == "MH12AB1234"
    assert result.format_status == FormatStatus.NO_MATCH            # still describes normalized_text
    assert result.candidate_format_status == FormatStatus.MATCHED
    assert result.recognition_status == ReadStatus.NEEDS_REVIEW
    assert result.reason_codes == [ReasonCode.TEXT_CLEANED, ReasonCode.FORMAT_NOT_SUPPORTED]


def test_clean_reads_keep_previous_decisions():
    accepted = assess_ocr_read("mh 12-ab 1234", 0.92)
    assert accepted.recognition_status == ReadStatus.READ and accepted.candidate_text is None
    review = assess_ocr_read("MH12AB1234", None, review_only=True)
    assert review.reason_codes == [ReasonCode.OCR_REVIEW_ONLY] and review.text_cleanup == []


def test_contract_rejects_inconsistent_cleanup_fields():
    base = assess_ocr_read("IND MH12AB1234", None, review_only=True).model_dump()
    for bad in (
        {"text_cleanup": []},                                   # candidate without steps
        {"candidate_text": None},                               # steps without candidate
        {"reason_codes": [ReasonCode.OCR_REVIEW_ONLY]},         # missing TEXT_CLEANED
        {"candidate_text": base["normalized_text"]},            # candidate equal to normalized
        {"candidate_format_status": None},                      # candidate without its own format
    ):
        with pytest.raises(ValidationError):
            OcrAssessment(**{**base, **bad})
    plain = assess_ocr_read("MH12AB1234", None, review_only=True).model_dump()
    with pytest.raises(ValidationError):
        OcrAssessment(**{**plain, "reason_codes": [ReasonCode.OCR_REVIEW_ONLY, ReasonCode.TEXT_CLEANED]})


def test_current_schema_is_default_and_saved_10_11_results_remain_readable():
    plate = {
        "raw_text": "MH12AB1234", "normalized_text": "MH12AB1234", "ocr_confidence": None,
        "format_status": "MATCHED", "recognition_status": "NEEDS_REVIEW",
        "reason_codes": ["OCR_CONFIDENCE_MISSING"], "plate_id": "plate-1",
        "bbox": {"x1": 1, "y1": 1, "x2": 20, "y2": 10}, "detection_confidence": 0.9,
    }
    for version in ("1.0", "1.1"):
        old = AnprImageResult.model_validate({
            "schema_version": version, "request_id": "r", "status": "NEEDS_REVIEW",
            "image_width": 40, "image_height": 20, "plates": [plate], "processing_ms": 1.0,
        })
        assert old.schema_version == version and old.plates[0].candidate_text is None
    assert AnprImageResult(request_id="r", status="NO_PLATE", image_width=4, image_height=4,
                           processing_ms=0.0).schema_version == "1.3"


def test_evaluation_scores_candidate_only_when_asked_and_rejects_tampered_candidates():
    from tests.test_anpr_evaluation import prediction, record
    from src.anpr.evaluation import EvaluationError, evaluate_predictions

    records = [record("a")]
    cleaned = {"a": prediction("IND MH12AB1234", confidence=None)}
    assert evaluate_predictions(records, cleaned)["overall"]["correct_plate_reads"] == 0
    report = evaluate_predictions(records, cleaned, text_source="candidate")
    assert report["overall"]["correct_plate_reads"] == 1 and report["text_source"] == "candidate"
    assert report["overall"]["accepted_reads"] == 0          # cleaned reads are never accepted

    tampered = cleaned["a"].model_copy(deep=True)
    tampered.plates[0].candidate_text = "MH12AB9999"  # derivation check must catch this
    with pytest.raises(EvaluationError):
        evaluate_predictions(records, {"a": tampered}, text_source="candidate")
    with pytest.raises(EvaluationError):
        evaluate_predictions(records, cleaned, text_source="other")
