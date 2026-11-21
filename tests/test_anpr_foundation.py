"""ANPR evidence preservation, uncertainty handling and result boundary tests."""

import pytest
from pydantic import ValidationError

from src.anpr.contracts import AnprImageResult, FormatStatus, PlateBox, PlateResult, ReadStatus
from src.anpr.validation import assess_ocr_read, normalize_plate_text


def make_plate(**overrides):
    assessment = assess_ocr_read("mh 12-ab 1234", 0.92)
    values = {
        **assessment.model_dump(),
        "plate_id": "plate-1",
        "bbox": {"x1": 10, "y1": 20, "x2": 180, "y2": 75},
        "detection_confidence": 0.95,
    }
    values.update(overrides)
    return PlateResult(**values)


def test_normalization_preserves_original_evidence_and_does_not_guess():
    raw = " mh 12-ab 1234\n"
    result = assess_ocr_read(raw, 0.92)
    assert result.raw_text == raw
    assert result.normalized_text == "MH12AB1234"
    assert result.format_status == FormatStatus.MATCHED
    assert result.recognition_status == ReadStatus.READ
    assert normalize_plate_text("MH12AB12O4") == "MH12AB12O4"
    assert assess_ocr_read("MH12AB12O4", 0.99).recognition_status == ReadStatus.NEEDS_REVIEW


@pytest.mark.parametrize("raw", ["MH12AB1234!", "ＭＨ12AB1234", "MH12Aß1234", "24BH1234AA"])
def test_unsupported_text_is_not_silently_rewritten_to_a_match(raw):
    result = assess_ocr_read(raw, 0.99)
    assert result.format_status == FormatStatus.NO_MATCH
    assert result.recognition_status == ReadStatus.NEEDS_REVIEW
    assert "FORMAT_NOT_SUPPORTED" in result.reason_codes


@pytest.mark.parametrize("confidence", [None, 0.4])
def test_format_match_does_not_override_uncertain_ocr(confidence):
    result = assess_ocr_read("MH12AB1234", confidence)
    assert result.format_status == FormatStatus.MATCHED
    assert result.recognition_status == ReadStatus.NEEDS_REVIEW
    assert result.reason_codes


def test_empty_read_is_unreadable_and_preserves_raw_text():
    result = assess_ocr_read(" -- \n", 0.99)
    assert result.raw_text == " -- \n"
    assert result.normalized_text == ""
    assert result.recognition_status == ReadStatus.UNREADABLE
    assert result.reason_codes == ["NO_TEXT"]


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.1, 1.1, "0.9"])
def test_invalid_ocr_scores_cannot_produce_a_read(value):
    with pytest.raises(ValueError):
        assess_ocr_read("MH12AB1234", value)


def test_result_round_trip_preserves_bbox_and_ocr_fields():
    result = AnprImageResult(
        request_id="request-1",
        status="COMPLETE",
        image_width=640,
        image_height=480,
        plates=[make_plate()],
        processing_ms=12.5,
    )
    payload = result.model_dump(mode="json")
    assert payload["schema_version"] == "1.0"
    assert payload["plates"][0]["bbox"] == {"x1": 10, "y1": 20, "x2": 180, "y2": 75}
    assert payload["plates"][0]["recognition_status"] == "READ"
    assert AnprImageResult.model_validate_json(result.model_dump_json()) == result


def test_boxes_must_have_area_and_remain_within_image():
    with pytest.raises(ValidationError):
        PlateBox(x1=10, y1=20, x2=10, y2=75)
    with pytest.raises(ValidationError):
        AnprImageResult(
            request_id="request-1", status="COMPLETE", image_width=100,
            image_height=100, plates=[make_plate()], processing_ms=12.5,
        )


def test_completed_result_requires_actual_plate_evidence():
    with pytest.raises(ValidationError):
        AnprImageResult(
            request_id="request-1", status="COMPLETE", image_width=640,
            image_height=480, plates=[], processing_ms=12.5,
        )
    with pytest.raises(ValidationError):
        AnprImageResult(
            request_id="request-1", status="NO_PLATE", image_width=640,
            image_height=480, plates=[make_plate()], processing_ms=12.5,
        )


def test_uncertain_plate_cannot_be_published_as_complete():
    uncertain = assess_ocr_read("MH12AB1234", 0.25)
    plate = make_plate(**uncertain.model_dump())
    with pytest.raises(ValidationError):
        AnprImageResult(
            request_id="request-1", status="COMPLETE", image_width=640,
            image_height=480, plates=[plate], processing_ms=12.5,
        )
