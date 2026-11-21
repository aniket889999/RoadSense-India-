"""Metric denominator, matching and review-decision regressions."""

import pytest

from src.anpr.contracts import AnprImageResult, PlateBox, PlateResult
from src.anpr.dataset import PreparedRecord
from src.anpr.evaluation import EvaluationError, evaluate_predictions
from src.anpr.validation import assess_ocr_read


def record(image_id="image-1", *, text="MH12AB1234", conditions=None):
    return PreparedRecord(
        image_id=image_id, image_path=f"images/val/{image_id}.png", split="val",
        group_id=image_id, source_id=image_id, source_image_path=f"{image_id}.png",
        image_sha256="a" * 64, prepared_image_sha256="b" * 64, width=80, height=40,
        conditions=conditions or ["clear"],
        plates=[] if text is None else [{
            "bbox": {"x1": 5, "y1": 5, "x2": 70, "y2": 30}, "text": text,
        }],
    )


def prediction(text="MH12AB1234", *, confidence=0.9, status=None):
    assessment = assess_ocr_read(text, confidence)
    plate = PlateResult(
        **assessment.model_dump(), plate_id="plate-1", detection_confidence=0.95,
        bbox={"x1": 5, "y1": 5, "x2": 70, "y2": 30},
    )
    return AnprImageResult(
        request_id="prediction-1", status=status or (
            "COMPLETE" if plate.recognition_status == "READ" else "NEEDS_REVIEW"
        ), image_width=80, image_height=40, plates=[plate], processing_ms=10.0,
    )


def test_missing_predictions_remain_in_accuracy_and_cer_denominators():
    records = [record("a"), record("b", conditions=["blur"])]
    report = evaluate_predictions(records, {"a": prediction()})
    overall = report["overall"]
    assert overall["ground_truth_plates"] == 2
    assert overall["detection_recall"] == 0.5
    assert overall["exact_plate_match_accuracy"] == 0.5
    assert overall["character_error_rate"] == 0.5
    assert overall["missing_prediction_images"] == 1
    assert report["by_condition"]["blur"]["exact_plate_match_accuracy"] == 0.0


def test_unavailable_runtime_never_scores_as_success_or_inference_latency():
    unavailable = AnprImageResult(
        request_id="unavailable", status="UNAVAILABLE", image_width=80, image_height=40,
        processing_ms=1.0, warnings=["Models unavailable"],
    )
    overall = evaluate_predictions([record()], {"image-1": unavailable})["overall"]
    assert overall["unavailable_images"] == 1
    assert overall["exact_plate_match_accuracy"] == 0.0
    assert overall["character_error_rate"] == 1.0
    assert overall["detection_precision"] is None
    assert overall["wrong_accepted_read_rate"] is None
    assert overall["mean_processing_ms"] is None


def test_wrong_but_format_matching_read_is_counted_as_wrong_accepted():
    overall = evaluate_predictions([record()], {"image-1": prediction("MH12AB1235")})["overall"]
    assert overall["exact_plate_match_accuracy"] == 0.0
    assert overall["character_errors"] == 1
    assert overall["wrong_accepted_reads"] == 1
    assert overall["wrong_accepted_read_rate"] == 1.0


def test_false_positive_accepted_read_on_negative_image_is_not_ignored():
    overall = evaluate_predictions([record(text=None)], {"image-1": prediction()})["overall"]
    assert overall["false_positive_detections"] == 1
    assert overall["detection_precision"] == 0.0
    assert overall["detection_recall"] is None
    assert overall["wrong_accepted_reads"] == 1


def test_uncertain_but_correct_text_has_accuracy_without_acceptance():
    overall = evaluate_predictions([record()], {"image-1": prediction(confidence=0.2)})["overall"]
    assert overall["exact_plate_match_accuracy"] == 1.0
    assert overall["acceptance_coverage"] == 0.0
    assert overall["accepted_reads"] == 0


def test_mislocalized_plate_cannot_match_just_because_its_text_is_correct():
    observed = prediction()
    observed.plates[0].bbox = PlateBox(x1=71, y1=31, x2=79, y2=39)
    overall = evaluate_predictions([record()], {"image-1": observed})["overall"]
    assert overall["matched_detections"] == 0
    assert overall["exact_plate_match_accuracy"] == 0.0
    assert overall["wrong_accepted_reads"] == 1


def test_predictions_cannot_include_another_split_or_hide_raw_ocr_changes():
    with pytest.raises(EvaluationError, match="outside"):
        evaluate_predictions([record()], {"test-image": prediction()})
    observed = prediction()
    observed.plates[0].normalized_text = "MH12AB1235"
    with pytest.raises(EvaluationError, match="raw OCR"):
        evaluate_predictions([record()], {"image-1": observed})
