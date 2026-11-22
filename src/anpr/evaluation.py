"""Offline scoring of frozen ANPR predictions, including missed plates."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import ValidationError

from src.anpr.contracts import AnprImageResult, PlateBox, ReadStatus, ResultStatus
from src.anpr.dataset import PreparedRecord, summarize_evidence_assurance
from src.anpr.validation import check_plate_format, derive_candidate_text, normalize_plate_text

TEXT_SOURCES = ("normalized", "candidate")


class EvaluationError(ValueError):
    """Prediction evidence cannot be matched to the labelled evaluation set."""


def load_predictions(path: Path) -> dict[str, AnprImageResult]:
    predictions = {}
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict) or set(item) != {"image_id", "result"}:
                    raise EvaluationError("Each prediction row requires exactly image_id and result.")
                image_id = item["image_id"]
                if not isinstance(image_id, str) or not image_id or image_id in predictions:
                    raise EvaluationError("Prediction image IDs must be nonempty and unique.")
                predictions[image_id] = AnprImageResult.model_validate(item["result"])
    except (OSError, UnicodeError, ValueError, TypeError, ValidationError) as exc:
        raise EvaluationError(f"Invalid predictions file: {exc}") from exc
    return predictions


def character_edit_distance(expected: str, observed: str) -> int:
    """Levenshtein distance with memory proportional to the shorter string."""
    if len(expected) < len(observed):
        expected, observed = observed, expected
    prior = list(range(len(observed) + 1))
    for row, expected_character in enumerate(expected, start=1):
        current = [row]
        for column, observed_character in enumerate(observed, start=1):
            current.append(min(
                current[-1] + 1,
                prior[column] + 1,
                prior[column - 1] + (expected_character != observed_character),
            ))
        prior = current
    return prior[-1]


def _iou(left: PlateBox, right: PlateBox) -> float:
    intersection = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1)) * max(
        0.0, min(left.y2, right.y2) - max(left.y1, right.y1)
    )
    union = (
        (left.x2 - left.x1) * (left.y2 - left.y1)
        + (right.x2 - right.x1) * (right.y2 - right.y1) - intersection
    )
    return intersection / union


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _observed_text(plate, text_source: str) -> str:
    normalized = normalize_plate_text(plate.raw_text)
    if text_source == "candidate" and plate.candidate_text is not None:
        return plate.candidate_text
    return normalized


def _score(
    records: Sequence[PreparedRecord], predictions: Mapping[str, AnprImageResult], threshold: float,
    text_source: str = "normalized",
) -> dict:
    counts = {
        "images": len(records), "ground_truth_plates": 0, "predicted_plates": 0,
        "matched_detections": 0, "correct_plate_reads": 0, "plates_with_text": 0,
        "accepted_reads": 0, "accepted_matched_reads": 0, "wrong_accepted_reads": 0,
        "expected_characters": 0, "character_errors": 0,
        "missing_prediction_images": 0, "unavailable_images": 0,
    }
    latencies = []
    for record in records:
        ground_truth = record.plates
        counts["ground_truth_plates"] += len(ground_truth)
        result = predictions.get(record.image_id)
        if result is None:
            counts["missing_prediction_images"] += 1
        elif result.status == ResultStatus.UNAVAILABLE:
            counts["unavailable_images"] += 1
        else:
            latencies.append(result.processing_ms)
        plates = result.plates if result else []
        counts["predicted_plates"] += len(plates)

        # Greedy one-to-one IoU matching; text is never used to choose a match.
        candidates = sorted(
            ((-_iou(gt.bbox, plate.bbox), gt_index, predicted_index)
             for gt_index, gt in enumerate(ground_truth)
             for predicted_index, plate in enumerate(plates)
             if _iou(gt.bbox, plate.bbox) >= threshold)
        )
        matches = {}
        used_predictions = set()
        for _, gt_index, predicted_index in candidates:
            if gt_index not in matches and predicted_index not in used_predictions:
                matches[gt_index] = predicted_index
                used_predictions.add(predicted_index)
        counts["matched_detections"] += len(matches)

        correct_accepted = set()
        for gt_index, gt in enumerate(ground_truth):
            expected = normalize_plate_text(gt.text)
            plate = plates[matches[gt_index]] if gt_index in matches else None
            observed = _observed_text(plate, text_source) if plate else ""
            counts["expected_characters"] += len(expected)
            counts["character_errors"] += character_edit_distance(expected, observed)
            counts["plates_with_text"] += bool(observed)
            counts["correct_plate_reads"] += bool(plate and expected == observed)
            if plate and plate.recognition_status == ReadStatus.READ:
                counts["accepted_matched_reads"] += 1
                if expected == observed:
                    correct_accepted.add(matches[gt_index])

        for predicted_index, plate in enumerate(plates):
            if plate.recognition_status == ReadStatus.READ:
                counts["accepted_reads"] += 1
                if predicted_index not in correct_accepted:
                    counts["wrong_accepted_reads"] += 1

    tp = counts["matched_detections"]
    gt_count = counts["ground_truth_plates"]
    return {
        **counts,
        "false_positive_detections": counts["predicted_plates"] - tp,
        "missed_plates": gt_count - tp,
        "detection_precision": _ratio(tp, counts["predicted_plates"]),
        "detection_recall": _ratio(tp, gt_count),
        "exact_plate_match_accuracy": _ratio(counts["correct_plate_reads"], gt_count),
        "character_error_rate": _ratio(counts["character_errors"], counts["expected_characters"]),
        "read_coverage": _ratio(counts["plates_with_text"], gt_count),
        "acceptance_coverage": _ratio(counts["accepted_matched_reads"], gt_count),
        "wrong_accepted_read_rate": _ratio(counts["wrong_accepted_reads"], counts["accepted_reads"]),
        "latency_sample_count": len(latencies),
        "mean_processing_ms": sum(latencies) / len(latencies) if latencies else None,
    }


def evaluate_predictions(
    records: Sequence[PreparedRecord], predictions: Mapping[str, AnprImageResult], *, iou_threshold: float = 0.5,
    text_source: str = "normalized",
) -> dict:
    """Score all labelled plates; absent/unavailable outputs remain misses.

    CER includes full deletions for unmatched ground-truth plates and can exceed
    1.0. Undefined rates are null, not a claimed zero error rate. False-positive
    accepted reads affect wrong-accepted metrics even on negative images.
    """
    if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, (int, float)):
        raise EvaluationError("IoU threshold must be numeric.")
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise EvaluationError("IoU threshold must lie in (0, 1].")
    if text_source not in TEXT_SOURCES:
        raise EvaluationError(f"text_source must be one of {TEXT_SOURCES}.")
    if not records:
        raise EvaluationError("Evaluation requires at least one labelled image.")
    identities = {record.image_id for record in records}
    if len(identities) != len(records):
        raise EvaluationError("Ground-truth image IDs must be unique.")
    if set(predictions) - identities:
        raise EvaluationError("Predictions contain images outside the selected evaluation split.")
    for record in records:
        result = predictions.get(record.image_id)
        if result is None:
            continue
        if (result.image_width, result.image_height) != (record.width, record.height):
            raise EvaluationError("Prediction dimensions differ from ground-truth image dimensions.")
        for plate in result.plates:
            if plate.normalized_text != normalize_plate_text(plate.raw_text):
                raise EvaluationError("Prediction normalized text differs from its raw OCR evidence.")
            if plate.candidate_text is not None and (
                derive_candidate_text(plate.normalized_text) != (plate.candidate_text, plate.text_cleanup)
                or check_plate_format(plate.candidate_text) != plate.candidate_format_status
            ):
                raise EvaluationError("Prediction candidate text is not derived from its raw OCR evidence.")

    conditions = sorted({condition for record in records for condition in record.conditions})
    assurance = summarize_evidence_assurance([record.model_dump(mode="json") for record in records])
    return {
        "schema_version": "1.0", "iou_threshold": float(iou_threshold),
        "matching_policy": "greedy_descending_iou_without_text_v1",
        "text_source": text_source,
        "reference_evidence": assurance,
        "metric_interpretation": (
            "accuracy_against_human_verified_labels"
            if assurance["all_plate_texts_human_verified"] else "agreement_with_supplied_labels"
        ),
        "overall": _score(records, predictions, iou_threshold, text_source),
        "by_condition": {
            condition: _score(
                [record for record in records if condition in record.conditions], predictions, iou_threshold, text_source,
            )
            for condition in conditions
        },
    }
