"""Re-apply the CURRENT text policy to frozen VALIDATION OCR reads (no inference).

The OCR engine output (``raw_text``), boxes and detection scores are taken unchanged
from a frozen predictions file. Only the deterministic post-OCR assessment
(normalization, candidate cleanup, format check, review reasons) is recomputed, so
the effect of a text-policy change is measured on exactly the same reads.

The validation split is hard-coded; predictions containing any other image are refused.

Writes two new files under ``outputs/``: re-assessed predictions (schema 1.2, contains
OCR strings, keep out of Git) and a string-free JSON report comparing scoring of
``normalized_text`` and ``candidate_text``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPLIT = "val"


def reassess(result, *, minimum_confidence: float, review_only: bool):
    """Return a schema-1.2 result with only the text assessment recomputed."""
    from src.anpr.contracts import AnprImageResult, PlateResult, ReadStatus, ResultStatus
    from src.anpr.validation import assess_ocr_read

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
    return AnprImageResult(schema_version="1.2", request_id=result.request_id, status=status,
                           image_width=result.image_width, image_height=result.image_height, plates=plates,
                           processing_ms=result.processing_ms, warnings=list(result.warnings))


def cleanup_effects(records, frozen, reassessed, normalize) -> dict:
    """Per reference plate: did cleanup fix, break or not change the exact match?"""
    from src.anpr.evaluation import _iou

    effects = Counter()
    steps = Counter()
    for record in records:
        new = reassessed.get(record.image_id)
        plates_new = new.plates if new else []
        for plate in plates_new:
            for step in plate.text_cleanup:
                steps[step.value] += 1
        # Same greedy one-to-one IoU matching as src.anpr.evaluation.
        candidates = sorted((-_iou(g.bbox, p.bbox), gi, pi) for gi, g in enumerate(record.plates)
                            for pi, p in enumerate(plates_new) if _iou(g.bbox, p.bbox) >= 0.5)
        matches, used = {}, set()
        for _, gi, pi in candidates:
            if gi not in matches and pi not in used:
                matches[gi] = pi
                used.add(pi)
        for gi, gt in enumerate(record.plates):
            plate = plates_new[matches[gi]] if gi in matches else None
            if plate is None or plate.candidate_text is None:
                continue
            expected = normalize(gt.text)
            before = plate.normalized_text == expected
            after = plate.candidate_text == expected
            effects["fixed" if after and not before else "broke" if before and not after
                    else "still_correct" if before else "still_wrong"] += 1
    return {"plates_with_cleanup": sum(effects.values()), **{k: effects[k] for k in ("fixed", "broke", "still_wrong", "still_correct")},
            "cleanup_steps_applied": dict(steps)}


def _new_output(path: Path, suffix: str) -> Path:
    path = path if path.is_absolute() else ROOT / path
    if ".." in path.parts or path.suffix.lower() != suffix or not path.resolve().is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError(f"{path} must be a new {suffix} file inside outputs/")
    if path.exists():
        raise ValueError(f"{path} already exists")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Re-apply the current text policy to frozen VALIDATION reads.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="frozen predictions JSONL (with .provenance.json)")
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args(argv)

    from src.anpr.dataset import load_manifest, sha256_file
    from src.anpr.evaluation import evaluate_predictions, load_predictions
    from src.anpr.validation import normalize_plate_text

    out_predictions = _new_output(args.output_predictions, ".jsonl")
    out_report = _new_output(args.output_report, ".json")
    provenance = json.loads(args.predictions.with_suffix(".provenance.json").read_text())
    runtime = provenance["runtime"]
    review_only = bool(runtime.get("ocr_review_only", False))
    threshold = float(provenance["minimum_ocr_confidence_threshold"])

    records = [r for r in load_manifest(args.dataset / "manifest.jsonl", prepared=True) if r.split == SPLIT]
    frozen = load_predictions(args.predictions)
    if set(frozen) - {r.image_id for r in records}:
        print("ABORT: predictions contain images outside the validation split", file=sys.stderr)
        return 1
    reassessed = {image_id: reassess(result, minimum_confidence=threshold, review_only=review_only)
                  for image_id, result in frozen.items()}

    before = evaluate_predictions(records, frozen)["overall"]
    normalized = evaluate_predictions(records, reassessed, text_source="normalized")
    candidate = evaluate_predictions(records, reassessed, text_source="candidate")
    if (normalized["overall"]["correct_plate_reads"], normalized["overall"]["character_errors"]) != (
            before["correct_plate_reads"], before["character_errors"]):
        print("ABORT: re-assessment changed normalized-text scoring", file=sys.stderr)
        return 1

    keys = ("ground_truth_plates", "correct_plate_reads", "exact_plate_match_accuracy", "character_errors",
            "character_error_rate", "accepted_reads", "wrong_accepted_reads", "acceptance_coverage")
    report = {
        "schema_version": "1.0", "split": SPLIT, "policy": "candidate_cleanup_v1",
        "policy_steps": ["remove dot-like marks . , : · • ° (never letter/digit look-alikes)",
                         "remove a leading IND token only if the remainder matches IN_COMMON_V1"],
        "source_predictions_sha256": sha256_file(args.predictions),
        "ocr_review_only": review_only,
        "normalized_text": {k: normalized["overall"][k] for k in keys},
        "candidate_text": {k: candidate["overall"][k] for k in keys},
        "candidate_by_condition": {c: {k: v[k] for k in keys} for c, v in candidate["by_condition"].items()},
        "normalized_by_condition": {c: {k: v[k] for k in keys} for c, v in normalized["by_condition"].items()},
        "cleanup_effects": cleanup_effects(records, frozen, reassessed, normalize_plate_text),
        "interpretation": ("Validation-derived policy measured on the same frozen reads; agreement with "
                           "publisher_mixed labels. Requires confirmation on independent data."),
    }
    out_predictions.parent.mkdir(parents=True, exist_ok=True)
    with out_predictions.open("x", encoding="utf-8") as handle:
        for image_id in sorted(reassessed):
            handle.write(json.dumps({"image_id": image_id, "result": reassessed[image_id].model_dump(mode="json")}) + "\n")
    with out_report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
