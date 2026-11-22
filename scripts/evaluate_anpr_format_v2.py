"""Measure the IN_PLATE_V2 format check on frozen VALIDATION reads (no inference).

Re-assesses frozen RapidOCR and EasyOCR validation predictions with the current text
policy and reports, without any plate strings:

* how the 94 reference labels fall into the supported patterns;
* whether the review text's pattern / state-code result separates correct from wrong reads;
* whether look-alike suggestions contain the reference reading;
* whether the stricter UNKNOWN_STATE_CODE reason changes automatic acceptance (EasyOCR).

The validation split is hard-coded; predictions with any other image are refused.
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


def _greedy_matches(gt_plates, predicted, iou):
    candidates = sorted((-iou(g.bbox, p.bbox), gi, pi) for gi, g in enumerate(gt_plates)
                        for pi, p in enumerate(predicted) if iou(g.bbox, p.bbox) >= 0.5)
    matches, used = {}, set()
    for _, gi, pi in candidates:
        if gi not in matches and pi not in used:
            matches[gi] = pi
            used.add(pi)
    return matches


def format_bucket(fmt) -> str:
    if fmt is None:
        return "EMPTY"
    if fmt.pattern == "STATE_SERIES":
        return "STATE_KNOWN" if fmt.state_code_known else "STATE_UNKNOWN"
    return fmt.pattern or "NO_PATTERN"


def analyze_engine(records, predictions, normalize) -> dict:
    from src.anpr.evaluation import _iou

    from src.anpr.rto import lookup_origin

    table = Counter()
    suggestions = Counter()
    origin = Counter()
    for record in records:
        result = predictions.get(record.image_id)
        plates = result.plates if result else []
        matches = _greedy_matches(record.plates, plates, _iou)
        for gi, gt in enumerate(record.plates):
            if gi not in matches:
                table[("MISSED", "missed")] += 1
                continue
            plate = plates[matches[gi]]
            expected = normalize(gt.text)
            review = plate.candidate_text if plate.candidate_text is not None else plate.normalized_text
            correct = review == expected
            table[(format_bucket(plate.plate_format), "correct" if correct else "wrong")] += 1
            ref_origin = lookup_origin(expected)
            got_origin = plate.plate_format.registration_origin if plate.plate_format else None
            if ref_origin is not None:
                same = got_origin is not None and got_origin.rto_code == ref_origin.rto_code
                origin[("correct" if correct else "wrong", "origin_match" if same else "origin_differs")] += 1
            options = plate.plate_format.lookalike_suggestions if plate.plate_format else []
            if options:
                suggestions["with_suggestions_" + ("correct" if correct else "wrong")] += 1
                if not correct and expected in options:
                    suggestions["wrong_with_reference_in_suggestions"] += 1
                    if len(options) == 1:
                        suggestions["wrong_with_single_suggestion_equal_reference"] += 1
    buckets = sorted({b for b, _ in table})
    return {
        "review_text_by_format": {b: {"correct": table[(b, "correct")], "wrong": table[(b, "wrong")],
                                      "missed": table[(b, "missed")]} for b in buckets},
        "suggestions": {k: suggestions[k] for k in (
            "with_suggestions_wrong", "with_suggestions_correct",
            "wrong_with_reference_in_suggestions", "wrong_with_single_suggestion_equal_reference")},
        "state_rto_origin_vs_reference": {
            f"{a}_read_{b}": origin[(a, b)] for a in ("correct", "wrong") for b in ("origin_match", "origin_differs")
        },
    }


def acceptance(records, predictions, normalize) -> dict:
    from src.anpr.evaluation import evaluate_predictions

    overall = evaluate_predictions(records, predictions)["overall"]
    return {k: overall[k] for k in ("accepted_reads", "wrong_accepted_reads", "acceptance_coverage")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure IN_PLATE_V2 on frozen VALIDATION reads.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rapidocr", type=Path, required=True)
    parser.add_argument("--easyocr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSON file under outputs/")
    args = parser.parse_args(argv)

    from src.anpr.dataset import load_manifest, sha256_file
    from src.anpr.evaluation import load_predictions
    from src.anpr.reassessment import reassess_result
    from src.anpr.validation import assess_plate_format_v2, normalize_plate_text

    output = args.output if args.output.is_absolute() else ROOT / args.output
    if (output.suffix != ".json" or ".." in output.parts or output.exists()
            or not output.resolve().is_relative_to((ROOT / "outputs").resolve())):
        print("ABORT: output must be a new .json file inside outputs/", file=sys.stderr)
        return 1
    records = [r for r in load_manifest(args.dataset / "manifest.jsonl", prepared=True) if r.split == SPLIT]
    ids = {r.image_id for r in records}

    reference_patterns = Counter(format_bucket(assess_plate_format_v2(normalize_plate_text(p.text), "normalized"))
                                 for r in records for p in r.plates)
    from src.anpr.rto import load_table, lookup_origin

    ref_origins = [lookup_origin(normalize_plate_text(p.text)) for r in records for p in r.plates]
    report = {"schema_version": "1.0", "split": SPLIT, "profile": "IN_PLATE_V2",
              "reference_labels_by_format": dict(sorted(reference_patterns.items())),
              "reference_rto_coverage": {
                  "state_series_labels": sum(o is not None for o in ref_origins),
                  "rto_in_table": sum(bool(o and o.rto_in_table) for o in ref_origins),
                  "rto_table_revision": (load_table() or {}).get("source", {}).get("revision_id"),
              },
              "engines": {}}
    for name, path in (("rapidocr", args.rapidocr), ("easyocr", args.easyocr)):
        frozen = load_predictions(path)
        if set(frozen) - ids:
            print(f"ABORT: {name} predictions contain images outside the validation split", file=sys.stderr)
            return 1
        provenance = json.loads(path.with_suffix(".provenance.json").read_text())
        review_only = bool(provenance["runtime"].get("ocr_review_only", False))
        threshold = float(provenance["minimum_ocr_confidence_threshold"])
        current = {k: reassess_result(v, minimum_confidence=threshold, review_only=review_only)
                   for k, v in frozen.items()}
        report["engines"][name] = {
            "predictions_sha256": sha256_file(path), "ocr_review_only": review_only,
            "acceptance_frozen": acceptance(records, frozen, normalize_plate_text),
            "acceptance_current_policy": acceptance(records, current, normalize_plate_text),
            **analyze_engine(records, current, normalize_plate_text),
        }
    report["interpretation"] = ("Agreement with publisher_mixed labels on frozen validation reads; "
                                "a known state code or supported pattern does not verify a registration.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
