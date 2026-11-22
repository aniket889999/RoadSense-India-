"""Categorize OCR failures in frozen ANPR VALIDATION predictions (no inference).

Reads an existing predictions JSONL and the prepared manifest, matches plates with
the same greedy IoU policy as ``src.anpr.evaluation``, and assigns every validation
reference plate exactly one failure category. Totals are cross-checked against
``evaluate_predictions``; the script aborts if they disagree.

The validation split is hard-coded: there is no option to analyze the held-out test
split, and test rows are never selected.

Outputs (both must be new files under ``outputs/``):
  --output   aggregate JSON only (counts, rates, character-confusion pairs); contains
             no plate strings and is suitable for quoting in documentation.
  --details  optional per-plate JSONL WITH reference/observed strings for local
             review; keep it out of Git.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import re
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPLIT = "val"
IOU_THRESHOLD = 0.5
CONFUSABLE_PAIRS = frozenset(frozenset(p) for p in (
    "O0", "D0", "Q0", "I1", "L1", "B8", "S5", "Z2", "G6", "A4",
))
CATEGORIES = (
    "CORRECT",
    "MISSED_DETECTION",      # no predicted box with IoU >= 0.5
    "EMPTY_READ",            # matched box, OCR returned no characters
    "REORDERED",             # same characters, different order (e.g. line order)
    "FRAGMENT",              # read is a contiguous piece of the reference, >= 2 chars missing
    "DROPPED_CHARS",         # only deletions, otherwise
    "EXTRA_CHARS",           # only insertions (spurious characters)
    "CONFUSABLE_ONLY",       # only substitutions, all within look-alike pairs
    "MIXED_ERRORS",          # anything else
)
_NON_ALNUM = re.compile(r"[^A-Z0-9]")
EMBLEM_TOKEN = "IND"  # text on Indian high-security registration plates, not part of the registration


def strip_non_alnum(text: str) -> str:
    return _NON_ALNUM.sub("", text)


def strip_non_alnum_and_leading_emblem(text: str) -> str:
    text = strip_non_alnum(text)
    return text[len(EMBLEM_TOKEN):] if text.startswith(EMBLEM_TOKEN) else text


COUNTERFACTUALS = {
    "strip_non_alnum": strip_non_alnum,
    "strip_non_alnum_and_leading_IND": strip_non_alnum_and_leading_emblem,
}
HEIGHT_BUCKETS = ((0, 40, "<40px"), (40, 80, "40-79px"), (80, 10 ** 9, ">=80px"))


# ── text alignment ───────────────────────────────────────────────────────────

def align(expected: str, observed: str) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Levenshtein alignment as (op, expected_char, observed_char); op in = S D I."""
    rows, cols = len(expected) + 1, len(observed) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1,
                             cost[i - 1][j - 1] + (expected[i - 1] != observed[j - 1]))
    ops = []
    i, j = len(expected), len(observed)
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (expected[i - 1] != observed[j - 1]):
            ops.append(("=" if expected[i - 1] == observed[j - 1] else "S", expected[i - 1], observed[j - 1]))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("D", expected[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, observed[j - 1]))
            j -= 1
    return ops[::-1]


def categorize(expected: str, observed: Optional[str]) -> str:
    """Assign one category; ``observed=None`` means no matched detection."""
    if observed is None:
        return "MISSED_DETECTION"
    if observed == expected:
        return "CORRECT"
    if not observed:
        return "EMPTY_READ"
    if sorted(observed) == sorted(expected):
        return "REORDERED"
    if len(expected) - len(observed) >= 2 and observed in expected:
        return "FRAGMENT"
    kinds = {op for op, _, _ in align(expected, observed) if op != "="}
    if kinds == {"D"}:
        return "DROPPED_CHARS"
    if kinds == {"I"}:
        return "EXTRA_CHARS"
    if kinds == {"S"}:
        subs = [(e, o) for op, e, o in align(expected, observed) if op == "S"]
        if all(frozenset((e, o)) in CONFUSABLE_PAIRS for e, o in subs):
            return "CONFUSABLE_ONLY"
    return "MIXED_ERRORS"


# ── matching (same policy as src.anpr.evaluation) ─────────────────────────────

def _iou(a, b) -> float:
    inter = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


def match(gt_boxes: Sequence, pred_boxes: Sequence, threshold: float = IOU_THRESHOLD) -> Dict[int, int]:
    candidates = sorted((-_iou(g, p), gi, pi)
                        for gi, g in enumerate(gt_boxes) for pi, p in enumerate(pred_boxes)
                        if _iou(g, p) >= threshold)
    matches, used = {}, set()
    for _, gi, pi in candidates:
        if gi not in matches and pi not in used:
            matches[gi] = pi
            used.add(pi)
    return matches


def height_bucket(height: float) -> str:
    for low, high, name in HEIGHT_BUCKETS:
        if low <= height < high:
            return name
    return HEIGHT_BUCKETS[-1][2]


# ── analysis ─────────────────────────────────────────────────────────────────

def analyze(records: Iterable, predictions: Mapping, layouts: Mapping[str, str], normalize) -> Tuple[dict, List[dict]]:
    """Return (aggregate report without plate strings, per-plate detail rows)."""
    details: List[dict] = []
    confusions: Counter = Counter()
    deleted: Counter = Counter()
    inserted: Counter = Counter()
    position_errors: Counter = Counter()
    position_totals: Counter = Counter()
    for record in records:
        if record.split != SPLIT:
            raise ValueError("analyze() received a non-validation record")
        result = predictions.get(record.image_id)
        plates = result.plates if result else []
        matches = match([g.bbox for g in record.plates], [p.bbox for p in plates])
        plate_layouts = layouts.get(record.image_id, [])
        condition = "low_light" if "low_light" in record.conditions else "other"
        for gi, gt in enumerate(record.plates):
            layout = plate_layouts[gi] if gi < len(plate_layouts) else "unknown"
            expected = normalize(gt.text)
            plate = plates[matches[gi]] if gi in matches else None
            observed = normalize(plate.raw_text) if plate else None
            category = categorize(expected, observed)
            ops = align(expected, observed or "")
            errors = sum(op != "=" for op, _, _ in ops)
            for op, e, o in ops:
                if op == "S":
                    confusions[f"{e}->{o}"] += 1
                elif op == "D":
                    deleted[e] += 1
                elif op == "I":
                    inserted[o] += 1
            if observed is not None and len(observed) == len(expected):
                for index, (e, o) in enumerate(zip(expected, observed)):
                    position_totals[index] += 1
                    position_errors[index] += e != o
            details.append({
                "image_id": record.image_id, "layout": layout, "condition": condition,
                "gt_height_px": round(gt.bbox.y2 - gt.bbox.y1, 1),
                "height_bucket": height_bucket(gt.bbox.y2 - gt.bbox.y1),
                "category": category, "edit_distance": errors, "expected_len": len(expected),
                "reference": expected, "observed": observed,
            })

    def summarize(rows: List[dict]) -> dict:
        counts = Counter(r["category"] for r in rows)
        chars = sum(r["expected_len"] for r in rows)
        edits = sum(r["edit_distance"] for r in rows)
        return {
            "plates": len(rows), "correct": counts["CORRECT"],
            "exact_rate": counts["CORRECT"] / len(rows) if rows else None,
            "expected_characters": chars, "character_errors": edits,
            "cer": edits / chars if chars else None,
            "categories": {c: counts[c] for c in CATEGORIES},
        }

    def by(key: str) -> dict:
        return {value: summarize([r for r in details if r[key] == value])
                for value in sorted({r[key] for r in details})}

    report = {
        "schema_version": "1.0", "split": SPLIT, "iou_threshold": IOU_THRESHOLD,
        "matching_policy": "greedy_descending_iou_without_text_v1",
        "category_definitions": {
            "FRAGMENT": "observed is a contiguous substring of the reference with >= 2 characters missing",
            "REORDERED": "same multiset of characters in a different order",
            "CONFUSABLE_ONLY": "substitutions only, each within " + ", ".join(sorted("/".join(sorted(p)) for p in CONFUSABLE_PAIRS)),
        },
        "overall": summarize(details),
        "by_layout": by("layout"), "by_condition": by("condition"), "by_height_bucket": by("height_bucket"),
        "layout_x_condition": {f"{l}|{c}": summarize([r for r in details if r["layout"] == l and r["condition"] == c])
                               for l in sorted({r["layout"] for r in details})
                               for c in sorted({r["condition"] for r in details})},
        "top_substitutions": dict(confusions.most_common(15)),
        "top_deleted_characters": dict(deleted.most_common(10)),
        "top_inserted_characters": dict(inserted.most_common(10)),
        "same_length_position_error_rate": {
            str(i): {"errors": position_errors[i], "plates": position_totals[i],
                     "rate": position_errors[i] / position_totals[i]}
            for i in sorted(position_totals)
        },
        "diagnostic_counterfactuals": {
            "note": ("NOT applied to the pipeline. Exact matches if a post-OCR rule were applied to the same "
                     "frozen validation reads; derived on validation, so any adopted rule must be frozen and "
                     "confirmed on independent data."),
            "baseline_exact": sum(r["observed"] == r["reference"] for r in details),
            **{name: sum(r["observed"] is not None and fn(r["observed"]) == r["reference"] for r in details)
               for name, fn in COUNTERFACTUALS.items()},
            "references_containing_emblem_token": sum(EMBLEM_TOKEN in r["reference"] for r in details),
        },
        "labels_note": "Agreement with publisher_mixed labels; not human-verified accuracy.",
    }
    return report, details


def load_layouts(plates_csv: Path, records: Iterable) -> Dict[str, List[str]]:
    """Per-plate layout from the source CSV, paired by plate order and confirmed by box.

    A plate is labelled only if the CSV row at the same index has a box within 1 px of
    the manifest box; otherwise it is "unknown" (never guessed).
    """
    rows: Dict[str, List[dict]] = {}
    with plates_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(Path(row["image"]).stem, []).append(row)
    layouts: Dict[str, List[str]] = {}
    for record in records:
        stem = record.image_id[len("witvix-"):] if record.image_id.startswith("witvix-") else None
        source = sorted(rows.get(stem, []), key=lambda r: int(r.get("plate_index") or 0))
        labels = []
        for index, plate in enumerate(record.plates):
            label = "unknown"
            if index < len(source):
                r = source[index]
                try:
                    coords = [float(r[k]) for k in ("x_min", "y_min", "x_max", "y_max")]
                except (KeyError, TypeError, ValueError):
                    coords = None
                box = (plate.bbox.x1, plate.bbox.y1, plate.bbox.x2, plate.bbox.y2)
                if coords and max(abs(a - b) for a, b in zip(coords, box)) <= 1.0:
                    label = r.get("layout") or "unknown"
            labels.append(label)
        layouts[record.image_id] = labels
    return layouts


def _new_output(path: Path, suffix: str) -> Path:
    path = path if path.is_absolute() else ROOT / path
    outputs = (ROOT / "outputs").resolve()
    if ".." in path.parts or path.suffix.lower() != suffix or not path.resolve().is_relative_to(outputs):
        raise ValueError(f"{path} must be a new {suffix} file inside outputs/")
    if path.exists():
        raise ValueError(f"{path} already exists; choose a new path")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Categorize OCR failures in frozen VALIDATION predictions.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--plates-csv", type=Path, required=True, help="raw Witvix plates.csv (read-only)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", type=Path)
    args = parser.parse_args(argv)

    from src.anpr.dataset import load_manifest
    from src.anpr.evaluation import evaluate_predictions, load_predictions
    from src.anpr.validation import normalize_plate_text

    output = _new_output(args.output, ".json")
    details_path = _new_output(args.details, ".jsonl") if args.details else None
    records = [r for r in load_manifest(args.dataset / "manifest.jsonl", prepared=True) if r.split == SPLIT]
    predictions = load_predictions(args.predictions)
    layouts = load_layouts(args.plates_csv, records)
    report, details = analyze(records, predictions, layouts, normalize_plate_text)

    frozen = evaluate_predictions(records, predictions, iou_threshold=IOU_THRESHOLD)["overall"]
    mine = report["overall"]
    if (mine["correct"], mine["character_errors"], mine["plates"]) != (
            frozen["correct_plate_reads"], frozen["character_errors"], frozen["ground_truth_plates"]):
        print(f"ABORT: totals disagree with evaluate_predictions: {mine} vs {frozen}", file=sys.stderr)
        return 1
    report["cross_check"] = {"matches_evaluate_predictions": True, "correct_plate_reads": frozen["correct_plate_reads"],
                             "character_errors": frozen["character_errors"]}
    from src.anpr.dataset import sha256_file
    report["predictions_sha256"] = sha256_file(args.predictions)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if details_path:
        with details_path.open("x", encoding="utf-8") as handle:
            for row in details:
                handle.write(json.dumps(row) + "\n")
    print(json.dumps({k: report[k] for k in ("overall", "by_layout", "by_condition", "diagnostic_counterfactuals", "cross_check")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
