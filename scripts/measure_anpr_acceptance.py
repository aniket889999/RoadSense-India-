"""Measure risk vs coverage of automatic acceptance on the VALIDATION split (no policy change).

RapidOCR runs in review-only mode in the application, so its score is withheld and the
frozen predictions carry no score. This script re-runs the SAME detector, crops and
RapidOCR configuration as ``scripts/serve_anpr_local.py`` on validation images, with the
score retained, and:

1. checks that every raw OCR string equals the frozen review-only run (determinism);
2. for candidate acceptance rules (score threshold alone, and format v2 gate + score),
   reports accepted plates, wrong accepted plates, coverage, and a 95% upper bound on the
   wrong-acceptance rate (exact one-sided binomial / Clopper-Pearson).

It changes no application setting. The held-out test split cannot be selected. Outputs are
new files under ``outputs/``; the per-plate file contains OCR strings and stays out of Git.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPLIT = "val"
MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def upper_bound_95(errors: int, n: int) -> Optional[float]:
    """One-sided 95% Clopper-Pearson upper bound on a binomial rate (bisection, no SciPy)."""
    if n == 0:
        return None
    if errors >= n:
        return 1.0
    from math import comb

    def cdf(p: float) -> float:
        return sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(errors + 1))

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if cdf(mid) > 0.05:
            lo = mid
        else:
            hi = mid
    return hi


def risk_coverage(rows: List[dict], total_plates: int, gate: bool) -> List[dict]:
    """Accept plates with score >= t (and passing the format gate if requested), for each t."""
    pool = [r for r in rows if r["score"] is not None and (r["gate"] if gate else True)]
    thresholds = sorted({round(r["score"], 4) for r in pool})
    table = []
    for t in thresholds:
        accepted = [r for r in pool if r["score"] >= t]
        wrong = sum(not r["correct"] for r in accepted)
        table.append({"threshold": t, "accepted": len(accepted), "wrong_accepted": wrong,
                      "coverage": len(accepted) / total_plates,
                      "wrong_rate_upper95": upper_bound_95(wrong, len(accepted))})
    return table


def summarize(table: List[dict]) -> dict:
    zero = [row for row in table if row["wrong_accepted"] == 0]
    best = max(zero, key=lambda row: row["accepted"]) if zero else None
    return {"max_coverage_with_zero_wrong": best,
            "lowest_threshold_row": table[0] if table else None}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validation-only acceptance risk/coverage (no policy change).")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True, help="frozen RapidOCR review-only validation predictions")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--output-rows", type=Path, required=True, help="new .jsonl under outputs/ (contains OCR text)")
    parser.add_argument("--output-report", type=Path, required=True, help="new .json under outputs/ (no strings)")
    args = parser.parse_args(argv)

    for path, suffix in ((args.output_rows, ".jsonl"), (args.output_report, ".json")):
        full = path if path.is_absolute() else ROOT / path
        if full.suffix != suffix or full.exists() or not full.resolve().is_relative_to((ROOT / "outputs").resolve()):
            print(f"ABORT: {path} must be a new {suffix} file inside outputs/", file=sys.stderr)
            return 1

    import importlib.util
    spec = importlib.util.spec_from_file_location("serve_anpr_local", ROOT / "scripts" / "serve_anpr_local.py")
    launcher = importlib.util.module_from_spec(spec)
    sys.modules["serve_anpr_local"] = launcher
    spec.loader.exec_module(launcher)
    os.environ.update({k: v for k, v in launcher.build_environment("rapidocr", args.device, inherited={}).items()
                       if k.startswith("ROADSENSE_ANPR_")})

    from src.anpr.dataset import load_manifest, sha256_file
    from src.anpr.detector import UltralyticsPlateDetector
    from src.anpr.evaluation import _iou, load_predictions
    from src.anpr.ocr import RapidOcrPlateRecognizer
    from src.anpr.pipeline import AnprImageService
    from src.anpr.runtime_contracts import AnprRuntime
    from src.anpr.validation import assess_ocr_read, normalize_plate_text

    records = [r for r in load_manifest(args.dataset / "manifest.jsonl", prepared=True) if r.split == SPLIT]
    frozen = load_predictions(args.frozen)
    if set(frozen) - {r.image_id for r in records}:
        print("ABORT: frozen predictions contain non-validation images", file=sys.stderr)
        return 1

    runtime = AnprRuntime(detector=UltralyticsPlateDetector(repo_root=ROOT),
                          recognizer=RapidOcrPlateRecognizer(repo_root=ROOT, review_only=False))
    service = AnprImageService(lambda: runtime)

    rows, mismatches, total_plates = [], 0, 0
    for record in records:
        path = args.dataset / record.image_path
        result = service.recognize_bytes(path.read_bytes(), MEDIA[path.suffix.lower()], f"accept-{record.image_id}")
        old = frozen.get(record.image_id)
        old_plates = old.plates if old else []
        if [p.raw_text for p in result.plates] != [p.raw_text for p in old_plates]:
            mismatches += 1
        total_plates += len(record.plates)
        candidates = sorted((-_iou(g.bbox, p.bbox), gi, pi) for gi, g in enumerate(record.plates)
                            for pi, p in enumerate(result.plates) if _iou(g.bbox, p.bbox) >= 0.5)
        matches, used = {}, set()
        for _, gi, pi in candidates:
            if gi not in matches and pi not in used:
                matches[gi] = pi
                used.add(pi)
        matched_pred = set(matches.values())
        for pi, plate in enumerate(result.plates):
            review = assess_ocr_read(plate.raw_text, None, review_only=True)
            text = review.candidate_text if review.candidate_text is not None else review.normalized_text
            fmt = review.plate_format
            gate = bool(fmt and (fmt.pattern == "BH_SERIES" or (fmt.pattern == "STATE_SERIES" and fmt.state_code_known)))
            gi = next((g for g, p in matches.items() if p == pi), None)
            expected = normalize_plate_text(record.plates[gi].text) if gi is not None else None
            rows.append({"image_id": record.image_id, "plate_id": plate.plate_id, "matched": pi in matched_pred,
                         "score": plate.ocr_confidence, "gate": gate,
                         "correct": expected is not None and text == expected,
                         "review_text": text, "reference": expected})

    report = {
        "schema_version": "1.0", "split": SPLIT, "device": args.device, "ocr_engine": "rapidocr",
        "frozen_predictions_sha256": sha256_file(args.frozen),
        "determinism": {"images": len(records), "images_with_raw_text_mismatch": mismatches},
        "reference_plates": total_plates, "predicted_plates": len(rows),
        "predicted_plates_with_score": sum(r["score"] is not None for r in rows),
        "note": ("Unmatched predicted boxes count as wrong when accepted. Validation-derived: a threshold "
                 "chosen here must be frozen and confirmed on independent data. No policy was changed."),
        "score_only": summarize(risk_coverage(rows, total_plates, gate=False)),
        "format_gate_plus_score": summarize(risk_coverage(rows, total_plates, gate=True)),
        "curves": {"score_only": risk_coverage(rows, total_plates, gate=False),
                   "format_gate_plus_score": risk_coverage(rows, total_plates, gate=True)},
    }
    out_rows = args.output_rows if args.output_rows.is_absolute() else ROOT / args.output_rows
    out_report = args.output_report if args.output_report.is_absolute() else ROOT / args.output_report
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    with out_rows.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with out_report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("determinism", "score_only", "format_gate_plus_score")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
