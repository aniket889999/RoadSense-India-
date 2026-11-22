"""Independent passenger-car evaluation set: template, label validation and frozen evaluation.

  template  write an empty manifest CSV with the required columns
  validate  check permissions, double labels, adjudication and condition tags
  evaluate  run the FROZEN pipeline once and report agreement by condition (no tuning)

The manifest and images contain real plates: keep them in a Git-ignored folder such as
``data/raw/car_eval_v1/``. See docs/anpr/car_evaluation_protocol.md.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COLUMNS = ("image_file", "permission", "labeler_a", "label_a", "labeler_b", "label_b", "adjudicated",
           "blur", "angle", "lighting", "plate_layout", "vehicle_type", "notes")
VOCAB = {
    "permission": {"own_vehicle", "owner_consent", "public_permitted"},
    "blur": {"none", "mild", "strong"},
    "angle": {"frontal", "oblique_lt30", "oblique_ge30"},
    "lighting": {"day", "night", "glare", "shade"},
    "plate_layout": {"single_line", "two_line"},
    "vehicle_type": {"car", "suv", "other"},
}
TAGS = ("blur", "angle", "lighting", "plate_layout", "vehicle_type")
MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def norm(text: str) -> str:
    from src.anpr.validation import normalize_plate_text
    return normalize_plate_text(text or "")


def validate_rows(rows: List[Dict[str, str]], image_dir: Path) -> dict:
    """Return {'errors': [...], 'stats': {...}}; errors are row-level, never plate strings."""
    errors: List[str] = []
    agree = disagree = 0
    tags = {t: Counter() for t in TAGS}
    seen = set()
    for i, row in enumerate(rows, start=2):  # CSV line numbers
        where = f"line {i}"
        missing = [c for c in COLUMNS if c not in row]
        if missing:
            errors.append(f"{where}: missing columns {missing}")
            continue
        for field, allowed in VOCAB.items():
            if row[field].strip() not in allowed:
                errors.append(f"{where}: {field} must be one of {sorted(allowed)}")
        name = row["image_file"].strip()
        if not name or Path(name).name != name or Path(name).suffix.lower() not in MEDIA:
            errors.append(f"{where}: image_file must be a plain .jpg/.jpeg/.png file name")
        elif not (image_dir / name).is_file():
            errors.append(f"{where}: image file not found")
        if name in seen:
            errors.append(f"{where}: duplicate image_file")
        seen.add(name)
        if not row["labeler_a"].strip() or not row["labeler_b"].strip():
            errors.append(f"{where}: both labeler ids are required")
        elif row["labeler_a"].strip() == row["labeler_b"].strip():
            errors.append(f"{where}: the two labels must come from different people")
        a, b, final = norm(row["label_a"]), norm(row["label_b"]), norm(row["adjudicated"])
        if not a or not b:
            errors.append(f"{where}: both independent labels are required")
            continue
        if a == b:
            agree += 1
            if final and final != a:
                errors.append(f"{where}: adjudicated text differs from two agreeing labels")
        else:
            disagree += 1
            if not final:
                errors.append(f"{where}: labels disagree; adjudicated text is required")
        for t in TAGS:
            tags[t][row[t].strip()] += 1
    total = agree + disagree
    return {"errors": errors, "stats": {
        "images": len(rows), "double_labelled": total,
        "labeler_agreement": agree / total if total else None, "needed_adjudication": disagree,
        "by_tag": {t: dict(sorted(c.items())) for t, c in tags.items()},
    }}


def reference(row: Dict[str, str]) -> str:
    a, b, final = norm(row["label_a"]), norm(row["label_b"]), norm(row["adjudicated"])
    return final or (a if a == b else "")


def summarize(results: List[dict]) -> dict:
    def block(items):
        n = len(items)
        hit = sum(r["hit"] for r in items)
        acc = [r for r in items if r["accepted"]]
        wrong = sum(not r["accepted_correct"] for r in acc)
        return {"images": n, "exact": hit, "exact_rate": hit / n if n else None,
                "no_plate": sum(r["no_plate"] for r in items),
                "accepted": len(acc), "wrong_accepted": wrong}
    out = {"overall": block(results), "by_tag": {}}
    for t in TAGS:
        groups = defaultdict(list)
        for r in results:
            groups[r[t]].append(r)
        out["by_tag"][t] = {k: block(v) for k, v in sorted(groups.items())}
    return out


def evaluate(rows, image_dir: Path, device: str, auto_accept: bool) -> List[dict]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("serve_anpr_local_car", ROOT / "scripts" / "serve_anpr_local.py")
    launcher = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = launcher
    spec.loader.exec_module(launcher)
    env = launcher.build_environment("rapidocr", device, inherited={}, auto_accept=auto_accept)
    os.environ.update({k: v for k, v in env.items() if k.startswith("ROADSENSE_ANPR_")})
    if not auto_accept:  # never inherit an acceptance policy from the shell
        os.environ.pop("ROADSENSE_ANPR_ACCEPTANCE", None)
    from src.anpr.contracts import ReadStatus
    from src.anpr.local_runtime import build_local_runtime
    from src.anpr.pipeline import AnprImageService

    service = AnprImageService(lambda: build_local_runtime(repo_root=ROOT))
    results = []
    for row in rows:
        path = image_dir / row["image_file"].strip()
        result = service.recognize_bytes(path.read_bytes(), MEDIA[path.suffix.lower()], "car-eval")
        expected = reference(row)
        texts = [(p.candidate_text if p.candidate_text is not None else p.normalized_text, p) for p in result.plates]
        accepted = [p for _, p in texts if p.recognition_status == ReadStatus.READ]
        results.append({
            **{t: row[t].strip() for t in TAGS},
            "hit": any(text == expected for text, _ in texts),
            "no_plate": not result.plates,
            "accepted": bool(accepted),
            "accepted_correct": all(p.normalized_text == expected for p in accepted),
            "image_file": row["image_file"].strip(), "reference": expected,
            "readings": [text for text, _ in texts],
        })
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("template")
    t.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("validate")
    v.add_argument("--manifest", type=Path, required=True)
    v.add_argument("--images", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--manifest", type=Path, required=True)
    e.add_argument("--images", type=Path, required=True)
    e.add_argument("--device", choices=("mps", "cpu"), default="mps")
    e.add_argument("--auto-accept", action="store_true")
    e.add_argument("--frozen-pipeline-confirmed", action="store_true",
                   help="required: confirms no setting will be tuned on this set after seeing results")
    e.add_argument("--output", type=Path, required=True, help="new .json report under outputs/ (no plate strings)")
    args = parser.parse_args(argv)

    if args.command == "template":
        if args.output.exists():
            print("ABORT: output exists", file=sys.stderr)
            return 1
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(COLUMNS)
        print(f"wrote {args.output}")
        return 0

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    checked = validate_rows(rows, args.images)
    if args.command == "validate":
        print(json.dumps(checked, indent=2))
        return 1 if checked["errors"] else 0
    if checked["errors"]:
        print("ABORT: manifest invalid; run validate first", file=sys.stderr)
        return 1
    if not args.frozen_pipeline_confirmed:
        print("ABORT: this is an independent test set. Re-run with --frozen-pipeline-confirmed only after the "
              "pipeline and policy are frozen; do not tune on its results.", file=sys.stderr)
        return 1
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.suffix != ".json" or output.exists() or not output.resolve().is_relative_to((ROOT / "outputs").resolve()):
        print("ABORT: --output must be a new .json file inside outputs/", file=sys.stderr)
        return 1
    results = evaluate(rows, args.images, args.device, args.auto_accept)
    report = {"schema_version": "1.0", "set": "car_eval", "auto_accept": args.auto_accept,
              "labeling": checked["stats"], **summarize(results)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    details = output.with_suffix(".details.jsonl")
    with details.open("x", encoding="utf-8") as handle:
        for r in results:
            handle.write(json.dumps(r) + "\n")
    print(json.dumps({k: report[k] for k in ("overall", "by_tag")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
