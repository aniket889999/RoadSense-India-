"""Score frozen ANPR predictions without loading models or running inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anpr.dataset import DatasetValidationError, load_manifest, sha256_file, validate_prepared_dataset
from src.anpr.evaluation import EvaluationError, evaluate_predictions, load_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--text-source", choices=("normalized", "candidate"), default="normalized",
                        help="score normalized OCR text (default) or the cleaned review candidate when present")
    parser.add_argument("--output", type=Path, required=True, help="A new JSON file under outputs/.")
    args = parser.parse_args()
    try:
        output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        outputs_root = REPO_ROOT / "outputs"
        if (
            ".." in output.parts or not output.is_relative_to(outputs_root)
            or output == outputs_root or output.suffix.lower() != ".json"
        ):
            raise EvaluationError("Output must be a new JSON file inside repository outputs/.")
        current = REPO_ROOT
        for part in output.relative_to(REPO_ROOT).parts:
            current /= part
            if current.is_symlink():
                raise EvaluationError("Evaluation output components cannot be symlinks.")
        if output.exists():
            raise EvaluationError("The evaluation report already exists; choose a new output path.")

        metadata = validate_prepared_dataset(args.dataset)
        records = [record for record in load_manifest(args.dataset / "manifest.jsonl", prepared=True)
                   if record.split == args.split]
        predictions = load_predictions(args.predictions)
        result = evaluate_predictions(records, predictions, iou_threshold=args.iou_threshold, text_source=args.text_source)
        policy = {"split": args.split, "iou_threshold": args.iou_threshold, "normalization": "IN_COMMON_V1"}
        if args.text_source != "normalized":
            policy["text_source"] = args.text_source  # keeps the historical config hash for the default
        report = {
            **result,
            "split": args.split,
            "dataset_fingerprint": metadata["dataset_fingerprint"],
            "predictions_sha256": sha256_file(args.predictions),
            "evaluation_config_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
            "evaluation_git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except (DatasetValidationError, EvaluationError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"ANPR prediction evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "report_path": str(output), "split": args.split,
        "metric_interpretation": report["metric_interpretation"], "overall": report["overall"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
