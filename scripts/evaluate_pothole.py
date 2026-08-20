import sys, os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import hashlib
from datetime import datetime
import sys as sys_module

from src.ml.dataset_validation import validate_prepared_yolo_dataset
from src.ml.metrics import extract_detection_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing test_metrics.json")
    args = parser.parse_args()

    if not os.path.exists(args.weights) or not os.path.isfile(args.weights):
        print(f"Weights artifact missing or invalid: {args.weights}")
        sys_module.exit(1)

    is_valid, err = validate_prepared_yolo_dataset(args.dataset)
    if not is_valid:
        print(f"Dataset Validation Failed: {err}")
        sys_module.exit(1)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    pothole_yaml = os.path.abspath(os.path.join(args.dataset, "pothole.yaml"))

    metrics = model.val(data=pothole_yaml, split="test")
    try:
        results_dict = metrics.results_dict
    except AttributeError:
        results_dict = None

    if not isinstance(results_dict, dict):
        print("Error: Metrics results_dict is missing or not a dictionary")
        sys_module.exit(1)

    results = extract_detection_metrics(results_dict)

    if not results:
        print("No recognized detection metrics found in results")
        sys_module.exit(1)

    prep_meta_path = os.path.join(args.dataset, "manifests", "preparation_metadata.json")
    fingerprint = "Unknown"
    if os.path.exists(prep_meta_path):
        with open(prep_meta_path, "r") as f:
            fingerprint = json.load(f).get("dataset_fingerprint", "Unknown")

    h = hashlib.sha256()
    with open(args.weights, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)

    from datetime import timezone
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_sha256": h.hexdigest(),
        "dataset_fingerprint": fingerprint,
        "split": "test",
        "metrics": results
    }

    run_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.weights)))) or "unknown_run"
    out_dir = REPO_ROOT / "outputs" / "evaluation" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "test_metrics.json"

    if out_path.exists() and not args.overwrite:
        print(f"Error: Evaluation metrics already exist at {out_path}.")
        print("Use --overwrite to intentionally replace them.")
        sys_module.exit(1)

    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Evaluation completed. Saved to {out_path}")

if __name__ == "__main__":
    main()
