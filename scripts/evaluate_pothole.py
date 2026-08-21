import sys, os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import hashlib
import math
import re
import platform
import subprocess
from datetime import datetime, timezone
import sys as sys_module

from src.ml.dataset_validation import validate_prepared_yolo_dataset
from src.ml.metrics import extract_detection_metrics

HEX64_REGEX = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_ARTIFACT_KEYS = [
    "base_weights",
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "train_config.yaml",
    "dataset_preparation_metadata.json"
]

def is_valid_sha256(val: object) -> bool:
    return isinstance(val, str) and bool(re.fullmatch(r"[0-9a-f]{64}", val))

def compute_file_sha256(filepath: Path) -> str:
    if not filepath.exists() or not filepath.is_file() or filepath.is_symlink():
        raise FileNotFoundError(f"File missing, not a regular file, or is a symlink: {filepath}")
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if not is_valid_sha256(digest):
        raise ValueError(f"Invalid SHA-256 computed for {filepath}: {digest}")
    return digest

def has_symlink_components(path: Path, base: Path = None) -> bool:
    """Checks if path or any of its directory/file components is a symlink without resolving beforehand."""
    try:
        base_res = base.resolve() if base is not None else None
        if not path.is_absolute():
            start = base_res if base_res is not None else Path.cwd().resolve()
            curr = start
            for part in path.parts:
                curr = curr / part
                if curr.is_symlink():
                    return True
            return False
        else:
            if base_res is not None:
                try:
                    rel = path.relative_to(base_res)
                    curr = base_res
                    for part in rel.parts:
                        curr = curr / part
                        if curr.is_symlink():
                            return True
                    return False
                except ValueError:
                    pass
            curr = Path(path.parts[0])
            for part in path.parts[1:]:
                curr = curr / part
                if curr.is_symlink():
                    return True
            return False
    except Exception:
        return True

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned YOLOv8n pothole model on test split (strictly one-time)")
    parser.add_argument("--weights", required=True, help="Path to fine-tuned model checkpoint (.pt) - must be <training_run_dir>/weights/best.pt")
    parser.add_argument("--dataset", required=True, help="Path to prepared YOLO dataset")
    parser.add_argument("--device", required=True, help="Compute device for evaluation (e.g., 'mps', 'cuda', 'cpu')")
    parser.add_argument("--batch", type=int, required=True, help="Batch size for evaluation")
    parser.add_argument("--imgsz", type=int, required=True, help="Image size for evaluation (e.g. 640 or 320)")
    args = parser.parse_args()

    # 1. Validate command argument values BEFORE any reservation or irreversible locking
    if not args.device or not args.device.strip():
        print(f"Error: Invalid --device argument: '{args.device}'. Must be non-empty (e.g. 'mps', 'cuda', 'cpu').")
        sys_module.exit(1)

    if args.batch <= 0:
        print(f"Error: Invalid --batch argument: {args.batch}. Must be a positive integer.")
        sys_module.exit(1)

    if args.imgsz <= 0:
        print(f"Error: Invalid --imgsz argument: {args.imgsz}. Must be a positive integer.")
        sys_module.exit(1)

    # 2. Check repository output root directory safety (reject symlinks pointing outside repository)
    resolved_repo_root = REPO_ROOT.resolve()
    for sub in ["outputs", "outputs/training", "outputs/evaluation"]:
        p = REPO_ROOT / sub
        if p.is_symlink():
            target = p.resolve()
            try:
                target.relative_to(resolved_repo_root)
            except ValueError:
                print(f"Error: Directory '{p}' is a symlink pointing outside the repository ({target}).")
                sys_module.exit(1)

    # 3. Check for symlinks in weights path components before resolving
    raw_weights_path = Path(args.weights)
    if has_symlink_components(raw_weights_path, base=REPO_ROOT):
        print(f"Error: Selected weights path '{raw_weights_path}' or one of its components is a symlink.")
        sys_module.exit(1)

    # 4. Confine evaluation to a real local training run inside outputs/training
    training_root = (REPO_ROOT / "outputs" / "training").resolve()
    resolved_weights_path = raw_weights_path.resolve()

    if not resolved_weights_path.exists() or not resolved_weights_path.is_file():
        print(f"Error: Weights artifact missing or not a regular file: {args.weights}")
        sys_module.exit(1)

    # Require weights path to be exactly <training_run_dir>/weights/best.pt
    if resolved_weights_path.name != "best.pt" or resolved_weights_path.parent.name != "weights":
        print(f"Error: Selected weights path must be exactly '<training_run_dir>/weights/best.pt', got: {resolved_weights_path}")
        sys_module.exit(1)

    training_run_dir = resolved_weights_path.parent.parent.resolve()
    try:
        rel_to_training_root = training_run_dir.relative_to(training_root)
        if rel_to_training_root == Path("."):
            raise ValueError("Training run directory cannot be the training root itself.")
    except Exception:
        print(f"Error: Training run directory '{training_run_dir}' must be strictly inside local training outputs ({training_root}).")
        sys_module.exit(1)

    expected_best_pt = (training_run_dir / "weights" / "best.pt").resolve()
    if resolved_weights_path != expected_best_pt:
        print(f"Error: Checkpoint path mismatch! Expected {expected_best_pt}, got {resolved_weights_path}")
        sys_module.exit(1)

    run_id = training_run_dir.name

    # 5. Resolve and validate dataset structure and preparation metadata
    resolved_dataset_path = Path(args.dataset).resolve()
    is_valid, err = validate_prepared_yolo_dataset(str(resolved_dataset_path))
    if not is_valid:
        print(f"Dataset Validation Failed: {err}")
        sys_module.exit(1)

    prep_meta_path = resolved_dataset_path / "manifests" / "preparation_metadata.json"
    if not prep_meta_path.exists() or not prep_meta_path.is_file() or prep_meta_path.is_symlink():
        print(f"Error: Dataset preparation metadata missing or is a symlink at {prep_meta_path}")
        sys_module.exit(1)

    try:
        with open(prep_meta_path, "r", encoding="utf-8") as f:
            prep_meta = json.load(f)
    except Exception as e:
        print(f"Error reading dataset preparation metadata ({prep_meta_path}): {e}")
        sys_module.exit(1)

    dataset_fingerprint = prep_meta.get("dataset_fingerprint")
    if not is_valid_sha256(dataset_fingerprint):
        print(f"Error: Dataset fingerprint '{dataset_fingerprint}' is missing or not a valid 64-character SHA-256 hash.")
        sys_module.exit(1)

    test_counts = prep_meta.get("counts", {}).get("split_distribution", {}).get("test")
    if not isinstance(test_counts, dict):
        print("Error: Missing 'test' split distribution in dataset preparation metadata.")
        sys_module.exit(1)

    required_count_keys = ["images", "positive_images", "d40_instances"]
    for ck in required_count_keys:
        val = test_counts.get(ck)
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            print(f"Error: Test count '{ck}' is invalid in dataset preparation metadata: {val}")
            sys_module.exit(1)

    # 6. Strict Checkpoint & Artifact Provenance Verification
    model_meta_path = training_run_dir / "model_metadata.json"
    if not model_meta_path.exists() or not model_meta_path.is_file() or model_meta_path.is_symlink():
        print(f"Error: Model training metadata missing or is a symlink at {model_meta_path}")
        sys_module.exit(1)

    try:
        with open(model_meta_path, "r", encoding="utf-8") as f:
            model_meta = json.load(f)
    except Exception as e:
        print(f"Error reading model training metadata ({model_meta_path}): {e}")
        sys_module.exit(1)

    # Verify model_metadata run_directory
    meta_run_dir = model_meta.get("run_directory")
    if not meta_run_dir:
        print(f"Error: Missing 'run_directory' in model metadata at {model_meta_path}")
        sys_module.exit(1)

    try:
        resolved_meta_run_dir = Path(meta_run_dir).resolve()
    except Exception as e:
        print(f"Error resolving 'run_directory' ({meta_run_dir}): {e}")
        sys_module.exit(1)

    if resolved_meta_run_dir != training_run_dir:
        print("Error: 'run_directory' mismatch in model metadata!")
        print(f"  Metadata run_directory: {resolved_meta_run_dir}")
        print(f"  Actual training run dir: {training_run_dir}")
        sys_module.exit(1)

    training_dataset_fingerprint = model_meta.get("dataset_fingerprint")
    if training_dataset_fingerprint != dataset_fingerprint:
        print("Error: Dataset fingerprint mismatch between training run and evaluation dataset!")
        print(f"  Dataset fingerprint:        {dataset_fingerprint}")
        print(f"  Training model fingerprint: {training_dataset_fingerprint}")
        sys_module.exit(1)

    model_artifacts = model_meta.get("artifacts")
    if not isinstance(model_artifacts, dict):
        print(f"Error: Model metadata 'artifacts' missing or not a dictionary in {model_meta_path}")
        sys_module.exit(1)

    # Verify all 7 required artifact keys and hashes
    for art_key in REQUIRED_ARTIFACT_KEYS:
        if art_key not in model_artifacts:
            print(f"Error: Required artifact key '{art_key}' missing from model_metadata.json artifacts")
            sys_module.exit(1)
        recorded_hash = model_artifacts[art_key]
        if not is_valid_sha256(recorded_hash):
            print(f"Error: Artifact '{art_key}' has invalid SHA-256 hash in model_metadata.json: {recorded_hash!r}")
            sys_module.exit(1)

    # Resolve each artifact file on disk and verify it is not a symlink and hash matches
    def resolve_artifact_file(key: str) -> Path:
        if key == "base_weights":
            res_bw = model_meta.get("resolved_base_weights") or model_meta.get("base_weights")
            if res_bw and Path(res_bw).is_absolute():
                return Path(res_bw)
            elif res_bw:
                return (REPO_ROOT / res_bw).resolve()
            return (REPO_ROOT / "models" / "yolov8n.pt").resolve()
        elif key == "weights/best.pt":
            return training_run_dir / "weights" / "best.pt"
        elif key == "weights/last.pt":
            return training_run_dir / "weights" / "last.pt"
        elif key == "results.csv":
            return training_run_dir / "results.csv"
        elif key == "args.yaml":
            return training_run_dir / "args.yaml"
        elif key == "train_config.yaml":
            return training_run_dir / "train_config.yaml"
        elif key == "dataset_preparation_metadata.json":
            return training_run_dir / "dataset_preparation_metadata.json"
        else:
            return training_run_dir / key

    for art_key in REQUIRED_ARTIFACT_KEYS:
        art_path = resolve_artifact_file(art_key)
        if art_path.is_symlink():
            print(f"Error: Training artifact '{art_key}' at '{art_path}' is a symlink.")
            sys_module.exit(1)
        if not art_path.exists() or not art_path.is_file():
            print(f"Error: Required training artifact '{art_key}' missing on disk at {art_path}")
            sys_module.exit(1)
        try:
            disk_hash = compute_file_sha256(art_path)
        except Exception as e:
            print(f"Error computing hash for artifact '{art_key}' ({art_path}): {e}")
            sys_module.exit(1)

        recorded_hash = model_artifacts[art_key]
        if disk_hash != recorded_hash:
            print(f"Error: Hash mismatch for training artifact '{art_key}'!")
            print(f"  Disk hash:     {disk_hash}")
            print(f"  Recorded hash: {recorded_hash}")
            sys_module.exit(1)

    weights_sha256 = model_artifacts["weights/best.pt"]

    try:
        eval_git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=str(REPO_ROOT)).decode().strip()
    except Exception:
        eval_git_sha = "unknown"

    # 7. Atomic One-Time Evaluation Reservation (exist_ok=False)
    eval_parent_dir = REPO_ROOT / "outputs" / "evaluation"
    eval_parent_dir.mkdir(parents=True, exist_ok=True)
    eval_out_dir = eval_parent_dir / run_id

    try:
        os.mkdir(str(eval_out_dir))
    except FileExistsError:
        print(f"Error: Evaluation directory '{eval_out_dir}' already exists.")
        print("Held-out test split evaluation is strictly one-time per run to prevent test-set data snooping.")
        sys_module.exit(1)
    except OSError as e:
        print(f"Error reserving evaluation directory '{eval_out_dir}': {e}")
        sys_module.exit(1)

    # Create evaluation_attempt.json using exclusive creation (O_CREAT | O_EXCL via "x" mode)
    attempt_file = eval_out_dir / "evaluation_attempt.json"
    attempt_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_git_sha": eval_git_sha,
        "run_id": run_id,
        "weights_path": str(resolved_weights_path),
        "weights_sha256": weights_sha256,
        "dataset_path": str(resolved_dataset_path),
        "dataset_fingerprint": dataset_fingerprint,
        "evaluation_params": {
            "device": args.device,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "split": "test"
        }
    }

    try:
        with open(attempt_file, "x", encoding="utf-8") as f:
            json.dump(attempt_record, f, indent=4)
    except FileExistsError:
        print(f"Error: Attempt ledger '{attempt_file}' already exists.")
        sys_module.exit(1)

    # 8. Model Loading & Evaluation Execution
    import torch
    import ultralytics
    try:
        import torchvision
        torchvision_version = getattr(torchvision, "__version__", "unknown")
    except ImportError:
        torchvision_version = "unknown"

    from ultralytics import YOLO

    model = YOLO(str(resolved_weights_path))
    pothole_yaml = os.path.abspath(os.path.join(str(resolved_dataset_path), "pothole.yaml"))

    metrics_obj = model.val(
        data=pothole_yaml,
        split="test",
        device=args.device,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(eval_parent_dir),
        name=run_id,
        exist_ok=True
    )

    try:
        results_dict = metrics_obj.results_dict
    except AttributeError:
        results_dict = None

    if not isinstance(results_dict, dict):
        print("Error: Metrics results_dict is missing or not a dictionary from Ultralytics val()")
        sys_module.exit(1)

    raw_metrics = extract_detection_metrics(results_dict)
    required_metric_keys = ["precision", "recall", "mAP50", "mAP50-95"]
    clean_metrics = {}

    for k in required_metric_keys:
        if k not in raw_metrics:
            print(f"Error: Required detection metric '{k}' missing in results: {raw_metrics}")
            sys_module.exit(1)
        val = raw_metrics[k]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            print(f"Error: Required metric '{k}' is non-numeric, boolean, or not finite (NaN/Inf): {val}")
            sys_module.exit(1)
        clean_metrics[k] = float(val)

    training_provenance = {
        "training_run_directory": str(training_run_dir),
        "training_git_sha": model_meta.get("git_sha", "unknown"),
        "training_dataset_fingerprint": training_dataset_fingerprint,
        "model_metadata_sha256": compute_file_sha256(model_meta_path),
        "training_artifacts": model_artifacts
    }

    report = {
        "evaluation_git_sha": eval_git_sha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weights_path": str(resolved_weights_path),
        "weights_sha256": weights_sha256,
        "dataset_path": str(resolved_dataset_path),
        "dataset_fingerprint": dataset_fingerprint,
        "evaluation_params": {
            "device": args.device,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "split": "test"
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": getattr(torch, "__version__", "unknown"),
            "torchvision": torchvision_version,
            "ultralytics": getattr(ultralytics, "__version__", "unknown")
        },
        "test_split_counts": test_counts,
        "training_provenance": training_provenance,
        "metrics": clean_metrics
    }

    metrics_file = eval_out_dir / "test_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"Test evaluation completed successfully. Saved report to {metrics_file}")

if __name__ == "__main__":
    main()
