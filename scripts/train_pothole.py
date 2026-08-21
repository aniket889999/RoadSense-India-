import sys, os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import yaml
import json
import shutil
import hashlib
import platform
import subprocess
import sys as sys_module

from src.ml.dataset_validation import validate_prepared_yolo_dataset

def compute_sha256(filepath: str) -> str:
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        raise FileNotFoundError(f"Required artifact file missing or invalid: {filepath}")
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if len(digest) != 64:
        raise ValueError(f"Invalid SHA-256 digest computed for {filepath}: {digest}")
    return digest

def main():
    parser = argparse.ArgumentParser(description="Train custom YOLOv8n pothole model")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--dataset", required=True, help="Path to prepared YOLO dataset")
    parser.add_argument("--allow-weight-download", action="store_true", help="Allow downloading weights if missing")
    args = parser.parse_args()

    is_valid, err = validate_prepared_yolo_dataset(args.dataset)
    if not is_valid:
        print(f"Dataset Validation Failed: {err}")
        sys_module.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_weights_cfg = cfg["model"].get("base_weights", "models/yolov8n.pt")

    # Resolve relative to REPO_ROOT if it's a relative path
    resolved_weights = str(REPO_ROOT / base_weights_cfg) if not os.path.isabs(base_weights_cfg) else base_weights_cfg

    if not os.path.exists(resolved_weights):
        if not args.allow_weight_download:
            print(f"Error: Base weights file {resolved_weights} not found locally.")
            print("Use --allow-weight-download to implicitly fetch from the Ultralytics registry.")
            sys_module.exit(1)
        else:
            print(f"Weights {resolved_weights} not found. Falling back to Ultralytics registry name: yolov8n.pt")
            resolved_weights = "yolov8n.pt"

    # Lazy import Ultralytics
    from ultralytics import YOLO

    pothole_yaml = os.path.abspath(os.path.join(args.dataset, "pothole.yaml"))

    model = YOLO(resolved_weights)

    # Normalize resolved_weights to an absolute resolved file path on disk
    if not os.path.isabs(resolved_weights) or not os.path.isfile(resolved_weights):
        candidates = [
            resolved_weights,
            getattr(model, "ckpt_path", None),
            getattr(model, "weights", None),
            str(REPO_ROOT / "yolov8n.pt"),
            str(Path.cwd() / "yolov8n.pt")
        ]
        found_path = None
        for cand in candidates:
            if cand and isinstance(cand, (str, Path, os.PathLike)) and not isinstance(cand, bool):
                p = Path(cand)
                if p.exists() and p.is_file():
                    found_path = str(p.resolve())
                    break
        if not found_path:
            print(f"Error: Base weights could not be resolved to a regular local file: {resolved_weights}")
            sys_module.exit(1)
        resolved_weights = found_path
    else:
        resolved_weights = str(Path(resolved_weights).resolve())

    out_base = REPO_ROOT / "outputs" / "training"
    out_base.mkdir(parents=True, exist_ok=True)

    results = model.train(
        data=pothole_yaml,
        epochs=cfg["training"]["epochs"],
        batch=cfg["training"]["batch"],
        patience=cfg["training"]["patience"],
        device=cfg["training"]["device"],
        imgsz=cfg["model"]["image_size"],
        seed=cfg["experiment"]["seed"],
        project=str(out_base),
        name=cfg["experiment"]["name"],
        deterministic=True
    )

    if not hasattr(model, "trainer") or not hasattr(model.trainer, "save_dir"):
        print("Error: Ultralytics did not return a valid trainer.save_dir")
        sys_module.exit(1)

    actual_run_dir = str(model.trainer.save_dir)

    prep_meta_path = os.path.join(args.dataset, "manifests", "preparation_metadata.json")
    if not os.path.exists(prep_meta_path) or not os.path.isfile(prep_meta_path):
        print(f"Error: Dataset preparation metadata missing at {prep_meta_path}")
        sys_module.exit(1)

    with open(prep_meta_path, "r", encoding="utf-8") as f:
        prep_meta = json.load(f)

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=str(REPO_ROOT)).decode().strip()
    except Exception:
        git_sha = "unknown"

    import torch
    import ultralytics
    try:
        import torchvision
        torchvision_version = getattr(torchvision, "__version__", "unknown")
    except ImportError:
        torchvision_version = "unknown"

    best_pt = os.path.join(actual_run_dir, "weights", "best.pt")
    last_pt = os.path.join(actual_run_dir, "weights", "last.pt")
    results_csv = os.path.join(actual_run_dir, "results.csv")
    args_yaml = os.path.join(actual_run_dir, "args.yaml")
    train_config_path = os.path.join(actual_run_dir, "train_config.yaml")
    dataset_prep_path = os.path.join(actual_run_dir, "dataset_preparation_metadata.json")

    shutil.copy2(args.config, train_config_path)
    shutil.copy2(prep_meta_path, dataset_prep_path)

    # Strictly verify and hash all required artifacts
    required_artifact_files = {
        "base_weights": resolved_weights,
        "weights/best.pt": best_pt,
        "weights/last.pt": last_pt,
        "results.csv": results_csv,
        "args.yaml": args_yaml,
        "train_config.yaml": train_config_path,
        "dataset_preparation_metadata.json": dataset_prep_path
    }

    artifacts_map = {}
    for key, fpath in required_artifact_files.items():
        if not os.path.exists(fpath) or not os.path.isfile(fpath):
            print(f"Error: Required post-training artifact '{key}' missing or not a regular file at {fpath}")
            sys_module.exit(1)
        try:
            artifacts_map[key] = compute_sha256(fpath)
        except Exception as e:
            print(f"Error computing SHA-256 for required artifact '{key}' ({fpath}): {e}")
            sys_module.exit(1)

    from src.ml.metadata import build_training_metadata
    metadata = build_training_metadata(
        git_sha=git_sha,
        prep_meta=prep_meta,
        cfg=cfg,
        run_dir=actual_run_dir,
        artifacts=artifacts_map,
        env={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": getattr(torch, "__version__", "unknown"),
            "torchvision": torchvision_version,
            "ultralytics": getattr(ultralytics, "__version__", "unknown")
        },
        resolved_base_weights=resolved_weights
    )

    with open(os.path.join(actual_run_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Training completed. Outputs saved to {actual_run_dir}")

if __name__ == "__main__":
    main()
