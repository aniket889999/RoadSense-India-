import sys, os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import yaml
import json
import shutil
import hashlib
from datetime import datetime
import sys as sys_module

from src.ml.dataset_validation import validate_prepared_yolo_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--allow-weight-download", action="store_true", help="Allow downloading weights if missing")
    args = parser.parse_args()

    is_valid, err = validate_prepared_yolo_dataset(args.dataset)
    if not is_valid:
        print(f"Dataset Validation Failed: {err}")
        sys_module.exit(1)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    base_weights_cfg = cfg["model"].get("base_weights", "yolov8n.pt")

    # Resolve relative to REPO_ROOT if it's a path
    resolved_weights = str(REPO_ROOT / base_weights_cfg) if not os.path.isabs(base_weights_cfg) else base_weights_cfg

    if not os.path.exists(resolved_weights):
        if not args.allow_weight_download:
            print(f"Error: Base weights file {resolved_weights} not found locally.")
            print("Use --allow-weight-download to implicitly fetch from the Ultralytics registry.")
            sys_module.exit(1)
        else:
            print(f"Weights {resolved_weights} not found. Falling back to Ultralytics registry name: yolov8n.pt")
            resolved_weights = "yolov8n.pt" # Official registry name
            cfg["model"]["base_weights"] = resolved_weights # record choice in metadata

    # Lazy import Ultralytics
    from ultralytics import YOLO

    pothole_yaml = os.path.abspath(os.path.join(args.dataset, "pothole.yaml"))

    model = YOLO(resolved_weights)

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
        print("Ultralytics did not return a valid trainer.save_dir")
        sys_module.exit(1)

    actual_run_dir = str(model.trainer.save_dir)

    prep_meta = {}
    prep_meta_path = os.path.join(args.dataset, "manifests", "preparation_metadata.json")
    if os.path.exists(prep_meta_path):
        with open(prep_meta_path, "r") as f:
            prep_meta = json.load(f)

    import platform
    import subprocess

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=str(REPO_ROOT)).decode().strip()
    except Exception:
        git_sha = "unknown"

    import torch
    import ultralytics

    best_pt = os.path.join(actual_run_dir, "weights", "best.pt")
    weight_hash = ""
    if os.path.exists(best_pt):
        h = hashlib.sha256()
        with open(best_pt, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        weight_hash = h.hexdigest()
    else:
        print(f"Error: Expected artifact {best_pt} missing.")
        sys_module.exit(1)

    # Compute sha256 for artifacts
    train_config_path = os.path.join(actual_run_dir, "train_config.yaml")
    dataset_prep_path = os.path.join(actual_run_dir, "dataset_preparation_metadata.json")

    def get_hash(path):
        if not os.path.exists(path): return "Missing"
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""): h.update(chunk)
        return h.hexdigest()

    shutil.copy2(args.config, train_config_path)
    shutil.copy2(prep_meta_path, dataset_prep_path)

    from src.ml.metadata import build_training_metadata
    metadata = build_training_metadata(
        git_sha=git_sha,
        prep_meta=prep_meta,
        cfg=cfg,
        run_dir=actual_run_dir,
        artifacts={
            "weights/best.pt": weight_hash,
            "train_config.yaml": get_hash(train_config_path),
            "dataset_preparation_metadata.json": get_hash(dataset_prep_path)
        },
        env={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__
        }
    )

    with open(os.path.join(actual_run_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Training completed. Outputs saved to {actual_run_dir}")

if __name__ == "__main__":
    main()
