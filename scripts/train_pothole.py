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
from datetime import datetime, timezone
import sys as sys_module

from src.ml.dataset_validation import validate_prepared_yolo_dataset

def is_valid_sha256(val: object) -> bool:
    import re
    return isinstance(val, str) and bool(re.fullmatch(r"[0-9a-f]{64}", val))

def compute_sha256(filepath: str) -> str:
    if not os.path.exists(filepath) or not os.path.isfile(filepath) or os.path.islink(filepath):
        raise FileNotFoundError(f"Required artifact file missing, invalid, or is a symlink: {filepath}")
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if len(digest) != 64:
        raise ValueError(f"Invalid SHA-256 digest computed for {filepath}: {digest}")
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
    parser = argparse.ArgumentParser(description="Train custom YOLOv8n pothole model")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--dataset", required=True, help="Path to prepared YOLO dataset")
    parser.add_argument("--allow-weight-download", action="store_true", help="Allow downloading weights if missing")
    parser.add_argument("--resume-from", default=None, help="Path to interrupted training checkpoint (.pt) to resume - must be <training_run_dir>/weights/last.pt")
    args = parser.parse_args()

    # 1. Reject incompatible options
    if args.resume_from and args.allow_weight_download:
        print("Error: --allow-weight-download cannot be used with --resume-from.")
        sys_module.exit(1)

    # 2. Dataset validation (required in both fresh and resume modes)
    dataset_path = Path(args.dataset) if Path(args.dataset).is_absolute() else (REPO_ROOT / args.dataset)
    is_valid, err = validate_prepared_yolo_dataset(str(dataset_path))
    if not is_valid:
        print(f"Dataset Validation Failed: {err}")
        sys_module.exit(1)

    config_path = Path(args.config) if Path(args.config).is_absolute() else (REPO_ROOT / args.config)
    if not config_path.exists() or not config_path.is_file():
        print(f"Error: Training config file missing: {config_path}")
        sys_module.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 3. Handle resume mode vs fresh training mode
    if args.resume_from:
        raw_resume_path = Path(args.resume_from) if Path(args.resume_from).is_absolute() else (REPO_ROOT / args.resume_from)

        # Lexical and component symlink checks before any resolving
        if raw_resume_path.is_symlink() or has_symlink_components(raw_resume_path, base=REPO_ROOT):
            print(f"Error: Resume checkpoint path '{raw_resume_path}' or one of its components is a symlink.")
            sys_module.exit(1)

        if not raw_resume_path.exists() or not raw_resume_path.is_file():
            print(f"Error: Resume checkpoint file '{raw_resume_path}' does not exist or is not a regular file.")
            sys_module.exit(1)

        resolved_resume_path = raw_resume_path.resolve()

        # Require checkpoint to be named exactly last.pt inside weights/
        if resolved_resume_path.name != "last.pt" or resolved_resume_path.parent.name != "weights":
            print(f"Error: Resume checkpoint must be named exactly 'last.pt' inside a 'weights' directory, got: {resolved_resume_path}")
            sys_module.exit(1)

        # Confine training run directory strictly inside outputs/training/
        training_root = (REPO_ROOT / "outputs" / "training").resolve()
        training_run_dir = resolved_resume_path.parent.parent.resolve()

        try:
            rel = training_run_dir.relative_to(training_root)
            if rel == Path("."):
                raise ValueError("Training run directory cannot be the training root itself.")
        except Exception:
            print(f"Error: Resumed training run directory '{training_run_dir}' must be strictly inside local training outputs ({training_root}).")
            sys_module.exit(1)

        # Preflight: validate base weights exist as a regular, non-symlink file
        base_weights_cfg = cfg["model"].get("base_weights", "models/yolov8n.pt")
        raw_bw_path = (REPO_ROOT / base_weights_cfg) if not os.path.isabs(base_weights_cfg) else Path(base_weights_cfg)
        if raw_bw_path.is_symlink() or has_symlink_components(raw_bw_path, base=REPO_ROOT):
            print(f"Error: Base weights path '{raw_bw_path}' or one of its components is a symlink.")
            sys_module.exit(1)

        resolved_bw_path = raw_bw_path.resolve()
        if not resolved_bw_path.exists() or not resolved_bw_path.is_file():
            print(f"Error: Configured base weights file missing or not a regular file: {resolved_bw_path}")
            sys_module.exit(1)
        resolved_weights = str(resolved_bw_path)

        # Preflight CPU-only checkpoint inspection before importing Ultralytics.
        # weights_only=False is explicitly required because Ultralytics checkpoints contain
        # model structure/tasks classes; this is safe because the path is strictly confined
        # to the local, trusted outputs/training directory verified above.
        import torch
        try:
            ckpt = torch.load(
                str(resolved_resume_path),
                map_location="cpu",
                weights_only=False
            )
        except Exception as e:
            print(f"Error loading checkpoint {resolved_resume_path}: {e}")
            sys_module.exit(1)

        if not isinstance(ckpt, dict):
            print(f"Error: Checkpoint {resolved_resume_path} is not a valid state dictionary.")
            sys_module.exit(1)

        epoch = ckpt.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            print(f"Error: Checkpoint epoch '{epoch}' is invalid (must be a non-negative integer).")
            sys_module.exit(1)

        optimizer = ckpt.get("optimizer")
        if not optimizer:
            print("Error: Checkpoint is missing optimizer state or optimizer state is empty.")
            sys_module.exit(1)

        train_args = ckpt.get("train_args")
        if not isinstance(train_args, dict) and not hasattr(train_args, "get"):
            print("Error: Checkpoint is missing a valid train_args mapping.")
            sys_module.exit(1)

        # Enforce all mandatory saved training arguments in resume mode
        mandatory_fields = ["data", "project", "save_dir", "model", "epochs", "batch", "patience", "device", "imgsz", "seed", "name"]
        for field in mandatory_fields:
            if field not in train_args or train_args.get(field) is None:
                print(f"Error: Mandatory field '{field}' missing from checkpoint train_args.")
                sys_module.exit(1)

        total_epochs = train_args["epochs"]
        if not isinstance(total_epochs, int) or isinstance(total_epochs, bool) or total_epochs <= 0:
            print(f"Error: Checkpoint train_args has invalid total epochs: {total_epochs}")
            sys_module.exit(1)

        if epoch >= total_epochs:
            print(f"Error: Checkpoint completed all {total_epochs} epochs (current epoch={epoch}); cannot resume.")
            sys_module.exit(1)

        # Provenance alignment checks: Compare checkpoint train_args with current dataset and config
        pothole_yaml = (dataset_path / "pothole.yaml").resolve()
        saved_data = train_args["data"]
        resolved_saved_data = Path(saved_data).resolve()
        if resolved_saved_data != pothole_yaml:
            print(f"Error: Dataset mismatch between checkpoint and command arguments!")
            print(f"  Checkpoint data: {resolved_saved_data}")
            print(f"  Supplied data:   {pothole_yaml}")
            sys_module.exit(1)

        # Verify hyperparameters alignment
        param_checks = [
            ("epochs", cfg["training"]["epochs"]),
            ("batch", cfg["training"]["batch"]),
            ("patience", cfg["training"]["patience"]),
            ("device", cfg["training"]["device"]),
            ("imgsz", cfg["model"]["image_size"]),
            ("seed", cfg["experiment"]["seed"]),
            ("name", cfg["experiment"]["name"])
        ]
        for key, expected_val in param_checks:
            actual_val = train_args[key]
            if str(actual_val) != str(expected_val):
                print(f"Error: Parameter mismatch for '{key}' in resume preflight!")
                print(f"  Checkpoint train_args: {actual_val}")
                print(f"  Supplied config:       {expected_val}")
                sys_module.exit(1)

        saved_project = train_args["project"]
        if Path(saved_project).resolve() != (REPO_ROOT / "outputs" / "training").resolve():
            print(f"Error: Project output directory mismatch: {saved_project}")
            sys_module.exit(1)

        saved_save_dir = train_args["save_dir"]
        if Path(saved_save_dir).resolve() != training_run_dir:
            print(f"Error: Checkpoint save_dir mismatch: {saved_save_dir} != {training_run_dir}")
            sys_module.exit(1)

        saved_model = train_args["model"]
        resolved_saved_model = Path(saved_model).resolve() if Path(saved_model).exists() else (REPO_ROOT / saved_model).resolve()
        if resolved_saved_model != resolved_bw_path:
            print(f"Error: Base weights path mismatch: {resolved_saved_model} != {resolved_bw_path}")
            sys_module.exit(1)

        # Canonical SHA-256 of saved train_args
        train_args_canonical = json.dumps(train_args, sort_keys=True, default=str)
        train_args_sha256 = hashlib.sha256(train_args_canonical.encode("utf-8")).hexdigest()

        resume_checkpoint_sha256 = compute_sha256(str(resolved_resume_path))
        resume_info = {
            "resumed": True,
            "resume_checkpoint": str(resolved_resume_path),
            "resume_checkpoint_sha256": resume_checkpoint_sha256,
            "resume_timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint_epoch": epoch,
            "next_epoch": epoch + 1,
            "checkpoint_train_args_sha256": train_args_sha256
        }

        # Lazy import Ultralytics after all preflight checks pass
        from ultralytics import YOLO

        print(f"Resuming training from checkpoint: {resolved_resume_path}")
        model = YOLO(str(resolved_resume_path))
        results = model.train(resume=True)

        if not hasattr(model, "trainer") or not hasattr(model.trainer, "save_dir"):
            print("Error: Ultralytics did not return a valid trainer.save_dir upon resume")
            sys_module.exit(1)

        actual_run_dir = str(Path(model.trainer.save_dir).resolve())
        if actual_run_dir != str(training_run_dir):
            print(f"Error: Ultralytics trainer save_dir ({actual_run_dir}) does not match expected run directory ({training_run_dir})")
            sys_module.exit(1)

    else:
        # Fresh training mode
        resume_info = None

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

        pothole_yaml = os.path.abspath(os.path.join(str(dataset_path), "pothole.yaml"))

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

    # 4. Post-training metadata and artifact verification
    prep_meta_path = os.path.join(str(dataset_path), "manifests", "preparation_metadata.json")
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

    shutil.copy2(str(config_path), train_config_path)
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
        if not os.path.exists(fpath) or not os.path.isfile(fpath) or os.path.islink(fpath):
            print(f"Error: Required post-training artifact '{key}' missing, invalid, or is a symlink at {fpath}")
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
        resolved_base_weights=resolved_weights,
        resume_info=resume_info
    )

    with open(os.path.join(actual_run_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Training completed successfully. Outputs saved to {actual_run_dir}")

if __name__ == "__main__":
    main()
