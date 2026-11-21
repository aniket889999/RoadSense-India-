"""Reproducible training wrapper for a dedicated single-class YOLO plate detector.

This module intentionally has no top-level Torch or Ultralytics imports so
``--dry-run`` preflight validation executes without loading ML engines or
producing filesystem side effects. Actual training runs only when explicitly
requested with ``execute=True`` (CLI ``--execute``), validates strictly against
the ``val`` split, and never invokes ``test`` split evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import time
from typing import Any, Callable, Dict, Mapping, Optional

import yaml

from src.anpr.dataset import REPO_ROOT, validate_prepared_dataset
from src.anpr.detector import REQUIRED_CLASS_NAME, _validate_single_plate_class, verify_confinement_under_models
from src.anpr.runtime_contracts import RuntimeUnavailable

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_SUPPORTED_DEVICES = {"cpu", "mps", "cuda"}

REQUIRED_TRAINING_ARTIFACTS = (
    "base_weights",
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "train_config.yaml",
    "dataset_preparation_metadata.json",
)

# Forbidden pothole checkpoint name tokens to prevent accidental substitution
_FORBIDDEN_BASE_WEIGHT_TOKENS = ("pothole",)

# Safe default plate-detector augmentation bounds that forbid text-mirroring flips
# and aggressive perspective/shear/copy-paste transforms.
DEFAULT_SAFE_AUGMENTATION: Dict[str, float] = {
    "fliplr": 0.0,
    "flipud": 0.0,
    "degrees": 3.0,
    "translate": 0.05,
    "scale": 0.15,
    "shear": 0.0,
    "perspective": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
    "hsv_h": 0.01,
    "hsv_s": 0.25,
    "hsv_v": 0.25,
}

# Upper bounds on geometric/combinatorial augmentations so plate text is never
# mirrored, inverted, sheared beyond legibility, or spliced via copy-paste/mixup.
_MAX_SAFE_AUGMENTATION_BOUNDS: Dict[str, float] = {
    "fliplr": 0.0,
    "flipud": 0.0,
    "degrees": 10.0,
    "translate": 0.15,
    "scale": 0.30,
    "shear": 2.0,
    "perspective": 0.0005,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
    "hsv_h": 0.05,
    "hsv_s": 0.50,
    "hsv_v": 0.50,
}


class AnprTrainingError(RuntimeError):
    """Raised when ANPR detector training preflight, execution, or provenance checks fail."""


@dataclass(frozen=True)
class AnprTrainingPlan:
    """Verified preflight plan produced by dry-run and consumed by execute mode."""

    repo_root: Path
    config_path: Path
    config_sha256: str
    dataset_dir: Path
    dataset_yaml_path: Path
    dataset_metadata: Mapping[str, Any]
    dataset_fingerprint: str
    base_weights_relative: str
    base_weights_path: Path
    base_weights_sha256: str
    output_project_dir: Path
    run_name: str
    run_directory: Path
    device: str
    seed: int
    image_size: int
    epochs: int
    batch: int
    patience: int
    workers: int
    class_name: str
    augmentation: Mapping[str, float]
    raw_config: Mapping[str, Any]

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "mode": "dry_run",
            "run_name": self.run_name,
            "run_directory": str(self.run_directory),
            "output_project_dir": str(self.output_project_dir),
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "dataset_dir": str(self.dataset_dir),
            "dataset_yaml_path": str(self.dataset_yaml_path),
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_counts": dict(self.dataset_metadata.get("split_counts", {})),
            "evidence_assurance": dict(self.dataset_metadata.get("evidence_assurance", {})),
            "base_weights": self.base_weights_relative,
            "resolved_base_weights": str(self.base_weights_path),
            "base_weights_sha256": self.base_weights_sha256,
            "class_mapping": {"0": self.class_name},
            "device": self.device,
            "seed": self.seed,
            "image_size": self.image_size,
            "epochs": self.epochs,
            "batch": self.batch,
            "patience": self.patience,
            "workers": self.workers,
            "validation_split": "val",
            "test_evaluation_invoked": False,
            "augmentation": dict(self.augmentation),
        }


def _normalize_repo_root(repo_root: Optional[Path] = None) -> Path:
    root = Path(os.path.abspath(os.fspath(repo_root or REPO_ROOT)))
    try:
        info = os.lstat(root)
    except OSError as exc:
        raise AnprTrainingError(f"Repository root is not accessible: {root} ({exc})") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise AnprTrainingError(f"Repository root must be a real, non-symlink directory: {root}")
    return root


def _ensure_no_symlink_chain_under_repo(target: Path, repo_root: Path, label: str) -> None:
    try:
        rel = target.relative_to(repo_root)
    except ValueError as exc:
        raise AnprTrainingError(
            f"{label} must stay inside the repository root ({repo_root}): {target}"
        ) from exc

    current = repo_root
    for part in rel.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            try:
                info = os.lstat(current)
            except OSError as exc:
                raise AnprTrainingError(f"Unable to inspect {label} component {current}: {exc}") from exc
            if stat.S_ISLNK(info.st_mode):
                raise AnprTrainingError(f"{label} contains a forbidden symlink component: {current}")


def _compute_regular_file_sha256(path: Path, label: str = "file") -> str:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise AnprTrainingError(f"Required {label} is missing or unreadable: {path} ({exc})") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise AnprTrainingError(f"Required {label} must be a regular, non-symlink file: {path}")

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(65536):
                digest.update(chunk)
    except OSError as exc:
        raise AnprTrainingError(f"Unable to hash {label} at {path}: {exc}") from exc
    return digest.hexdigest()


def _validate_positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnprTrainingError(f"{field} must be an integer.")
    if allow_zero and value < 0:
        raise AnprTrainingError(f"{field} must be a non-negative integer.")
    if not allow_zero and value <= 0:
        raise AnprTrainingError(f"{field} must be a positive integer.")
    return value


def _validate_augmentation_config(raw_aug: Any) -> Dict[str, float]:
    if raw_aug is None:
        return dict(DEFAULT_SAFE_AUGMENTATION)
    if not isinstance(raw_aug, Mapping):
        raise AnprTrainingError("training config 'augmentation' section must be a mapping.")

    unknown_keys = set(raw_aug.keys()) - set(_MAX_SAFE_AUGMENTATION_BOUNDS.keys())
    if unknown_keys:
        raise AnprTrainingError(
            f"Unsupported augmentation keys in training config: {sorted(unknown_keys)}"
        )

    resolved: Dict[str, float] = dict(DEFAULT_SAFE_AUGMENTATION)
    for key, raw_val in raw_aug.items():
        if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
            raise AnprTrainingError(f"augmentation.{key} must be a finite numeric value.")
        val = float(raw_val)
        if not math.isfinite(val) or val < 0.0:
            raise AnprTrainingError(f"augmentation.{key} must be a non-negative finite number.")
        max_allowed = _MAX_SAFE_AUGMENTATION_BOUNDS[key]
        if val > max_allowed:
            if key in {"fliplr", "flipud"}:
                raise AnprTrainingError(
                    f"augmentation.{key} must be 0.0 for ANPR training to prevent text-mirroring flips."
                )
            raise AnprTrainingError(
                f"augmentation.{key}={val} exceeds safe ANPR plate-text limit ({max_allowed})."
            )
        resolved[key] = val

    return resolved


def _validate_device_setting(raw_device: Any, *, check_hardware: bool = False) -> str:
    if not isinstance(raw_device, str) or not raw_device.strip():
        raise AnprTrainingError("Training device must be a non-empty string ('cpu', 'mps', or 'cuda').")
    device = raw_device.strip().lower()
    if device not in _SUPPORTED_DEVICES:
        raise AnprTrainingError(
            f"Unsupported training device {raw_device!r}. Allowed values: cpu, mps, cuda."
        )
    if check_hardware and device in {"mps", "cuda"}:
        try:
            import torch
        except ImportError as exc:
            raise AnprTrainingError(
                f"PyTorch is required to train on device={device!r}, but torch is not installed."
            ) from exc
        if device == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is None or not mps_backend.is_available():
                raise AnprTrainingError(
                    "Configured training device 'mps' is unavailable in this environment."
                )
        elif device == "cuda" and not torch.cuda.is_available():
            raise AnprTrainingError(
                "Configured training device 'cuda' is unavailable in this environment."
            )
    return device


def _resolve_git_sha(repo_root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        sha = output.decode("utf-8").strip()
        if not re.fullmatch(r"[0-9a-f]{40,64}", sha):
            raise AnprTrainingError("Cannot establish a valid training Git SHA.")
        return sha
    except (OSError, subprocess.SubprocessError) as exc:
        raise AnprTrainingError("Cannot establish training Git provenance.") from exc


def build_training_plan(
    *,
    config_path: Path | str,
    dataset_dir: Path | str,
    run_name_override: Optional[str] = None,
    device_override: Optional[str] = None,
    base_weights_override: Optional[str] = None,
    base_weights_sha256_override: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> AnprTrainingPlan:
    """Validate dataset, config, local base weights, output run path and device without loading YOLO."""
    root = _normalize_repo_root(repo_root)

    # 1. Resolve and verify training config YAML inside repository
    raw_cfg_path = Path(config_path)
    resolved_cfg_path = (
        Path(os.path.abspath(os.fspath(raw_cfg_path)))
        if raw_cfg_path.is_absolute()
        else Path(os.path.abspath(os.fspath(root / raw_cfg_path)))
    )
    _ensure_no_symlink_chain_under_repo(resolved_cfg_path, root, "training config path")
    config_sha256 = _compute_regular_file_sha256(resolved_cfg_path, label="training config file")

    try:
        parsed_cfg = yaml.safe_load(resolved_cfg_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise AnprTrainingError(f"Unable to parse training config YAML: {exc}") from exc

    if not isinstance(parsed_cfg, Mapping):
        raise AnprTrainingError("Training config file must contain a YAML mapping.")

    experiment_cfg = parsed_cfg.get("experiment")
    model_cfg = parsed_cfg.get("model")
    training_cfg = parsed_cfg.get("training")
    if not isinstance(experiment_cfg, Mapping) or not isinstance(model_cfg, Mapping) or not isinstance(training_cfg, Mapping):
        raise AnprTrainingError(
            "Training config must define 'experiment', 'model', and 'training' mappings."
        )

    # 2. Validate run name and ensure output directory under outputs/training/anpr/ does not exist
    raw_run_name = run_name_override if run_name_override is not None else experiment_cfg.get("name")
    if not isinstance(raw_run_name, str) or not _RUN_NAME_RE.fullmatch(raw_run_name.strip()):
        raise AnprTrainingError(
            f"Run name must match {_RUN_NAME_RE.pattern!r}, got {raw_run_name!r}."
        )
    run_name = raw_run_name.strip()
    seed = _validate_positive_int(experiment_cfg.get("seed", 42), "experiment.seed", allow_zero=True)

    output_project_dir = root / "outputs" / "training" / "anpr"
    run_directory = output_project_dir / run_name
    _ensure_no_symlink_chain_under_repo(run_directory, root, "training run directory")
    if run_directory.exists() or run_directory.is_symlink():
        raise AnprTrainingError(
            f"Training run directory already exists: {run_directory}. "
            "Choose a new run name to preserve existing outputs."
        )

    # 3. Validate single-class license_plate model configuration and local base weights
    class_name = model_cfg.get("class_name", REQUIRED_CLASS_NAME)
    if class_name != REQUIRED_CLASS_NAME:
        raise AnprTrainingError(
            f"model.class_name must be exactly {REQUIRED_CLASS_NAME!r}, got {class_name!r}."
        )

    image_size = _validate_positive_int(model_cfg.get("image_size", 640), "model.image_size")

    raw_base_weights = (
        base_weights_override
        if base_weights_override is not None
        else model_cfg.get("base_weights")
    )
    if not isinstance(raw_base_weights, str) or not raw_base_weights.strip():
        raise AnprTrainingError("model.base_weights must be a non-empty repository-relative path under models/.")
    base_weights_rel = raw_base_weights.strip()

    lowered_weights = base_weights_rel.lower()
    if any(token in lowered_weights for token in _FORBIDDEN_BASE_WEIGHT_TOKENS):
        raise AnprTrainingError(
            f"Refusing to substitute pothole weights ({base_weights_rel!r}) for ANPR plate-detector training."
        )

    raw_expected_sha = (
        base_weights_sha256_override
        if base_weights_sha256_override is not None
        else model_cfg.get("base_weights_sha256")
    )
    if not isinstance(raw_expected_sha, str) or not _SHA256_RE.fullmatch(raw_expected_sha.strip()):
        raise AnprTrainingError(
            "model.base_weights_sha256 must be a 64-character lowercase hex SHA-256 digest."
        )
    expected_base_sha256 = raw_expected_sha.strip()

    try:
        verified_base_weights = verify_confinement_under_models(
            base_weights_rel,
            repo_root=root,
            expect_directory=False,
            setting_name="model.base_weights",
        )
    except Exception as exc:
        raise AnprTrainingError(f"Invalid local base weights path: {exc}") from exc

    actual_base_sha256 = _compute_regular_file_sha256(verified_base_weights, label="base weights")
    if actual_base_sha256 != expected_base_sha256:
        raise AnprTrainingError(
            f"Base weights SHA-256 mismatch for {verified_base_weights}: "
            f"expected {expected_base_sha256}, actual {actual_base_sha256}."
        )

    # 4. Validate training hyperparameters & plate-safe augmentations
    epochs = _validate_positive_int(training_cfg.get("epochs"), "training.epochs")
    batch = _validate_positive_int(training_cfg.get("batch"), "training.batch")
    patience = _validate_positive_int(
        training_cfg.get("patience", 15), "training.patience", allow_zero=True
    )
    workers = _validate_positive_int(
        training_cfg.get("workers", 2), "training.workers", allow_zero=True
    )

    raw_device = device_override if device_override is not None else training_cfg.get("device", "cpu")
    device = _validate_device_setting(raw_device, check_hardware=False)
    augmentation = _validate_augmentation_config(parsed_cfg.get("augmentation"))

    # 5. Validate prepared dataset using Codex's canonical validator
    raw_dataset_path = Path(dataset_dir)
    resolved_dataset_dir = (
        Path(os.path.abspath(os.fspath(raw_dataset_path)))
        if raw_dataset_path.is_absolute()
        else Path(os.path.abspath(os.fspath(root / raw_dataset_path)))
    )
    _ensure_no_symlink_chain_under_repo(resolved_dataset_dir, root, "prepared dataset directory")

    dataset_metadata = validate_prepared_dataset(resolved_dataset_dir)
    dataset_fingerprint = dataset_metadata.get("dataset_fingerprint")
    if not isinstance(dataset_fingerprint, str) or not _SHA256_RE.fullmatch(dataset_fingerprint):
        raise AnprTrainingError("Validated dataset metadata did not return a valid 64-character dataset_fingerprint.")

    dataset_yaml_path = resolved_dataset_dir / "anpr.yaml"

    return AnprTrainingPlan(
        repo_root=root,
        config_path=resolved_cfg_path,
        config_sha256=config_sha256,
        dataset_dir=resolved_dataset_dir,
        dataset_yaml_path=dataset_yaml_path,
        dataset_metadata=dataset_metadata,
        dataset_fingerprint=dataset_fingerprint,
        base_weights_relative=base_weights_rel,
        base_weights_path=verified_base_weights,
        base_weights_sha256=actual_base_sha256,
        output_project_dir=output_project_dir,
        run_name=run_name,
        run_directory=run_directory,
        device=device,
        seed=seed,
        image_size=image_size,
        epochs=epochs,
        batch=batch,
        patience=patience,
        workers=workers,
        class_name=class_name,
        augmentation=augmentation,
        raw_config=dict(parsed_cfg),
    )


def _assert_trained_model_is_single_plate_class(model: Any) -> None:
    """Require the same class schema accepted by the deployed plate adapter."""
    try:
        _validate_single_plate_class(model)
    except RuntimeUnavailable as exc:
        raise AnprTrainingError(f"Trained plate detector class verification failed: {exc}") from exc


def _verify_plan_inputs(plan: AnprTrainingPlan) -> None:
    if _compute_regular_file_sha256(plan.config_path, "training config") != plan.config_sha256:
        raise AnprTrainingError("Training configuration changed after preflight.")
    metadata = validate_prepared_dataset(plan.dataset_dir)
    if metadata["dataset_fingerprint"] != plan.dataset_fingerprint:
        raise AnprTrainingError("Prepared ANPR dataset changed after preflight.")


def execute_training_plan(
    plan: AnprTrainingPlan,
    *,
    yolo_loader: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    """Execute dedicated plate-detector training and write verified provenance metadata.

    Trains using local base weights, validates strictly on the ``val`` split,
    and never invokes ``test`` split evaluation.
    """
    if plan.run_directory.exists() or plan.run_directory.is_symlink():
        raise AnprTrainingError(
            f"Training run directory already exists: {plan.run_directory}. "
            "Refusing to overwrite prior training outputs."
        )

    # Re-verify base weights hash right before loading
    current_base_sha = _compute_regular_file_sha256(plan.base_weights_path, label="base weights")
    if current_base_sha != plan.base_weights_sha256:
        raise AnprTrainingError("Base weights changed after preflight validation; aborting training.")

    _verify_plan_inputs(plan)
    training_git_sha = _resolve_git_sha(plan.repo_root)

    _validate_device_setting(plan.device, check_hardware=(yolo_loader is None))

    if yolo_loader is None:
        try:
            from ultralytics import YOLO as _YOLO
        except ImportError as exc:
            raise AnprTrainingError(
                "Ultralytics is required for --execute training, but it is not installed locally."
            ) from exc
        loader: Callable[[str], Any] = _YOLO
    else:
        loader = yolo_loader

    plan.output_project_dir.mkdir(parents=True, exist_ok=True)
    _ensure_no_symlink_chain_under_repo(plan.output_project_dir, plan.repo_root, "output project directory")

    started_wall = datetime.now(timezone.utc).isoformat()
    started_perf = time.perf_counter()

    try:
        model = loader(str(plan.base_weights_path))
    except Exception as exc:
        raise AnprTrainingError(
            f"Unable to initialize YOLO from local base weights {plan.base_weights_path}: {exc}"
        ) from exc

    train_kwargs: Dict[str, Any] = {
        "data": str(plan.dataset_yaml_path),
        "epochs": plan.epochs,
        "batch": plan.batch,
        "patience": plan.patience,
        "device": plan.device,
        "imgsz": plan.image_size,
        "seed": plan.seed,
        "workers": plan.workers,
        "project": str(plan.output_project_dir),
        "name": plan.run_name,
        "exist_ok": False,
        "deterministic": True,
        "val": True,
        "split": "val",
        "pretrained": True,
        # Ultralytics AMP validation can fetch a separate reference checkpoint.
        "amp": False,
        "verbose": False,
        **plan.augmentation,
    }

    try:
        model.train(**train_kwargs)
    except Exception as exc:
        raise AnprTrainingError(f"Plate detector training execution failed: {exc}") from exc

    elapsed_seconds = round(max(0.0, time.perf_counter() - started_perf), 4)
    completed_wall = datetime.now(timezone.utc).isoformat()

    trainer = getattr(model, "trainer", None)
    save_dir_attr = getattr(trainer, "save_dir", None) if trainer is not None else None
    actual_run_dir = (
        Path(os.path.abspath(os.fspath(save_dir_attr)))
        if save_dir_attr is not None
        else plan.run_directory
    )
    if actual_run_dir != plan.run_directory:
        raise AnprTrainingError(
            f"Trainer save_dir ({actual_run_dir}) did not match expected run directory ({plan.run_directory})."
        )
    if not actual_run_dir.is_dir() or actual_run_dir.is_symlink():
        raise AnprTrainingError(
            f"Expected training output directory was not created as a real directory: {actual_run_dir}"
        )

    _assert_trained_model_is_single_plate_class(model)
    _verify_plan_inputs(plan)

    # Copy config and dataset preparation metadata into the run directory before hashing all required artifacts
    copied_config_path = actual_run_dir / "train_config.yaml"
    copied_prep_meta_path = actual_run_dir / "dataset_preparation_metadata.json"
    source_prep_meta_path = plan.dataset_dir / "preparation_metadata.json"

    shutil.copy2(plan.config_path, copied_config_path)
    shutil.copy2(source_prep_meta_path, copied_prep_meta_path)

    artifact_paths: Dict[str, Path] = {
        "base_weights": plan.base_weights_path,
        "weights/best.pt": actual_run_dir / "weights" / "best.pt",
        "weights/last.pt": actual_run_dir / "weights" / "last.pt",
        "results.csv": actual_run_dir / "results.csv",
        "args.yaml": actual_run_dir / "args.yaml",
        "train_config.yaml": copied_config_path,
        "dataset_preparation_metadata.json": copied_prep_meta_path,
    }

    artifact_hashes: Dict[str, str] = {}
    for key in REQUIRED_TRAINING_ARTIFACTS:
        target_path = artifact_paths[key]
        _ensure_no_symlink_chain_under_repo(target_path, plan.repo_root, f"training artifact {key}")
        artifact_hashes[key] = _compute_regular_file_sha256(target_path, label=f"training artifact '{key}'")

    def installed_version(package: str) -> str:
        try:
            return version(package)
        except PackageNotFoundError:
            return "not_installed"

    metadata: Dict[str, Any] = {
        "schema_version": "1.0",
        "task": "detection",
        "domain": "anpr_license_plate",
        "class_mapping": {"0": REQUIRED_CLASS_NAME},
        "run_id": plan.run_name,
        "run_directory": str(actual_run_dir),
        "git_sha": training_git_sha,
        "dataset_dir": str(plan.dataset_dir),
        "dataset_yaml_path": str(plan.dataset_yaml_path),
        "dataset_fingerprint": plan.dataset_fingerprint,
        "group_policy": plan.dataset_metadata.get("group_policy"),
        "split_counts": plan.dataset_metadata.get("split_counts"),
        "evidence_assurance": plan.dataset_metadata.get("evidence_assurance"),
        "validation_split": "val",
        "test_evaluation_invoked": False,
        "base_weights": plan.base_weights_relative,
        "resolved_base_weights": str(plan.base_weights_path),
        "base_weights_sha256": plan.base_weights_sha256,
        "config_path": str(plan.config_path),
        "config_sha256": plan.config_sha256,
        "device": plan.device,
        "seed": plan.seed,
        "hyperparameters": {
            "image_size": plan.image_size,
            "epochs": plan.epochs,
            "batch": plan.batch,
            "patience": plan.patience,
            "workers": plan.workers,
        },
        "augmentation": dict(plan.augmentation),
        "started_at": started_wall,
        "completed_at": completed_wall,
        "elapsed_seconds": elapsed_seconds,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": installed_version("torch"),
            "ultralytics": installed_version("ultralytics"),
        },
        "artifacts": artifact_hashes,
    }

    metadata_path = actual_run_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def run_anpr_detector_training(
    *,
    config_path: Path | str,
    dataset_dir: Path | str,
    execute: bool = False,
    run_name_override: Optional[str] = None,
    device_override: Optional[str] = None,
    base_weights_override: Optional[str] = None,
    base_weights_sha256_override: Optional[str] = None,
    repo_root: Optional[Path] = None,
    yolo_loader: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    """Build and validate the training plan; execute training only when ``execute=True``."""
    plan = build_training_plan(
        config_path=config_path,
        dataset_dir=dataset_dir,
        run_name_override=run_name_override,
        device_override=device_override,
        base_weights_override=base_weights_override,
        base_weights_sha256_override=base_weights_sha256_override,
        repo_root=repo_root,
    )
    if not execute:
        return plan.to_summary_dict()
    return execute_training_plan(plan, yolo_loader=yolo_loader)
