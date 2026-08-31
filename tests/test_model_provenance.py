import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.ml.model_provenance import (
    FrozenBaselineVerificationError,
    load_frozen_baseline_config,
    verify_frozen_baseline,
)


RUN_ID = "frozen_test_run"
REQUIRED_ARTIFACT_KEYS = (
    "base_weights",
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "train_config.yaml",
    "dataset_preparation_metadata.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_frozen_repository(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo_root = tmp_path / "repo"
    config_path = repo_root / "configs" / "inference" / "frozen_baseline.yaml"
    run_dir = repo_root / "outputs" / "training" / RUN_ID
    weights_dir = run_dir / "weights"
    base_weights = repo_root / "models" / "yolov8n.pt"
    weights_dir.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    base_weights.parent.mkdir(parents=True)

    base_weights.write_bytes(b"base weights")
    (weights_dir / "best.pt").write_bytes(b"best weights")
    (weights_dir / "last.pt").write_bytes(b"last weights")
    (run_dir / "results.csv").write_text("epoch,loss\n1,0.2\n", encoding="utf-8")
    (run_dir / "args.yaml").write_text("epochs: 50\n", encoding="utf-8")
    (run_dir / "train_config.yaml").write_text("model: yolov8n\n", encoding="utf-8")

    fingerprint = hashlib.sha256(b"dataset fingerprint").hexdigest()
    copied_prep_meta = run_dir / "dataset_preparation_metadata.json"
    copied_prep_meta.write_text(
        json.dumps({"dataset_fingerprint": fingerprint}), encoding="utf-8"
    )

    artifacts = {
        "base_weights": sha256(base_weights),
        "weights/best.pt": sha256(weights_dir / "best.pt"),
        "weights/last.pt": sha256(weights_dir / "last.pt"),
        "results.csv": sha256(run_dir / "results.csv"),
        "args.yaml": sha256(run_dir / "args.yaml"),
        "train_config.yaml": sha256(run_dir / "train_config.yaml"),
        "dataset_preparation_metadata.json": sha256(copied_prep_meta),
    }
    assert tuple(artifacts) == REQUIRED_ARTIFACT_KEYS

    git_sha = "1" * 40
    metadata = {
        "task": "detection",
        "class_mapping": {"0": "pothole"},
        "git_sha": git_sha,
        "dataset_fingerprint": fingerprint,
        "run_directory": str(run_dir),
        "base_weights": "models/yolov8n.pt",
        "resolved_base_weights": str(base_weights),
        "artifacts": artifacts,
    }
    metadata_path = run_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    frozen_config = {
        "schema_version": 1,
        "frozen_baseline": {
            "run_id": RUN_ID,
            "run_directory": f"outputs/training/{RUN_ID}",
            "checkpoint_path": f"outputs/training/{RUN_ID}/weights/best.pt",
            "checkpoint_sha256": artifacts["weights/best.pt"],
            "model_metadata_path": f"outputs/training/{RUN_ID}/model_metadata.json",
            "model_metadata_sha256": sha256(metadata_path),
            "base_weights_path": "models/yolov8n.pt",
            "task": "detection",
            "class_mapping": {"0": "pothole"},
            "git_sha": git_sha,
            "dataset_fingerprint": fingerprint,
            "artifact_hashes": artifacts,
        },
    }
    config_path.write_text(yaml.safe_dump(frozen_config, sort_keys=False), encoding="utf-8")
    return repo_root, config_path, run_dir, metadata_path


def test_frozen_provenance_passes_without_model_loading(tmp_path):
    repo_root, config_path, run_dir, metadata_path = make_frozen_repository(tmp_path)

    config = load_frozen_baseline_config(config_path, repo_root=repo_root)
    info = verify_frozen_baseline(config, repo_root=repo_root)

    assert info.run_id == RUN_ID
    assert info.training_run_directory == run_dir
    assert info.checkpoint_path == run_dir / "weights" / "best.pt"
    assert info.model_metadata_path == metadata_path
    assert info.task == "detection"
    assert info.class_mapping == {"0": "pothole"}
    assert info.base_weights_path == repo_root / "models" / "yolov8n.pt"


def test_altered_checkpoint_is_rejected(tmp_path):
    repo_root, config_path, run_dir, _ = make_frozen_repository(tmp_path)
    config = load_frozen_baseline_config(config_path, repo_root=repo_root)
    (run_dir / "weights" / "best.pt").write_bytes(b"altered checkpoint")

    with pytest.raises(FrozenBaselineVerificationError, match="checkpoint SHA-256 mismatch"):
        verify_frozen_baseline(config, repo_root=repo_root)


def test_altered_model_metadata_is_rejected(tmp_path):
    repo_root, config_path, _, metadata_path = make_frozen_repository(tmp_path)
    config = load_frozen_baseline_config(config_path, repo_root=repo_root)
    metadata_path.write_text('{"altered": true}', encoding="utf-8")

    with pytest.raises(FrozenBaselineVerificationError, match="metadata SHA-256 mismatch"):
        verify_frozen_baseline(config, repo_root=repo_root)


def test_external_checkpoint_path_is_rejected_during_config_load(tmp_path):
    repo_root, config_path, _, _ = make_frozen_repository(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["frozen_baseline"]["checkpoint_path"] = str(tmp_path / "outside" / "best.pt")
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(FrozenBaselineVerificationError, match="inside the repository root|checkpoint_path must be exactly"):
        load_frozen_baseline_config(config_path, repo_root=repo_root)


def test_symlinked_weights_directory_is_rejected(tmp_path):
    repo_root, config_path, run_dir, _ = make_frozen_repository(tmp_path)
    config = load_frozen_baseline_config(config_path, repo_root=repo_root)

    weights_dir = run_dir / "weights"
    outside_weights = tmp_path / "outside_weights"
    weights_dir.rename(outside_weights)
    weights_dir.symlink_to(outside_weights, target_is_directory=True)

    with pytest.raises(FrozenBaselineVerificationError, match="symlink"):
        verify_frozen_baseline(config, repo_root=repo_root)


def test_non_checkpoint_artifact_mismatch_is_rejected(tmp_path):
    repo_root, config_path, run_dir, _ = make_frozen_repository(tmp_path)
    config = load_frozen_baseline_config(config_path, repo_root=repo_root)
    (run_dir / "results.csv").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(FrozenBaselineVerificationError, match="Artifact SHA-256 mismatch for 'results.csv'"):
        verify_frozen_baseline(config, repo_root=repo_root)


def test_metadata_semantics_are_checked_after_metadata_hash_is_re_pinned(tmp_path):
    repo_root, config_path, _, metadata_path = make_frozen_repository(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["task"] = "classification"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    payload["frozen_baseline"]["model_metadata_sha256"] = sha256(metadata_path)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = load_frozen_baseline_config(config_path, repo_root=repo_root)
    with pytest.raises(FrozenBaselineVerificationError, match="model_metadata.task"):
        verify_frozen_baseline(config, repo_root=repo_root)
