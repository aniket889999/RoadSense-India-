import sys, os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ml.metadata import build_training_metadata
import datetime

def test_build_training_metadata_exact_fields():
    prep_meta = {
        "dataset_fingerprint": "abc",
        "grouping_method": "xyz",
        "limitation": "foo",
        "counts": {
            "split_distribution": {"train": 0.7, "val": 0.15, "test": 0.15},
            "train": {"images": 10, "positive_images": 5, "d40_instances": 8},
            "val": {"images": 2, "positive_images": 1, "d40_instances": 1},
            "test": {"images": 2, "positive_images": 1, "d40_instances": 1}
        }
    }

    cfg = {
        "model": {"base_weights": "yolov8n.pt"},
        "experiment": {"seed": 42},
        "training": {"epochs": 1}
    }

    artifacts = {
        "weights/best.pt": "hash1",
        "train_config.yaml": "hash2",
        "dataset_preparation_metadata.json": "hash3"
    }

    env = {"python": "3.10", "platform": "macOS", "torch": "2.0", "ultralytics": "8.0"}

    out = build_training_metadata("commit123", prep_meta, cfg, "/run", artifacts, env)

    assert out["rdd2022_reference"] == "Arya et al. (2024), DOI: 10.1002/gdj3.260"
    assert out["dataset_reference"] == "DOI: 10.6084/m9.figshare.21431547.v1"
    assert out["dataset_fingerprint"] == "abc"
    assert out["grouping_method"] == "xyz"
    assert out["residual_leakage_limitation"] == "foo"

    assert out["split_counts"]["train"]["images"] == 10

    assert out["hyperparameters"]["model"]["base_weights"] == "yolov8n.pt"
    assert out["git_sha"] == "commit123"
    assert out["environment"]["python"] == "3.10"
    assert out["run_directory"] == "/run"
    assert out["artifacts"]["weights/best.pt"] == "hash1"
    assert out["artifacts"]["train_config.yaml"] == "hash2"
