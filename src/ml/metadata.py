import json
import os
from typing import Dict, Any

def write_metadata_json(output_dir: str, filename: str, metadata: Dict[str, Any]):
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

def build_training_metadata(
    git_sha: str,
    prep_meta: Dict[str, Any],
    cfg: Dict[str, Any],
    run_dir: str,
    artifacts: Dict[str, str],
    env: Dict[str, str]
) -> Dict[str, Any]:
    return {
        "task": "detection",
        "class_mapping": {0: "pothole"},
        "rdd2022_reference": "Arya et al. (2024), DOI: 10.1002/gdj3.260",
        "dataset_reference": "DOI: 10.6084/m9.figshare.21431547.v1",
        "dataset_fingerprint": prep_meta.get("dataset_fingerprint", "Unknown"),
        "grouping_method": prep_meta.get("grouping_method", "Unknown"),
        "residual_leakage_limitation": prep_meta.get("limitation") or prep_meta.get("residual_leakage_limitation", "Unknown"),
        "split_distribution": prep_meta.get("counts", {}).get("split_distribution", {}),
        "split_counts": prep_meta.get("counts", {}),
        "base_weights": cfg["model"]["base_weights"],
        "fine_tuned": True,
        "seed": cfg["experiment"]["seed"],
        "hyperparameters": {
            "experiment": cfg["experiment"],
            "model": cfg["model"],
            "training": cfg["training"]
        },
        "run_directory": run_dir,
        "timestamp": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        "git_sha": git_sha,
        "environment": env,
        "artifacts": artifacts,
        "limitations": "Model not yet connected to UI. Requires external evaluation on novel real-world sequences."
    }
