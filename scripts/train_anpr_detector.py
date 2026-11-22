"""CLI wrapper for dedicated single-class YOLO license-plate detector training.

Defaults to ``--dry-run`` validation if ``--execute`` is not supplied. Never
downloads weights or invokes held-out ``test`` split evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anpr.dataset import DatasetValidationError
from src.anpr.training import AnprTrainingError, run_anpr_detector_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or train a dedicated single-class (license_plate) YOLO detector "
            "using verified local base weights and a prepared ANPR dataset."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/training/anpr_plate_detector.yaml",
        help="Repository-relative or absolute path to ANPR detector training YAML config.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to prepared ANPR YOLO dataset directory containing anpr.yaml and manifest.jsonl.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional override for experiment.name (creates outputs/training/anpr/<run-name>).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override ('cpu', 'mps', or 'cuda').",
    )
    parser.add_argument(
        "--base-weights",
        default=None,
        help="Optional repository-relative path under models/ overriding model.base_weights.",
    )
    parser.add_argument(
        "--base-weights-sha256",
        default=None,
        help="Optional 64-character lowercase SHA-256 hex digest overriding model.base_weights_sha256.",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset, config, local base weights, output directory and device without loading YOLO.",
    )
    mode_group.add_argument(
        "--execute",
        action="store_true",
        help="Run actual YOLO detector training on the train/val splits and write model_metadata.json.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    execute_mode = bool(args.execute)
    try:
        summary = run_anpr_detector_training(
            config_path=args.config,
            dataset_dir=args.dataset,
            execute=execute_mode,
            run_name_override=args.run_name,
            device_override=args.device,
            base_weights_override=args.base_weights,
            base_weights_sha256_override=args.base_weights_sha256,
        )
    except (AnprTrainingError, DatasetValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
