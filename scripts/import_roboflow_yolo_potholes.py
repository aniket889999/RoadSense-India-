import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
from src.ml.roboflow_yolo_import import run_roboflow_yolo_import

def main():
    parser = argparse.ArgumentParser(
        description="Import Roboflow-exported RDD2022 YOLO dataset into a leak-safe, contiguous-sequence D40 pothole dataset."
    )
    parser.add_argument("--source-root", required=True, help="Path to Roboflow YOLO dataset root containing data.yaml and train/valid/test folders")
    parser.add_argument("--output-dir", required=True, help="Path to destination YOLO dataset directory under data/processed/")
    parser.add_argument("--max-consecutive-gap", type=int, default=1, help="Max numeric gap between frame IDs to consider part of the same contiguous sequence run (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for group splitting (default: 42)")
    parser.add_argument("--dry-run", action="store_true", help="Run full preflight, contiguous-run grouping, and split calculation without writing dataset files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory via atomic backup/promotion")
    args = parser.parse_args()

    success, summary, err = run_roboflow_yolo_import(
        source_root=args.source_root,
        output_dir=args.output_dir,
        max_consecutive_gap=args.max_consecutive_gap,
        seed=args.seed,
        dry_run=args.dry_run,
        overwrite=args.overwrite
    )

    if not success:
        print(f"Import Failed: {err}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("=== DRY RUN SUMMARY ===")
        print(f"Source Root: {summary['source_root']}")
        print(f"Proposed Output Directory: {summary['output_dir']}")
        print(f"Dataset Fingerprint: {summary['fingerprint']}")
        print("Counts & Split Distribution:")
        print(json.dumps(summary["counts"], indent=2))
        print("Dry run completed successfully. No dataset files were written.")
    else:
        print(f"Import completed successfully. Dataset created at: {summary['output_dir']}")
        print(f"Dataset Fingerprint: {summary['fingerprint']}")

if __name__ == "__main__":
    main()
