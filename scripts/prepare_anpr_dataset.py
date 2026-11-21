"""Audit a labelled ANPR manifest and optionally prepare grouped YOLO splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anpr.dataset import DatasetValidationError, prepare_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Audit only; this is the default.")
    mode.add_argument("--execute", action="store_true", help="Create a new prepared dataset.")
    args = parser.parse_args()
    try:
        metadata = prepare_dataset(
            args.manifest, args.source_root, args.output_dir, seed=args.seed, dry_run=not args.execute,
        )
    except (DatasetValidationError, OSError) as exc:
        print(f"ANPR dataset preparation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"dry_run": not args.execute, **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
