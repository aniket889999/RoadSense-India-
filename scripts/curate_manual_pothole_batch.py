"""Create a local, human-confirmed curation pool from a RoadSense annotation kit.

This command does not train a model, run inference, modify the frozen RDD2022
dataset, or use any held-out test split.  It creates a separate local batch
only after the caller explicitly supplies ``--write``.
"""

import sys
import os
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import json

from src.ml.manual_curation import prepare_manual_curation_batch


def _exit(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _read_regular_local_file(path_value: str, *, label: str) -> bytes:
    path = Path(path_value).expanduser()
    try:
        info = os.lstat(path)
    except OSError as exc:
        _exit(f"{label} is missing or unreadable: {exc}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        _exit(f"{label} must be a regular local file, not a symlink or directory.")
    try:
        return path.read_bytes()
    except OSError as exc:
        _exit(f"Unable to read {label}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a local-only human-confirmed pothole curation pool from an annotation kit. "
            "It never trains or evaluates a model."
        )
    )
    parser.add_argument("--annotation-kit", required=True, help="Path to a RoadSense annotation_kit.zip")
    parser.add_argument("--annotations", required=True, help="Path to a completed strict manual annotations CSV")
    parser.add_argument(
        "--frame-review",
        required=True,
        help="Path to strict frame review CSV: frame_index,review_status,note",
    )
    parser.add_argument(
        "--recording-id",
        required=True,
        help="Safe local recording group ID; all exported frames remain one future split group.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Exact new output: data/interim/manual_curation/<recording-id>",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate and summarize only; write nothing.")
    mode.add_argument("--write", action="store_true", help="Explicitly create the local curation batch.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing batch only with --write, using guarded atomic promotion.",
    )
    args = parser.parse_args()

    if args.overwrite and not args.write:
        _exit("--overwrite is valid only together with --write.")

    annotation_kit = _read_regular_local_file(args.annotation_kit, label="annotation kit")
    annotations = _read_regular_local_file(args.annotations, label="manual annotations CSV")
    frame_review = _read_regular_local_file(args.frame_review, label="frame review CSV")

    summary, errors = prepare_manual_curation_batch(
        annotation_kit,
        annotations,
        frame_review,
        args.recording_id,
        output_dir=args.output_dir,
        write=args.write,
        overwrite=args.overwrite,
        repo_root=REPO_ROOT,
    )
    if errors:
        for error in errors:
            print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)

    if args.dry_run:
        print("Manual curation dry run passed. No files were written.")
    else:
        print("Manual curation batch created locally.")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("No model training, inference, frozen-dataset modification, or held-out test evaluation occurred.")


if __name__ == "__main__":
    main()
