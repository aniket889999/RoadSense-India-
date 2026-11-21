"""CLI for auditing the Witvix ANPR dataset and writing an interim intake manifest.

Defaults to ``--dry-run`` (read-only audit, no writes). Writing the interim
manifest requires ``--execute`` alongside both explicit provenance
acknowledgement flags:
``--accept-publisher-labels`` and ``--accept-vehicle-proxy-groups``.
Never prints plate registration numbers in its report output.
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
from src.anpr.witvix_intake import (
    _safe_interim_manifest_path,
    build_witvix_records,
    write_witvix_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a Witvix LPD+LPR source directory and optionally write an "
            "interim JSONL manifest under repository data/interim/."
        )
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Path to the read-only Witvix archive directory.",
    )
    parser.add_argument(
        "--output-manifest",
        default="data/interim/witvix_anpr_manifest.jsonl",
        help="Repository-relative or absolute path under data/interim/ for the output .jsonl manifest.",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source files and print aggregate report without writing any files (default).",
    )
    mode_group.add_argument(
        "--execute",
        action="store_true",
        help="Write the validated interim manifest under data/interim/ (requires both acknowledgement flags).",
    )

    parser.add_argument(
        "--accept-publisher-labels",
        action="store_true",
        help=(
            "Acknowledge that plate labels have publisher_mixed provenance "
            "(58%% human checked without per-row flags)."
        ),
    )
    parser.add_argument(
        "--accept-vehicle-proxy-groups",
        action="store_true",
        help=(
            "Acknowledge that grouping uses publisher_vehicle_proxy because "
            "capture-session IDs are not supplied."
        ),
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    repo_root: Optional[Path] = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    execute_mode = bool(args.execute)

    try:
        if execute_mode:
            missing_flags = []
            if not args.accept_publisher_labels:
                missing_flags.append("--accept-publisher-labels")
            if not args.accept_vehicle_proxy_groups:
                missing_flags.append("--accept-vehicle-proxy-groups")
            if missing_flags:
                raise DatasetValidationError(
                    "--execute requires explicit provenance acknowledgement flags: "
                    + ", ".join(missing_flags)
                )
            # Validate target output path before scanning source files
            target_path = _safe_interim_manifest_path(args.output_manifest, repo_root=repo_root)
        else:
            target_path = _safe_interim_manifest_path(args.output_manifest, repo_root=repo_root)

        records, report = build_witvix_records(Path(args.source_root))

        if execute_mode:
            written_path, manifest_sha256 = write_witvix_manifest(
                records,
                target_path,
                repo_root=repo_root,
            )
            report["mode"] = "execute"
            report["output_manifest"] = str(written_path)
            report["output_manifest_sha256"] = manifest_sha256
        else:
            report["mode"] = "dry_run"
            report["output_manifest"] = str(target_path)
            report["output_manifest_written"] = False
    except DatasetValidationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
